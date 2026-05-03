"""
Expert Parameter Transfer Module

This module implements efficient cross-node expert parameter transfer functionality, supporting:
1. All-Reduce communication strategy
2. Point-to-point communication strategy
3. Parameter compression and serialization
4. Error handling and retry mechanisms
5. TEGroupedMLP specific parameter handling
6. Megatron parallel_state integration
"""

import logging
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter
import zlib
import os
from megatron.training import print_rank_0

try:
    from megatron.core import parallel_state
    from megatron.core.transformer.moe.moe_utils import ModelCommProcessGroups
    HAVE_MEGATRON = True
except ImportError:
    HAVE_MEGATRON = False

logger = logging.getLogger(__name__)


class ExpertParameterTransfer:
    """
    Expert Parameter Transfer Manager
    
    Responsible for managing expert parameter transfer between nodes, supporting multiple communication strategies
    """
    
    def __init__(self, 
                 use_allreduce: bool = True,
                 compression_ratio: float = 0.8,
                 max_retries: int = 3,
                 timeout: float = 30.0,
                 verbose: bool = True,
                 expert_type: str = 'TEGroupedMLP',
                 model_comm_pgs: Optional[ModelCommProcessGroups] = None):
        """
        Initialize expert parameter transfer manager
        
        Args:
            use_allreduce: Whether to use All-Reduce communication strategy
            compression_ratio: Parameter compression ratio
            max_retries: Maximum number of retries
            timeout: Timeout in seconds
            verbose: Whether to output detailed logs
            expert_type: Type of expert implementation ('TEGroupedMLP', 'GroupedMLP', 'SequentialMLP')
            model_comm_pgs: Megatron communication group manager
        """
        self.use_allreduce = use_allreduce
        self.compression_ratio = compression_ratio
        self.max_retries = max_retries
        self.timeout = timeout
        self.verbose = verbose
        self.expert_type = expert_type
        self.model_comm_pgs = model_comm_pgs
        
        # Communication group information
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = f"cuda:{self.local_rank}"
        
        # Initialize expert parallel communication groups
        self._setup_expert_parallel_groups()
        
        if self.verbose:
            print_rank_0(f"Expert parameter transfer manager initialized")
            print_rank_0(f"Expert type: {expert_type}")
            print_rank_0(f"Communication strategy: {'All-Reduce' if use_allreduce else 'Point-to-point'}")
            print_rank_0(f"Compression ratio: {compression_ratio}")
            print_rank_0(f"Maximum retries: {max_retries}")
    
    def _setup_expert_parallel_groups(self):
        """Setup expert parallel communication groups"""
        self.ep_group = None
        self.ep_tp_group = None
        self.ep_dp_group = None
        
        assert HAVE_MEGATRON, "Megatron not available"
            
        if self.model_comm_pgs:
            # Use provided communication groups
            self.ep_group = self.model_comm_pgs.ep
            self.ep_tp_group = self.model_comm_pgs.expt_tp
            self.ep_dp_group = self.model_comm_pgs.expt_dp
            
        if self.verbose and self.ep_group:
            ep_size = self.ep_group.size()
            ep_rank = self.ep_group.rank()
            logger.info(f"Expert parallel groups initialized: EP size={ep_size}, EP rank={ep_rank}")
                
    def _serialize_single_expert_parameters(self, expert_params: Dict[str, torch.Tensor]) -> bytes:
        """Serialize single expert parameters"""
        # Convert parameters to serializable format
        serializable_params = {}
        for param_name, param_tensor in expert_params.items():
            serializable_params[param_name] = {
                    'data': param_tensor.detach().cpu().to(torch.float32).numpy(),
                    'shape': param_tensor.shape,
                    'dtype': str(param_tensor.dtype)
                }
        
        # Use pickle for serialization
        return pickle.dumps(serializable_params)
    
    def _deserialize_single_expert_parameters(self, serialized_data: bytes) -> Dict[str, torch.Tensor]:
        """Deserialize single expert parameters"""
        serializable_params = pickle.loads(serialized_data)
        
        expert_params = {}
        for param_name, param_info in serializable_params.items():
            param_type_str = param_info['dtype'].split('.')[1]
            param_type = getattr(torch, param_type_str)
            tensor = torch.from_numpy(param_info['data']).to(param_type).to(self.device)
            expert_params[param_name] = tensor
        
        return expert_params
    
    def _compress_parameters(self, serialized_data: bytes) -> bytes:
        """Compress parameter data"""
        compressed_data = zlib.compress(serialized_data, level=int(self.compression_ratio * 9))
        
        if self.verbose:
            original_size = len(serialized_data)
            compressed_size = len(compressed_data)
            compression_ratio = compressed_size / original_size
            logger.info(f"Parameter compression completed: {original_size} -> {compressed_size} bytes (compression ratio: {compression_ratio:.2f})")
        
        return compressed_data
    
    def _decompress_parameters(self, compressed_data: bytes) -> bytes:
        """Decompress parameter data"""
        return zlib.decompress(compressed_data)
    
    def _send_parameters_to_rank(self, 
                               expert_params: Dict[str, torch.Tensor],
                               target_rank: int,
                               expert_id) -> bool:
        """Send parameters to specified rank"""
        for retry in range(self.max_retries):
            try:
                # Serialize parameters
                serialized_params = self._serialize_single_expert_parameters(expert_params)
                compressed_params = self._compress_parameters(serialized_params)

                
                # Send data length
                data_length = torch.tensor(len(compressed_params), dtype=torch.long, device=self.device)
                dist.isend(data_length, dst=target_rank, tag=expert_id)
                
                # Send parameter data
                param_tensor = torch.frombuffer(compressed_params, dtype=torch.uint8).clone().to(self.device)
                dist.isend(param_tensor, dst=target_rank, tag=expert_id)
                
                if self.verbose:
                    logger.info(f"Successfully sent parameters to rank {target_rank}")

                break
                
            except Exception as e:
                logger.warning(f"Failed to send parameters to rank {target_rank} (retry {retry + 1}/{self.max_retries}): {str(e)}")
                if retry < self.max_retries - 1:
                    time.sleep(1.0)  # Wait 1 second before retry
        
        return False
    
    def _receive_parameters_from_rank(self, source_rank: int, expert_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """Receive parameters from specified rank"""
        try:
            # Receive data length
            data_length = torch.tensor(0, dtype=torch.long, device=self.device)
            dist.irecv(data_length, src=source_rank, tag=expert_id)
            
            # Receive parameter data
            param_tensor = torch.empty(data_length.item(), dtype=torch.uint8, device=self.device)
            dist.irecv(param_tensor, src=source_rank, tag=expert_id)
            
            # Decompress and deserialize
            compressed_data = param_tensor.cpu().numpy().tobytes()
            decompressed_data = self._decompress_parameters(compressed_data)
            expert_params = self._deserialize_single_expert_parameters(decompressed_data)
            
            if self.verbose:
                logger.info(f"Successfully received parameters from rank {source_rank}")
            
            return expert_params
            
        except Exception as e:
            logger.error(f"Failed to receive parameters from rank {source_rank}: {str(e)}")
            return None
    
    def get_local_expert_index(self, expert_id: int, current_expert_assignments: List[Dict[str, Any]]) -> int:
        """
        Get the local index of an expert based on current expert assignments
        
        Args:
            expert_id: Global expert ID
            current_expert_assignments: Current expert assignments
            
        Returns:
            Local index of the expert, or -1 if not found
        """
        for idx, assignment in enumerate(current_expert_assignments):
            if assignment['expert_id'] == expert_id:
                return idx
        return -1  # Expert not found in current assignments
    
    def extract_mlp_parameters(self, model: torch.nn.Module, expert_id: int, current_expert_assignments: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Extract MLP expert parameters from model based on current expert assignments
        
        Args:
            model: Model containing expert layers
            expert_id: Global expert ID to extract
            current_expert_assignments: Current expert assignments for this node
            
        Returns:
            Dictionary of expert parameters
        """
        expert_params = {}
        
        # Get local index of the expert
        local_expert_idx = self.get_local_expert_index(expert_id, current_expert_assignments)
        assert local_expert_idx >= 0, f"Expert {expert_id} not found in current assignments"
        
        # Iterate through all MoE layers in the model
        # NOTE Only support TEGroupedMLP now
        for name, module in model.named_modules():
            if hasattr(module, 'experts'):
                experts_module = module.experts
                
                if hasattr(experts_module, 'linear_fc1') and hasattr(experts_module, 'linear_fc2'):
                    # TEGroupedMLP implementation
                    # Extract parameters from linear_fc1
                    fc1_params = dict(experts_module.linear_fc1.named_parameters())
                    for param_name, param_tensor in fc1_params.items():
                        if param_name.startswith('weight') and param_name[6:].isdigit():
                            param_expert_idx = int(param_name[6:])
                            if param_expert_idx == local_expert_idx:
                                expert_params[f"{name}.linear_fc1.{param_name}"] = param_tensor.clone()
                                break
                    
                    # Extract parameters from linear_fc2
                    fc2_params = dict(experts_module.linear_fc2.named_parameters())
                    for param_name, param_tensor in fc2_params.items():
                        if param_name.startswith('weight') and param_name[6:].isdigit():
                            param_expert_idx = int(param_name[6:])
                            if param_expert_idx == local_expert_idx:
                                expert_params[f"{name}.linear_fc2.{param_name}"] = param_tensor.clone()
                                break
                    
                    if self.verbose:
                        logger.info(f"Extracted TEGroupedMLP parameters for expert {expert_id} (local_idx={local_expert_idx}) from layer {name}")
        
        return expert_params
    
    def apply_mlp_parameters(self, 
                            model: torch.nn.Module, 
                            expert_id: int, 
                            expert_params: Dict[str, torch.Tensor],
                            migration_map: Dict[int, int],
                            current_expert_assignments: List[Dict[str, Any]]) -> None:
        """
        Apply MLP expert parameters to model based on current expert assignments
        
        Args:
            model: Model containing expert layers
            expert_id: Global expert ID to apply
            expert_params: Expert parameters to apply
            migration_map: Mapping add_expert_id to removed_expert_id
            current_expert_assignments: Current expert assignments for this node
        """
        # Get local index of the expert
        removed_expert_id = migration_map[expert_id]
        local_expert_idx = self.get_local_expert_index(removed_expert_id, current_expert_assignments) 
        postfix = list(expert_params.keys())[0].split('.')[-1]
        
        # Iterate through all MoE layers in the model
        for name, module in model.named_modules():
            if hasattr(module, 'experts'):
                experts_module = module.experts
                
                if hasattr(experts_module, 'linear_fc1') and hasattr(experts_module, 'linear_fc2'):
                    # TEGroupedMLP implementation
                    # Apply parameters to linear_fc1
                    fc1_params = dict(experts_module.linear_fc1.named_parameters())
                    for param_name, param_tensor in fc1_params.items():
                        param_expert_idx = int(param_name[6:])
                        if param_expert_idx == local_expert_idx:
                            full_param_name = f"{name}.linear_fc1.{postfix}"
                            expert_param = expert_params[full_param_name]
                            param_tensor.data.copy_(expert_param)
                            break
                    
                    # Apply parameters to linear_fc2
                    fc2_params = dict(experts_module.linear_fc2.named_parameters())
                    for param_name, param_tensor in fc2_params.items():
                        param_expert_idx = int(param_name[6:])
                        if param_expert_idx == local_expert_idx:
                            full_param_name = f"{name}.linear_fc2.{postfix}"
                            expert_param = expert_params[full_param_name]
                            param_tensor.data.copy_(expert_param)
                            break
                    
                    if self.verbose:
                        logger.info(f"Applied TEGroupedMLP parameters for expert {expert_id} (local_idx={local_expert_idx}) to layer {name}")
    
    def migrate_expert_parameters(self, 
                                expert_id: int, 
                                src_rank: int, 
                                dst_rank: int,
                                expert_params: Dict[str, torch.Tensor] = None) -> Optional[Dict[str, torch.Tensor]]:
        """
        Migrate a single expert's parameters from src_rank to dst_rank
        Uses point-to-point communication for direct transfers
        """
        current_rank = dist.get_rank()
        
        if current_rank == src_rank:
            # Source rank: send parameters
            self._send_parameters_to_rank(expert_params, dst_rank, expert_id)
            if self.verbose:
                logger.info(f"Sent expert {expert_id} parameters to rank {dst_rank}")

            return None
            
        elif current_rank == dst_rank:
            # Destination rank: receive parameters
            received_params = self._receive_parameters_from_rank(src_rank, expert_id)
            if self.verbose:
                logger.info(f"Received expert {expert_id} parameters from rank {src_rank}")

            return received_params
            
        else:
            # Other ranks: no action needed
            return None
        
            
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """Get transfer statistics"""
        return {
            'expert_type': self.expert_type,
            'compression_ratio': self.compression_ratio,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'ep_group_size': self.ep_group.size() if self.ep_group else 0,
            'ep_group_rank': self.ep_group.rank() if self.ep_group else 0
        }
