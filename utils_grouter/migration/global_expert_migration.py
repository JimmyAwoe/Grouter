"""
Global Expert Migration Module

This module implements global expert migration functionality in Megatron-LM framework, supporting:
1. Expert parameter migration based on pre-computed expert-node mapping configuration
2. Cross-node expert parameter redistribution
3. Efficient communication strategies (All-Reduce and point-to-point communication)
4. Seamless integration with Megatron-LM framework
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter
from pathlib import Path

from megatron.core import parallel_state
from megatron.core.transformer.moe.moe_utils import ModelCommProcessGroups
from megatron.training import print_rank_0

from .expert_parameter_transfer import ExpertParameterTransfer

logger = logging.getLogger(__name__)


class GlobalExpertMigration:
    """
    Global Expert Migration Manager
    
    Responsible for managing expert parameter migration and redistribution across nodes, supporting:
    - Reading expert placement configuration
    - Executing expert parameter migration
    - Managing communication groups and synchronization
    """
    
    def __init__(self, 
                 expert_placement_config_path: str,
                 model_comm_pgs: Optional[ModelCommProcessGroups],
                 parameter_transfer: ExpertParameterTransfer,
                 num_experts: int,
                 verbose: bool = True):
        """
        Initialize global expert migration manager
        
        Args:
            expert_placement_config_path: Expert placement configuration JSON file path
            model_comm_pgs: Megatron communication group manager
            parameter_transfer: Help transfer parameter
            num_experts: The number of experts
            verbose: Whether to output detailed logs
        """
        self.expert_placement_config_path = expert_placement_config_path
        self.model_comm_pgs = model_comm_pgs
        self.verbose = verbose
        self.num_experts = num_experts
        
        
        # Get current node's expert assignment information
        self.current_node_id = f"node_{int(os.environ.get('NODE_ID', 0))}"

        self.node_id = int(self.current_node_id[self.current_node_id.find('_')+1:])
        self.gpu_id = torch.cuda.current_device()

        world_size = dist.get_world_size()

        self.load_migration_config(0)

        self.experts_per_node = len(self.expert_placement_config['node_0'])
        self.experts_per_gpu = self.experts_per_node // 8 if self.experts_per_node > 8 else self.experts_per_node // world_size



        # At the start of training, expert is placed in order 
        self.global_expert_assignments = {}
        for eid in range(num_experts):
            self.global_expert_assignments[eid] = {
                'gpu_id': eid // self.experts_per_gpu - eid // self.experts_per_node * 8,
                'node_id': f'node_{eid // self.experts_per_node}'
            }

        self.current_expert_assignments = []
        self.rank = dist.get_rank()

        for eid in range(self.rank * self.experts_per_gpu, (self.rank+ 1) * self.experts_per_gpu):
            self.current_expert_assignments.append({
                'expert_id': eid,
            })
        
        # Communication group information
        self.ep_group = model_comm_pgs.ep if model_comm_pgs else None
        self.tp_group = model_comm_pgs.expt_tp if model_comm_pgs else None
        self.dp_group = model_comm_pgs.expt_dp if model_comm_pgs else None
        
        # Initialize parameter transfer module
        self.parameter_transfer = parameter_transfer
        
        if self.verbose:
            print_rank_0(f"Global expert migration manager initialized")
            logger.info(f"Current node ID: {self.current_node_id}")
            logger.info(f"Current node expert assignments: {len(self.current_expert_assignments)} experts")
    
    def get_expert_migration_plan(self) -> Dict[str, List]:
        """
        Generate expert migration plan
        
        Returns:
            Dictionary containing migration plan with format:
            {
                'migrations': [
                    {
                        'expert_id': int,
                        'from_node': str,
                        'to_node': str,
                        'from_gpu': int,
                        'to_gpu': int
                    }
                ],
                'local_experts_to_remove': List[int],
                'local_experts_to_add': List[int],
            }
        """
        migration_plan = {
            'migrations': [],
            'local_experts_to_remove': [],
            'local_experts_to_add': []
        }
        
        # Get current gpu's expert ID set
        current_expert_ids = {exp['expert_id'] for exp in self.current_expert_assignments}

        # Iterate through all nodes to find experts that need migration
        for target_node_id, target_experts in self.expert_placement_config.items():
            for target_expert in target_experts:

                # Convert node IDs to ranks 
                node_id = int(target_node_id.split('_')[1])
                dst_rank = node_id * 8 + target_expert['gpu_id']

                expert_id = target_expert['expert_id']

                if self.rank == dst_rank:           
                    # Append expert will be migrated to this rank
                    if expert_id not in current_expert_ids:
                        migration_plan['migrations'].append({
                            'expert_id': expert_id,
                            'from_node': self.global_expert_assignments[expert_id]['node_id'],
                            'to_node': self.current_node_id,
                            'from_gpu': self.global_expert_assignments[expert_id]['gpu_id'],
                            'to_gpu': self.gpu_id
                        })
                        migration_plan['local_experts_to_add'].append(expert_id)
                    continue
                
                # If this expert is currently on our node, it needs to be migrated out
                if expert_id in current_expert_ids:
                    migration_plan['migrations'].append({
                        'expert_id': expert_id,
                        'from_node': self.current_node_id,
                        'to_node': target_node_id,
                        'from_gpu': self.gpu_id,
                        'to_gpu': target_expert['gpu_id']
                    })
                    migration_plan['local_experts_to_remove'].append(expert_id)
        
        self.migration_map = {add_expert_id: remove_expert_id
                              for add_expert_id , remove_expert_id
                              in zip(migration_plan['local_experts_to_add'], 
                                     migration_plan['local_experts_to_remove'])}
        
        if self.verbose:
            print_rank_0(f"Expert migration plan generation completed:")
            logger.info(f"  Number of experts to migrate: {len(migration_plan['migrations'])}")
            logger.info(f"  Local experts to remove: {migration_plan['local_experts_to_remove']}")
        
        return migration_plan
    
    def get_global_migration_plan(self, current_expert_mapping: Dict = None) -> Dict[str, List]:
        """
        Get expert migration plans in all ranks

        Args:
            current_expert_mapping: The expert mapping at current step
        
        Returns:
            A dictionary contains the mapping relationship between experts
        """
        if current_expert_mapping is None:
            current_expert_mapping = {}
            for eid in range(self.num_experts):
                current_expert_mapping[eid] = eid


        world_size = dist.get_world_size()
        gathered_migration_plan = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_migration_plan, self.migration_map)
        plans = [plan for local_plan in gathered_migration_plan for plan in local_plan]
        round_migrtion_plans = {eid: replaced_eid for plan in plans for eid, replaced_eid in plan.items()}
        global_migration_plans = {eid: current_expert_mapping[replaced_eid] 
                                  for eid, replaced_eid in round_migrtion_plans}
        
        for eid in range(self.num_experts):
            if eid not in global_migration_plans:
                global_migration_plans[eid] = eid

        return global_migration_plans
    
    def load_migration_config(self, migration_steps):
        """Load expert placement configuration"""
        dir_path = Path(self.expert_placement_config_path)
        step_path = Path(f'migration_plan_{migration_steps}_step.json')
        with open(dir_path / step_path, 'r', encoding='utf-8') as f:
            self.expert_placement_config = json.load(f)
        
        for node_id, experts in self.expert_placement_config.items():
            self.expert_placement_config[node_id] = sorted(experts, key=lambda x: x['expert_id'])

    
    def migrate_experts(self, 
                       model: torch.nn.Module,
                       migration_steps: int,
                       migration_plan: Optional[Dict[str, Any]] = None) -> bool:
        """
        Execute expert migration
        
        Args:
            model: Model containing expert layers
            migration_steps: The current training steps
            migration_plan: Optional migration plan, if None will be auto-generated
            
        Returns:
            Whether migration was successful
        """
        self.load_migration_config(migration_steps)

        if migration_plan is None:
            migration_plan = self.get_expert_migration_plan()
        
        if self.verbose:
            print_rank_0("Starting expert migration...")
            
        # 1. Collect expert parameters that need migration
        expert_params_to_migrate = self._collect_expert_parameters(model, migration_plan)
            
        # 2. Execute cross-node parameter transfer
        self._transfer_expert_parameters(expert_params_to_migrate, migration_plan)
            
        # 3. Update local expert parameters
        self._update_local_expert_parameters(model)
            
        # 4. Update expert assignments to reflect new placement
        self.update_expert_assignments()
            
        # 5. Synchronize all nodes
        if dist.is_initialized():
            dist.barrier()
            
        if self.verbose:
            print_rank_0("Expert migration completed")
            
        return True
    
    def _collect_expert_parameters(self, 
                                 model: torch.nn.Module,
                                 migration_plan: Dict[str, Any]) -> Dict[int, Dict[str, torch.Tensor]]:
        """Collect expert parameters that need migration"""
        expert_params = {}
        
        for expert_id in migration_plan['local_experts_to_remove']:
            expert_params[expert_id] = self.parameter_transfer.extract_mlp_parameters(
                model, expert_id, self.current_expert_assignments
            )
        
        return expert_params
    
    def _transfer_expert_parameters(self, 
                                  expert_params: Dict[int, Dict[str, torch.Tensor]],
                                  migration_plan: Dict[str, Any]) -> None:
        """Execute cross-node expert parameter transfer"""
        # Use parameter transfer module
        if self.verbose:
            print_rank_0("Transferring expert parameters using parameter transfer module...")
        
        # Store received parameters for later application
        received_params = {}
        
        # Process each expert migration
        for migration in migration_plan.get('migrations', []):
            expert_id = migration['expert_id']
            from_node = int(migration['from_node'].split('_')[1])
            to_node = int(migration['to_node'].split('_')[1])
            from_gpu = migration['from_gpu']
            to_gpu = migration['to_gpu']
            
            # Convert node IDs to ranks (simplified mapping)
            src_rank = from_node * 8 + from_gpu
            dst_rank = to_node * 8 + to_gpu 
            
            if expert_id in expert_params:
                # Migrate expert parameters using parameter transfer module
                migrated_params = self.parameter_transfer.migrate_expert_parameters(
                    expert_id, src_rank, dst_rank, expert_params[expert_id]
                )
            else:
                # Receive expert parameters from other device
                migrated_params = self.parameter_transfer.migrate_expert_parameters(
                    expert_id, src_rank, dst_rank 
                )

                if self.verbose:
                    logger.info(f"Migrated expert {expert_id} from {from_node}:GPU{from_gpu} to {to_node}:GPU{to_gpu}")

                
            if migrated_params:
                # Store migrated parameters for experts that should be on this node
                received_params[expert_id] = migrated_params
                    
        
        # Store received parameters for local update
        if received_params:
            self._store_received_parameters(received_params)
        
        if self.verbose:
            print_rank_0("Parameter transfer module migration completed")
    
    def _store_received_parameters(self, expert_params: Dict[int, Dict[str, torch.Tensor]]) -> None:
        """Store received parameters for local update"""
        # Store in instance variable for later use
        if not hasattr(self, '_received_expert_params'):
            self._received_expert_params = {}
        self._received_expert_params.update(expert_params)
    
    def _update_local_expert_parameters(self, model: torch.nn.Module) -> None:
        """Update local expert parameters"""
        # Update parameters for experts that should be on this node
        for expert_id in self._received_expert_params.keys():
            self.parameter_transfer.apply_mlp_parameters(
                model, expert_id, 
                self._received_expert_params[expert_id],
                self.migration_map,
                self.current_expert_assignments
            )
        
        # Clear received parameters
        self._received_expert_params = {}
    
    def update_expert_assignments(self) -> None:
        """Update expert assignments after migration"""
        self.global_expert_assignments = {}
        for node_id, experts in self.expert_placement_config.items():
            for expert in experts:
                expert_id = expert['expert_id']
                self.global_expert_assignments[expert_id] = {
                    'gpu_id': expert['gpu_id'],
                    'node_id': node_id
                }
        
        for added_expert_id, removed_expert_id in self.migration_map.items():
            for idx, e in enumerate(self.current_expert_assignments):
                if e['expert_id'] == removed_expert_id:
                    self.current_expert_assignments[idx] = {'expert_id': added_expert_id}
            
    
    

