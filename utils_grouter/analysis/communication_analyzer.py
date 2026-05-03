"""
GPU communication analyzer for EP optimization results.

This module analyzes communication patterns within and across nodes based on
expert-sample assignments and expert placements. It focuses on:
1. Intra-node communication (within each node)
2. Expert co-location optimization (multiple experts on same GPU)
3. DeepEP-style cross-node communication (token routing through corresponding GPUs)
"""

import json
import logging
from typing import Dict, List, Tuple, Set
import numpy as np
from dataclasses import dataclass, asdict

from ..utils.data_structures import Sample

logger = logging.getLogger(__name__)


@dataclass
class CommunicationStats:
    """Communication statistics for a specific scope"""
    num_micro_batch: int
    total_tokens: int
    inter_node_communication_tokens: int  # Actual cross-node communication tokens
    intra_node_communication_tokens: int  # Actual intra-node communication tokens
    total_gpu_load: Dict
    average_load_entropy: float

    def __post_init__(self):
        self.communication_tokens = self.inter_node_communication_tokens + self.intra_node_communication_tokens
        self.communication_ratio = self.communication_tokens / self.total_tokens
        self.inter_communication_ratio = self.inter_node_communication_tokens / self.total_tokens
        self.intra_communication_ratio = self.intra_node_communication_tokens / self.total_tokens


@dataclass
class CommunicationResult:
    """Communication analysis result for a specific GPU"""
    total_tokens: int
    inter_node_communication_tokens: int  # Actual inter-node communication tokens
    intra_node_communication_tokens: int  # Actual intra-node communication tokens
    load_per_gpu: Dict # The load for each gpu

    def __post_init__(self):

        gpu_load = np.array(list(self.load_per_gpu.values()))
        gpu_load_ratio = gpu_load / sum(gpu_load)
        self.entropy = - np.sum(np.log2(gpu_load_ratio) / len(gpu_load))



class CommunicationAnalyzer:
    """
    Analyzes communication patterns for EP optimization results.
    
    This class computes communication costs based on:
    - Sample-to-GPU assignments
    - Expert-to-GPU placements  
    - DeepEP routing patterns
    - Expert co-location optimizations
    """
    
    def __init__(self, num_nodes: int, topk: int, micro_batch_size: int):
        """
        Initialize communication analyzer.
        
        Args:
            num_nodes: Number of nodes in the system
            topk: Number of top experts per token
            micro_batch_size: The input data size for each micro batch per GPU
        """
        self.num_nodes = num_nodes
        self.topk = topk
        self.gpus_per_node = 8
        self.micro_batch_size = micro_batch_size
        self.node_gpu_tuple = [(node_id, gpu_id) for node_id in range(self.num_nodes) for gpu_id in range(self.gpus_per_node)]
        
        logger.debug(f"CommunicationAnalyzer initialized: {num_nodes} nodes, topk={topk}")
    
    def analyze_communication(self,
                                gpu_assignments: Dict[Tuple[int, int], List[Sample]],
                                expert_placements: Dict[Tuple[int, int], List[int]],
                                ) -> Dict[int, CommunicationResult]:
        """
        Analyze communication patterns for all GPUs.
        
        Args:
            gpu_assignments: (node_id, gpu_id) -> assigned samples
            expert_placements: (node_id, gpu_id) -> assigned experts
            
        Returns:
            Dictionary mapping (node_id, gpu_id) -> communication analysis result
        """
        assert len(gpu_assignments[(0,0)]) % self.micro_batch_size == 0, "The gpu assignments should be consist of micro batchs"
        num_batch = len(gpu_assignments[(0,0)]) // self.micro_batch_size

        micro_batch_communication = {}

        for micro_batch_id in range(num_batch):
            # Use micro batch size to get the range of this micro batch size
            micro_batch_range = slice(micro_batch_id * self.micro_batch_size, (micro_batch_id + 1) * self.micro_batch_size)

            # Recover the samples and experts placement in this micro batch
            micro_batch_samples = {node_gpu_comb: samples[micro_batch_range] 
                                   for node_gpu_comb, samples in gpu_assignments.items()}
            micro_batch_experts = {node_gpu_comb: experts[micro_batch_range] 
                                   for node_gpu_comb, experts in expert_placements.items()}

            # Start to compute communication in this micro batch
            micro_batch_communication[micro_batch_id] = self._analyze_single_micro_batch_communication(
                micro_batch_samples, micro_batch_experts, 
            )
            
            if (micro_batch_id + 1) % 10 == 0:
                logger.info(f"Communication analysis completed for {micro_batch_id + 1}/{num_batch} micro batch")

        logger.info(f"Communication analysis completed")

        return micro_batch_communication
    
    def _analyze_single_micro_batch_communication(self,
                                        batch_samples: Dict[Tuple[int, int], List[Sample]],
                                        batch_experts: Dict[Tuple[int, int], List[int]],) -> CommunicationResult:
        """
        Analyze communication for a single micro batch.
        
        Args:
            all_gpu_assignments: GPU assignments for this micro batch
            all_expert_placements: Expert placements for this micro batch
            
        Returns:
            Communication analysis result for this micro batch
        """
        
        total_tokens = 0
        inter_node_communication_tokens = 0
        intra_node_communication_tokens = 0
        load_per_gpu = {}

        
        # Analyze intra comm first
        for sample_node_gpu_comb, samples in batch_samples.items():
            _, sample_gpu = sample_node_gpu_comb
            for sample in samples:
                sample_tokens = sample.token_count
                total_tokens += sample_tokens
            
                # Reshape dispatch IDs to [seq_len, topk]
                dispatch_matrix = np.array(sample.dispatch_ids, dtype=np.uint8).reshape(-1, self.topk)

                # Calculate communication costs for each target node/GPU combination
                for target_node_gpu_comb in self.node_gpu_tuple:
                    _, target_gpu = target_node_gpu_comb
                    experts_in_gpu = batch_experts[target_node_gpu_comb]

                    # Initialize load_per_gpu
                    if target_node_gpu_comb not in load_per_gpu.keys():
                        load_per_gpu[target_node_gpu_comb] = 0
                    
                    # Each experts needed count
                    load_per_gpu[target_node_gpu_comb] += np.isin(dispatch_matrix, experts_in_gpu).sum()

                    if target_node_gpu_comb == sample_node_gpu_comb:
                        # In the sample's gpu, no communication
                        continue
                
                    if sample_gpu == target_gpu:
                        # Because of DeepEP's type communication, no intra comm
                        continue

                    intra_node_comm = np.any(np.isin(dispatch_matrix, experts_in_gpu), axis=1).sum() 
                    intra_node_communication_tokens += intra_node_comm


        # Start to statitic inter node communication

        # First group sample and experts by node
        sample_grouped_by_nodes = {node_id: [] for node_id in range(self.num_nodes)}
        expert_grouped_by_nodes = {node_id: [] for node_id in range(self.num_nodes)}
        
        for sample_node_gpu_comb, samples in batch_samples.items():
            sample_node_id, _ = sample_node_gpu_comb
            for sample in samples:
                sample_grouped_by_nodes[sample_node_id].append(sample)
        
        for expert_node_gpu_comb, experts in batch_experts.items():
            expert_node_id, _ = expert_node_gpu_comb
            for expert in experts:
                expert_grouped_by_nodes[expert_node_id].append(expert)
        
        for node_id in range(self.num_nodes):
            samples_in_node = sample_grouped_by_nodes[node_id]
            for samples in samples_in_node: 

                dispatch_matrix = np.array(sample.dispatch_ids, dtype=np.uint8).reshape(-1, self.topk)
                for nid, experts in expert_grouped_by_nodes.items():
                    if nid == node_id:
                        continue
                     
                    inter_node_comm = np.any(np.isin(dispatch_matrix, experts), axis=1).sum()
                    inter_node_communication_tokens += inter_node_comm

        return CommunicationResult(
            total_tokens=total_tokens,
            inter_node_communication_tokens=inter_node_communication_tokens,
            intra_node_communication_tokens=intra_node_communication_tokens,
            load_per_gpu=load_per_gpu,
        )
    
    def compute_aggregate_stats(self, micro_batch_results: Dict[int, CommunicationResult]) -> CommunicationStats:
        """
        Compute aggregate communication statistics across all micro batch.
        
        Args:
            micro_batch_results: Results from analyze_gpu_communication
            
        Returns:
            Aggregate communication statistics
        """
        num_micro_batch = len(micro_batch_results.keys())

        total_tokens = sum(result.total_tokens for result in micro_batch_results.values())
        
        # Aggregate actual communication tokens
        total_inter_node_communication = sum(result.inter_node_communication_tokens for result in micro_batch_results.values())
        total_intra_node_communication = sum(result.intra_node_communication_tokens for result in micro_batch_results.values())

        total_gpu_load = {}
        for node_gpu_comb in self.node_gpu_tuple:
            total_gpu_load[node_gpu_comb] = sum(result.load_per_gpu[node_gpu_comb] for result in micro_batch_results.values())
        
        load_balance_entropy = sum(result.entropy for result in micro_batch_results.values()) / num_micro_batch

        
        return CommunicationStats(
            num_micro_batch=num_micro_batch,
            total_tokens=total_tokens,
            inter_node_communication_tokens=total_inter_node_communication,
            intra_node_communication_tokens=total_intra_node_communication,
            total_gpu_load=total_gpu_load,
            average_load_entropy=load_balance_entropy
        )
    
    def generate_communication_json(self, aggregate_stats: CommunicationStats) -> Dict:
        """
        Generate communication analysis results in JSON format.
        
        Args:
            aggregate_stats: Aggregate statistics
            
        Returns:
            Dictionary containing all communication analysis results
        """
        # Create comprehensive JSON structure
        json_result = {
            "analysis_metadata": {
                "num_nodes": self.num_nodes,
                "topk": self.topk,
                "micro_batch_size": self.micro_batch_size,
                "num_micro_batch": aggregate_stats.num_micro_batch
            },
            "communication_analysis": {
                "total_tokens": aggregate_stats.total_tokens,
                "communication_tokens": int(aggregate_stats.communication_tokens),
                "communication_ratio": float(aggregate_stats.communication_ratio),
                "inter_communication_ratio": float(aggregate_stats.inter_communication_ratio),
                "intra_communication_ratio": float(aggregate_stats.intra_communication_ratio),
                "inter_node_communication_tokens": int(aggregate_stats.inter_node_communication_tokens),
                "intra_node_communication_tokens": int(aggregate_stats.intra_node_communication_tokens),
                "average_load_entropy": float(aggregate_stats.average_load_entropy),
            },
        }
        
        return json_result
    
    def save_communication_json(self, aggregate_stats: CommunicationStats,
                              micro_batch_results: Dict[int, CommunicationResult],
                              output_path: str) -> None:
        """
        Save communication analysis results to JSON file.
        
        Args:
            aggregate_stats: Aggregate statistics
            micro_batch_results: Results from analyze_communication
            output_path: Path to save JSON file
        """
        json_result = self.generate_communication_json(aggregate_stats)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Communication analysis JSON saved to: {output_path}")
