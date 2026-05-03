"""
Node-level assignment optimizer

Assign pooled micro-batch samples to nodes with capacity constraints (8 * micro_batch_size per node)
to minimize estimated cross-node communication based on DeepEP-style routing.

Heuristic implementation using expanded-cost Hungarian assignment:
- Left side: samples in a pooled micro batch from all nodes
- Right side: node slots (each node replicated by its capacity for the batch)
- Edge weight: number of token destinations (top-k experts per token) that are not on the target node
"""

from typing import Dict, List, Tuple, Set
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..utils.data_structures import Sample


class NodeAssigner:
    """Assign samples to nodes under capacity constraints to minimize cross-node communication."""

    def __init__(self, num_nodes: int, experts_by_node: Dict[int, Set[int]], topk: int):
        self.num_nodes = num_nodes
        self.experts_by_node = experts_by_node
        self.topk = topk

    def assign(self,
               pooled_samples: List[Sample],
               micro_batch_size: int) -> Dict[int, List[Sample]]:
        """
        Assign pooled samples to nodes with capacity = 8 * micro_batch_size per node.

        Args:
            pooled_samples: Combined samples from all nodes for one global micro step
            micro_batch_size: Micro batch size per GPU

        Returns:
            Mapping node_id -> assigned samples for this step (length <= capacity per node)
        """
        if len(pooled_samples) == 0:
            return {nid: [] for nid in range(self.num_nodes)}

        capacity_per_node = 8 * micro_batch_size
        total_capacity = capacity_per_node * self.num_nodes
        num_samples = min(len(pooled_samples), total_capacity)
        samples = pooled_samples[:num_samples]

        # Build expanded right-side slots (node replicated by capacity)
        node_slots: List[int] = [i for i in range(self.num_nodes)]

        # Build cost matrix: shape [num_samples, num_nodes]
        # Cost(sample i -> slot j) = number of token expert destinations not on node node_slots[j]
        cost_matrix = np.zeros((num_samples, self.num_nodes), dtype=np.int32)

        # Pre-cache experts per node as numpy arrays for vectorized isin
        experts_per_node = {
            nid: np.array(sorted(list(experts)), dtype=np.int32)
            for nid, experts in self.experts_by_node.items()
        }

        # For each sample, compute token->expert matrix of shape [seq_len, topk]
        # dispatch_ids length should be seq_len * topk
        for i, sample in enumerate(samples):
            # reshape to [seq_len, topk]
            if self.topk <= 0:
                raise ValueError("topk must be positive for node assignment cost computation")
            seq_topk = np.asarray(sample.dispatch_ids, dtype=np.uint8).reshape(-1, self.topk)

            # For each slot/node, count how many experts per token land outside the node
            for j, node_id in enumerate(node_slots):
                sample_comm = 0
                for target_node_id in range(self.num_nodes):
                    if target_node_id == node_id:
                        continue
                    node_experts = experts_per_node[target_node_id]
                    # isin returns match per expert; invert to count non-local expert sends per token
                    token_need_comm = np.any(np.isin(seq_topk, node_experts, assume_unique=True), axis=1)
                    comm = token_need_comm.sum()
                    # total non-local sends across tokens
                    sample_comm += comm
                cost_matrix[i, j] = sample_comm

        cost_matrix = np.repeat(cost_matrix, 8, axis=1)
        # Hungarian assignment on square matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments: Dict[int, List[Sample]] = {nid: [] for nid in range(self.num_nodes)}
        for i, j in zip(row_ind, col_ind):
            node_id = j // 8
            assignments[node_id].append(samples[i])

        return assignments


