import torch
import numpy as np
from typing import Tuple


def compute_expert_affinity_matrix(
    grouter,
    dataloader,
    num_experts: int,
    total_steps: int,
    device: str,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute expert affinity matrix based on co-activation frequency
    
    Args:
        grouter: The grouter model
        dataloader: Data loader for inference
        num_experts: Number of experts
        total_steps: Number of steps to compute affinity
        device: Device to run computation on
        verbose: Whether to print verbose information
    
    Returns:
        affinity_matrix: numpy array of shape (num_experts, num_experts)
                       where affinity_matrix[i, j] represents co-activation frequency
                       between expert i and expert j
    """
    grouter.eval()
    affinity_matrix = np.zeros((num_experts, num_experts), dtype=np.float32)
    total_tokens = 0
    
    step_count = 0
    with torch.no_grad():
        for batch in dataloader:
            if step_count >= total_steps:
                break

            batch = {key: value.to(device) for key, value in batch.items()}
            topk_idx = grouter(input_ids=batch['tokens'],
                        attention_mask=batch['attention_mask'].float(),
                        position_ids=batch['position_ids'])[0]
            if isinstance(topk_idx, tuple):
                topk_idx = topk_idx[0]
            
            # Create co-activation matrix from topk_idx
            # topk_idx shape: [batch_size * seq_len, topk]
            batch_size, topk = topk_idx.shape
            
            # Create binary matrix indicating which experts are activated
            routing_map = torch.zeros(batch_size, num_experts, device=device)
            routing_map.scatter_(1, topk_idx, 1)
            
            # Compute co-activation matrix for this batch
            batch_affinity = torch.matmul(routing_map.float().T, routing_map.float())
            affinity_matrix += batch_affinity.cpu().numpy()
            
            total_tokens += batch_size
            step_count += 1
            
            if verbose and step_count % 20 == 0:
                print(f"Processed {step_count} steps, {total_tokens} tokens")
    
    # Normalize by total tokens to get co-activation frequency
    if total_tokens > 0:
        affinity_matrix = affinity_matrix / total_tokens
    
    # Set diagonal to 0 (self-affinity is not meaningful for grouping)
    np.fill_diagonal(affinity_matrix, 0)
    
    if verbose:
        print(f"Computed affinity matrix for {num_experts} experts")
        print(f"Total tokens processed: {total_tokens}")
        print(f"Max affinity: {affinity_matrix.max():.6f}")
        print(f"Min affinity: {affinity_matrix.min():.6f}")
    
    return affinity_matrix


def get_affinity_based_groups(
    affinity_matrix: np.ndarray,
    source_num_experts: int,
    target_num_experts: int
) -> dict:
    """
    Group experts based on affinity matrix using greedy selection
    
    Args:
        affinity_matrix: Expert affinity matrix
        source_num_experts: Number of source experts
        target_num_experts: Number of target experts
    
    Returns:
        mapping: dict, {target_expert_id: [source_expert_ids]}
    """
    experts_per_target = source_num_experts // target_num_experts
    assert source_num_experts % target_num_experts == 0, \
        "source_num_experts must be divisible by target_num_experts"
    
    mapping = {}
    used_experts = set()
    
    for target_id in range(target_num_experts):
        if len(used_experts) == source_num_experts:
            break
            
        # Find the expert with highest total affinity to unused experts
        available_experts = [i for i in range(source_num_experts) if i not in used_experts]
        
        if not available_experts:
            break
            
        # Select the first expert as seed
        seed_expert = available_experts[0]
        group = [seed_expert]
        used_experts.add(seed_expert)
        
        # Greedily add experts with highest affinity to the group
        while len(group) < experts_per_target and len(used_experts) < source_num_experts:
            remaining_experts = [i for i in range(source_num_experts) if i not in used_experts]
            
            if not remaining_experts:
                break
            
            # Compute average affinity to the entire group
            remaining_idx = np.array(remaining_experts)
            avg_affinities = affinity_matrix[remaining_idx][:, group].mean(axis=1)
            best_idx = np.argmax(avg_affinities)
            best_expert = remaining_experts[best_idx]
            
            group.append(best_expert)
            used_experts.add(best_expert)
        
        mapping[target_id] = group
    
    return mapping
