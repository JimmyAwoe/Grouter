import torch
import os
import torch.distributed as dist
from utils_grouter.grouter import grouter
import argparse
from transformers import AutoTokenizer
from utils_grouter.utils import create_megatron_dataloader
from utils_grouter.core.config import OptimizationConfig
from megatron.training.tokenizer import build_tokenizer
import csv
import numpy as np
import json

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--bf16", action="store_true")

    # grouter
    parser.add_argument("--grouter-config-path", type=str)
    parser.add_argument("--grouter-checkpoint-path", type=str)
    parser.add_argument("--grouter-bias-checkpoint-path", type=str, default=None)

    # dataset
    parser.add_argument("--data-prefix", type=str, nargs='*', required=True,
                       help="Data prefix for Megatron dataset")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for Megatron dataset processing")
    parser.add_argument("--tokenizer-type", type=str, default=None,
                       help="Tokenizer type for Megatron")
    parser.add_argument("--tokenizer-model", type=str, default=None,
                       help="Tokenizer model for Megatron")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Directory to save/load checkpoints")
    parser.add_argument("--output-csv", type=str, required=True,
                       help="Path to save load balancing results as CSV")
    parser.add_argument("--total-steps", type=int, default=100,
                       help="Total number of training steps")
    
    # log
    parser.add_argument("--save-detail", action='store_true')
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(args)
    return args
    

def test_lb(count_matrix: torch.Tensor, args):
    lb_entropy = - (torch.log2(count_matrix) * count_matrix).sum(axis=1)
    has_nan = torch.isnan(lb_entropy).any()

    lb_entropy = torch.tensor(float('nan')) if has_nan else torch.mean(lb_entropy)
    lb_range = (count_matrix.max(axis=1)[0] - count_matrix.min(axis=1)[0]).mean()
    expect_load = 1 / args.num_experts
    lb_maxvio = ((count_matrix.max(axis=-1)[0] - expect_load) / expect_load).mean() 
    return lb_entropy, lb_range, lb_maxvio


def compute_expert_loads(grt, dataloader, num_experts, max_length, batch_size, topk, total_steps, device, verbose=False):
    """
    Compute load distribution for each expert
    
    Args:
        grt: grouter model
        dataloader: data loader
        num_experts: total number of experts
        max_length: max sequence length
        batch_size: batch size
        topk: topk value
        total_steps: total steps
        device: device
        verbose: whether to print verbose information
    
    Returns:
        expert_loads: numpy array of shape (num_experts,), load for each expert (normalized frequency)
        lb_metrics: dict, containing load balancing metrics (lb_entropy, lb_range, lb_maxvio)
    """
    expert_counts = torch.zeros(num_experts, device=device)
    lb_entropy = []
    lb_range = []
    lb_maxvio = []
    
    data = iter(dataloader)
    
    for i, batch in enumerate(data):
        if i >= total_steps:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            topk_idx = grt(input_ids=batch['tokens'],
                           attention_mask=batch['attention_mask'].float(),
                           position_ids=batch['position_ids'])[0]
            if isinstance(topk_idx, tuple):
                topk_idx = topk_idx[0]
        topk_idx = topk_idx.reshape(max_length, batch_size, topk).permute(1,0,2)
        topk_idx = topk_idx.contiguous().view(batch_size,-1)
        
        # Count how many times each expert is selected
        for idx in topk_idx:
            expert_counts += torch.bincount(idx, minlength=num_experts)
        
        # Compute load balancing metrics
        count_matrix = torch.vstack([torch.bincount(idx, minlength=num_experts) for idx in topk_idx])
        count_matrix = count_matrix / count_matrix.sum(dim=-1, keepdim=True)
        
        # Create temporary args object for test_lb function
        class TempArgs:
            pass
        temp_args = TempArgs()
        temp_args.num_experts = num_experts
        
        lb_e, lb_r, lb_m = test_lb(count_matrix, temp_args)
        lb_entropy.append(lb_e)
        lb_range.append(lb_r)
        lb_maxvio.append(lb_m)
        
        if verbose:
            print(f"index: {i} | lb_entropy: {lb_e} | lb_range: {lb_r} | lb_maxvio: {lb_m}")
    
    # Normalize expert loads
    expert_loads = expert_counts / expert_counts.sum()
    
    # Compute average metrics
    lb_metrics = {
        'lb_entropy': torch.tensor(lb_entropy).mean().item(),
        'lb_range': torch.tensor(lb_range).mean().item(),
        'lb_maxvio': torch.tensor(lb_maxvio).mean().item(),
        'entropy_list': lb_entropy,
        'range_list': lb_range,
        'maxvio_list': lb_maxvio
    }
    
    return expert_loads.cpu().numpy(), lb_metrics


def main(args):
    rank = int(os.environ.get('RANK'))
    world_size = int(os.environ.get('WORLD_SIZE'))
    device = f"cuda:{rank}"

    with open(args.grouter_config_path, 'r') as f:
        grouter_config = json.load(f)
    # load grouter
    grt = grouter(**grouter_config, output_logits=True)
    
    args.topk = grouter_config["topk"]
    num_experts = grouter_config["target_num_experts"] 
    args.num_experts = int(num_experts)

    grt = grt.to(device=device)
    if args.bf16:
        grt = grt.to(torch.bfloat16)
    if args.grouter_checkpoint_path.split('.')[1] == 'pth':
        checkpoint = torch.load(args.grouter_checkpoint_path, map_location=device)
    elif args.grouter_checkpoint_path.split('.')[1] == 'pt':
        checkpoint = torch.load(args.grouter_checkpoint_path, 
                                map_location=device)['grouter_states'][1]
    if "bias" not in checkpoint:
        checkpoint["bias"] = torch.zeros(grt.num_experts)
    grt.load_state_dict(checkpoint)

    if args.grouter_bias_checkpoint_path is not None:
        bias_checkpoint = torch.load(args.grouter_bias_checkpoint_path, map_location=device)
        grt.bias.data = bias_checkpoint
        grt.bias_routing = True


    # load dataset
    config = OptimizationConfig(
        num_nodes=1,
        num_experts=args.num_experts,
        training_config=None,
        experts_per_gpu=None,
        network_topology=None,
        split_config="1,0,0",
        random_seed=getattr(args, 'random_seed', 42),
        train_iters=args.total_steps,
        global_batch_size=args.batch_size * world_size,
        tokenizer_type=args.tokenizer_type,
        tokenizer_model=args.tokenizer_model
    )
    config.rank = 1
    config.make_vocab_size_divisible_by = 128
    config.tensor_model_parallel_size = 1
    config.trust_remote_code = True
    teacher_tokenizer = build_tokenizer(config)

    dataloader = create_megatron_dataloader(args, teacher_tokenizer, config, rank, world_size, 0)

    # Use the new interface function
    expert_loads, lb_metrics = compute_expert_loads(
        grt=grt,
        dataloader=dataloader,
        num_experts=args.num_experts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        topk=args.topk,
        total_steps=args.total_steps,
        device=device,
        verbose=args.verbose
    )
    
    # Gather expert loads from all ranks
    expert_loads_tensor = torch.from_numpy(expert_loads).to(device)
    gathered_expert_loads = [torch.zeros_like(expert_loads_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_expert_loads, expert_loads_tensor)
    
    # Average expert loads across all ranks
    all_expert_loads = torch.stack(gathered_expert_loads).mean(dim=0).cpu().numpy()
    
    # Gather metrics from all ranks
    lb_entropy_tensor = torch.tensor(lb_metrics['entropy_list'], device=device)
    lb_range_tensor = torch.tensor(lb_metrics['range_list'], device=device)
    lb_maxvio_tensor = torch.tensor(lb_metrics['maxvio_list'], device=device)
    
    gathered_entropy = [torch.zeros_like(lb_entropy_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_entropy, lb_entropy_tensor)
    
    gathered_range = [torch.zeros_like(lb_range_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_range, lb_range_tensor)

    gathered_maxvio = [torch.zeros_like(lb_maxvio_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_maxvio, lb_maxvio_tensor)
    
    # Convert to numpy arrays and flatten
    all_entropy = torch.cat(gathered_entropy).cpu().numpy()
    all_range = torch.cat(gathered_range).cpu().numpy()
    all_maxvio = torch.cat(gathered_maxvio).cpu().numpy()
    
    
    # Only rank 0 saves the results
    if rank == 0:
        load_balance_entropy_bound = np.log2(args.num_experts)
        if args.save_detail:
            save_results_to_csv(all_entropy, all_range, all_maxvio, args.output_csv)
        
        # Print expert load information
        print("\n=== Expert Load Distribution ===")
        print(f"Expert loads (normalized): {all_expert_loads}")
        print(f"Max load: {all_expert_loads.max():.6f}, Min load: {all_expert_loads.min():.6f}")
        print(f"Load std: {all_expert_loads.std():.6f}")
        
        print(f"\n=== Load Balance Metrics ===")
        print(f"Total num experts {args.num_experts}, Topk: {args.topk}"
              f" Total Samples {len(all_entropy) * args.batch_size}\n"
              f" Average Load Balance Entropy: {(all_entropy.mean())}\n"
              f" Average Load Balance Range: {all_range.mean()}\n"
              f" Average MaxVio: {all_maxvio.mean()}\n"
              f" The Load Balance Entropy Ratio: {1 - all_entropy.mean() / load_balance_entropy_bound}")
    
    return all_expert_loads if rank == 0 else None


def save_results_to_csv(entropy_values, range_values, maxvio, output_path):
    """Save load balancing results to CSV file"""
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['batch_index', 'lb_entropy', 'lb_range'])
        
        for i, (entropy, range_val, mv) in enumerate(zip(entropy_values, range_values, maxvio)):
            writer.writerow([i, entropy, range_val, mv])
    
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    args = parse_args(None)
    main(args)
    dist.destroy_process_group()