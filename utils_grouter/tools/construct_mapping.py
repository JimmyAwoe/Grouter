import torch
import os
import torch.distributed as dist
import argparse
import json
import numpy as np
from utils_grouter.grouter import grouter as grt
from utils_grouter.utils.megatron_dataloader import create_megatron_dataloader
from utils_grouter.core.config import OptimizationConfig
from megatron.training.tokenizer import build_tokenizer
from utils_grouter.tools.lb_test import compute_expert_loads
from utils_grouter.tools.expert_affinity import compute_expert_affinity_matrix, get_affinity_based_groups


def parse_args(args):
    parser = argparse.ArgumentParser()
    # grouter config
    parser.add_argument("--grouter-config-path", type=str, required=True)
    parser.add_argument("--grouter-checkpoint-path", type=str, required=True)
    
    # dataset config
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--data-prefix", type=str, nargs='*', required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--tokenizer-type", type=str, default=None)
    parser.add_argument("--tokenizer-model", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=100)
    
    # mapping config
    parser.add_argument("--target-num-experts", type=int, required=True, 
                       help="Target number of experts after folding")
    parser.add_argument("--output-mapping", type=str, required=True,
                       help="Path to save the expert mapping JSON file")
    parser.add_argument("--mapping-strategy", type=str, default='balanced',
                       choices=['balanced', 'greedy', 'affinity', 'random'],
                       help="Mapping strategy to use")
    
    # training config
    parser.add_argument("--bf16", action='store_true')
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(args)
    return args


def create_load_balanced_mapping(expert_loads, source_num_experts, target_num_experts, strategy='balanced', 
                                affinity_matrix=None):
    """
    Create expert mapping based on load distribution or affinity
    
    Args:
        expert_loads: numpy array, load for each expert (normalized frequency)
        source_num_experts: number of source experts
        target_num_experts: number of target experts
        strategy: mapping strategy, 'balanced', 'greedy', or 'affinity'
        affinity_matrix: expert affinity matrix (required for 'affinity' strategy)
    
    Returns:
        mapping: dict, {target_expert_id: [source_expert_ids]}
    """
    # Sort experts by load
    exp_idx = np.argsort(expert_loads)  # Sort from low to high
    experts_per_target = source_num_experts // target_num_experts
    remainder = source_num_experts % target_num_experts
    assert remainder == 0, "The source_num_experts must be divisible by target_num_experts"
    
    if strategy == 'balanced':
        # Balanced strategy: pair high-load and low-load experts
        mapping = {}
        
        source_idx = 0
        for target_id in range(target_num_experts):
            # Take experts from both ends: half low-load, half high-load
            low_end = exp_idx[source_idx:source_idx + experts_per_target // 2]
            high_end = exp_idx[-(experts_per_target - experts_per_target // 2):]
            
            # Update expert_indices, remove used high-load experts
            exp_idx = exp_idx[:-(experts_per_target - experts_per_target // 2)]
            
            mapping[target_id] = np.concatenate([low_end, high_end]).tolist()
            
    elif strategy == 'greedy':
        # Greedy strategy: sequential allocation
        mapping = {idx: exp_idx[idx * experts_per_target: (idx + 1) *experts_per_target].tolist() 
                   for idx in range(target_num_experts)}
    
    elif strategy == 'affinity':
        # Affinity strategy: group experts based on co-activation frequency
        if affinity_matrix is None:
            raise ValueError("affinity_matrix is required for 'affinity' strategy")
        mapping = get_affinity_based_groups(affinity_matrix, source_num_experts, target_num_experts)
    
    # Compute total load for each target expert
    if expert_loads is not None:
        target_loads = {}
        for target_id, source_ids in mapping.items():
            target_loads[target_id] = sum(expert_loads[sid] for sid in source_ids)
    else:
        target_loads = None
    
    return mapping, target_loads


def save_mapping_to_json(mapping, output_path, grouter_config, target_num_experts):
    """Save mapping to JSON file"""
    grouter_config['target_num_experts'] = str(target_num_experts)
    grouter_config['mapping'] = {}
    grouter_config['mapping'][f'{target_num_experts}'] = mapping
    with open(output_path, 'w') as f:
        json.dump(grouter_config, f, indent=2)
    
    print(f"Mapping saved to {output_path}")


def main(args):
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    
    # Load grouter
    with open(args.grouter_config_path, 'r') as f:
        grouter_config = json.load(f)
    grouter = grt(**grouter_config, output_logits=True)

    args.topk = grouter_config["topk"]
    num_experts = grouter_config["target_num_experts"] if "target_num_experts" in grouter_config else grouter_config["num_experts"]
    args.num_experts = int(num_experts)

    if args.mapping_strategy == 'random':
        experts_per_target = args.num_experts // args.target_num_experts 
        remainder = args.num_experts % args.target_num_experts
        exp_idx = np.arange(args.num_experts)
        assert remainder == 0, "The source_num_experts must be divisible by target_num_experts"
        mapping = {idx: exp_idx[idx * experts_per_target: (idx + 1) * experts_per_target].tolist() 
                   for idx in range(args.target_num_experts)}
        if rank == 0:
            save_mapping_to_json(mapping, args.output_mapping, grouter_config, args.target_num_experts)
        return 
    
    grouter = grouter.to(device=device)
    if args.bf16:
        grouter = grouter.to(torch.bfloat16)
    
    # Load checkpoint
    if args.grouter_checkpoint_path.split('.')[1] == 'pth':
        checkpoint = torch.load(args.grouter_checkpoint_path, map_location=device)
    elif args.grouter_checkpoint_path.split('.')[1] == 'pt':
        checkpoint = torch.load(args.grouter_checkpoint_path, 
                                map_location=device)['grouter_states']
    grouter.load_state_dict(checkpoint)
    grouter.eval()
    
    # Load dataset
    config = OptimizationConfig(
        num_nodes=1,
        num_experts=args.num_experts,
        training_config=None,
        experts_per_gpu=None,
        network_topology=None,
        split_config="1,0,0",
        random_seed=args.random_seed,
        train_iters=args.total_steps,
        global_batch_size=args.batch_size * world_size,
        tokenizer_type=args.tokenizer_type,
        tokenizer_model=args.tokenizer_model
    )
    config.rank = rank
    config.make_vocab_size_divisible_by = 128
    config.tensor_model_parallel_size = 1
    config.trust_remote_code = True
    tokenizer = build_tokenizer(config)

    args.checkpoint_dir = os.path.dirname(args.output_mapping)
    dataloader = create_megatron_dataloader(args, tokenizer, config, rank, world_size, 0)

    # Compute expert loads
    if rank == 0:
        print(f"Computing expert loads for {args.num_experts} experts...")
    
    if not args.mapping_strategy == 'affinity':
        expert_loads, lb_metrics = compute_expert_loads(
            grt=grouter,
            dataloader=dataloader,
            num_experts=args.num_experts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            topk=args.topk,
            total_steps=args.total_steps,
            device=device,
            verbose=args.verbose
        )
    
    # Compute affinity matrix if using affinity strategy
    affinity_matrix = None
    if args.mapping_strategy == 'affinity':
        if rank == 0:
            print(f"Computing expert affinity matrix...")
        expert_loads = None 
        affinity_matrix= compute_expert_affinity_matrix(
            grouter=grouter,
            dataloader=dataloader,
            num_experts=args.num_experts,
            total_steps=args.total_steps,
            device=device,
            verbose=args.verbose
        )
    
    # Synchronize expert loads across multiple GPUs
    if world_size > 1 and args.mapping_strategy != 'affinity':
        expert_loads_tensor = torch.from_numpy(expert_loads).to(device)
        gathered_expert_loads = [torch.zeros_like(expert_loads_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_expert_loads, expert_loads_tensor)
        expert_loads = torch.stack(gathered_expert_loads).mean(dim=0).cpu().numpy()
    
    # Only create and save mapping on rank 0
    if rank == 0:
        if args.mapping_strategy != 'affinity':
            print("\n=== Expert Load Distribution ===")
            print(f"Expert loads: {expert_loads}")
            print(f"Max load: {expert_loads.max():.6f}, Min load: {expert_loads.min():.6f}")
            print(f"Load std: {expert_loads.std():.6f}")
        
        print(f"\n=== Creating mapping from {args.num_experts} to {args.target_num_experts} experts using {args.mapping_strategy} strategy ===")
        
        # Create mapping
        mapping, target_loads = create_load_balanced_mapping(
            expert_loads=expert_loads,
            source_num_experts=args.num_experts,
            target_num_experts=args.target_num_experts,
            strategy=args.mapping_strategy,
            affinity_matrix=affinity_matrix,
        )
        if target_loads is not None: 
            print(f"\n=== Target Load Balance ===")
            target_load_array = np.array(list(target_loads.values()))
            print(f"Target loads: {target_load_array}")
            print(f"Max target load: {target_load_array.max():.6f}")
            print(f"Min target load: {target_load_array.min():.6f}")
            print(f"Target load std: {target_load_array.std():.6f}")
        
        
        save_mapping_to_json(mapping, args.output_mapping, grouter_config, args.target_num_experts)
        print(f"\nMapping successfully created and saved!")
    

if __name__ == "__main__":
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        dist.init_process_group(backend='nccl')
    
    args = parse_args(None)
    main(args)
    
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        dist.destroy_process_group()

