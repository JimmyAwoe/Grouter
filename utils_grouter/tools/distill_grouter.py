import sys
from pathlib import Path
import os
link_path = os.path.abspath(sys.argv[0])
link_dir = os.path.dirname(link_path)    

if link_dir not in sys.path:
    sys.path.insert(0, link_dir)

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, AutoConfig, get_cosine_schedule_with_warmup
#from datasets import load_dataset
#from datasets.distributed import split_dataset_by_node
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import grouter, logits_hook_utils_func, get_model_path, get_key_info
from accelerate import Accelerator
import torch.distributed as dist
import argparse
import json
import time
import csv
from pathlib import Path
from megatron.training.tokenizer import build_tokenizer

# Import Megatron dataloader
from utils.common.megatron_dataloader import create_megatron_dataloader
from utils.core.config import OptimizationConfig
MEGATRON_AVAILABLE = True

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--moe-layer-start", default=0, type=int)
    parser.add_argument("--moe-layer-end", default=1, type=int)
    parser.add_argument("--max-length", default=1024, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--grouter-type", type=str)
    parser.add_argument("--model-config-path", type=str)
    parser.add_argument("--total-steps", default=1000, type=int)
    parser.add_argument("--warmup", default=100, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument("--use-auto", action="store_true")
    parser.add_argument("--transpose", action="store_true")
    
    # Megatron dataset options
    parser.add_argument("--use-megatron", action="store_true",
                       help="Use Megatron dataset processing instead of HuggingFace datasets")
    parser.add_argument("--data-prefix", type=str, nargs='*', required=True,
                       help="Data prefix for Megatron dataset")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for Megatron dataset processing")
    parser.add_argument("--tokenizer-type", type=str, default=None,
                       help="Tokenizer type for Megatron")
    parser.add_argument("--tokenizer-model", type=str, default=None,
                       help="Tokenizer model for Megatron")
    
    # Checkpoint related arguments
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Directory to save/load checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=1000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--resume-from-last-ckpt", action="store_true",
                       help="Auto find and load last ckpt in checkpoint-dir")
    parser.add_argument("--save-latest", action="store_true",
                       help="Save latest checkpoint in addition to interval checkpoints")
    
    # Gradient accumulation arguments
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help="Number of gradient accumulation steps before optimizer step")
    parser.add_argument("--effective-batch-size", type=int, default=None,
                       help="Effective batch size (micro_batch_size * gradient_accumulation_steps * world_size). If set, gradient_accumulation_steps will be calculated automatically")
    
    # Logging arguments
    parser.add_argument("--log-csv", action="store_true",
                       help="Save training logs to CSV file for plotting")
    parser.add_argument("--log-interval", type=int, default=1,
                       help="Log every N steps (default: 1, log every step)")
    
    args = parser.parse_args(args)
    
    # Calculate gradient accumulation steps if effective batch size is specified
    if args.effective_batch_size is not None:
        if args.effective_batch_size % (args.batch_size * int(os.environ.get('WORLD_SIZE', 1))) != 0:
            raise ValueError(f"Effective batch size ({args.effective_batch_size}) must be divisible by micro_batch_size * world_size ({args.batch_size} * {int(os.environ.get('WORLD_SIZE', 1))})")
        args.gradient_accumulation_steps = args.effective_batch_size // (args.batch_size * int(os.environ.get('WORLD_SIZE', 1)))
        print(f"Calculated gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    
    return args


def save_checkpoint(checkpoint_dir, step, grouter_dict, opt_dict, sch_dict, 
                   loss_list, args, rank, world_size):
    """Save training checkpoint"""
    if rank != 0:
        return
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        'step': step,
        'args': vars(args),
        'loss_list': loss_list,
        'timestamp': time.time(),
        'world_size': world_size,
        'rank': rank
    }
    
    # Save grouter models
    grouter_states = {}
    for layer_idx, model in grouter_dict.items():
        grouter_states[layer_idx] = model.state_dict()
    checkpoint['grouter_states'] = grouter_states
    
    # Save optimizer states
    optimizer_states = {}
    for layer_idx, optimizer in opt_dict.items():
        optimizer_states[layer_idx] = optimizer.state_dict()
    checkpoint['optimizer_states'] = optimizer_states
    
    # Save scheduler states
    scheduler_states = {}
    for layer_idx, scheduler in sch_dict.items():
        scheduler_states[layer_idx] = scheduler.state_dict()
    checkpoint['scheduler_states'] = scheduler_states
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save latest checkpoint symlink
    latest_path = checkpoint_dir / "latest_checkpoint.pt"
    if latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(checkpoint_path.name)
    
    # Save metadata
    metadata = {
        'last_step': step,
        'checkpoint_files': [f"checkpoint_step_{step}.pt"],
        'args': vars(args),
        'timestamp': time.time()
    }
    metadata_path = checkpoint_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Checkpoint saved at step {step}: {checkpoint_path}")
    
    # Clean up old checkpoints
    cleanup_old_checkpoints(checkpoint_dir, keep_last_n=1)


def load_checkpoint(checkpoint_path, grouter_dict, opt_dict, sch_dict, 
                   device, rank, world_size):
    """Load training checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None, None, None, 0, []
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Verify world size compatibility
    if checkpoint.get('world_size') != world_size:
        print(f"Warning: Checkpoint world_size ({checkpoint['world_size']}) != current world_size ({world_size})")
    
    # Load grouter states
    for layer_idx, state_dict in checkpoint['grouter_states'].items():
        if layer_idx in grouter_dict:
            grouter_dict[layer_idx].load_state_dict(state_dict)
            print(f"Loaded grouter state for layer {layer_idx}")
    
    # Load optimizer states
    for layer_idx, state_dict in checkpoint['optimizer_states'].items():
        if layer_idx in opt_dict:
            opt_dict[layer_idx].load_state_dict(state_dict)
            print(f"Loaded optimizer state for layer {layer_idx}")
    
    # Load scheduler states
    for layer_idx, state_dict in checkpoint['scheduler_states'].items():
        if layer_idx in sch_dict:
            sch_dict[layer_idx].load_state_dict(state_dict)
            print(f"Loaded scheduler state for layer {layer_idx}")
    
    step = checkpoint['step']
    loss_list = checkpoint.get('loss_list', [])
    
    print(f"Resumed from step {step} with {len(loss_list)} loss records")
    return grouter_dict, opt_dict, sch_dict, step, loss_list


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory"""
    if not checkpoint_dir:
        return None
    
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    # Try to find latest checkpoint symlink
    latest_path = checkpoint_dir / "latest_checkpoint.pt"
    if latest_path.exists() and latest_path.is_symlink():
        return str(latest_path.resolve())
    
    # Fallback: find checkpoint with highest step number
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if not checkpoint_files:
        return None
    
    # Extract step numbers and find the highest
    step_numbers = []
    for file_path in checkpoint_files:
        try:
            step = int(file_path.stem.split('_')[-1])
            step_numbers.append((step, file_path))
        except (ValueError, IndexError):
            continue
    
    if not step_numbers:
        return None
    
    # Return the checkpoint with highest step number
    latest_checkpoint = max(step_numbers, key=lambda x: x[0])[1]
    return str(latest_checkpoint)


def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3):
    """Clean up old checkpoints, keeping only the last N ones"""
    if not checkpoint_dir:
        return
    
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if len(checkpoint_files) <= keep_last_n:
        return
    
    # Sort by step number and remove old ones
    step_numbers = []
    for file_path in checkpoint_files:
        try:
            step = int(file_path.stem.split('_')[-1])
            step_numbers.append((step, file_path))
        except (ValueError, IndexError):
            continue
    
    if len(step_numbers) <= keep_last_n:
        return
    
    # Sort by step number (ascending) and remove old ones
    step_numbers.sort(key=lambda x: x[0])
    files_to_remove = step_numbers[:-keep_last_n]
    
    for _, file_path in files_to_remove:
        try:
            file_path.unlink()
            print(f"Removed old checkpoint: {file_path}")
        except Exception as e:
            print(f"Failed to remove checkpoint {file_path}: {e}")


def dilute_grouter(args):
    rank = int(os.environ.get('RANK'))
    world_size = int(os.environ.get('WORLD_SIZE'))
    device = f'cuda:{rank}'
    
    model_dir, _, save_dir = get_model_path(args.model_name)
    experts_per_layer, n_routed_experts, padding_idx, hidden_size, vocab_size = get_key_info(args.model_name)
    # load teacher model and tokenizer
    if args.use_megatron:

        config = OptimizationConfig(
            num_nodes=1,
            num_experts=n_routed_experts,
            training_config=None,
            experts_per_gpu=None,
            network_topology=None,
            split_config="1,0,0",
            random_seed=getattr(args, 'random_seed', 42),
            train_iters=args.total_steps,
            global_batch_size=args.batch_size * world_size * args.gradient_accumulation_steps,
            tokenizer_type=args.tokenizer_type,
            tokenizer_model=args.tokenizer_model
        )
        config.rank = 1
        config.make_vocab_size_divisible_by = 128
        config.tensor_model_parallel_size = 1
        config.trust_remote_code = True
        teacher_tokenizer = build_tokenizer(config)
    else:
        teacher_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if args.use_auto:
        device_map = "auto"
        teacher_model = AutoModelForCausalLM.from_pretrained(model_dir, 
                                                            trust_remote_code=True,
                                                            torch_dtype=torch.bfloat16,
                                                            device_map=device_map)
        teacher_model.config._attn_implementation = "flash_attention_2"
    else:
        device_map = device
        teacher_model = AutoModelForCausalLM.from_pretrained(model_dir, 
                                                            trust_remote_code=True,
                                                            torch_dtype=torch.bfloat16,
                                                            device_map=device_map).to(device)

        teacher_model.config._attn_implementation = "flash_attention_2"

    # Initialize training state
    current_step = 0
    loss_list = []
    grouter_dict = {}
    opt_dict = {}
    sch_dict = {}
    latest_checkpoint = None
    for layer_idx in range(args.moe_layer_start, args.moe_layer_end):
        grouter_dict[layer_idx] = grouter(topk=n_routed_experts, 
                                          num_experts=experts_per_layer, 
                                          scoring_fun='softmax', 
                                          model_config_path=args.model_config_path,
                                          structure_type=args.grouter_type).to(device)
        opt_dict[layer_idx] = torch.optim.AdamW(grouter_dict[layer_idx].parameters(), lr=args.lr)
        sch_dict[layer_idx] = get_cosine_schedule_with_warmup(opt_dict[layer_idx], 
                                                                num_warmup_steps=args.warmup, 
                                                                num_training_steps=args.total_steps)
    
    # Check for checkpoint to resume from
    if args.resume_from:
        checkpoint_path = args.resume_from
        print(f"Resuming from checkpoint: {checkpoint_path}")
        grouter_dict, opt_dict, sch_dict, current_step, loss_list = load_checkpoint(
            checkpoint_path, grouter_dict, opt_dict, sch_dict, device, rank, world_size
        )
        if grouter_dict is None:
            print("Failed to load checkpoint, starting from scratch")
            current_step = 0
            loss_list = []
    elif args.checkpoint_dir:
        # Try to find latest checkpoint
        if latest_checkpoint or args.resume_from_last_ckpt:
            latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
            print(f"Found latest checkpoint: {latest_checkpoint}")
            grouter_dict, opt_dict, sch_dict, current_step, loss_list = load_checkpoint(
                latest_checkpoint, grouter_dict, opt_dict, sch_dict, device, rank, world_size
            )
            if grouter_dict is None:
                print("Failed to load checkpoint, starting from scratch")
                current_step = 0
                loss_list = []
    
    # Calculate consumed_samples for dataloader
    # consumed_samples = current_step * effective_batch_size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
    consumed_samples = current_step * effective_batch_size
    
    # load data
    if args.use_megatron:
        if not MEGATRON_AVAILABLE:
            raise ImportError("Megatron dataloader not available. Please install Megatron-LM properly.")
        
        print(f"Rank {rank}: Using Megatron dataset processing with data distribution...")
        print(f"Rank {rank}: Resuming from step {current_step}, consumed_samples={consumed_samples}")
        args.num_experts = n_routed_experts
        dataloader = create_megatron_dataloader(args, teacher_tokenizer, config, rank, world_size, consumed_samples)
        
        # 验证数据分布（可选，用于调试）
        if rank == 0:
            print("Verifying data distribution across ranks...")
            # 这里可以添加验证逻辑
    else:
        print("Using HuggingFace datasets...")
        data = load_dataset(args.data_path, streaming=True)['train']
        def tokenizer_func(data):
            output = teacher_tokenizer(data['text'], 
                                truncation=True,
                                max_length=args.max_length,
                                padding="max_length")
            return output
        dataset = data.map(tokenizer_func, remove_columns=["url", "timestamp", "text"])
        #val_dataset = dataset['validation']
        dataset = split_dataset_by_node(dataset, rank, world_size)
        collator_fun = DataCollatorForLanguageModeling(teacher_tokenizer, mlm=False)
        #val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collator_fun)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator_fun)


    # add hook to get scores
    logits_dict = {}
    fetch_logits_hook = logits_hook_utils_func(logits_dict, experts_per_layer, device, transpose=args.transpose)

    
    for layer_idx in range(args.moe_layer_start, args.moe_layer_end):
        if args.model_name == 'gpt-oss-120b':
            teacher_model.model.layers[layer_idx].mlp.router.register_forward_hook(fetch_logits_hook(f"gate_{layer_idx}"))
            continue
        teacher_model.model.layers[layer_idx].mlp.gate.register_forward_hook(fetch_logits_hook(f"gate_{layer_idx}"))

    # generate grouter (only if not loaded from checkpoint)
    
    # start dilute
    if rank == 0:
        pbar = tqdm(total=args.total_steps, initial=current_step, ncols=100, desc="Dilution")
        
        # Initialize CSV logging if enabled
        csv_writer = None
        csv_file = None
        if args.log_csv:
            csv_path = os.path.join(args.checkpoint_dir, f"training_log_{args.grouter_type}.csv")
            if args.resume_from or args.resume_from_last_ckpt:
                csv_file = open(csv_path, 'a', newline='')
            else:
                csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            # Write header
            csv_writer.writerow([
                'step', 'loss', 'learning_rate', 'gradient_accumulation_steps', 
                'effective_batch_size', 'memory_gb', 'timestamp'
            ])
            print(f"CSV logging enabled: {csv_path}")
    
    temperature = 2
    
    # Create dataloader iterator (consumed_samples already handled by dataloader)
    dataloader_iter = iter(dataloader)
    
    # Initialize gradient accumulation variables
    accumulation_step = 0
    accumulated_loss = 0.0
    
    batch_idx = 0
    while True:
        # Get next batch
        batch = next(dataloader_iter)
        
        # Calculate actual step number (only increment when we do optimizer step)
        actual_step = current_step + (batch_idx // args.gradient_accumulation_steps)
        batch = {key: value.to(device) for key, value in batch.items()}
        
        teacher_inputs = {
            'input_ids': batch['tokens'], 
            'attention_mask': batch['attention_mask'].float(), 
            'labels': batch['labels'],
        }
        
        with torch.no_grad():
            # update scores_dict
            teacher_model(**teacher_inputs)
    
        # Forward pass and loss computation for each layer
        layer_losses = []
        for layer_idx in range(args.moe_layer_start, args.moe_layer_end):
            teacher_logits = logits_dict[f"gate_{layer_idx}"]
            teacher_scores = F.softmax(teacher_logits / temperature, dim=-1)
            _, glogits = grouter_dict[layer_idx](input_ids=batch['tokens'], 
                                                #attention_mask=batch['attention_mask'].float(), 
                                                #position_ids=batch['position_ids'],
                                                )
            gscores = F.log_softmax(glogits / temperature, dim=-1)

            loss = F.kl_div(gscores, teacher_scores, reduction="batchmean")  # [B*T, E]
            if dist.get_rank() == 0:
                torch.set_printoptions(threshold=float('inf'), edgeitems=torch.inf)
                print("teacher_logits", teacher_logits[0])
                print("glogits", glogits[0].dtype)
            loss = loss * (temperature ** 2)
            
            # Scale loss by gradient accumulation steps
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            layer_losses.append(loss.item())
        
        # Accumulate loss for logging
        accumulated_loss += sum(layer_losses)
        accumulation_step += 1
        
        # Check if we should do optimizer step
        if accumulation_step % args.gradient_accumulation_steps == 0:
            # All-reduce gradients across all layers
            for layer_idx in range(args.moe_layer_start, args.moe_layer_end):
                for p in grouter_dict[layer_idx].parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                
                # Step optimizer and scheduler
                opt_dict[layer_idx].step()
                opt_dict[layer_idx].zero_grad()
                sch_dict[layer_idx].step()
            
            # Logging and checkpointing
            if rank == 0:
                avg_loss = accumulated_loss / args.gradient_accumulation_steps
                current_lr = opt_dict[args.moe_layer_start].param_groups[0]['lr']
                effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
                memory_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                
                # 更新进度条
                pbar.set_postfix({
                    "loss": avg_loss, 
                    "lr": current_lr,
                })
                loss_list.append(avg_loss)
                pbar.update(1)
                
                # 写入CSV日志
                if csv_writer and actual_step % args.log_interval == 0:
                    csv_writer.writerow([
                        actual_step, avg_loss, current_lr, args.gradient_accumulation_steps,
                        effective_batch_size, memory_gb, time.time()
                    ])
                    csv_file.flush()  # 确保数据写入文件
            
            # Reset accumulation variables
            accumulated_loss = 0.0
            accumulation_step = 0
            
            # Save checkpoint at regular intervals
            if args.checkpoint_dir and actual_step > 0 and actual_step % args.checkpoint_interval == 0:
                save_checkpoint(
                    args.checkpoint_dir, actual_step, grouter_dict, opt_dict, sch_dict,
                    loss_list, args, rank, world_size
                )
        
        # Increment batch index
        batch_idx += 1
        
        # Check if training is complete
        if actual_step >= args.total_steps:
            if rank == 0:
                print("finish training")
                pbar.close()
                
                # Close CSV file if opened
                if csv_file:
                    csv_file.close()
                    print(f"Training log saved to: {csv_path}")
                
                # Save final loss history
                loss_path = os.path.join(save_dir, f"{args.grouter_type}_dilute_loss.txt") 
                os.makedirs(os.path.dirname(loss_path), exist_ok=True)
                with open(loss_path, 'w') as f:
                    for num in loss_list:
                        f.write(f"{num}\n")
                
                # Save final checkpoint
                if args.checkpoint_dir:
                    save_checkpoint(
                        args.checkpoint_dir, actual_step, grouter_dict, opt_dict, sch_dict,
                        loss_list, args, rank, world_size
                    )
            
            # Save final models
            for layer_idx in range(args.moe_layer_start, args.moe_layer_end):
                torch.save(grouter_dict[layer_idx].state_dict(), os.path.join(save_dir, f"{args.grouter_type}_{layer_idx}.pth"))
            break
        
if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    args = parse_args(None)
    dilute_grouter(args)
    dist.destroy_process_group()
    