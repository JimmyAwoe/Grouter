import torch
import torch.distributed as dist
import os
import argparse
import json
from utils_grouter.grouter import grouter
from utils_grouter.utils import create_megatron_dataloader
from utils_grouter.core.config import OptimizationConfig
from utils_grouter.grouter.structure.general_structure import GrouterScore, GrouterMHAConfig, GrouterMLAConfig
import copy
from megatron.training.tokenizer import build_tokenizer
import numpy as np
from torch.optim import AdamW
import logging
from transformers import get_cosine_schedule_with_warmup

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true")

    # grouter
    parser.add_argument("--grouter-config-path", type=str, required=True)
    parser.add_argument("--grouter-checkpoint-path", type=str, default=None)

    # dataset
    parser.add_argument("--data-prefix", type=str, nargs='*', required=True,
                       help="Data prefix for Megatron dataset")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for Megatron dataset processing")
    parser.add_argument("--tokenizer-type", type=str, default=None,
                       help="Tokenizer type for Megatron")
    parser.add_argument("--tokenizer-model", type=str, default=None,
                       help="Tokenizer model for Megatron")
    
    # training
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help="Number of steps to accumulate gradients before updating")
    parser.add_argument("--save-interval", type=int, default=100000)
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save checkpoints")
    parser.add_argument("--loss-type", choices=['aux_loss', 'maxvio', 'maxvio_continuous', 'variance', 'kl', 'importance', 'combined'])
    parser.add_argument("--importance-loss-weight", type=float, default=1.0,
                       help="Weight for expert importance balance loss (used in 'combined' mode)")
    parser.add_argument("--finetune-mode", type=str, default='bias_only', choices=['bias_only', 'last_layer', 'full'],
                       help="Finetuning mode: bias_only, last_layer, or full")
    parser.add_argument("--finetune-optim", type=str, default='gradient', choices=['gradient', 'ste'],
                       help="ste is the same update method used in aux loss free paper")
    
    # log
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(args)
    return args

class LossFreeBiasOptimizer:
    """
    Optimizer for updating linear layer's bias to achieve MoE load balancing,
    implementing the Loss-Free strategy (b ← b - alpha·sign(F - Q)) with STE.
    """
    def __init__(self, bias: torch.nn.Parameter):
        self.num_experts = len(bias)
        Q = torch.full((self.num_experts,), 1 / self.num_experts)
        
        # Initialize core components
        self.Q = Q.to(bias.device)  # Move Q to target device
        self.b = bias  # Reference to linear layer's bias (to be updated)

    def step(self, F: torch.Tensor, alpha: float):
        """Perform one optimization step to update the linear layer's bias."""
        F = F.to(self.b.device)

        # STE update: F - Q 
        load_diff = F - self.Q
        dist.all_reduce(load_diff, op=dist.ReduceOp.AVG)

        # Compute update term: -α * sign(load_diff) (core Loss-Free rule)
        update_term = -alpha * torch.sign(load_diff)
        #update_term = -alpha * load_diff / torch.norm(load_diff)

        # Update bias: b ← b + update_term
        with torch.no_grad():
            self.b.add_(update_term) 

    def zero_grad(self):
        """Dummy method for consistency with PyTorch optimizer workflow"""
        pass

class LoadBalanceLoss:
    """Load Balance Loss for MoE routing"""
    
    def __init__(self, num_experts, topk):
        self.num_experts = num_experts
        self.topk = topk
        
    def compute_load_balance_loss(self, logits, topk_indices):
        # Get routing probabilities
        num_experts = logits.shape[-1]
        routing_probs = torch.softmax(logits, dim=-1)
        routing_probs = routing_probs.mean(axis=1)
        
        # Compute expert selection frequencies
        select_count = torch.vstack([torch.bincount(idx, minlength=num_experts) for idx in topk_indices])
        select_freq = select_count / select_count.sum(dim=-1, keepdim=True)

        
        return (routing_probs * select_freq).mean(axis=0).sum()
    
    def compute_maxvio(self, logits, topk_indices):
        """Compute maximum violation from uniform distribution"""
        num_experts = logits.shape[-1]

        select_count = torch.vstack([torch.bincount(idx, minlength=num_experts) for idx in topk_indices])
        #select_count = select_count.view(-1)
        select_freq = select_count / select_count.sum(dim=-1, keepdim=True)

        expected_freq = 1 / num_experts
        return ((select_freq.max(axis=-1)[0] - expected_freq) / expected_freq).mean() 
    
    def compute_load(self, logits, topk_indices):

        num_experts = logits.shape[-1]

        # Compute expert selection frequencies
        select_count = torch.vstack([torch.bincount(idx, minlength=num_experts) for idx in topk_indices])
        select_freq = select_count / select_count.sum(dim=-1, keepdim=True)
        return select_freq.mean(axis=0)


def save_checkpoint(model, step, output_dir, rank, finetune_mode):
    """Save model checkpoint"""
    if rank == 0:
        if finetune_mode == 'bias_only':
            checkpoint = model.state_dict()['bias']
            checkpoint_path = os.path.join(output_dir, f'checkpoint_step_{step}_bias.pt')
        else:
            checkpoint = model.state_dict()
            checkpoint_path = os.path.join(output_dir, f'checkpoint_step_{step}.pth')
        os.makedirs(output_dir, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


def main(args):
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    device = f"cuda:{rank}"

    assert os.path.exists(args.output_dir), "The save path doesn't exist"
    
    dist.init_process_group(backend='nccl')
    
    # Setup logging
    if rank == 0:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info(f"Starting training with {world_size} processes")
        logger.setLevel(logging.INFO)
    
    # Load grouter model
    with open(args.grouter_config_path, 'r') as f:
        grouter_config = json.load(f)
    grt = grouter(**grouter_config, output_logits=True)
    args.topk = grouter_config["topk"]
    num_experts = grouter_config["target_num_experts"] if "target_num_experts" in grouter_config else grouter_config["num_experts"]
    args.num_experts = int(num_experts)
    
    grt = grt.to(device=device)
    if args.bf16:
        grt = grt.to(torch.bfloat16)
    
    # Load checkpoint
    if args.grouter_checkpoint_path is None:
        assert args.finetune_mode == 'full', "Learn from scratch must set fully trainable"
    elif args.grouter_checkpoint_path.endswith('.pth'):
        checkpoint = torch.load(args.grouter_checkpoint_path, map_location=device)
    elif args.grouter_checkpoint_path.endswith('.pt'):
        checkpoint = torch.load(args.grouter_checkpoint_path, map_location=device)
        if 'grouter_states' in checkpoint:
            checkpoint = checkpoint['grouter_states'][1]
    
    if args.grouter_checkpoint_path is not None:
        if "bias" not in checkpoint:
            checkpoint["bias"] = torch.zeros(grt.num_experts)
        grt.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.grouter_checkpoint_path}")
    
    # Setup dataset
    config = OptimizationConfig(
        num_nodes=1,
        num_experts=args.num_experts,
        training_config=None,
        experts_per_gpu=None,
        network_topology=None,
        split_config="1,0,0",
        random_seed=getattr(args, 'random_seed', 42),
        train_iters=args.max_steps,
        global_batch_size=args.batch_size * world_size,
        tokenizer_type=args.tokenizer_type,
        tokenizer_model=args.tokenizer_model
    )
    config.rank = rank
    config.make_vocab_size_divisible_by = 128
    config.tensor_model_parallel_size = 1
    config.trust_remote_code = True
    
    tokenizer = build_tokenizer(config)
    args.checkpoint_dir = args.output_dir
    dataloader = create_megatron_dataloader(args, tokenizer, config, rank, world_size, 0)
    
    # Initialize load balance loss
    lb_loss_fn = LoadBalanceLoss(args.num_experts, args.topk)
    
    # Setup trainable parameters
    trainable_params = []
    
    if args.finetune_mode == 'bias_only':
        # Option 1: Only bias
        if grt.bias_routing != True:
            grt.bias_routing = True
            if rank == 0:
                logger.info("The bias_routing is not True and will be set automatically")

        for p in grt.parameters():
            p.requires_grad = False
        grt.bias.to(device)
        grt.bias.to(torch.float32)
        grt.bias.requires_grad = True
        trainable_params = [grt.bias]
    elif args.finetune_mode == 'last_layer':
        # Option 2: Last layer (weight + bias)
        for p in grt.parameters():
            p.requires_grad = False
        grt.grouter_structure.score.score.weight.requires_grad = True
        trainable_params = [
            grt.grouter_structure.score.score.weight,
        ]
    elif args.finetune_mode == 'full':
        # Option 3: All parameters (full training/finetuning)
        for p in grt.parameters():
            p.requires_grad = True
        grt.bias.requires_grad = False
        trainable_params = list(grt.parameters())

    total_params = sum(p.numel() for p in trainable_params)
    if rank == 0:
        logger.info(f"Total trainable parameters: {total_params:,}")

    # Setup optimizer and scheduler
    if args.finetune_optim == "gradient":
        optimizer = AdamW(trainable_params, 
                        lr=args.learning_rate, 
                        weight_decay=args.weight_decay)

        # Cosine annealing scheduler
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=args.warmup_steps, 
                                                    num_training_steps=args.max_steps)
    
    elif args.finetune_optim == "ste":
        assert args.finetune_mode == "bias_only", "ste update method is only suitable for bias only mode"
        optimizer = LossFreeBiasOptimizer(grt.bias)
        f_list = []

        dummy_optimizer = AdamW(trainable_params, 
                        lr=args.learning_rate, 
                        weight_decay=args.weight_decay)

        scheduler = get_cosine_schedule_with_warmup(dummy_optimizer, 
                                                    num_warmup_steps=args.warmup_steps, 
                                                    num_training_steps=args.max_steps)
        
    # Training loop
    data_iter = iter(dataloader)
    step = 0
    accumulated_steps = 0
    
    if rank == 0:
        logger.info("Starting training loop...")
        logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        batch = {key: value.to(device) for key, value in batch.items()}
        
        # Forward pass
        topk_idx, logits = grt(
            input_ids=batch['tokens'],
            attention_mask=batch['attention_mask'].float(),
            position_ids=batch['position_ids']
        )
        
        # Compute loss
        if isinstance(topk_idx, tuple):
            topk_idx = topk_idx[0] # Get balanced topk
        topk_idx = topk_idx.reshape(args.max_length, args.batch_size, args.topk).permute(1,0,2)
        topk_idx = topk_idx.contiguous().view(args.batch_size, -1)

        logits = logits.reshape(args.max_length, args.batch_size, args.num_experts).permute(1,0,2)

        aux_loss = lb_loss_fn.compute_load_balance_loss(logits, topk_idx)
        maxvio = lb_loss_fn.compute_maxvio(logits, topk_idx)
        F = lb_loss_fn.compute_load(logits, topk_idx)
        
        if args.finetune_optim == 'ste':
            f_list.append(F)
        elif args.loss_type == 'aux_loss':
            loss = aux_loss
        elif args.loss_type == 'maxvio':
            loss = maxvio
        else:
            raise ValueError(f"Unknown loss type: {args.loss_type}")


        # Backward pass
        if accumulated_steps == 0 and args.finetune_optim == "gradient":
            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            optimizer.zero_grad()
            loss.backward()
        
        accumulated_steps += 1
        
        # Only update weights after accumulating enough gradients
        if accumulated_steps == args.gradient_accumulation_steps:
            if args.finetune_optim == 'ste':
                if len(f_list) == 1:
                    avg_f = f_list[0]
                else:
                    stacked = torch.stack(f_list, dim=-1)  # 在最后新增一个维度堆叠
                    avg_f = torch.mean(stacked, dim=-1)   
                current_lr = scheduler.get_last_lr()[0]
                optimizer.step(avg_f, current_lr)
                f_list = []
            else:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(grt.parameters(), max_norm=1.0)

                for p in grt.parameters():
                    if p.requires_grad is True:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            
                optimizer.step()
            scheduler.step()
            
            accumulated_steps = 0
            step += 1
            
            # Logging
            if step % args.log_interval == 0 and rank == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {step}/{args.max_steps} | "
                    f"AuxLoss: {aux_loss.item():.6f} | "
                    f"Maxvio: {maxvio.item():.6f} | "
                    f"LR: {current_lr:.2e}"
                )
            
            # Save checkpoint
            if step % args.save_interval == 0:
                save_checkpoint(grt, step, args.output_dir, rank, args.finetune_mode)
    
    # Final checkpoint
    if rank == 0:
        save_checkpoint(grt, step, args.output_dir, rank, args.finetune_mode)
    
    if rank == 0:
        logger.info("Training completed!")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_args(None)
    main(args)
