import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, Optional, Tuple
import os
import sys
import random
import numpy as np

# Add grouter path
current_dir = os.path.dirname(os.path.abspath(__file__))
grouter_path = os.path.join(current_dir, '..')
if grouter_path not in sys.path:
    sys.path.insert(0, grouter_path)

from grouter.general_router import Grouter
from .grouter_distillation_hooks import register_grouter_hooks, clear_grouter_hooks
import json

from transformers import get_cosine_schedule_with_warmup

# Import Megatron parallel state utilities
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.utils import is_pp_first_stage


class GrouterDistillationTrainer:
    """Grouter distillation trainer"""
    
    def __init__(self, args, model_config, device: str):
        self.args = args
        self.model_config = model_config
        self.device = device
        self.logits_dict = {}
        
        # Get dstillation setting from model config
        self.moe_layer_start = model_config.grouter_moe_layer_start
        self.moe_layer_end = model_config.grouter_moe_layer_end
        self.distillation_temperature = model_config.grouter_distillation_temperature
        self.grouter_init_seed = model_config.grouter_init_seed  # Fixed seed for grouter initialization
        self.finetune_scores = args.grouter_distillation_finetune_scores

        self.distillation_lr = args.lr
        self.warmup_iters = args.lr_warmup_iters
        self.train_iters = args.train_iters

        # Cauculate gradient_accumulation_steps
        data_parallel_size = args.world_size // (args.tensor_model_parallel_size * args.pipeline_model_parallel_size)
        self.gradient_accumulation_steps = args.global_batch_size // (args.micro_batch_size * data_parallel_size)

        # initialization grouter
        self.grouter_config_path = self.model_config.grouter_config_path

        # log
        self.checkpoint_dir = model_config.grouter_checkpoint_dir
        self.checkpoint_interval = model_config.grouter_checkpoint_interval
        self.resume_from = model_config.grouter_resume_from
        
        # loss log
        self.loss_log_file = os.path.join(self.checkpoint_dir, "grouter_distillation_loss.log")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        with open(self.loss_log_file, 'w') as f:
            pass

        self.num_experts = model_config.num_moe_experts
        
        # Initialize Grouter models
        self.loss_dict = {"grouter distillation loss": 0.0}
        # Initialize layer-wise loss tracking for multi-score distillation
        self.layer_loss_dict = {}

        if self.args.pipeline_model_parallel_size > 1:
            self.comm_group = self._get_distillation_comm_group()
        else:
            self.comm_group = None

        self.accumulation_step = 0
        
        self._initialize_grouter_models()
    
    def _initialize_grouter_models(self):
        """Initialize Grouter models"""
        
        # Read JSON configuration file
        with open(self.grouter_config_path, 'r', encoding='utf-8') as f:
            grouter_config = json.load(f)
        
        torch.manual_seed(self.grouter_init_seed)
        torch.cuda.manual_seed_all(self.grouter_init_seed)
        np.random.seed(self.grouter_init_seed)
        random.seed(self.grouter_init_seed)
        
        # Create Grouter model using configuration from JSON
        distillation_layer_num = self.moe_layer_end - self.moe_layer_start
        grouter_model = Grouter(**grouter_config, output_logits=True).to(self.device)
            
        # Create optimizer
        if self.finetune_scores:
            for p in grouter_model.parameters():
                p.requires_grad = False
            grouter_model.grouter_structure.score.score.weight.requires_grad = True
            trainable_param = [grouter_model.grouter_structure.score.score.weight]
        else:
            trainable_param = grouter_model.parameters()
                
        optimizer = torch.optim.AdamW(
            trainable_param, 
            lr=self.distillation_lr
        )
            
        # Create learning rate scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_iters,  # Default warmup steps
            num_training_steps=self.train_iters # Default total training steps
        )
            
        self.grouter_model = grouter_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        if self.resume_from:
            self.load_checkpoint(self.resume_from)

    
    def register_hooks(self, model):
        """Register hooks for the model"""
        register_grouter_hooks(
            model, 
            self.logits_dict, 
            self.device,
            self.moe_layer_start,
            self.moe_layer_end
        )
    
    def clear_hooks(self, model):
        """Clear model hooks"""
        clear_grouter_hooks(
            model,
            self.moe_layer_start,
            self.moe_layer_end
        )
    
    def _get_distillation_comm_group(self):
        """
        Get the appropriate communication group for distillation synchronization.
        
        Returns:
            torch.distributed.ProcessGroup or None: The communication group to use
        """
        world_group = dist.group.WORLD
        world_size = dist.get_world_size(world_group)
        all_ranks = list(range(world_size))
    
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_rank = dist.get_rank(pp_group)
    
        first_stage_ranks = []
        for rank in all_ranks:
            with torch.no_grad():
                all_pp_ranks = [torch.tensor(-1, device=self.device) for _ in range(world_size)]
                dist.all_gather(all_pp_ranks, torch.tensor(pp_rank, device=self.device), group=world_group)
            for r, pp_r in zip(all_ranks, all_pp_ranks):
                if pp_r.item() == 0:
                    first_stage_ranks.append(r)
        first_stage_ranks = sorted(list(set(first_stage_ranks)))
    
        first_pp_group = dist.new_group(ranks=first_stage_ranks, backend='nccl')
        return first_pp_group
    
    def distill_step(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, float]:
        """
        Execute one distillation training step
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
        Returns:
            Loss dictionary
        """
        # Only participate if we're in the pipeline stage that contains the MoE layers we want to distill
        pp_group = parallel_state.get_pipeline_model_parallel_group()
            
        # Check if current PP stage contains the MoE layer we want to distill
        should_participate = is_pp_first_stage(pp_group)
        
        if not should_participate:
            # Return empty loss dict for non-participating stages
            return {}
        
        teacher_layer_scores = {}
        for layer_idx in range(self.moe_layer_start, self.moe_layer_end):
            # Get teacher logits (detached to prevent gradient flow to teacher model)
            teacher_logits = self.logits_dict[f"gate_{layer_idx}"].detach()
            teacher_scores = teacher_logits.view(-1, self.num_experts)
            teacher_scores = F.softmax(teacher_scores / self.distillation_temperature, dim=-1)
            teacher_layer_scores[layer_idx] = teacher_scores
            
        # Get student logits
        _, student_logits = self.grouter_model(input_ids)

        student_scores = F.log_softmax(student_logits / self.distillation_temperature, dim=-1)
            
        # Compute KL divergence loss with mean reduction
        loss = F.kl_div(student_scores, teacher_scores, reduction="batchmean")
        loss = loss * (self.distillation_temperature ** 2)

        # losses equal to loss defaultly
        losses = loss

        # Backward pass
        loss = loss / self.gradient_accumulation_steps
        loss.backward()

        self.loss_dict["grouter distillation loss"] += loss.item()
        self.accumulation_step += 1
            
        if self.accumulation_step % self.gradient_accumulation_steps == 0:
            # Gradient synchronization - only within participating ranks
            for p in self.grouter_model.parameters():
                if p.grad is not None:
                    if self.comm_group is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=self.comm_group)
                    else:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            real_step = self.accumulation_step // self.gradient_accumulation_steps
            if real_step % self.checkpoint_interval == 0:
                self.save_checkpoint(real_step)
            
            # log loss with lr and iteration
            losses = self.loss_dict["grouter distillation loss"]
            if dist.get_rank() == 0:
                current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else 0.0
                with open(self.loss_log_file, 'a') as f:
                    # Single-score distillation: log only total loss
                    f.write(f"Step {real_step} | LR {current_lr:.6f} | Distillation Loss {losses:.6f}\n")
            
            self.loss_dict["grouter distillation loss"] = 0.0
            # Reset layer-wise loss tracking
            self.layer_loss_dict = {}
        return {"grouter distillation loss": torch.tensor(losses, device=self.device)}
            
    def save_checkpoint(self, step: int):
        """Save checkpoint"""
        if dist.get_rank() != 0:
            return
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'step': step,
            'grouter_states': self.grouter_model.state_dict(),
            'optimizers': self.optimizer.state_dict(),
            'schedulers': self.scheduler.state_dict()
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"grouter_checkpoint_step_{step}.pt")
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise RuntimeError("The grouter checkpoint path is not exist.")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        state_dict = checkpoint['grouter_states']
        state_dict = checkpoint['grouter_states']
        if "bias" not in state_dict:
            state_dict["bias"] = torch.zeros(self.grouter_model.num_experts)
        self.grouter_model.load_state_dict(state_dict)
        
        # Load optimizer states
        if not self.finetune_scores:
            optim = checkpoint['optimizers']
            self.optimizer.load_state_dict(optim)
        
            # Load scheduler states
            scheduler = checkpoint['schedulers']
            self.scheduler.load_state_dict(scheduler)
        
        self.accumulation_step = self.gradient_accumulation_steps * checkpoint['step']
