"""
Megatron-LM Integration Module

This module provides integration interfaces with Megatron-LM framework, supporting:
1. Model initialization integration
2. Training process integration
3. Expert migration interface
"""

import logging
import os
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.training import print_rank_0
from megatron.core.transformer.moe.moe_utils import get_default_model_comm_pgs

from .global_expert_migration import GlobalExpertMigration
from .expert_parameter_transfer import ExpertParameterTransfer

logger = logging.getLogger(__name__)


class ExpertMigrationManager:
    """
    Expert Migration Manager
    
    Responsible for managing integration with Megatron-LM framework
    """
    
    def __init__(self, args):
        """
        Initialize expert migration manager
        
        Args:
            args: Megatron command line arguments
        """
        self.args = args
        self.expert_placement_path = getattr(args, 'grouter_expert_placement_path', None)
        self.migration_strategy = getattr(args, 'grouter_migration_strategy', 'allreduce')
        self.compression_ratio = getattr(args, 'grouter_compression_ratio', 0.8)
        self.timeout = getattr(args, 'grouter_migration_timeout', 30.0)
        self.max_retries = getattr(args, 'grouter_migration_max_retries', 3)
        self.verbose = getattr(args, 'grouter_migration_verbose', False)

        assert os.path.exists(self.expert_placement_path), f"Expert placement configuration file does not exist: {self.expert_placement_path}"
            
        
        # Initialize migration components
        self.migration_manager = None
        self.parameter_transfer = None
        
        # Create communication group manager
        model_comm_pgs = get_default_model_comm_pgs()

        # Create parameter transfer manager (for fallback scenarios)
        self.parameter_transfer = ExpertParameterTransfer(
            use_allreduce=(self.migration_strategy == 'allreduce'),
            compression_ratio=self.compression_ratio,
            max_retries=self.max_retries,
            timeout=self.timeout,
            verbose=self.verbose,
            expert_type='TEGroupedMLP', # now only support TEGroupedMLP
            model_comm_pgs=model_comm_pgs
        )
            
        # Create global expert migration manager
        self.migration_manager = GlobalExpertMigration(
            expert_placement_config_path=self.expert_placement_path,
            model_comm_pgs=model_comm_pgs,
            parameter_transfer=self.parameter_transfer,
            num_experts=args.num_experts,
            verbose=self.verbose
        )
            
        if self.verbose:
            print_rank_0("Grouter expert migration components initialized")
            if self.migration_manager.parameter_transfer is not None:
                print_rank_0("Expert parameter transfer module is active")
                print_rank_0(f"Expert type: {self.migration_manager.parameter_transfer.expert_type}")
            else:
                print_rank_0("Expert parameter transfer module fallback mode")
    
    def migrate_experts(self, model: nn.Module, migration_steps: int, migration_plan: Dict=None) -> bool:
        """
        Execute expert migration
        
        Args:
            model: Model containing expert layers
            migration_steps: The current training steps
            migration_plan: The plan for migration
            
        Returns:
            Whether migration was successful
        """
        if self.verbose:
            print_rank_0("Starting Grouter expert migration...")
            
        # Execute expert migration
        success = self.migration_manager.migrate_experts(model, migration_steps, migration_plan)
            
        return success
    
    def get_migration_plan(self) -> Optional[Dict[str, Any]]:
        """Get migration plan"""
        
        return self.migration_manager.get_expert_migration_plan()
    
    def get_optimized_expert_placement(self) -> Optional[Dict[str, Any]]:
        """Get optimized expert placement configuration"""
        
        return self.migration_manager.get_optimized_expert_placement()
