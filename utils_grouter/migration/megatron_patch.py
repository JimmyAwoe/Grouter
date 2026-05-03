"""
Megatron-LM Patch Module

This module provides patches for Megatron-LM framework to integrate Grouter expert migration functionality
"""

import logging
from typing import Optional
import torch.nn as nn

from megatron.core.transformer.moe.moe_utils import ModelCommProcessGroups

from ..megatron_integration import (
    add_grouter_expert_migration_args,
    integrate_grouter_migration_in_model_init,
    integrate_grouter_migration_in_training_start
)

logger = logging.getLogger(__name__)


def patch_megatron_arguments(parser):
    """
    Add Grouter expert migration parameters to Megatron argument parser
    
    Args:
        parser: Megatron's ArgumentParser instance
    """
    add_grouter_expert_migration_args(parser)


def patch_megatron_model_initialization(model: nn.Module, 
                                       args,
                                       model_comm_pgs: Optional[ModelCommProcessGroups] = None) -> bool:
    """
    Integrate Grouter expert migration during Megatron model initialization
    
    Args:
        model: Model containing expert layers
        args: Megatron command line arguments
        model_comm_pgs: Model communication group manager
        
    Returns:
        Whether integration was successful
    """
    return integrate_grouter_migration_in_model_init(model, args, model_comm_pgs)


def patch_megatron_training_start(model: nn.Module,
                                 args,
                                 model_comm_pgs: Optional[ModelCommProcessGroups] = None) -> bool:
    """
    Integrate Grouter expert migration at Megatron training start
    
    Args:
        model: Model containing expert layers
        args: Megatron command line arguments
        model_comm_pgs: Model communication group manager
        
    Returns:
        Whether integration was successful
    """
    return integrate_grouter_migration_in_training_start(model, args, model_comm_pgs)


def apply_megatron_patches():
    """Apply all Megatron patches"""
    logger.info("Applying Grouter expert migration Megatron patches...")
    
    # Can add more patch logic here
    # For example: monkey patching Megatron functions
    
    logger.info("Megatron patches applied")