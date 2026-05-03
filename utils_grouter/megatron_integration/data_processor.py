"""
Megatron dataset processing utilities

This module provides utilities to work with Megatron's processed datasets
and handle tokenizer integration.
"""

import logging
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from ..core.config import OptimizationConfig

from megatron.core.datasets import indexed_dataset
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
from megatron.training.tokenizer import build_tokenizer
MEGATRON_AVAILABLE = True

logger = logging.getLogger(__name__)

class TokenizerFactory:
    """
    Factory for creating appropriate tokenizer instances
    """
    
    @staticmethod
    def create_tokenizer(config: OptimizationConfig):
        """
        Create tokenizer instance based on configuration
        
        Args:
            config: Configuration containing tokenizer settings
            
        Returns:
            Tokenizer instance
        """
        assert MEGATRON_AVAILABLE, "The Megatron is not available."

        # In a real implementation, you'd create the actual Megatron tokenizer
        # based on the tokenizer type specified in config
        logger.info("Creating Megatron tokenizer")

        # Adding needed argument in build tokenizer, this is for compatibility
        # with build_tokenizer.
        config.rank = 1
        config.make_vocab_size_divisible_by = 128
        config.tensor_model_parallel_size = 1
        config.trust_remote_code = True

        # Build tokenizer
        return build_tokenizer(config)

def create_tokenizer(config: OptimizationConfig) -> Any:
    """Create tokenizer instance"""
    return TokenizerFactory.create_tokenizer(config)
