"""
Megatron-based dataloader for grouter distillation training
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any, List, Tuple
import logging
from pathlib import Path

try:
    # Works when imported as utils.utils.megatron_dataloader
    from ..core.config import OptimizationConfig, DataPaths, MegatronPaths
except ImportError:
    # Fallback for environments importing this module as utils.megatron_dataloader
    from core.config import OptimizationConfig, DataPaths, MegatronPaths
from megatron.core.datasets import indexed_dataset
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.utils import Split
from megatron.training.utils import get_blend_and_blend_per_split

logger = logging.getLogger(__name__)


class MegatronPretrainingSampler:
    """
    Megatron-style data sampler for distributed training
    Based on Megatron's MegatronPretrainingSampler
    """
    
    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]

class MegatronGrouterDataset:
    """
    DataLoader for Megatron-processed grouter training data
    """
    
    def __init__(self, config: OptimizationConfig, data_paths: DataPaths, tokenizer, sequence_length, args=None):
        """
        Initialize Megatron data loader
        
        Args:
            config: Optimization configuration
            data_paths: Data path configuration
            tokenizer: Tokenizer instance
            sequence_length: Sequence length
            args: Command line arguments
        """
        self.config = config
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.args = args
        self.gpt_config = self._create_gpt_dataset_config()
    
    def _create_gpt_dataset_config(self):
        """
        Create GPTDatasetConfig exactly like Megatron's core_gpt_dataset_config_from_args
        """
        blend: Optional[Tuple[List[str], Optional[List[float]]]]
        blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
        dataset_path = MegatronPaths(data_path=self.args.data_prefix)
        blend, blend_per_split = get_blend_and_blend_per_split(dataset_path)

        # Create GPTDatasetConfig - split_matrix will be calculated automatically in __post_init__
        config = GPTDatasetConfig(
            random_seed=self.config.random_seed,
            sequence_length=self.sequence_length,
            blend=blend,  
            blend_per_split=blend_per_split,
            split=self.config.split_config,  # This will be parsed in __post_init__
            num_dataset_builder_threads=1,
            path_to_cache=os.path.join(self.data_paths.output_dir, "cache"),
            mmap_bin_files=True,
            tokenizer=self.tokenizer,
            reset_position_ids=True,
            reset_attention_mask=True,
            eod_mask_loss=True,
            create_attention_mask=True
        )
        
        return config
    
    def _build_megatron_datasets(self):
        """
        Build train dataset using BlendedMegatronDatasetBuilder
        exactly like Megatron's train_valid_test_datasets_provider
        """
        logger.info("Building datasets with BlendedMegatronDatasetBuilder")
        
        # Calculate sample sizes for training
        train_samples = (self.config.train_iters + 100) * self.config.global_batch_size # ensure enough samples
        valid_samples = 0
        test_samples = 0
        train_val_test_num_samples = [train_samples, valid_samples, test_samples]
        
        def is_dataset_built_on_rank():
            return True  # Always build on current rank for our use case
        
        # Build datasets exactly like Megatron
        builder = BlendedMegatronDatasetBuilder(
            GPTDataset,
            train_val_test_num_samples,
            is_dataset_built_on_rank,
            self.gpt_config
        )
        
        train_ds, _, _= builder.build()
        
        logger.info(f"Built dataset with {len(train_ds) if train_ds else 0} samples")
        
        return train_ds


def build_pretraining_data_loader(dataset, consumed_samples, micro_batch_size, 
                                 data_parallel_rank, data_parallel_size, 
                                 num_workers=0, pin_memory=True, drop_last=True):
    """
    Build dataloader given an input dataset using Megatron-style sampler
    Based on Megatron's build_pretraining_data_loader
    """
    if dataset is None:
        return None
    
    # Create Megatron-style sampler
    batch_sampler = MegatronPretrainingSampler(
        total_samples=len(dataset),
        consumed_samples=consumed_samples,
        micro_batch_size=micro_batch_size,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
        drop_last=drop_last
    )
    
    # Create DataLoader with Megatron-style sampler (no collate_fn needed)
    # PyTorch automatically handles dictionary batching
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )


def create_megatron_dataloader(args, tokenizer, config, rank: int = 0, world_size: int = 1, consumed_samples: int = 0) -> DataLoader:
    """
    Factory function to create Megatron DataLoader using Megatron-style data processing
    
    Args:
        args: Command line arguments
        tokenizer: Tokenizer instance
        config: OptimizationConfig instance
        rank: Current process rank
        world_size: Total number of processes
        consumed_samples: Number of samples already consumed (for checkpoint resuming)
        
    Returns:
        DataLoader instance
    """
    # Create data paths
    data_paths = DataPaths(
        dataset_path=None,
        predispatch_path=None,
        output_dir=args.checkpoint_dir,
        prefix=args.data_prefix
    )
    
    # Create dataset wrapper
    loader = MegatronGrouterDataset(config, data_paths, tokenizer, args.max_length, args)
    
    # Build the actual Megatron dataset
    dataset = loader._build_megatron_datasets()
    
    # Use Megatron-style data loader with proper rank-based sampling
    micro_batch_size = args.batch_size
    
    dataloader = build_pretraining_data_loader(
        dataset=dataset,
        consumed_samples=consumed_samples,
        micro_batch_size=micro_batch_size,
        data_parallel_rank=rank,
        data_parallel_size=world_size,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with Megatron
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Drop last incomplete batch
    )
    
    logger.info(f"Rank {rank}: Created Megatron DataLoader with micro_batch_size={micro_batch_size}, "
                f"dataset_size={len(dataset) if hasattr(dataset, '__len__') else 'unknown'}")
    
    return dataloader


def create_megatron_eval_dataloader(args, tokenizer, config, rank: int = 0, world_size: int = 1, consumed_samples: int = 0) -> DataLoader:
    """
    Create a DataLoader for evaluation using Megatron dataset processing
    
    Args:
        args: Command line arguments
        tokenizer: Tokenizer instance
        config: OptimizationConfig instance
        rank: Current process rank
        world_size: Total number of processes
        consumed_samples: Number of samples already consumed (for checkpoint resuming)
        
    Returns:
        DataLoader instance for evaluation
    """
    # Create data paths
    data_paths = DataPaths(
        dataset_path=args.data_path,
        predispatch_path=None,
        output_dir=args.checkpoint_dir,
        prefix=args.data_prefix
    )
    
    # Create dataset wrapper
    loader = MegatronGrouterDataset(config, data_paths, tokenizer, args.max_length, args)
    
    # Build the actual Megatron dataset
    dataset = loader._build_megatron_datasets()
    
    # Use Megatron-style data loader for evaluation
    micro_batch_size = args.batch_size
    
    dataloader = build_pretraining_data_loader(
        dataset=dataset,
        consumed_samples=consumed_samples,
        micro_batch_size=micro_batch_size,
        data_parallel_rank=rank,
        data_parallel_size=world_size,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with Megatron
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False  # Don't drop last batch for evaluation
    )
    
    logger.info(f"Created Megatron Eval DataLoader with micro_batch_size={micro_batch_size}, "
                f"dataset_size={len(dataset) if hasattr(dataset, '__len__') else 'unknown'}")
    
    return dataloader
