"""
Megatron dataset processing pipeline for grouter EP optimization

This module implements the complete Megatron data processing pipeline to convert
IndexedDataset binary files into processed training datasets, exactly like Megatron's
train_valid_test_datasets_provider function, and then serializes the results back
to IndexedDataset format for efficient loading.
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from ..core.config import OptimizationConfig, DataPaths, MegatronPaths
from ..utils.data_structures import AlignedDispatchGPTDataset, AlignedDispatchBlendedDataset

from megatron.core.datasets import indexed_dataset
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.utils import Split
from megatron.training.utils import get_blend_and_blend_per_split
MEGATRON_AVAILABLE = True

logger = logging.getLogger(__name__)


class MegatronDatasetProcessor:
    """
    Full Megatron dataset processing pipeline
    
    This class replicates Megatron's train_valid_test_datasets_provider functionality:
    1. Reads IndexedDataset binary files
    2. Creates GPTDatasetConfig
    3. Uses BlendedMegatronDatasetBuilder to build train/valid/test datasets  
    4. Processes all samples through GPTDataset.__getitem__
    5. Serializes processed data back to IndexedDataset format
    """
    
    def __init__(self, config: OptimizationConfig, data_paths: DataPaths, tokenizer=None):
        if not MEGATRON_AVAILABLE:
            raise ImportError("Megatron-LM is required for dataset processing")
            
        self.config = config
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        
        # Dataset split configuration - matches Megatron defaults
        self.split_config = config.split_config  # train, valid, test
        
    def process_node_datasets(self, node_ids: List[int]) -> Dict[int, Dict[str, str]]:
        """
        Process datasets for all nodes through complete Megatron pipeline
        
        Args:
            node_ids: List of node IDs to process
            
        Returns:
            Dict mapping node_id to dict of split names to processed dataset paths
        """
        logger.info(f"Processing {len(node_ids)} node datasets through Megatron pipeline")
        
        processed_paths = {}
        for node_id in node_ids:
            logger.info(f"Processing node {node_id} through Megatron pipeline")
            node_paths = self._process_single_node_dataset(node_id)
            processed_paths[node_id] = node_paths
            
        logger.info("Completed Megatron pipeline processing for all nodes")
        return processed_paths
    
    def _process_single_node_dataset(self, node_id: int) -> Dict[str, str]:
        """
        Process a single node's dataset through complete Megatron pipeline
        
        Args:
            node_id: Node ID to process
            
        Returns:
            Dict mapping split names to processed dataset paths
        """
        # Input paths
        prefix = self.data_paths.prefix
        if len(self.data_paths.prefix) == 1:
            input_data_prefix = [1.0, os.path.join(self.data_paths.node_data_dir, f"{prefix[0]}_node_{node_id}_data")]
            input_dispatch_prefix = [1.0, os.path.join(self.data_paths.node_dispatch_dir, f"{prefix[0]}_node_{node_id}_dispatch")]
        else:
            add_data_prefix = lambda x: os.path.join(self.data_paths.node_data_dir, f"{x}_node_{node_id}_data")
            add_dispatch_prefix = lambda x: os.path.join(self.data_paths.node_dispatch_dir, f"{x}_node_{node_id}_dispatch")
            input_data_prefix = [add_data_prefix(x) if i % 2 == 1 else x for i, x in enumerate(prefix)]
            input_dispatch_prefix = [add_dispatch_prefix(x) if i % 2 == 1 else x for i, x in enumerate(prefix)]
        self.data_paths.data_path = input_data_prefix

        # Verify input files exist
        for i, (data, dispatch) in enumerate(zip(input_data_prefix, input_dispatch_prefix)):
            if i % 2 == 1:
                if not self._verify_input_files(data, dispatch):
                    raise FileNotFoundError(f"Input files not found for node {node_id}")
        
        data_path = MegatronPaths(data_path=input_data_prefix)
        
        # Create output directory
        output_dir = os.path.join(self.data_paths.output_dir, "megatron_processed", f"node_{node_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Create GPTDatasetConfig exactly like Megatron
        gpt_config = self._create_gpt_dataset_config(data_path)
        gpt_config.no_document_idx_shuffle = True
        
        # Step 2: Use BlendedMegatronDatasetBuilder to build datasets
        train_ds, valid_ds, test_ds = self._build_megatron_datasets(gpt_config)

        # Step 3: Create aligned dispatch datasets for each split
        dispatch_datasets = {}
        for split_name, dataset in [("train", train_ds), ("valid", valid_ds), ("test", test_ds)]:
            if dataset is not None:
                dispatch_datasets[split_name] = self._align_dispatch_token_dataset(dataset, input_dispatch_prefix)
        
        # Step 4: Process all samples and serialize to IndexedDataset
        processed_paths = {}
        splits = [("train", train_ds), ("valid", valid_ds), ("test", test_ds)]
        
        for split_name, dataset in splits:
            if dataset is not None:
                dispatch_dataset = dispatch_datasets[split_name]
                processed_path = self._serialize_processed_dataset(
                    dataset, dispatch_dataset, node_id, split_name, output_dir
                )
                processed_paths[split_name] = processed_path
        
        logger.info(f"Node {node_id} processed through Megatron pipeline")
        return processed_paths


    def _align_dispatch_token_dataset(self, token_dataset: BlendedDataset, dispatch_prefix: List[str]) -> BlendedDataset:
        """Align dispatch result for each token in input BlendedDataset"""
        dispatch_path = MegatronPaths(dispatch_prefix)

        # Get path and weights for each dataset
        # Now only consider blended dataset, i.e., the dispatch_prefix should be 
        # like ['weight1', 'prefix1', 'weight2', 'prefix2',...] 
        prefixs, _ = get_blend_and_blend_per_split(dispatch_path)[0]

        dispatch_datasets = []
        for prefix in prefixs:
            dataset = indexed_dataset.IndexedDataset(prefix)
            dispatch_datasets.append(dataset)
        
        # Create aligned dispatch dataset by processing each sample through the same pipeline
        aligned_dispatch_dataset = self._create_aligned_dispatch_dataset(
            token_dataset, dispatch_datasets 
        )
        
        return aligned_dispatch_dataset

    def _create_aligned_dispatch_dataset(self, token_dataset: BlendedDataset, 
                                       dispatch_datasets: List[indexed_dataset.IndexedDataset]) -> BlendedDataset:
        """
        Create a dispatch dataset that aligns with the processed token dataset
        
        This method replicates the exact data processing pipeline that Megatron uses
        to ensure perfect alignment between tokens and dispatch information.
        """
        logger.info("Creating aligned dispatch dataset")
        
        # Create a new BlendedDataset with the same structure but dispatch data
        aligned_datasets = []
        
        for i, gpt_dataset in enumerate(token_dataset.datasets):
            # Create a dispatch dataset that mirrors the GPT dataset structure
            aligned_dispatch_dataset = self._create_aligned_gpt_dispatch_dataset(
                gpt_dataset, dispatch_datasets[i] 
            )
            aligned_datasets.append(aligned_dispatch_dataset)
        
        # Create new BlendedDataset with aligned dispatch data
        # We need to create a custom dataset class that behaves like BlendedDataset
        # but returns dispatch information instead of tokens
        # Get the BlendedDataset information directly from token_dataset.

        aligned_blended_dataset = AlignedDispatchBlendedDataset(
            aligned_datasets,
            token_dataset.weights,
            token_dataset.size,
            token_dataset.config,
            token_dataset.dataset_index,
            token_dataset.dataset_sample_index
        )
        
        return aligned_blended_dataset

    def _create_aligned_gpt_dispatch_dataset(self, gpt_dataset: GPTDataset, dispatch_dataset):
        """
        Create a dispatch dataset that aligns with a single GPT dataset
        
        This replicates the exact token processing logic to ensure alignment.
        """
        # Create a custom dataset class that processes dispatch data the same way
        # GPTDataset processes token data
        aligned_dataset = AlignedDispatchGPTDataset(
            gpt_dataset, dispatch_dataset, self.config.topk, self.config.dummy_expert_id, self.config.eod_token_id
        )
        
        return aligned_dataset

    def _verify_input_files(self, data_prefix: str, dispatch_prefix: str) -> bool:
        """Verify that input IndexedDataset files exist"""
        required_files = [
            f"{data_prefix}.bin", f"{data_prefix}.idx",
            f"{dispatch_prefix}.bin", f"{dispatch_prefix}.idx"
        ]
        return all(os.path.exists(f) for f in required_files)
    
    def _create_gpt_dataset_config(self, dataset_path: MegatronPaths) -> Any:
        """
        Create GPTDatasetConfig exactly like Megatron's core_gpt_dataset_config_from_args
        """
        if not MEGATRON_AVAILABLE:
            raise ImportError("Megatron required")

        blend: Optional[Tuple[List[str], Optional[List[float]]]]
        blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
        blend, blend_per_split = get_blend_and_blend_per_split(dataset_path)
        
        # Create GPTDatasetConfig - split_matrix will be calculated automatically in __post_init__
        config = GPTDatasetConfig(
            random_seed=self.config.random_seed,
            sequence_length=self.config.training_config.sequence_length,
            blend=blend,  
            blend_per_split=blend_per_split,
            split=self.split_config,  # This will be parsed in __post_init__
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
    
    def _build_megatron_datasets(self, config: GPTDatasetConfig) -> Tuple[BlendedDataset]:
        """
        Build train/valid/test datasets using BlendedMegatronDatasetBuilder
        exactly like Megatron's train_valid_test_datasets_provider
        """
        logger.info("Building datasets with BlendedMegatronDatasetBuilder")
        
        # Calculate sample sizes for each split based on sequence length and available data
        # This mimics how Megatron calculates train_val_test_num_samples
        train_samples, valid_samples, test_samples = self._get_train_valid_test_num_samples()
        
        train_val_test_num_samples = [train_samples, valid_samples, test_samples]
        
        def is_dataset_built_on_rank():
            return True  # Always build on current rank for our use case
        
        # Build datasets exactly like Megatron
        builder = BlendedMegatronDatasetBuilder(
            GPTDataset,
            train_val_test_num_samples,
            is_dataset_built_on_rank,
            config
        )
        
        train_ds, valid_ds, test_ds = builder.build()
        
        logger.info(f"Built datasets - Train: {len(train_ds) if train_ds else 0}, "
                   f"Valid: {len(valid_ds) if valid_ds else 0}, "
                   f"Test: {len(test_ds) if test_ds else 0}")
        
        return train_ds, valid_ds, test_ds


    def _get_train_valid_test_num_samples(self):
        """Train/valid/test num samples."""

        # Number of train/valid/test samples.
        if self.config.train_samples:
            train_samples = self.config.train_samples
        else:
            train_samples = self.config.train_iters * self.config.global_batch_size
        if self.config.full_validation:
            eval_samples = None
        else:
            #eval_iters = (self.config.train_iters // self.config.eval_interval + 1) * self.config.eval_iters
            eval_iters = (self.config.train_iters // self.config.eval_interval) * self.config.eval_iters
            eval_samples = eval_iters * self.config.global_batch_size
        test_iters = self.config.eval_iters

        return (train_samples, eval_samples, test_iters * self.config.global_batch_size)

    
    def _serialize_processed_dataset(self, dataset: BlendedDataset, dispatch_dataset: AlignedDispatchBlendedDataset,
                                      node_id: int, split_name: str, output_dir: str) -> str:
        """
        Serialize all samples to IndexedDataset
        
        Args:
            dataset: Token dataset (BlendedDataset)
            dispatch_dataset: Aligned dispatch dataset (AlignedDispatchBlendedDataset)
            node_id: Node ID being processed
            split_name: Split name (train/valid/test)
            output_dir: Output directory path
        """
        logger.info(f"Serializing {split_name} dataset for node {node_id} ({len(dataset)} samples)")
        
        # Output path
        output_prefix = os.path.join(output_dir, f"{split_name}_processed")
        
        # Create IndexedDataset builder
        tokens_builder = indexed_dataset.IndexedDatasetBuilder(
            f"{output_prefix}.bin", dtype=indexed_dataset.DType.optimal_dtype(self.tokenizer.vocab_size)
        )
        labels_builder = indexed_dataset.IndexedDatasetBuilder(
            f"{output_prefix}_labels.bin", dtype=indexed_dataset.DType.optimal_dtype(self.tokenizer.vocab_size)
        )
        dispatch_builder = indexed_dataset.IndexedDatasetBuilder(
            f"{output_prefix}_dispatch.bin", dtype=np.uint8 
        )
        
        # Process all samples through dataset.__getitem__ and get aligned dispatch info
        processed_samples = 0
        for i in range(len(dataset)):
            # Get processed sample from Megatron
            sample = dataset[i]
                
            # Extract tokens (this is the concatenated, processed sequence)
            if isinstance(sample, dict):
                tokens = sample.get('tokens', sample.get('text', None))
                # we only need to record the final token of labels and can recover another token from 
                # sample
                labels = sample.get('labels')[-1]
            else:
                tokens = sample
                
            # Get perfectly aligned dispatch information
            dispatch_info = dispatch_dataset[i]

            # Convert to tensor
            if not torch.is_tensor(dispatch_info):
                dispatch_info = torch.from_numpy(dispatch_info)
                
            # Ensure lengths match
            assert len(dispatch_info) == len(tokens) * self.config.topk, f"Length mismatch at sample {i}"
                
            # Add to builders
            tokens_builder.add_document(tokens, [self.config.training_config.sequence_length])
            labels_builder.add_document(labels, [1])

            dispatch_builder.add_document(dispatch_info, [self.config.training_config.sequence_length * self.config.topk])
                
            processed_samples += 1
                
            if processed_samples % 1000 == 0:
                logger.info(f"Processed {processed_samples}/{len(dataset)} samples")
                    
        # Finalize datasets
        tokens_builder.finalize(f"{output_prefix}.idx")
        labels_builder.finalize(f"{output_prefix}_labels.idx")

        dispatch_builder.finalize(f"{output_prefix}_dispatch.idx")
        
        logger.info(f"Serialized {processed_samples} processed samples to {output_prefix}")
        return output_prefix
    
    def validate_processed_datasets(self, node_ids: List[int]) -> bool:
        """Validate that all processed datasets are correctly formatted"""
        logger.info("Validating processed datasets")
        
        for node_id in node_ids:
            output_dir = os.path.join(self.data_paths.output_dir, "megatron_processed", f"node_{node_id}")
            
            for split_name in ["train", "valid", "test"]:
                data_path = os.path.join(output_dir, f"{split_name}_processed")
                dispatch_path = os.path.join(output_dir, f"{split_name}_processed_dispatch")
                
                if os.path.exists(f"{data_path}.bin"):
                    try:
                        data_ds = indexed_dataset.IndexedDataset(data_path)
                        dispatch_ds = indexed_dataset.IndexedDataset(dispatch_path)
                        
                        if len(data_ds) != len(dispatch_ds):
                            logger.error(f"Node {node_id} {split_name}: length mismatch")
                            return False
                            
                        logger.debug(f"Node {node_id} {split_name}: {len(data_ds)} samples validated")
                        
                    except Exception as e:
                        logger.error(f"Failed to validate node {node_id} {split_name}: {e}")
                        return False
        
        logger.info("All processed datasets validated successfully")
        return True
    
    def export_processing_info(self, node_ids: List[int], output_path: str):
        """Export information about the processed datasets"""
        info = {
            'processing_summary': {
                'total_nodes': len(node_ids),
                'megatron_pipeline': True,
                'sequence_length': self.config.training_config.sequence_length,
                'split_config': self.split_config
            },
            'node_datasets': {}
        }
        
        for node_id in node_ids:
            output_dir = os.path.join(self.data_paths.output_dir, "megatron_processed", f"node_{node_id}")
            node_info = {}
            
            for split_name in ["train", "valid", "test"]:
                data_path = os.path.join(output_dir, f"{split_name}_processed")
                
                if os.path.exists(f"{data_path}.bin"):
                    try:
                        dataset = indexed_dataset.IndexedDataset(data_path)
                        node_info[split_name] = {
                            'dataset_path': data_path,
                            'num_samples': len(dataset),
                            'total_tokens': int(np.sum(dataset.sequence_lengths))
                        }
                    except Exception as e:
                        node_info[split_name] = {'error': str(e)}
            
            info['node_datasets'][str(node_id)] = node_info
        
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Processing info exported to {output_path}")

def create_megatron_processor(config: OptimizationConfig, data_paths: DataPaths, tokenizer=None) -> MegatronDatasetProcessor:
    """
    Factory function to create Megatron dataset processor
    
    Args:
        config: Optimization configuration
        data_paths: Data path configuration  
        tokenizer: Megatron tokenizer instance
        
    Returns:
        Configured dataset processor
    """
    return MegatronDatasetProcessor(config, data_paths, tokenizer)