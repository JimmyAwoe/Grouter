#!/usr/bin/env python3
"""
Per-micro-batch assignment planner main entry
Provide command line interface to run assignment planning optimization
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Dict, List

from megatron.core.datasets import indexed_dataset

from utils_grouter.core.config import OptimizationConfig, DataPaths
from utils_grouter.core.optimizer import EPOptimizer
from utils_grouter.utils.data_structures import Sample, Cluster

# Setup logging

def setup_logger(level):
    # force overlap root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.NOTSET)

    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('assignment_planner.log')
        ]
    )



def load_config(config_path: str) -> OptimizationConfig:
    """Load optimization configuration from file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return OptimizationConfig.from_file(config_path)


def setup_data_paths(args) -> DataPaths:
    """Setup data paths configuration"""
    return DataPaths(
        dataset_path='',
        predispatch_path='',
        output_dir=args.output_dir
    )


def load_processed_datasets(output_dir: str, num_nodes: int, split: str) -> Dict[int, Dict[str, str]]:
    """Load processed dataset paths for all nodes"""
    logger.info(f"Loading processed {split} datasets for {num_nodes} nodes")
    
    node_to_paths: Dict[int, Dict[str, str]] = {}
    for node_id in range(num_nodes):
        node_dir = Path(output_dir) / 'megatron_processed' / f'node_{node_id}'
        data_prefix = str(node_dir / f'{split}_processed')
        dispatch_prefix = str(node_dir / f'{split}_processed_dispatch')
        labels_prefix = str(node_dir / f'{split}_processed_labels')
        
        if not os.path.exists(data_prefix + '.bin'):
            raise FileNotFoundError(f"Missing processed dataset for node {node_id}: {data_prefix}.bin")
        if not os.path.exists(dispatch_prefix + '.bin'):
            raise FileNotFoundError(f"Missing processed dispatch for node {node_id}: {dispatch_prefix}.bin")
        if not os.path.exists(labels_prefix + '.bin'):
            raise FileNotFoundError(f"Missing processed dispatch for node {node_id}: {labels_prefix}.bin")
        
        node_to_paths[node_id] = {'data': data_prefix, 'dispatch': dispatch_prefix, 'labels': dispatch_prefix}
    
    logger.info(f"Successfully loaded dataset paths for {len(node_to_paths)} nodes")
    return node_to_paths


def build_samples_from_dataset(token_dataset: indexed_dataset.IndexedDataset,
                               dispatch_dataset: indexed_dataset.IndexedDataset,
                               labels_dataset: indexed_dataset.IndexedDataset,
                               start_index: int,
                               sample_count: int,
                               node_id: int
                               ) -> List[Sample]:
    """Build Sample objects from Megatron-processed datasets"""
    samples: List[Sample] = []
    
    end_index = min(len(token_dataset), start_index + sample_count)
    for sample_index in range(start_index, end_index):
        # Get token data
        token_sample = token_dataset[sample_index]
        if isinstance(token_sample, dict):
            token_ids = token_sample.get('tokens', token_sample.get('text'))
        else:
            token_ids = token_sample
        
        # Get label data
        labels = labels_dataset[sample_index]
        
        # Get dispatch data
        dispatch_sample = dispatch_dataset[sample_index]
        
        # Create sample object
        samples.append(Sample(
            sample_id=sample_index,
            token_ids=token_ids.tolist(),
            token_count=len(token_ids),
            expert_preference_vector=None,
            dispatch_ids=dispatch_sample.tolist(),
            labels=labels.tolist(),
            cluster_id=node_id,
        ))
    
    return samples


def calculate_batch_statistics(node_id_to_dataset: Dict[int, indexed_dataset.IndexedDataset],
                               samples_per_node_per_batch: int) -> tuple:
    """Calculate dataset sizes and available batches"""
    dataset_sizes = {node_id: len(dataset) for node_id, dataset in node_id_to_dataset.items()}
    
    if not dataset_sizes:
        return dataset_sizes, 0
    
    # Find limiting factor
    min_batches = min(size // samples_per_node_per_batch for size in dataset_sizes.values())
    total_available_batches = max(0, min_batches)
    
    return dataset_sizes, total_available_batches


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Per-micro-batch assignment planner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--config', 
        required=True, 
        help='Path to optimization config (produced by clustering step)'
    )
    parser.add_argument(
        '--output-dir', 
        required=True, 
        help='Project output directory (processed datasets & plans)'
    )
    
    # Processing options
    parser.add_argument(
        '--max-batches', 
        type=int, 
        default=None,
        help='Maximum number of batches to process (default: process all available)'
    )
    parser.add_argument(
        '--split', 
        default='train', 
        choices=['train', 'valid', 'test'],
        help='Dataset split to process'
    )

    # logging
    parser.add_argument(
        '--log-level', 
        default='INFO',
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    setup_logger(args.log_level)

    global logger
    logger = logging.getLogger(__name__)

    logger.info("Starting per-micro-batch assignment planning")
    
    # Load configuration
    logger.info("Loading configuration")
    config = load_config(args.config)
    logger.info(f"Configuration loaded: {config.num_nodes} nodes, {config.num_experts} experts")
    
    # Setup data paths
    logger.info("Setting up data paths")
    data_paths = setup_data_paths(args)
    logger.info(f"Output directory: {data_paths.output_dir}")
    
    # Build EPOptimizer to re-use clusters and writer
    logger.info("Initializing EP optimizer")
    optimizer = EPOptimizer(config, data_paths)
    
    # Load cluster information
    logger.info("Loading cluster information")
    cluster_info_path = Path(args.output_dir) / 'config_cluster' / f"{config.output_prefix}_cluster_info.json"
    if not cluster_info_path.exists():
        raise FileNotFoundError(f"Cluster info not found: {cluster_info_path}")
    
    with open(cluster_info_path, 'r') as f:
        cluster_info = json.load(f)
    
    node_experts = {int(cluster['node_id']): set(cluster['target_experts'])
                    for cluster in cluster_info['clusters'] if cluster['node_id'] is not None}
    
    # Reconstruct minimal clusters for planning
    optimizer.clusters = [Cluster(
        cluster_id=node_id,
        samples=None,
        center_vector=None,
        target_experts=experts,
        node_id=node_id
    ) for node_id, experts in node_experts.items()]
    
    logger.info(f"Loaded cluster information for {len(optimizer.clusters)} clusters")
    
    # Load processed datasets
    node_to_prefixes = load_processed_datasets(args.output_dir, config.num_nodes, args.split)
    node_id_to_dataset = {node_id: indexed_dataset.IndexedDataset(paths['data'])
                          for node_id, paths in node_to_prefixes.items()}
    node_id_to_dispatch = {node_id: indexed_dataset.IndexedDataset(paths['dispatch'])
                           for node_id, paths in node_to_prefixes.items()}
    node_id_to_labels = {node_id: indexed_dataset.IndexedDataset(paths['labels'])
                           for node_id, paths in node_to_prefixes.items()}
    
    # Calculate batch statistics
    logger.info("Calculating batch statistics")
    micro_batch_size = config.training_config.micro_batch_size
    samples_per_node_per_batch = micro_batch_size * 8
    
    dataset_sizes, total_available_batches = calculate_batch_statistics(
        node_id_to_dataset, samples_per_node_per_batch
    )
    
    if total_available_batches == 0:
        logger.error("No complete batches can be formed from available data")
        logger.error(f"Dataset sizes: {dataset_sizes}")
        logger.error(f"Required samples per node per batch: {samples_per_node_per_batch}")
        return 1
    
    # Determine actual number of batches to process
    if args.max_batches is not None:
        batches_to_process = min(args.max_batches, total_available_batches)
    else:
        batches_to_process = total_available_batches
    
    logger.info(f"Dataset sizes: {dataset_sizes}")
    logger.info(f"Samples per node per batch: {samples_per_node_per_batch}")
    logger.info(f"Total available batches: {total_available_batches}")
    logger.info(f"Batches to process: {batches_to_process}")
    
    # Process batches
    logger.info("Starting batch processing")
    per_node_read_index = {node_id: 0 for node_id in node_id_to_dataset}
    
    for batch_id in range(batches_to_process):
        node_id_to_samples: Dict[int, List[Sample]] = {}
        
        for node_id, dataset in node_id_to_dataset.items():
            start_index = per_node_read_index[node_id]
            
            # Build samples for this batch
            samples = build_samples_from_dataset(
                token_dataset=dataset,
                dispatch_dataset=node_id_to_dispatch[node_id],
                labels_dataset=node_id_to_labels[node_id],
                start_index=start_index,
                sample_count=samples_per_node_per_batch,
                node_id=node_id
            )
            
            node_id_to_samples[node_id] = samples
            per_node_read_index[node_id] = start_index + len(samples)
        
        # Execute assignment planning for this batch
        optimizer.plan_micro_batch(node_id_to_samples, batch_id=batch_id)
        logger.info(f"Batch {batch_id}/{batches_to_process - 1} planned")
    
    # Finalize all IndexedDataset builders after processing all batches
    if optimizer.expert_sample_coordinator is not None:
        logger.info("Finalizing IndexedDataset builders...")
        optimizer.expert_sample_coordinator.finalize_builders()
        logger.info("IndexedDataset builders finalized successfully")
    
    logger.info(f"Assignment planning completed. Processed {batches_to_process} batches.")
    return 0


if __name__ == '__main__':
    sys.exit(main())


