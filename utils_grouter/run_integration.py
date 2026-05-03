#!/usr/bin/env python3
"""
Megatron Dataset Construction for Grouter EP Optimization

This script demonstrates how to use the Megatron dataset adapter
to process clustering optimization outputs and create Megatron-compatible datasets.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.config import OptimizationConfig, DataPaths
from megatron_integration import create_megatron_processor, create_tokenizer 


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    # force overlap root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.NOTSET)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('megatron_dataset_construction.log')
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
        dataset_path=args.dataset_path,
        predispatch_path=args.predispatch_path,
        output_dir=args.output_dir,
        prefix=args.data_prefix
    )


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Megatron Dataset Construction for Grouter EP Optimization")
    
    # Configuration
    parser.add_argument("--config", required=True, help="Path to optimization config file")
    
    # Data paths
    parser.add_argument("--dataset-path", required=True, help="Path to clustering optimization dataset output")
    parser.add_argument("--predispatch-path", required=True, help="Path to clustering optimization predispatch output")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed datasets")
    parser.add_argument("--data-prefix", required=True, nargs='*', help="Data file prefix for loading")
    
    # Processing options
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing datasets")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Initialize distributed process
    # This is because Megatron dataset processing code sometimes need to use 
    # torch.distributed.get_rank() and no other effect. You can just set 
    # --nproc-per-node=1 to minimize device load.
    torch.distributed.init_process_group()
    
    logger.info("Starting Megatron Dataset Construction for Grouter EP Optimization")
        
    # Load configuration
    logger.info("Loading configuration")
    config = load_config(args.config)
    logger.info(f"Configuration loaded: {config.num_nodes} nodes, {config.num_experts} experts")
        
    # Setup data paths
    logger.info("Setting up data paths")
    data_paths = setup_data_paths(args)
    logger.info(f"Output directory: {data_paths.output_dir}")
        
    # Create tokenizer
    logger.info("Creating tokenizer")
    tokenizer = create_tokenizer(config)
        
    # Create Megatron dataset processor
    logger.info("Creating Megatron dataset processor")
    processor = create_megatron_processor(config, data_paths, tokenizer)
        
    # Define node IDs
    node_ids = list(range(config.num_nodes))
    logger.info(f"Processing datasets for nodes: {node_ids}")
        
    # Process datasets through complete Megatron pipeline
    logger.info("Processing datasets through Megatron pipeline")
            
    # Process all node datasets
    processed_paths = processor.process_node_datasets(node_ids)
            
    # Validate processed datasets
    if not processor.validate_processed_datasets(node_ids):
        logger.error("Processed dataset validation failed")
        return 1
            
    # Export processing information
    info_path = os.path.join(data_paths.output_dir, "megatron_processed", "processing_info.json")
    processor.export_processing_info(node_ids, info_path)
            
    # Print summary
    total_datasets = sum(len(splits) for splits in processed_paths.values())
    logger.info(f"Successfully processed {len(processed_paths)} nodes with {total_datasets} dataset splits")
    logger.info(f"Processing info exported to: {info_path}")
            
    # Show detailed results
    for node_id, splits in processed_paths.items():
        logger.info(f"Node {node_id} splits: {list(splits.keys())}")

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    sys.exit(main())
