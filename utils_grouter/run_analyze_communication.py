#!/usr/bin/env python3
"""
Communication analysis script for EP optimization results.

This script analyzes communication patterns from completed assignment planning
and generates detailed reports on GPU communication efficiency.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

from utils_grouter.analysis import CommunicationAnalyzer
from utils_grouter.utils.data_structures import Sample
from utils_grouter.utils.gpu_data_reader import GPUDataReader


def setup_logging(level: str):
    """Setup logging configuration."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.NOTSET)

    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('communication_analysis.log')
        ]
    )


def load_gpu_data_directly(gpu_data_reader: GPUDataReader, 
                          num_nodes: int, 
                          topk: int) -> tuple:
    """
    Load GPU data directly from binary files for communication analysis.
    
    Args:
        gpu_data_reader: Reader for GPU data files
        num_nodes: Number of nodes in the system
        topk: Number of top experts per token
        
    Returns:
        Tuple of (gpu_assignments, expert_placements)
    """
    logger = logging.getLogger(__name__)
    
    gpu_assignments = {}
    expert_placements = {}
    
    for node_id in range(num_nodes):
        for gpu_id in range(8):
            gpu_key = (node_id, gpu_id)
            
            try:
                # Load expert placement (static)
                experts = gpu_data_reader.read_gpu_experts(node_id, gpu_id)
                expert_placements[gpu_key] = experts
                
                # Load dispatch data
                dispatch_data = gpu_data_reader.read_gpu_dispatch(node_id, gpu_id)
                
                # Create samples from dispatch data
                samples = []
                for i, dispatch_sequence in enumerate(dispatch_data):
                    # Convert to list if needed
                    dispatch_ids = dispatch_sequence.tolist() if hasattr(dispatch_sequence, 'tolist') else dispatch_sequence
                    
                    # Calculate token count from dispatch sequence length
                    token_count = len(dispatch_ids) // topk
                    
                    # Create sample object
                    sample = Sample(
                        sample_id=i,
                        token_ids=[],  # Not needed for communication analysis
                        token_count=token_count,
                        expert_preference_vector=None,
                        dispatch_ids=dispatch_ids,
                        cluster_id=0  # Not needed for communication analysis
                    )
                    samples.append(sample)
                
                gpu_assignments[gpu_key] = samples
                logger.debug(f"Loaded {len(samples)} samples for node {node_id}, GPU {gpu_id}")
                
            except Exception as e:
                logger.warning(f"Failed to load data for node {node_id}, GPU {gpu_id}: {e}")
                gpu_assignments[gpu_key] = []
                expert_placements[gpu_key] = []
    
    logger.info(f"Loaded data for {len(gpu_assignments)} GPUs")
    return gpu_assignments, expert_placements


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze communication patterns from EP optimization results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory containing GPU data and plans'
    )
    parser.add_argument(
        '--num-nodes',
        type=int,
        required=True,
        help='Number of nodes in the system'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=6,
        help='Number of top experts per token'
    )
    parser.add_argument(
        '--micro-batch-size',
        type=int,
        required=True,
        help='Micro batch size per GPU'
    )
    
    # Optional arguments
    parser.add_argument(
        '--json-file',
        default='communication_analysis_results.json',
        help='JSON output file name (used when --json-output is enabled)'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting communication analysis")
    
    # Setup paths
    output_dir = Path(args.output_dir)
    gpu_data_dir = output_dir / 'gpu_data'
    
    if not gpu_data_dir.exists():
        logger.error(f"GPU data directory not found: {gpu_data_dir}")
        return 1
    
    # Initialize GPU data reader
    gpu_data_reader = GPUDataReader(str(gpu_data_dir))
    
    # Load GPU data directly
    logger.info("Loading GPU data directly from binary files...")
    gpu_assignments, expert_placements = load_gpu_data_directly(
        gpu_data_reader, 
        args.num_nodes, 
        args.topk
    )

    # Initialize communication analyzer
    analyzer = CommunicationAnalyzer(num_nodes=args.num_nodes, 
                                     topk=args.topk, 
                                     micro_batch_size=args.micro_batch_size)
    
    # Analyze communication
    logger.info("Analyzing communication patterns...")
    micro_batch_results = analyzer.analyze_communication(gpu_assignments, expert_placements)
    
    # Compute aggregate statistics
    aggregate_stats = analyzer.compute_aggregate_stats(micro_batch_results)
    
    # Save JSON output if requested

    # Create analysis results directory
    analysis_dir = output_dir / 'analysis_results'
    analysis_dir.mkdir(exist_ok=True)
        
    # Save JSON results
    json_path = analysis_dir / args.json_file
    analyzer.save_communication_json(aggregate_stats, micro_batch_results, str(json_path))
        
    
    # Print summary
    logger.info(f"\nCommunication Analysis Summary:")
    logger.info(f"Total tokens: {aggregate_stats.total_tokens:,}")
    logger.info(f"Communication ratio: {aggregate_stats.communication_ratio:.4f}")
    logger.info(f"Intra-node communication tokens: {aggregate_stats.intra_node_communication_tokens:,}")
    logger.info(f"Inter-node communication tokens: {aggregate_stats.inter_node_communication_tokens:,}")
    logger.info(f"GPU load balance entropy: {aggregate_stats.average_load_entropy:,}")
    logger.info(f"Communication analysis JSON saved to: {json_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
