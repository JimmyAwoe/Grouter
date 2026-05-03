"""
EP communication optimization main entry
Provide command line interface to run EP optimization
"""

import argparse
import logging
import sys
from pathlib import Path
import json

from utils_grouter.core.optimizer import EPOptimizer
from utils_grouter.core.config import OptimizationConfig, DataPaths, NetworkTopology, TrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ep_optimization.log')
    ]
)

logger = logging.getLogger(__name__)


def create_config_from_args(args) -> OptimizationConfig:
    """Create configuration from command line arguments"""
    # Network topology
    network_topology = NetworkTopology(
        inter_node_bandwidth=args.inter_node_bandwidth,
        inter_node_latency=args.inter_node_latency,
        intra_node_bandwidth=args.intra_node_bandwidth
    )
    
    # Training configuration
    training_config = TrainingConfig(
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        sequence_length=args.sequence_length
    )
    
    # Optimization configuration
    config = OptimizationConfig(
        num_nodes=args.num_nodes,
        num_experts=args.num_experts,
        experts_per_gpu=args.experts_per_gpu,
        network_topology=network_topology,
        training_config=training_config,
        pca_dimensions=args.pca_dimensions,
        first_hier_cluster_num=args.first_hier_cluster_num,
        cosine_similarity_threshold=args.cosine_similarity_threshold,
        entropy_threshold=args.entropy_threshold,
        expert_balance_weight=args.expert_balance_weight,
        similarity_weight=args.similarity_weight,
        iterative_optimization_rounds=args.iterative_optimization_rounds,
        export_analysis_report=args.export_analysis_report,
        not_export_data=args.not_export_data,
        vocab_size=args.vocab_size,
        output_prefix=args.output_prefix,
        deepep_mode=args.deepep_mode,
        a2a_mode=args.a2a_mode
    )

    return config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='EP communication optimization tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset-path', 
        default=None,
        help='Original dataset path'
    )
    parser.add_argument(
        '--predispatch-path', 
        required=True,
        help='Predispatch results path'
    )
    parser.add_argument(
        '--output-dir', 
        required=True,
        help='Output directory'
    )
    
    # System configuration parameters
    parser.add_argument(
        '--num-nodes', 
        type=int, 
        default=None,
        help='Number of nodes'
    )
    parser.add_argument(
        '--num-experts', 
        type=int, 
        default=None,
        help='Total number of experts'
    )
    parser.add_argument(
        '--experts-per-gpu', 
        type=int, 
        default=None,
        help='Experts per GPU (auto-calculated)'
    )
    
    # Network topology parameters
    parser.add_argument(
        '--inter-node-bandwidth', 
        type=float, 
        default=None,
        help='Inter-node bandwidth (GB/s)'
    )
    parser.add_argument(
        '--inter-node-latency', 
        type=float, 
        default=None,
        help='Inter-node latency (microseconds)'
    )
    parser.add_argument(
        '--intra-node-bandwidth', 
        type=float, 
        default=None,
        help='Intra-node bandwidth (GB/s)'
    )
    
    # Training parameters
    parser.add_argument(
        '--micro-batch-size', 
        type=int, 
        default=None,
        help='Micro batch size per GPU'
    )
    parser.add_argument(
        '--global-batch-size', 
        type=int, 
        default=None,
        help='Global batch size'
    )
    parser.add_argument(
        '--sequence-length', 
        type=int, 
        default=None,
        help='Sequence length'
    )
    
    # Clustering parameters
    parser.add_argument(
        '--pca-dimensions', 
        type=int, 
        default=None,
        help='PCA dimensionality reduction'
    )
    parser.add_argument(
        '--first-hier-cluster-num', 
        type=int, 
        default=None,
        help='First hierarchical clustering number'
    )
    parser.add_argument(
        '--cosine-similarity-threshold', 
        type=float, 
        default=None,
        help='Cosine similarity threshold'
    )
    parser.add_argument(
        '--entropy-threshold', 
        type=float, 
        default=None,
        help='Entropy threshold'
    )
    
    # Expert allocation parameters
    parser.add_argument(
        '--expert-balance-weight', 
        type=float, 
        default=None,
        help='Expert concentration weight'
    )
    parser.add_argument(
        '--similarity-weight', 
        type=float, 
        default=None,
        help='Similarity weight'
    )
    
    # Iterative optimization parameters
    parser.add_argument(
        '--iterative-optimization-rounds',
        type=int,
        default=0,
        help='Number of iterative optimization rounds (0 to disable)'
    )
    
    # Other parameters
    parser.add_argument(
        '--config-file', 
        type=str,
        help='Configuration file path (JSON format)'
    )
    parser.add_argument(
        '--output-prefix', 
        type=str,
        help='Prefix to differentiate output file'
    )
    parser.add_argument(
        '--export-analysis-report', 
        action='store_true',
        help='Identify if output samples analysis'
    )
    parser.add_argument(
        '--not-export-data',
        action='store_true'
    )
    parser.add_argument(
        '--vocab-size', 
        type=int,
        help='Help output datasets'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument("--a2a-mode", action="store_true")
    parser.add_argument("--deepep-mode", action="store_true")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load or create configuration
    if args.config_file:
        logger.info(f"Loading configuration from file: {args.config_file}")
        config = OptimizationConfig.from_file(args.config_file)
    else:
        logger.info("Creating configuration from command line arguments")
        config = create_config_from_args(args)
    
    # Create data path configuration
    data_paths = DataPaths(
        dataset_path=args.dataset_path,
        predispatch_path=args.predispatch_path,
        output_dir=args.output_dir
    )
    
    # Create EP optimizer
    logger.info("Initializing EP optimizer...")
    optimizer = EPOptimizer(config, data_paths)
    
    # Execute optimization
    logger.info("Start executing EP optimization...")
    optimizer.optimize()
    
    # Output summary
    summary = optimizer.get_optimization_summary()
    logger.info("Optimization completed!")
    logger.info(f"Configuration summary: {json.dumps(summary, indent=2)}")
    
    # Save configuration: CLI --output-prefix wins, else config file value, else default
    output_prefix = args.output_prefix or getattr(config, "output_prefix", None) or "grt"
    config_path = Path(args.output_dir) / 'config_cluster' / f"{output_prefix}_optimization_config.json"
    config.save_to_file(str(config_path))
    logger.info(f"Configuration saved to: {config_path}")


if __name__ == "__main__":
    main()
