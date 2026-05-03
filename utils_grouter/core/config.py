"""
Configuration management module
Define core parameters for system parameterized design
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
import json
from pathlib import Path


@dataclass
class NetworkTopology:
    """Network topology configuration"""
    inter_node_bandwidth: float  # GB/s
    inter_node_latency: float    # microseconds
    intra_node_bandwidth: float  # GB/s
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkTopology':
        return cls(
            inter_node_bandwidth=data.get('inter_node_bandwidth', 100.0),
            inter_node_latency=data.get('inter_node_latency', 5.0),
            intra_node_bandwidth=data.get('intra_node_bandwidth', 1000.0)
        )


@dataclass
class TrainingConfig:
    """Training parameter configuration"""
    micro_batch_size: int
    global_batch_size: int
    sequence_length: int
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        return cls(
            micro_batch_size=data.get('micro_batch_size', 1),
            global_batch_size=data.get('global_batch_size', 8),
            sequence_length=data.get('sequence_length', 2048)
        )


@dataclass
class OptimizationConfig:
    """EP optimization configuration"""
    # Core parameters
    num_nodes: int = None                    # Number of nodes  
    num_experts: int = None                 # Total number of experts
    experts_per_gpu: Optional[int] = None   # Experts per GPU (auto-calculated)
    
    # Network parameters
    network_topology: NetworkTopology = None
    
    # Training parameters
    training_config: TrainingConfig = None
    
    # Clustering parameters
    pca_dimensions: int = None          # PCA dimensionality reduction
    first_hier_cluster_num: int = None  # First hierarchical clustering number
    cosine_similarity_threshold: float = None  # Cosine similarity threshold
    entropy_threshold: float = None    # Entropy threshold 
    deepep_mode: bool = True
    a2a_mode: bool = False

    
    # Expert allocation parameters
    expert_balance_weight: float = None  # Expert concentration weight
    similarity_weight: float = None      # Similarity weight
    
    # Iterative optimization parameters
    iterative_optimization_rounds: int = 0  # Number of iterative optimization rounds (0 to disable)

    # Logging
    export_analysis_report: bool = False # Whether export analysis for dataset
    not_export_data: bool = False # For the case only need analysis report

    # Tokenizer
    vocab_size: int = None # The vocab size for tokenizer
    eod_token_id: int = None # The eod token id
    tokenizer_type: str = None # The type of tokenizer for megatron initialization
    tokenizer_model: str = None # For megatron tokenizer initialization
    
    # Dataset integration with Megatron configuration
    # Just use the same config parameter with Megatron start script
    split_config: str = None  # Train, valid, test split ratios
    random_seed: int = None
    # How many samples needed for training 
    train_samples: int = None
    # Only work when train_samples is None
    train_iters: int = None
    global_batch_size: int = None
    # Only test once at the end of training
    full_validation: bool = None
    # Evaluation per interval steps
    eval_interval: int = None
    # How many steps for one evaluation
    eval_iters: int = None
    topk: int = None
    # This is for adding eod in dispatch data
    dummy_expert_id: List[int] = None

    # Output
    output_prefix: str = "grt" # Output prefix
    
    def __post_init__(self):
        """Automatically calculate experts per GPU"""
        if self.experts_per_gpu is None:
            self.experts_per_gpu = self.num_experts // (self.num_nodes * 8)
    
    def get(self, key: str, default: Any):
        if hasattr(self, key):
            return getattr(self, key)
        return default

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        """Create configuration from dictionary"""
        kw_args = {}
        for f in dataclasses.fields(cls):
            if f.name in data:
                if f.type == NetworkTopology:
                    kw_args[f.name] = NetworkTopology.from_dict(data[f.name])
                elif f.type == TrainingConfig:
                    kw_args[f.name] = TrainingConfig.from_dict(data[f.name])
                else:
                    kw_args[f.name] = data[f.name]
            else:
                kw_args[f.name] = None
        return cls(**kw_args)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'OptimizationConfig':
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'num_nodes': self.num_nodes,
            'num_experts': self.num_experts,
            'experts_per_gpu': self.experts_per_gpu,
            'network_topology': {
                'inter_node_bandwidth': self.network_topology.inter_node_bandwidth,
                'inter_node_latency': self.network_topology.inter_node_latency,
                'intra_node_bandwidth': self.network_topology.intra_node_bandwidth
            },
            'training_config': {
                'micro_batch_size': self.training_config.micro_batch_size,
                'global_batch_size': self.training_config.global_batch_size,
                'sequence_length': self.training_config.sequence_length
            },
            'pca_dimensions': self.pca_dimensions,
            'first_hier_cluster_num': self.first_hier_cluster_num,
            'cosine_similarity_threshold': self.cosine_similarity_threshold,
            'entropy_threshold': self.entropy_threshold,
            'expert_balance_weight': self.expert_balance_weight,
            'similarity_weight': self.similarity_weight,
            'iterative_optimization_rounds': self.iterative_optimization_rounds,
            'output_prefix': self.output_prefix,
            'deepep_mode': self.deepep_mode,
            'a2a_mode': self.a2a_mode,
            'export_analysis_report': self.export_analysis_report,
            'vocab_size': self.vocab_size,
        }
    
    def save_to_file(self, config_path: str):
        """Save configuration to file"""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class DataPaths:
    """Data path configuration"""
    dataset_path: str = None                    # Original dataset path
    predispatch_path: str = None              # Predispatch results path
    output_dir: str = None                      # Output directory

    # Integration argument
    prefix: str = None
    
    def __post_init__(self):
        """Ensure output directory exists"""
        if self.output_dir is not None:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    @property
    def node_data_dir(self) -> str:
        """Node data directory"""
        return str(Path(self.output_dir) / "node_data")
    
    @property
    def node_dispatch_dir(self) -> str:
        """Node dispatch directory"""
        return str(Path(self.output_dir) / "node_dispatch")
    
    @property
    def expert_placement_path(self) -> str:
        """Expert placement configuration file path"""
        return str(Path(self.output_dir) / "expert_placement.json")
    
    @property
    def cluster_info_path(self) -> str:
        """Cluster information file path"""
        return str(Path(self.output_dir) / "cluster_info.json")

@dataclass
class MegatronPaths:
    """For generating GPTdatasetConfig"""
    data_path: str = None
    data_args_path: str = None
    train_data_path: str = None
    valid_data_path: str = None
    test_data_path: str = None
    per_split_data_args_path: str = None

