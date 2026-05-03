"""
File I/O utility module
For reading predispatch results and saving optimization results
"""

import json
import gzip
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator, Callable
import logging

# Add Megatron path
import sys
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]  # .../general_router
_MEGATRON_ROOT = _PROJECT_ROOT / "Megatron-LM"
if str(_MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEGATRON_ROOT))

from megatron.core.datasets import indexed_dataset

from .data_structures import Sample, Cluster, ExpertAssignment, NodeData

logger = logging.getLogger(__name__)


class PredispatchReader:
    """Predispatch result reader"""
    
    def __init__(self, predispatch_path: str, dataset_path: str, num_experts: int, vocab_size: int):
        """
        Initialize reader
        
        Args:
            predispatch_path: Predispatch results directory path
            dataset_path: Original dataset path
            num_experts: Total number of experts
            vocab_size: The vocab size for tokenizer 
        """
        self.predispatch_path = Path(predispatch_path)
        if dataset_path is not None:
            self.dataset_path = Path(dataset_path)
        self.num_experts = num_experts
        self.token_ids_dtype = indexed_dataset.DType.optimal_dtype(vocab_size)
        
        # Check if paths exist
        if not self.predispatch_path.parent.exists():
            raise FileNotFoundError(f"Predispatch path does not exist: {predispatch_path}")
        if dataset_path is not None and not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    def read_dispatch_data(self, key: str = 'text') -> Tuple[List[np.ndarray], List[int]]:
        """
        Read dispatch data
        
        Args:
            key: Data key name, default is 'text'
            
        Returns:
            dispatch_sequences: Dispatch sequence for each sample
            sequence_lengths: Sequence length for each sample
        """
        bin_file = self.predispatch_path.parent / f"{self.predispatch_path.name}_{key}_dispatch_ids.bin"
        idx_file = self.predispatch_path.parent / f"{self.predispatch_path.name}_{key}_dispatch_ids.idx"
        
        if not bin_file.exists() or not idx_file.exists():
            raise FileNotFoundError(f"Dispatch files do not exist: {bin_file} or {idx_file}")
        
        # Use Megatron's IndexedDataset to read
        dataset = indexed_dataset.IndexedDataset(str(bin_file)[:-4])  # Remove .bin suffix
        
        dispatch_sequences = []
        sequence_lengths = []
        
        for i in range(len(dataset)):
            doc_indices = dataset.document_indices
            start_seq = int(doc_indices[i])
            end_seq = int(doc_indices[i + 1])
            sequences = dataset[start_seq:end_seq]
            
            # Each document now has only one sequence
            for seq in sequences:
                arr = np.array(seq, dtype=np.uint8)
                dispatch_sequences.append(arr)
                sequence_lengths.append(len(arr))
        
        logger.info(f"Read dispatch data for {len(dispatch_sequences)} samples")
        return dispatch_sequences, sequence_lengths
    
    def read_tokenized_data(self, key: str = 'text') -> Tuple[List[np.ndarray], List[int]]:
        """
        Read tokenized data from .bin and .idx files
        
        Args:
            key: Data key name, default is 'text'
            
        Returns:
            tokenized_sequences: Tokenized sequence for each sample
            sequence_lengths: Sequence length for each sample
        """
        bin_file = self.predispatch_path.parent / f"{self.predispatch_path.name}_{key}_tokenized.bin"
        idx_file = self.predispatch_path.parent / f"{self.predispatch_path.name}_{key}_tokenized.idx"
        
        if not bin_file.exists() or not idx_file.exists():
            raise FileNotFoundError(f"Tokenized files do not exist: {bin_file} or {idx_file}")
        
        # Use Megatron's IndexedDataset to read
        dataset = indexed_dataset.IndexedDataset(str(bin_file)[:-4])  # Remove .bin suffix
        
        tokenized_sequences = []
        sequence_lengths = []
        
        for i in range(len(dataset)):
            doc_indices = dataset.document_indices
            start_seq = int(doc_indices[i])
            end_seq = int(doc_indices[i + 1])
            sequences = dataset[start_seq:end_seq]
            
            # Each document now has only one sequence
            for seq in sequences:
                arr = np.array(seq, dtype=self.token_ids_dtype)
                tokenized_sequences.append(arr)
                sequence_lengths.append(len(arr))
        
        logger.info(f"Read tokenized data for {len(tokenized_sequences)} samples")
        return tokenized_sequences, sequence_lengths
    
    def read_dispatch_scores(self, key: str = 'text') -> Tuple[List[np.ndarray], List[int]]:
        """
        Read dispatch scores data from .bin and .idx files
        
        Args:
            key: Data key name, default is 'text'
            
        Returns:
            scores_sequences: Dispatch scores sequence for each sample
            sequence_lengths: Sequence length for each sample
        """
        bin_file = self.predispatch_path.parent / f"{self.predispatch_path.name}_{key}_dispatch_scores.bin"
        idx_file = self.predispatch_path.parent / f"{self.predispatch_path.name}_{key}_dispatch_scores.idx"
        
        if not bin_file.exists() or not idx_file.exists():
            logger.warning(f"Dispatch scores files do not exist: {bin_file} or {idx_file}, returning empty scores")
            return [], []
        
        # Use Megatron's IndexedDataset to read
        dataset = indexed_dataset.IndexedDataset(str(bin_file)[:-4])  # Remove .bin suffix
        
        scores_sequences = []
        sequence_lengths = []
        
        for i in range(len(dataset)):
            doc_indices = dataset.document_indices
            start_seq = int(doc_indices[i])
            end_seq = int(doc_indices[i + 1])
            sequences = dataset[start_seq:end_seq]
            
            # Each document now has only one sequence
            for seq in sequences:
                arr = np.array(seq, dtype=np.float32)
                scores_sequences.append(arr)
                sequence_lengths.append(len(arr))
        
        logger.info(f"Read dispatch scores data for {len(scores_sequences)} samples")
        return scores_sequences, sequence_lengths
    
    def read_original_dataset(self) -> Iterator[Dict[str, str]]:
        """
        Read original dataset
        
        Yields:
            Dictionary data for each sample
        """
        if self.dataset_path.suffix == '.gz':
            # Handle gzip compressed files
            with gzip.open(self.dataset_path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            yield data
                        except json.JSONDecodeError as e:
                            logger.warning(f"Line {line_num} JSON parsing failed: {e}")
                            continue
        else:
            # Handle plain text files
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            yield data
                        except json.JSONDecodeError as e:
                            logger.warning(f"Line {line_num} JSON parsing failed: {e}")
                            continue
    
    def load_samples(self, key: str = 'text') -> List[Sample]:
        """
        Load complete sample data
        
        Args:
            key: Data key name
            
        Returns:
            Sample list
        """
        # Read dispatch data
        dispatch_sequences, _ = self.read_dispatch_data(key)
        
        # Read tokenized data
        tokenized_sequences, tokenized_length = self.read_tokenized_data(key)
        
        # Read dispatch scores data (optional)
        scores_sequences, scores_length = self.read_dispatch_scores(key)
        
        # Check if scores data is available
        has_scores = len(scores_sequences) > 0
        
        samples = []
        for i, (dispatch_seq, tokenized_seq, tokenized_len) in enumerate(zip(dispatch_sequences, tokenized_sequences, tokenized_length)):
            # Calculate expert preference vector
            expert_preference_vector = self._compute_expert_preference_vector(dispatch_seq)
            
            # Get scores if available
            dispatch_scores = None
            if has_scores and i < len(scores_sequences):
                dispatch_scores = scores_sequences[i].tolist()
            
            # Create sample object
            sample = Sample(
                sample_id=i,
                token_ids=tokenized_seq.tolist(),
                token_count=tokenized_len,
                expert_preference_vector=expert_preference_vector,
                dispatch_ids=dispatch_seq.tolist(),
                dispatch_scores=dispatch_scores
            )
            samples.append(sample)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Loaded {i + 1} samples")
        
        logger.info(f"Total loaded {len(samples)} samples")
        if has_scores:
            logger.info(f"Scores data loaded for {len(scores_sequences)} samples")
        return samples
    
    def _compute_expert_preference_vector(self, dispatch_sequence: np.ndarray) -> np.ndarray:
        """
        Calculate expert preference vector
        
        Args:
            dispatch_sequence: Dispatch sequence
            
        Returns:
            Expert preference vector
        """
        # Count usage frequency of each expert
        expert_counts = np.bincount(dispatch_sequence, minlength=self.num_experts) 
        
        # Normalize to get probability distribution
        total_tokens = len(dispatch_sequence)
        if total_tokens > 0:
            preference_vector = expert_counts.astype(np.float32) / total_tokens
        else:
            preference_vector = np.zeros(self.num_experts, dtype=np.float32)
        
        return preference_vector


class ResultWriter:
    """Result writer"""
    
    def __init__(self, output_dir: str, prefix: str):
        """
        Initialize writer
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_prefix = prefix
        
        # Create subdirectories
        (self.output_dir / "node_data").mkdir(exist_ok=True)
        (self.output_dir / "node_dispatch").mkdir(exist_ok=True)
        (self.output_dir / "node_scores").mkdir(exist_ok=True)
        (self.output_dir / "config_cluster").mkdir(exist_ok=True)
    
    def save_cluster_info(self, clusters: List[Cluster], config: dict, 
                          pca_inverse_transform: Callable) -> str:
        """
        Save cluster information
        
        Args:
            clusters: Clustering results
            config: Configuration information
            pca_inverse_transform: The inverse map for pca
            
        Returns:
            Saved file path
        """
        cluster_info = {
            'config': config,
            'clusters': []
        }
        
        for cluster in clusters:
            cluster_data = {
                'cluster_id': cluster.cluster_id,
                'sample_count': cluster.sample_count,
                'node_id': cluster.node_id,
                'target_experts': [int(e) for e in cluster.target_experts],
                'center_vector': pca_inverse_transform(cluster.center_vector).tolist(),
                'average_entropy': float(cluster.average_entropy),
                'sample_ids': [s.sample_id for s in cluster.samples]
            }
            cluster_info['clusters'].append(cluster_data)
        
        output_path = self.output_dir / 'config_cluster' / f"{self.output_prefix}_cluster_info.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Cluster information saved to: {output_path}")
        return str(output_path)
    
    def save_expert_placement(self, expert_assignments: List[ExpertAssignment]) -> str:
        """
        Save expert placement configuration
        
        Args:
            expert_assignments: Expert allocation list
            
        Returns:
            Saved file path
        """
        placement_data = {}
        
        for assignment in expert_assignments:
            node_key = f"node_{assignment['node_id']}"
            if node_key not in placement_data:
                placement_data[node_key] = []
            
            placement_data[node_key].append({
                'expert_id': int(assignment['expert_id']),
                'gpu_id': assignment['gpu_id'],
                'cluster_id': assignment['cluster_id']
            })
        
        output_path = self.output_dir / 'config_cluster' / f"{self.output_prefix}_expert_placement.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(placement_data, f, indent=2)
        
        logger.info(f"Expert placement configuration saved to: {output_path}")
        return str(output_path)
    
    def save_node_data(self, node_data: NodeData, node_id: int, vocab_size: int) -> Tuple[str, str, Optional[str]]:
        """
        Save node data
        
        Args:
            node_data: Node data
            node_id: Node ID
            vocab_size: 
            
        Returns:
            (Data file path, dispatch file path, scores file path)
        """
        # Save tokenized data using IndexedDataset
        data_file_prefix = self.output_dir / "node_data" / f"{self.output_prefix}_node_{node_id}_data"
        data_builder = indexed_dataset.IndexedDatasetBuilder(
            str(data_file_prefix) + ".bin",
            dtype=indexed_dataset.DType.optimal_dtype(vocab_size)  
        )

        # Save dispatch data using IndexedDataset
        dispatch_file_prefix = self.output_dir / "node_dispatch" / f"{self.output_prefix}_node_{node_id}_dispatch"
        dispatch_builder = indexed_dataset.IndexedDatasetBuilder(
            str(dispatch_file_prefix) + ".bin",
            dtype=np.uint8  
        )
        
        # Check if scores data exists
        has_scores = any(sample.dispatch_scores is not None for sample in node_data.samples)
        scores_builder = None
        scores_file_prefix = None
        scores_file = None
        
        if has_scores:
            # Save dispatch scores data using IndexedDataset
            scores_file_prefix = self.output_dir / "node_scores" / f"{self.output_prefix}_node_{node_id}_scores"
            scores_builder = indexed_dataset.IndexedDatasetBuilder(
                str(scores_file_prefix) + ".bin",
                dtype=np.float32  
            )
        
        # Process each sample
        for sample in node_data.samples:
            # Save tokenized data
            token_ids_array = np.array(sample.token_ids, dtype=indexed_dataset.DType.optimal_dtype(vocab_size))
            data_builder.add_document(token_ids_array, [len(sample.token_ids)])
            
            # Save dispatch data
            dispatch_array = np.array(sample.dispatch_ids, dtype=np.uint8)
            dispatch_builder.add_document(dispatch_array, [len(dispatch_array)])
            
            # Save dispatch scores data if available
            if has_scores and sample.dispatch_scores is not None:
                scores_array = np.array(sample.dispatch_scores, dtype=np.float32)
                scores_builder.add_document(scores_array, [len(scores_array)])
        
        # Finalize all builders
        data_builder.finalize(str(data_file_prefix) + ".idx")
        dispatch_builder.finalize(str(dispatch_file_prefix) + ".idx")
        
        data_file = str(data_file_prefix) + ".bin"
        dispatch_file = str(dispatch_file_prefix) + ".bin"
        
        if has_scores and scores_builder is not None:
            scores_builder.finalize(str(scores_file_prefix) + ".idx")
            scores_file = str(scores_file_prefix) + ".bin"
            logger.info(f"Node {node_id} data saved to: {data_file}, {dispatch_file}, and {scores_file}")
        else:
            logger.info(f"Node {node_id} data saved to: {data_file} and {dispatch_file}")
        
        return data_file, dispatch_file, scores_file
    
    def save_optimization_stats(self, stats: Dict[str, float]) -> str:
        """
        Save optimization statistics
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            Saved file path
        """
        output_path = self.output_dir / 'config_cluster' / f"{self.output_prefix}_optimization_stats.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Optimization statistics saved to: {output_path}")
        return str(output_path)
    
    def save_validation_report(self, report: str) -> str:
        """
        Save validation report
        
        Args:
            report: Validation report content
            
        Returns:
            Saved file path
        """
        output_path = self.output_dir / 'config_cluster' / f"{self.output_prefix}_validation_report.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Validation report saved to: {output_path}")
        return str(output_path)


def load_config(config_path: str) -> dict:
    """
    Load configuration file
    
    Args:
        config_path: Configuration file path
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def save_config(config: dict, config_path: str):
    """
    Save configuration file
    
    Args:
        config: Configuration dictionary
        config_path: Configuration file path
    """
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
