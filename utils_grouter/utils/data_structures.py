"""
Data structure definition module
Define core data structures for samples, clusters, expert allocation, etc.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


@dataclass
class Sample:
    """Sample data structure"""
    sample_id: int                    # Sample ID
    token_ids: List[int]              # Token ids from tokenizer
    token_count: int
    expert_preference_vector: np.ndarray  # Expert preference vector (E-dimensional)
    dispatch_ids: List[int]           # Dispatch result for each token
    dispatch_scores: Optional[List[float]] = None  # Dispatch scores for each token (float32)
    cluster_id: Optional[int] = None  # Cluster ID

    # Megatron processed data fields
    labels: Optional[List[int]] = None           # Labels for language modeling
    
    entropy_type = 'cross_entropy'
    @property
    def entropy(self) -> float:
        """Calculate entropy of expert preference vector"""
        # Avoid log(0) case
        if self.entropy_type == 'cross_entropy':
            valid_probs = self.expert_preference_vector[self.expert_preference_vector > 0]
            if len(valid_probs) == 0:
                return 0.0
            return -np.sum(valid_probs * np.log2(valid_probs))
        elif self.entropy_type == 'maxvio':
            mean_load = self.expert_preference_vector.mean()
            return (self.expert_preference_vector.max() - mean_load) / mean_load
    
    @property
    def expert_concentration(self) -> float:
        """Calculate expert concentration (maximum probability value)"""
        return np.max(self.expert_preference_vector)

    def __repr__(self):
        """Only present important info"""
        return (f"Sample(id={self.sample_id}, "
                f"cluster={self.cluster_id}, entropy={self.entropy:.2f}, "
                f"concentration={self.expert_concentration:.2f})")


@dataclass
class Cluster:
    """Cluster data structure"""
    cluster_id: int                           # Cluster ID
    samples: List[Sample]                     # Samples in cluster
    center_vector: np.ndarray                 # Cluster center vector
    target_experts: Set[int]                  # Target expert set
    node_id: Optional[int] = None             # Corresponding node ID
    
    def __post_init__(self):
        """Tag samples and Calculate cluster center vector"""
        if self.samples:
            for s in self.samples:
                s.cluster_id = self.cluster_id
            if self.center_vector is None:
                vectors = [s.expert_preference_vector for s in self.samples]
                self.center_vector = np.mean(vectors, axis=0)
    
    @property
    def sample_count(self) -> int:
        """Sample count"""
        return len(self.samples)
    
    @property
    def average_entropy(self) -> float:
        """Average entropy"""
        if not self.samples:
            return 0.0
        return np.mean([s.entropy for s in self.samples])
    
    def add_samples(self, samples: List[Sample], update_center: bool=False):
        """Add sample to cluster"""
        self.samples.extend(samples)
        for sample in samples:
            sample.cluster_id = self.cluster_id
        # Update cluster center
        if self.samples and update_center:
            vectors = [s.expert_preference_vector for s in self.samples]
            self.center_vector = np.mean(vectors, axis=0)
    
    def remove_sample(self, sample: Sample):
        """Remove sample from cluster"""
        if sample in self.samples:
            self.samples.remove(sample)
            sample.cluster_id = None
            # Update cluster center
            if self.samples:
                vectors = [s.expert_preference_vector for s in self.samples]
                self.center_vector = np.mean(vectors, axis=0)

    def __repr__(self):
        """Only represent important info"""
        return f"Cluster(id={self.cluster_id}, samples_count={len(self.samples)})"


@dataclass
class ExpertAssignment:
    """Expert allocation structure"""
    expert_id: int                    # Expert ID
    node_id: int                      # Node ID
    gpu_id: int                       # GPU ID
    cluster_id: int                   # Corresponding cluster ID
    
    def __post_init__(self):
        """Validate allocation validity"""
        assert 0 <= self.node_id < 16, f"Node ID out of range: {self.node_id}"
        assert 0 <= self.gpu_id < 8, f"GPU ID out of range: {self.gpu_id}"


@dataclass
class NodeData:
    """Node data structure"""
    node_id: int                      # Node ID
    samples: List[Sample]             # Samples on node
    experts: Set[int]                 # Experts on node
    gpu_assignments: Dict[int, List[Sample]] = field(default_factory=dict)  # GPU allocation
    
    def __post_init__(self):
        """Initialize GPU allocation"""
        for gpu_id in range(8):
            self.gpu_assignments[gpu_id] = []
    
    def add_sample(self, sample: Sample, gpu_id: int):
        """Add sample to specified GPU"""
        if gpu_id not in self.gpu_assignments:
            self.gpu_assignments[gpu_id] = []
        self.gpu_assignments[gpu_id].append(sample)
        self.samples.append(sample)
    
    def get_gpu_sample_count(self, gpu_id: int) -> int:
        """Get sample count on specified GPU"""
        return len(self.gpu_assignments.get(gpu_id, []))

    def __repr__(self) -> str:
        return (f"NodeData(node_id={self.node_id}, "
                f"sample_count={len(self.samples)}, "
                f"expert_count={len(self.experts)}, "
                f"gpu_count={len(self.gpu_assignments)})")


@dataclass
class OptimizationResult:
    """Optimization result data structure"""
    # Node data files
    node_data_files: Dict[int, str]  # node_id -> data_file_path
    
    # Node dispatch files
    node_dispatch_files: Dict[int, str]  # node_id -> dispatch_file_path
    
    # Node scores files (optional)
    node_scores_files: Dict[int, Optional[str]]  # node_id -> scores_file_path (None if not available)
    
    # Expert placement configuration
    expert_placement_config: str  # JSON configuration file path
    
    # Cluster information
    cluster_info: str  # Cluster result file path
    
    # Optimization statistics
    optimization_stats: Dict[str, float]  # Various optimization metrics
    
    # Validation report
    validation_report: str  # Validation result report


@dataclass
class BatchAssignment:
    """Batch allocation structure"""
    batch_id: int                     # Batch ID
    node_assignments: Dict[int, List[Sample]]  # Node allocation
    gpu_assignments: Dict[Tuple[int, int], List[Sample]]  # (node_id, gpu_id) -> samples
    expert_placements: Dict[int, int]  # expert_id -> gpu_id
    
    def get_node_gpu_samples(self, node_id: int, gpu_id: int) -> List[Sample]:
        """Get samples on specified node and GPU"""
        return self.gpu_assignments.get((node_id, gpu_id), [])
    
    def get_node_samples(self, node_id: int) -> List[Sample]:
        """Get all samples on specified node"""
        return self.node_assignments.get(node_id, [])


class SamplePreferenceMatrix:
    """Sample preference matrix"""
    
    def __init__(self, samples: List[Sample]):
        self.samples = samples
        self.matrix = np.array([s.expert_preference_vector for s in samples])
        self.sample_ids = [s.sample_id for s in samples]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix shape (sample_count, expert_count)"""
        return self.matrix.shape
    
    def get_sample_vector(self, sample_id: int) -> Optional[np.ndarray]:
        """Get preference vector for specified sample"""
        try:
            idx = self.sample_ids.index(sample_id)
            return self.matrix[idx]
        except ValueError:
            return None
    
    def get_cluster_vectors(self, cluster: Cluster) -> np.ndarray:
        """Get preference vectors for all samples in cluster"""
        cluster_samples = [s for s in self.samples if s.cluster_id == cluster.cluster_id]
        if not cluster_samples:
            return np.empty((0, self.matrix.shape[1]))
        return np.array([s.expert_preference_vector for s in cluster_samples])
    
    def compute_pairwise_similarity(self, sample1_id: int, sample2_id: int) -> float:
        """Calculate cosine similarity between two samples"""
        vec1 = self.get_sample_vector(sample1_id)
        vec2 = self.get_sample_vector(sample2_id)
        if vec1 is None or vec2 is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class AlignedDispatchGPTDataset:
    """
    A dataset class that processes dispatch information exactly like GPTDataset processes tokens
    
    This ensures perfect alignment between processed tokens and dispatch information.
    """
    
    def __init__(self, gpt_dataset, dispatch_dataset, topk, dummy_expert_id, eod_id):
        self.gpt_dataset = gpt_dataset
        self.dispatch_dataset = dispatch_dataset
        self.topk = topk
        self.dummy_expert_id = dummy_expert_id
        self.eod_id = eod_id
        
        # Copy the same indices from GPT dataset
        # To see the logic behind, one can check megatron.core.datasets.readme.md
        self.document_index = gpt_dataset.document_index
        self.sample_index = gpt_dataset.sample_index
        self.shuffle_index = gpt_dataset.shuffle_index
        
        # Copy configuration
        self.config = gpt_dataset.config
        self.dataset = dispatch_dataset

        assert len(self.dummy_expert_id) == self.topk, "The dummy expert id must have topk experts."
        
    def __len__(self):
        return len(self.gpt_dataset)
    
    def __getitem__(self, idx):
        """
        Get dispatch information for the given index, processed exactly like tokens
        
        This method replicates the exact logic from GPTDataset._query_document_sample_shuffle_indices
        """
        # Use the same shuffle mapping as GPT dataset
        idx = self.shuffle_index[idx]
        
        # Get the beginning and end documents and offsets (same as GPT dataset)
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]
        
        document_ids = []
        dispatch_parts = []
        
        # Sample spans a single document
        if doc_index_beg == doc_index_end:
            # Add the document id
            document_ids.append(self.document_index[doc_index_beg])
            
            # Add the entire dispatch sample
            dispatch_parts.append(
                self.dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset * self.topk,
                    length=(doc_index_end_offset - doc_index_beg_offset) * self.topk,
                )
            )
        
        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                # Add the document id
                document_ids.append(self.document_index[i])
                
                # Add the dispatch sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = (
                    None
                    if i < doc_index_end
                    else doc_index_end_offset
                )
                length = length * self.topk if length is not None else None
                dispatch_parts.append(
                    self.dataset.get(self.document_index[i], offset=offset*self.topk, length=length)
                )


                if length is None:
                    # Dispatch data doesn't include eod, add dummy_expert_id to compensate
                    dispatch_parts.append(np.array(self.dummy_expert_id, dtype=np.uint8))
        
        
        length = sum(map(len, dispatch_parts))

        assert length // self.topk == len(self.gpt_dataset[idx]['tokens']) , \
            f"len(token_dataset) ({len(self.gpt_dataset[idx]['tokens'])}) != len(dispatch_parts) ({length // self.topk})"
        
        # Concatenate dispatch parts (same as token concatenation)
        aligned_dispatch = np.concatenate(dispatch_parts)
        
        return aligned_dispatch


class AlignedDispatchBlendedDataset:
    """
    A BlendedDataset-like class that returns aligned dispatch information
    
    This class maintains the same interface as BlendedDataset but returns
    dispatch information that perfectly aligns with the processed tokens.
    """
    
    def __init__(self, datasets, weights, size, config, dataset_index=None, dataset_sample_index=None):
        self.datasets = datasets
        self.weights = weights
        self.size = size
        self.config = config
        
        # Use provided indices if available, otherwise build new ones
        if dataset_index is not None and dataset_sample_index is not None:
            self.dataset_index = dataset_index
            self.dataset_sample_index = dataset_sample_index
        else:
            # Build the same indices as BlendedDataset
            self.dataset_index, self.dataset_sample_index = self._build_indices()
    
    def __len__(self):
        return len(self.dataset_index)
    
    def __getitem__(self, idx):
        """
        Get dispatch information for the given index
        
        This replicates the exact logic from BlendedDataset.__getitem__
        """
        dataset_id = self.dataset_index[idx]
        dataset_sample_id = self.dataset_sample_index[idx]
        
        # Return dispatch information instead of tokens
        return self.datasets[dataset_id][dataset_sample_id]
    
    def _build_indices(self):
        """
        Build the same indices as BlendedDataset
        
        This ensures the dispatch dataset follows the exact same sampling pattern.
        """
        # Replicate the exact BlendedDataset index building logic
        size = self.size if self.size is not None else sum(len(ds) for ds in self.datasets)
        
        # Build blending indices exactly like BlendedDataset
        if self.weights is not None:
            # Use the same blending logic as BlendedDataset
            dataset_index = np.zeros(size, dtype=np.int16)
            dataset_sample_index = np.zeros(size, dtype=np.int64)
            
            # Initialize counters for each dataset
            current_samples = np.zeros(len(self.datasets), dtype=np.int64)
            
            # For each sample, determine which dataset to sample from
            for sample_idx in range(size):
                # Find the dataset with maximum sampling error (same logic as BlendedDataset)
                sample_idx_double = max(sample_idx, 1.0)
                max_error_index = 0
                max_error = (self.weights[0] * sample_idx_double - 
                           float(current_samples[0]))
                
                for dataset_idx in range(1, len(self.datasets)):
                    error = (self.weights[dataset_idx] * sample_idx_double - 
                            float(current_samples[dataset_idx]))
                    if error > max_error:
                        max_error = error
                        max_error_index = dataset_idx
                
                # Assign the sample to the dataset with maximum error
                dataset_index[sample_idx] = max_error_index
                dataset_sample_index[sample_idx] = current_samples[max_error_index]
                
                # Update the counter
                current_samples[max_error_index] += 1
        else:
            # No weights specified - use sequential assignment
            dataset_index = np.zeros(size, dtype=np.int16)
            dataset_sample_index = np.zeros(size, dtype=np.int64)
            
            # Simple sequential assignment
            for i in range(size):
                dataset_index[i] = i % len(self.datasets)
                dataset_sample_index[i] = i // len(self.datasets)
        
        return dataset_index, dataset_sample_index

class MergedDataset(Dataset):
    def __init__(self, data_dataset, dispatch_dataset, new_key='dispatch_ids'):
        self.data_dataset = data_dataset
        self.dispatch_dataset = dispatch_dataset
        self.new_key = new_key
        assert len(data_dataset) == len(dispatch_dataset), "The length of dispatch_ids and data must be the same"
        
    def __len__(self):
        return len(self.data_dataset)
    
    def __getitem__(self, idx):
        data = self.data_dataset[idx]
        dispatch_id = self.dispatch_dataset[idx]
        data[self.new_key] = dispatch_id
        return data
