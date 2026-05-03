"""
Sample expert preference analysis module
Implement expert preference vector calculation and analysis
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..utils.data_structures import Sample, SamplePreferenceMatrix
from ..core.config import OptimizationConfig

logger = logging.getLogger(__name__)


class SamplePreferenceAnalyzer:
    """Sample expert preference analyzer"""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize analyzer
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.num_experts = config.num_experts
        self.pca_dimensions = config.pca_dimensions
        self.entropy_threshold = config.entropy_threshold
        
        # Initialize PCA and standardizer
        self.pca = PCA(n_components=self.pca_dimensions, random_state=42)
        self.scaler = StandardScaler()
        
        # Store analysis results
        self.preference_matrix: Optional[SamplePreferenceMatrix] = None
        self.pca_vectors: Optional[np.ndarray] = None
        self.normalized_vectors: Optional[np.ndarray] = None
        
    def analyze_samples(self, samples: List[Sample]) -> Dict[str, np.ndarray]:
        """
        Analyze expert preferences of samples
        
        Args:
            samples: Sample list
            
        Returns:
            Analysis result dictionary
        """
        logger.info(f"Start analyzing expert preferences for {len(samples)} samples")
        
        # Create preference matrix
        self.preference_matrix = SamplePreferenceMatrix(samples)

        # Calculate anomaly samples
        anomaly_mask = self._detect_anomalies(samples)
        
        # Calculate PCA dimensionality reduction vectors
        pca_vectors = self._compute_pca_vectors(anomaly_mask)
        
        # Store results
        self.pca_vectors = pca_vectors
        
        # Return analysis results
        results = {
            'preference_matrix': self.preference_matrix.matrix,
            'pca_vectors': pca_vectors,
            'anomaly_mask': anomaly_mask,
            'explained_variance_ratio': self.pca.explained_variance_ratio_ if self.pca_dimensions is not None else 1,
        }
        
        logger.info(f"Detected {np.sum(anomaly_mask)} anomaly samples")
        if self.pca_dimensions is not None:
            logger.info(f"PCA dimensionality reduction completed, preserved variance ratio: {np.sum(self.pca.explained_variance_ratio_):.3f}")
        
        return results
    
    def _compute_pca_vectors(self, excluded_samples: List[Sample]) -> np.ndarray:
        """
        Calculate PCA dimensionality reduction vectors
        
        Returns:
            PCA dimensionality reduced vectors
        """
        if self.preference_matrix is None:
            raise ValueError("Preference matrix not initialized")
        
        # Apply PCA dimensionality reduction
        if self.pca_dimensions is None:
            # Not carry out pca 
            return self.preference_matrix.matrix[~excluded_samples]
        else:
            pca_vectors = self.pca.fit_transform(self.preference_matrix.matrix[~excluded_samples])
            return pca_vectors
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Standardize vectors
        
        Args:
            vectors: Input vectors
            
        Returns:
            Standardized vectors
        """
        # Apply standardization
        normalized_vectors = self.scaler.fit_transform(vectors)
        
        return normalized_vectors
    
    def _detect_anomalies(self, samples: List[Sample]) -> np.ndarray:
        """
        Detect anomaly samples
        
        Args:
            samples: Sample list
            
        Returns:
            Anomaly sample mask (True means anomaly)
        """
        anomaly_mask = np.array([s.entropy >= self.config.entropy_threshold for s in samples], dtype=bool)
        
        if sum(anomaly_mask) > len(samples) * 0.9:
            logger.warning(f"The anomaly ratio is above 90%, try lower entropy-threshold.") 
        elif sum(anomaly_mask) < len(samples) * 0.05:
            logger.debug(f"The anomaly ratio is below 5%. May not filter any sample") 
        
        return anomaly_mask
    
    def get_sample_statistics(self, samples: List[Sample]) -> Dict[str, float]:
        """
        Get sample statistics
        
        Args:
            samples: Sample list
            
        Returns:
            Statistics dictionary
        """
        if not samples:
            return {}
        
        # Calculate various statistics
        expert_usage_counts = np.zeros(self.num_experts, dtype=int)
        total_tokens = 0
        
        for sample in samples:
            # Count expert usage
            expert_usage_counts += np.bincount(sample.dispatch_ids, minlength=self.num_experts)
            total_tokens += sample.token_count
        
        # Calculate statistics
        stats = {
            'total_samples': len(samples),
            'total_tokens': total_tokens,
            'average_tokens_per_sample': total_tokens / len(samples),
            'expert_usage_std': float(np.std(expert_usage_counts)),
            'expert_usage_min': int(np.min(expert_usage_counts)),
            'expert_usage_max': int(np.max(expert_usage_counts)),
            'expert_usage_mean': float(np.mean(expert_usage_counts)),
            'expert_usage_median': float(np.median(expert_usage_counts)),
            'unused_experts': int(np.sum(expert_usage_counts == 0)),
            'most_used_expert': int(np.argmax(expert_usage_counts)),
            'least_used_expert': int(np.argmin(expert_usage_counts))
        }
        
        return stats
    
    def get_expert_preference_distribution(self, samples: List[Sample]) -> Dict[str, np.ndarray]:
        """
        Get expert preference distribution
        
        Args:
            samples: Sample list
            
        Returns:
            Distribution information dictionary
        """
        if not samples:
            return {}
        
        # Collect preference vectors of all samples
        preference_vectors = np.array([s.expert_preference_vector for s in samples])
        
        # Calculate distribution statistics
        distribution_stats = {
            'mean_preference': np.mean(preference_vectors, axis=0),
            'std_preference': np.std(preference_vectors, axis=0),
            'min_preference': np.min(preference_vectors, axis=0),
            'max_preference': np.max(preference_vectors, axis=0),
            'median_preference': np.median(preference_vectors, axis=0),
            'expert_popularity_rank': np.argsort(-np.mean(preference_vectors, axis=0))
        }
        
        return distribution_stats
    
    def filter_samples_by_entropy(self, samples: List[Sample], 
                                 max_entropy: Optional[float] = None) -> Tuple[List[Sample], List[Sample]]:
        """
        Filter samples by entropy
        
        Args:
            samples: Sample list
            max_entropy: Maximum entropy threshold, if None use configuration threshold
            
        Returns:
            (Normal sample list, anomaly sample list)
        """
        if max_entropy is None:
            max_entropy = self.entropy_threshold
        
        normal_samples = []
        anomaly_samples = []
        
        for sample in samples:
            if sample.entropy <= max_entropy:
                normal_samples.append(sample)
            else:
                anomaly_samples.append(sample)
        
        logger.info(f"Entropy filtering completed: {len(normal_samples)} normal samples, {len(anomaly_samples)} anomaly samples")
        
        return normal_samples, anomaly_samples
    
    def get_sample_similarity_matrix(self, sample_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        Get sample similarity matrix
        
        Args:
            sample_ids: Specified sample ID list, if None use all samples
            
        Returns:
            Similarity matrix
        """
        if self.preference_matrix is None:
            raise ValueError("Preference matrix not initialized")
        
        if sample_ids is None:
            # Use all samples
            sample_indices = list(range(len(self.preference_matrix.samples)))
        else:
            # Use specified samples
            sample_indices = []
            for sample_id in sample_ids:
                try:
                    idx = self.preference_matrix.sample_ids.index(sample_id)
                    sample_indices.append(idx)
                except ValueError:
                    logger.warning(f"Sample ID {sample_id} does not exist")
                    continue
        
        n_samples = len(sample_indices)
        similarity_matrix = np.zeros((n_samples, n_samples))
        
        # Calculate pairwise similarity
        for i, idx1 in enumerate(sample_indices):
            for j, idx2 in enumerate(sample_indices):
                if i <= j:
                    sample1_id = self.preference_matrix.sample_ids[idx1]
                    sample2_id = self.preference_matrix.sample_ids[idx2]
                    similarity = self.preference_matrix.compute_pairwise_similarity(sample1_id, sample2_id)
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def export_analysis_report(self, samples: List[Sample], output_path: str):
        """
        Export analysis report
        
        Args:
            samples: Sample list
            output_path: Output file path
        """
        import json
        
        # Get statistics
        sample_stats = self.get_sample_statistics(samples)
        preference_dist = self.get_expert_preference_distribution(samples)

        if self.pca_dimensions is not None:
            pca_analysis = {
                'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(self.pca.explained_variance_ratio_).tolist()
            }
        else:
            pca_analysis = None
        # Prepare report data
        report = {
            'sample_statistics': sample_stats,
            'expert_preference_distribution': {
                'mean_preference': preference_dist['mean_preference'].tolist(),
                'std_preference': preference_dist['std_preference'].tolist(),
                'expert_popularity_rank': preference_dist['expert_popularity_rank'].tolist()
            },
            'pca_analysis': pca_analysis
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis report saved to: {output_path}")
