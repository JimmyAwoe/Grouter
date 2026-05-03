"""
Sample clustering optimization module
Implement three-stage clustering strategy: vector preprocessing, hybrid clustering algorithm, cluster-sample binding storage
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set, Callable
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment
import random

from ..utils.data_structures import Sample, Cluster, SamplePreferenceMatrix
from ..core.config import OptimizationConfig

logger = logging.getLogger(__name__)


class ClusterOptimizer:
    """Sample clustering optimizer"""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize clustering optimizer
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.num_nodes = config.num_nodes
        self.num_experts = config.num_experts
        self.experts_per_gpu = config.experts_per_gpu
        self.first_hier_cluster_num = config.first_hier_cluster_num
        self.cosine_similarity_threshold = config.cosine_similarity_threshold
        self.expert_balance_weight = config.expert_balance_weight
        self.similarity_weight = config.similarity_weight

        assert self.first_hier_cluster_num >= self.num_nodes, "The first hierarchical clustering number must be greater than the aiming cluster number"
        
        # Clustering results
        self.clusters: List[Cluster] = []
        self.backup_cluster: Optional[Cluster] = None
        
    def cluster_samples(self, samples: List[Sample], 
                       pca_vectors: np.ndarray,
                       anomaly_mask: np.ndarray,
                       pca_inverse_transform: Callable) -> List[Cluster]:
        """
        Execute three-stage clustering strategy
        
        Args:
            samples: Sample list
            pca_vectors: PCA dimensionality reduced vectors
            anomaly_mask: Anomaly mask for samples
            pca_inverse_transform: The inverse transform of PCA
            
        Returns:
            Clustering result list
        """
        logger.info(f"Start executing three-stage clustering strategy, target cluster count: {self.num_nodes}")
        
        # Stage 1: Vector preprocessing
        normal_samples, anomaly_samples = self._preprocess_vectors(samples, anomaly_mask)
        logger.info(f"Vector preprocessing completed: {len(normal_samples)} normal samples, {len(anomaly_samples)} anomaly samples")
        
        # Stage 2: Hybrid clustering algorithm
        clusters = self._hybrid_clustering(normal_samples, pca_vectors)
        logger.info(f"Hybrid clustering completed, generated {len(clusters)} clusters")
        
        # Stage 3: Cluster-sample binding storage
        self._bind_clusters_and_samples(clusters, normal_samples, anomaly_samples, pca_inverse_transform)
        
        # Allocate target experts
        self._assign_target_experts(pca_inverse_transform)
        
        # Validate clustering results
        self._validate_clustering()
        
        logger.info("Three-stage clustering strategy execution completed")
        return self.clusters
    
    def _preprocess_vectors(self, samples: List[Sample], 
                           anomaly_mask: np.ndarray) -> Tuple[List[Sample], List[Sample]]:
        """
        Stage 1: Vector preprocessing
        
        Args:
            samples: Sample list
            anomaly_mask: Anomaly mask for samples

        Returns:
            (Normal sample list, anomaly sample list)
        """
        # Filter samples by entropy
        normal_samples = np.array(samples)[~anomaly_mask].tolist()
        anomaly_samples = np.array(samples)[anomaly_mask].tolist()
        
        # Create backup cluster
        if anomaly_samples:
            self.backup_cluster = Cluster(
                cluster_id=-1,  # Backup cluster uses -1 as ID
                samples=anomaly_samples,
                center_vector=None,
                target_experts=set()
            )
            logger.info(f"Created backup cluster with {len(anomaly_samples)} anomaly samples")
        
        return normal_samples, anomaly_samples
    
    def _hybrid_clustering(self, samples: List[Sample], 
                          pca_vectors: np.ndarray) -> List[Cluster]:
        """
        Stage 2: Hybrid clustering algorithm
        
        Args:
            samples: Normal sample list
            pca_vectors: PCA dimensionality reduced vectors
            
        Returns:
            Clustering result list
        """
        if len(samples) < self.num_nodes:
            logger.warning(f"Sample count ({len(samples)}) less than target cluster count ({self.num_nodes})")
            # Create single cluster
            cluster = Cluster(
                cluster_id=0,
                samples=samples,
                center_vector=np.mean(pca_vectors, axis=0) if len(pca_vectors) > 0 else None,
                target_experts=set()
            )
            return [cluster]
        
        # Step 1: K-means++ initialization
        kmeans = KMeans(
            n_clusters=self.first_hier_cluster_num,
            init='k-means++',
            n_init=10,
            random_state=42
        )
        cluster_labels = kmeans.fit_predict(pca_vectors)
        
        # Create initial clusters
        initial_clusters = []
        for i in range(self.first_hier_cluster_num):
            cluster_samples = [s for j, s in enumerate(samples) if cluster_labels[j] == i]
            if cluster_samples:
                cluster_vectors = pca_vectors[cluster_labels == i]
                center_vector = np.mean(cluster_vectors, axis=0)
                
                cluster = Cluster(
                    cluster_id=i,
                    samples=cluster_samples,
                    center_vector=center_vector,
                    target_experts=set()
                )
                initial_clusters.append(cluster)
        
        logger.info(f"K-means++ initialization completed, generated {len(initial_clusters)} initial clusters")
        
        # Step 2: Hierarchical clustering refinement
        refined_clusters = self._refine_clusters_with_hierarchical(initial_clusters, pca_vectors, cluster_labels)
        
        return refined_clusters
    
    def _refine_clusters_with_hierarchical(self, initial_clusters: List[Cluster], 
                                         pca_vectors: np.ndarray,
                                         first_hier_cluster_labels: np.ndarray) -> List[Cluster]:
        """
        Refine initial clusters using hierarchical clustering
        
        Args:
            initial_clusters: Initial cluster list
            pca_vectors: PCA dimensionality reduced vectors
            first_hier_cluster_labels: The first hierarchical clustering labels
            
        Returns:
            Refined cluster list
        """
        # Calculate inter-cluster similarity matrix
        n_clusters = len(initial_clusters)
        similarity_matrix = np.zeros((n_clusters, n_clusters))
        
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                similarity = cosine_similarity(
                    [initial_clusters[i].center_vector], 
                    [initial_clusters[j].center_vector]
                )[0, 0]
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Use hierarchical clustering to merge similar clusters
        if n_clusters > self.num_nodes:
            # Need to merge clusters
            clustering = AgglomerativeClustering(
                n_clusters=self.num_nodes,
                linkage='complete',
                metric='precomputed'
            )
            
            # Convert similarity to distance
            distance_matrix = 1 - similarity_matrix
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Reorganize clusters
            merged_clusters = [[] for _ in range(self.num_nodes)]
            for i, label in enumerate(cluster_labels):
                merged_clusters[label].extend(initial_clusters[i].samples)
            
            # Create new clusters
            refined_clusters = []
            for i, cluster_samples in enumerate(merged_clusters):
                if cluster_samples:
                    # Calculate new center vector
                    first_hier_cluster_indices = np.arange(n_clusters)[cluster_labels == i]
                    cluster_vectors = pca_vectors[np.isin(first_hier_cluster_labels, first_hier_cluster_indices)]
                    center_vector = np.mean(cluster_vectors, axis=0)
                    
                    cluster = Cluster(
                        cluster_id=i,
                        samples=cluster_samples,
                        center_vector=center_vector,
                        target_experts=set()
                    )
                    refined_clusters.append(cluster)
            
            logger.info(f"Hierarchical clustering refinement completed, merged into {len(refined_clusters)} clusters")
            return refined_clusters
        else:
            # No need to merge, return directly
            return initial_clusters
    
    def _bind_clusters_and_samples(self, clusters: List[Cluster], 
                                  normal_samples: List[Sample], 
                                  anomaly_samples: List[Sample],
                                  pca_inverse_transform: Callable):
        """
        Stage 3: Cluster-sample binding storage
        
        Args:
            clusters: Clustering results
            normal_samples: Normal samples
            anomaly_samples: Anomaly samples
            pca_inverse_transform: The inverse transform of PCA
        """
        self.clusters = clusters
        
        # Allocate node ID for each cluster
        for i, cluster in enumerate(clusters):
            cluster.node_id = i
            logger.info(f"Cluster {cluster.cluster_id} allocated to node {cluster.node_id}, sample count: {cluster.sample_count}")
        
        # Handle samples in backup cluster
        if self.backup_cluster and anomaly_samples:
            self._distribute_backup_samples(normal_samples, anomaly_samples)
        
        # Calculate sample scores for each cluster and sort
        self._rank_cluster_samples(pca_inverse_transform)
    
    def _distribute_backup_samples(self, normal_samples: List[Sample], anomaly_samples: List[Sample]):
        """
        Distribute samples in backup cluster to various clusters (Optimized version)
    
        Args:
            normal_samples: Normaly sample list
            anomaly_samples: Anomaly sample list
        """
        if not anomaly_samples or not self.clusters:
            return
    
        # Pre-calculate all required information to avoid repeated computation
        num_clusters = len(self.clusters)
        num_anomaly_samples = len(anomaly_samples)
        anomaly_samples_length = np.array([s.token_count for s in anomaly_samples])
        cluster_length = np.array([sum([s.token_count for s in cluster.samples]) for cluster in self.clusters])
        total_length = sum(anomaly_samples_length) + sum(cluster_length)
    
        # Pre-calculate the number of samples to allocate for each cluster
        target_length_per_cluster = total_length // num_clusters
    
        # Pre-allocate sample-to-cluster mapping
        cluster_sample_mapping = [[] for _ in range(num_clusters)]
    
        # Calculate how many samples should be allocated to each cluster
        needed_length_per_cluster = np.array([target_length_per_cluster] * num_clusters) - cluster_length

        # Batch allocate samples to corresponding clusters
        sample_index = 0
        for cluster_idx, needed_length in enumerate(needed_length_per_cluster):
            if needed_length > 0:
                # Get remaining samples
                remaining_sample_length = anomaly_samples_length[sample_index:]
                remaining_samples = anomaly_samples[sample_index:]

                # Calculate samples needed
                cumsum = np.cumsum(remaining_sample_length)
                if len(np.where(cumsum > needed_length)[0]) > 0:
                    end_index = np.where(cumsum > needed_length)[0][0]
                else:
                    end_index = len(remaining_sample_length)

                # Add to cluster
                cluster_sample_mapping[cluster_idx].extend(remaining_samples[:end_index+1])
                sample_index += end_index + 1
            
                if sample_index >= num_anomaly_samples:
                    break
    
        # Batch update all clusters to reduce method call overhead
        for cluster_idx, samples_to_add in enumerate(cluster_sample_mapping):
            if samples_to_add:
                # Use extend instead of multiple add_sample calls
                self.clusters[cluster_idx].add_samples(samples_to_add)
    
        logger.info(f"Backup sample distribution completed, distributed {sample_index} samples")
    
    def _rank_cluster_samples(self, pca_inverse_transform: Callable):
        """
        Calculate sample scores for each cluster and sort

        Args:
            pca_inverse_transforme: The inverse transform for PCA

        """
        scaler = MinMaxScaler(feature_range=(-1, 1))
        for cluster in self.clusters:
            # Calculate score for each sample
            sample_scores = []
            inverse_center = pca_inverse_transform(cluster.center_vector)

            # Get similarity
            expert_preference_vectors = [s.expert_preference_vector for s in cluster.samples]
            similarity_vectors = cosine_similarity(expert_preference_vectors, [inverse_center]) 

            # Get expert concentration
            expert_concentration_vectors = np.array([[-s.expert_concentration for s in cluster.samples]]).T
            scaled_ec_vectors = scaler.fit_transform(expert_concentration_vectors)
            
            # Combining two features
            scores = self.similarity_weight * similarity_vectors + self.expert_balance_weight * scaled_ec_vectors

            sample_scores = [(sample, score.item()) for sample, score in zip(cluster.samples, scores)]
            
            # Sort by score
            sample_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Rearrange samples
            cluster.samples = [s[0] for s in sample_scores]
            
            logger.debug(f"Cluster {cluster.cluster_id} sample sorting completed, highest score: {sample_scores[0][1]:.3f}")
    
    def _assign_target_experts(self, pca_inverse_transform: Callable):
        """
        Allocate target experts for each cluster

        Args:
            pca_inverse_transforme: The inverse transform for PCA
        """
        # Calculate experts to allocate for each node
        experts_per_node = self.num_experts // self.num_nodes
        
        # Inverse center vector to orgint feature space
        inverse_centers = np.array([pca_inverse_transform(c.center_vector) for c in self.clusters])

        # Calculate expert assignments
        expert_assignments = self._assign_experts_to_clusters_with_hungarian(-inverse_centers, experts_per_node)

        # Allocate experts for each cluster
        for i, cluster in enumerate(self.clusters):
            cluster.target_experts = set(expert_assignments[i])
            
            logger.info(f"Cluster {cluster.cluster_id} allocated experts: {sorted(cluster.target_experts)}")


    def _assign_experts_to_clusters_with_hungarian(self, negative_expert_preference: np.ndarray, 
                                                   num_assignments_per_cluster: int) -> Dict:
        """
        Assigns a fixed number of experts to each cluster using a modified Hungarian algorithm.

        Args:
            negative_expert_preference: The negative level for a cluster to experts
            num_assignments_per_cluster: Number of experts to assign to each cluster
        """
        num_clusters, _ = negative_expert_preference.shape

        # Expand the cost matrix
        expanded_cost = np.repeat(negative_expert_preference, repeats=num_assignments_per_cluster, axis=0)
    
        # Apply the standard Hungarian algorithm to find optimal assignments in the expanded matrix
        row_indices, col_indices = linear_sum_assignment(expanded_cost)
    
        # Map the expanded row indices back to the original cluster indice
        cluster_assignments = [idx // num_assignments_per_cluster for idx in row_indices]

        return {
            cluster: [col for idx, col in enumerate(col_indices) if cluster_assignments[idx] == cluster]
            for cluster in range(num_clusters)
        }
    
    def _validate_clustering(self):
        """
        Validate clustering results
        """
        # Check cluster count
        if len(self.clusters) != self.num_nodes:
            logger.warning(f"Cluster count ({len(self.clusters)}) doesn't match target node count ({self.num_nodes})")
        
        # Check uniqueness of expert allocation
        all_target_experts = set()
        for cluster in self.clusters:
            if cluster.target_experts & all_target_experts:
                logger.warning(f"Cluster {cluster.cluster_id} expert set overlaps with other clusters")
            all_target_experts.update(cluster.target_experts)
        
        # Check uniqueness of sample allocation
        all_sample_ids = set()
        for cluster in self.clusters:
            cluster_sample_ids = {s.sample_id for s in cluster.samples}
            if cluster_sample_ids & all_sample_ids:
                logger.warning(f"Cluster {cluster.cluster_id} samples overlap with other clusters")
            all_sample_ids.update(cluster_sample_ids)
        
        logger.info("Clustering validation completed")
    
    def get_cluster_statistics(self) -> Dict[str, any]:
        """
        Get clustering statistics
        
        Returns:
            Statistics dictionary
        """
        if not self.clusters:
            return {}
        
        stats = {
            'total_clusters': len(self.clusters),
            'total_samples': sum(c.sample_count for c in self.clusters),
            'cluster_sizes': [c.sample_count for c in self.clusters],
            'average_cluster_size': float(np.mean([c.sample_count for c in self.clusters])),
            'cluster_size_std': float(np.std([c.sample_count for c in self.clusters])),
            'expert_coverage': len(set().union(*[c.target_experts for c in self.clusters])),
            'backup_samples': len(self.backup_cluster.samples) if self.backup_cluster else 0
        }
        
        return stats
    
    def export_clustering_report(self, output_path: str):
        """
        Export clustering report
        
        Args:
            output_path: Output file path
        """
        import json
        
        # Prepare report data
        report = {
            'clustering_config': {
                'num_nodes': self.num_nodes,
                'num_experts': self.num_experts,
                'experts_per_gpu': self.experts_per_gpu,
                'cosine_similarity_threshold': self.cosine_similarity_threshold
            },
            'clustering_statistics': self.get_cluster_statistics(),
            'clusters': []
        }
        
        for cluster in self.clusters:
            cluster_data = {
                'cluster_id': cluster.cluster_id,
                'node_id': cluster.node_id,
                'sample_count': cluster.sample_count,
                'target_experts': sorted([int(x) for x in cluster.target_experts]),
                'center_vector': cluster.center_vector.tolist() if cluster.center_vector is not None else None,
                'average_entropy': float(cluster.average_entropy),
            }
            report['clusters'].append(cluster_data)
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Clustering report saved to: {output_path}")
