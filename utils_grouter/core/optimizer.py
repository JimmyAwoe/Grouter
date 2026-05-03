"""
EP communication optimization main interface
Integrate all optimization modules, execute complete EP optimization process
"""

import logging
import time
from typing import Dict, List, Optional, Set
from pathlib import Path
import numpy as np
import json

from .config import OptimizationConfig, DataPaths
from ..analysis.sample_analyzer import SamplePreferenceAnalyzer
from ..analysis.cluster_optimizer import ClusterOptimizer
from ..utils.file_io import PredispatchReader, ResultWriter
from ..utils.data_structures import Sample, Cluster, OptimizationResult, NodeData
from ..placement.node_assigner import NodeAssigner as BatchNodeAssigner
from ..placement.expert_sample_coordinator import ExpertSampleCoordinator

logger = logging.getLogger(__name__)


class EPOptimizer:
    """EP communication optimization main interface"""
    
    def __init__(self, config: OptimizationConfig, data_paths: DataPaths):
        """
        Initialize EP optimizer
        
        Args:
            config: Optimization configuration
            data_paths: Data path configuration
        """
        self.config = config
        self.data_paths = data_paths
        
        # Initialize various modules
        self.sample_analyzer = SamplePreferenceAnalyzer(config)
        self.cluster_optimizer = ClusterOptimizer(config)
        
        # Initialize file I/O handlers
        self.predispatch_reader = PredispatchReader(
            data_paths.predispatch_path, 
            data_paths.dataset_path,
            self.config.num_experts,
            self.config.vocab_size,
        )
        self.result_writer = ResultWriter(data_paths.output_dir, config.output_prefix)
        
        # Store intermediate results
        self.samples: Optional[List[Sample]] = None
        self.clusters: Optional[List[Cluster]] = None
        self.analysis_results: Optional[Dict] = None
        # Batch-time optimizers
        self.batch_node_assigner: Optional[BatchNodeAssigner] = None
        self.expert_sample_coordinator: Optional[ExpertSampleCoordinator] = None

        
        logger.info("EP optimizer initialization completed")
    
    def optimize(self) -> OptimizationResult:
        """
        Execute complete EP optimization process
        
        Returns:
            Optimization result
        """
        start_time = time.time()
        logger.info("Start executing EP optimization process")
        
        # Step 1: Sample analysis
        logger.info("Step 1: Start sample expert preference analysis")
        self._analyze_samples()
            
        # Step 2: Sample clustering
        logger.info("Step 2: Start sample clustering optimization")
        self._cluster_samples()
        
        # Step 2.5: Iterative optimization of sample-node and expert-node assignment
        logger.info("Step 2.5: Start iterative optimization of assignment")
        self._sample_reassignment()
            
        # Step 3: Generate optimization results
        logger.info("Step 3: Generate optimization results")
        optimization_result = self._generate_results()

        total_time = time.time() - start_time
        logger.info(f"EP optimization process execution completed, total time: {total_time:.2f} seconds")
            
        if optimization_result:
            
            # Step 4: Save results
            logger.info("Step 4: Save optimization results")
            self._save_results(optimization_result)
            
            # Step 5: Generate validation report
            logger.info("Step 5: Generate validation report")
            self._generate_validation_report(optimization_result)
            
            
            return optimization_result
            
    
    def _analyze_samples(self):
        """Execute sample analysis"""
        # Load sample data
        logger.info("Loading sample data...")
        self.samples = self.predispatch_reader.load_samples()
        
        if not self.samples:
            raise ValueError("No sample data loaded")
        
        logger.info(f"Successfully loaded {len(self.samples)} samples")
        
        # Analyze sample expert preferences
        logger.info("Analyzing sample expert preferences...")
        self.analysis_results = self.sample_analyzer.analyze_samples(self.samples)
        
        # Export analysis report
        if self.config.export_analysis_report:
            analysis_report_path = Path(self.data_paths.output_dir)/ 'config_cluster' / f"{self.config.output_prefix}_sample_analysis_report.json"
            self.sample_analyzer.export_analysis_report(self.samples, str(analysis_report_path))
        
            logger.info("Sample analysis completed")
    
    def _cluster_samples(self):
        """Execute sample clustering"""
        if self.samples is None or self.analysis_results is None:
            raise ValueError("Sample data or analysis results not initialized")
        
        # Get PCA vectors and inverse transform
        pca_vectors = self.analysis_results['pca_vectors']
        if self.config.pca_dimensions is None:
            # For avoiding import error later
            pca_inverse_transform = lambda x: x
            self.sample_analyzer.pca.inverse_transform = pca_inverse_transform
        else:
            pca_inverse_transform = self.sample_analyzer.pca.inverse_transform

        # Get anomaly mask
        anomaly_mask = self.analysis_results['anomaly_mask']
        
        # Execute clustering
        logger.info("Executing sample clustering...")
        self.clusters = self.cluster_optimizer.cluster_samples(self.samples, pca_vectors, anomaly_mask, pca_inverse_transform)
        
        # Export clustering report
        clustering_report_path = Path(self.data_paths.output_dir) / 'config_cluster' / f"{self.config.output_prefix}_clustering_report.json"
        if self.config.export_analysis_report:
            self.cluster_optimizer.export_clustering_report(str(clustering_report_path))
        
        logger.info("Sample clustering completed")
    
    def _sample_reassignment(self):
        """
        optimize sample-to-node assignments
        
        """
        if self.clusters is None:
            raise ValueError("Clusters not initialized")
        
        # Extract all samples from clusters
        all_samples = []
        for cluster in self.clusters:
            all_samples.extend(cluster.samples)
        
        # Calculate topk from first sample
        if not all_samples or not all_samples[0].dispatch_ids:
            logger.warning("No samples or dispatch_ids found, skipping iterative optimization")
            return
        
        topk = len(all_samples[0].dispatch_ids) // (all_samples[0].token_count - 1)
        
        # Initialize expert placement from current clusters
        experts_by_node = {}
        for cluster in self.clusters:
            if cluster.node_id is not None:
                experts_by_node[cluster.node_id] = list(cluster.target_experts)
            
        logger.debug("Optimizing sample-to-node assignment...")
        sample_by_node = self._optimize_sample_assignment(all_samples, experts_by_node, topk)

        logger.info("Updating clusters with optimized assignment...")
        self._update_clusters_from_assignment(all_samples, sample_by_node, experts_by_node)
            
        logger.info("Iterative optimization completed")
    
    def _optimize_sample_assignment(
        self, 
        samples: List[Sample], 
        experts_by_node: Dict[int, Set[int]],
        topk: int
    ) -> Dict[int, int]:
        """
        Optimize sample-to-node assignment given fixed expert placement
        
        For each sample, assign it to the node that minimizes cross-node communication
        
        Args:
            samples: List of all samples
            experts_by_node: Current expert placement {node_id: list of expert_ids}
            topk: Number of experts per token
            
        Returns:
            sample_to_node: Mapping {sample_id: node_id}
        """
        samples_by_node = {node_id: [] for node_id in range(self.config.num_nodes)}
        
        for sample in samples:
            best_node = 0
            max_comm_cost = -float('inf')
            
            # Extract unique experts needed by this sample
            token_dispatch = np.array(sample.dispatch_ids).reshape(-1, topk)
            
            # Try each node and compute communication cost
            for node_id, node_experts in experts_by_node.items():
                # Communication cost = number of tokens needing communication
                # We use deepep pattern communication so the communication volume
                # compute is different to a2a
                if self.config.deepep_mode:
                    node_comm = np.any(np.isin(token_dispatch, node_experts), axis=1).sum()
                elif self.config.a2a_mode:
                    node_comm = np.isin(token_dispatch, node_experts).sum()
                else:
                    raise RuntimeError("Only support a2a and deepep mode.")
                # Select the most communication-hard node as the best node 
                if node_comm > max_comm_cost:
                    max_comm_cost = node_comm
                    best_node = node_id
            
            samples_by_node[best_node].append(sample)
        
        return samples_by_node
    
    def _optimize_expert_placement(
        self,
        samples_by_node: Dict[int, int],
        experts_by_node: Dict[int, Set[int]],
        topk: int
    ) -> Dict[int, Set[int]]:
        """
        Optimize expert-to-node placement given fixed sample assignment
        
        For each node, assign the most frequently used experts by its samples
        Ensure each expert is assigned to exactly one node
        
        Args:
            samples_by_node: Current sample assignment {node_id: list of sample_id}
            experts_by_node: Current expert placement {node_id: list of expert_ids}
            topk: Number of experts per token
            
        Returns:
            experts_by_node: Mapping {node_id: set of expert_ids}
        """
        # Count expert usage frequency per node
        cluster_comm_stats = self._compute_cross_node_communication_stats()['cluster_cross_node_communication']
        cluster_comm_ratio = {cluster_info['cluster_id']: cluster_info['cross_node_comm_ratio'] 
                            for cluster_info in cluster_comm_stats}
        heavy_load_cluster_id = max(cluster_comm_ratio.items(), key=lambda x: x[1])[0]
        cluster_samples = samples_by_node[heavy_load_cluster_id]
        cluster_sample_dispatch_id = [np.array(sample.dispatch_ids).reshape(-1, topk) for sample in cluster_samples]
        cluster_sample_dispatch_id = np.vstack(cluster_sample_dispatch_id)
        burden_expert_count = np.zeros(self.config.num_experts)
        for nid in experts_by_node:
            if nid == heavy_load_cluster_id:
                continue
            node_experts = experts_by_node[nid]
            burden_expert_token = cluster_sample_dispatch_id[np.isin(cluster_sample_dispatch_id, 
                                                               node_experts).sum(axis=-1) == 1]
            local_burden_count = np.bincount(burden_expert_token.reshape(-1), minlength=self.config.num_experts)
            burden_expert_count[node_experts] = local_burden_count[node_experts]
        
        most_burden_expert_id = np.argmax(burden_expert_count)
        for nid in experts_by_node:
            if most_burden_expert_id in experts_by_node[nid]:
                most_burden_node_id = nid
                break

        burden_cluster_center_vector = self.clusters[most_burden_node_id].center_vector
        exchange_experts_id = np.argmax(burden_cluster_center_vector[experts_by_node[heavy_load_cluster_id]])
        exchange_experts = experts_by_node[heavy_load_cluster_id][exchange_experts_id]
        experts_by_node[heavy_load_cluster_id].remove(exchange_experts)
        experts_by_node[heavy_load_cluster_id].append(most_burden_expert_id)
        experts_by_node[most_burden_node_id].remove(most_burden_expert_id)
        experts_by_node[most_burden_node_id].append(exchange_experts)
        
        #for eid in experts_by_node[heavy_load_cluster_id]:
            #target_experts = experts_by_node[heavy_load_cluster_id].remove(eid).append(most_burden_expert_id)
            #target_save_comm = np.any(np.isin(meaningful_tokens, target_experts), axis=1)
            #relax_expert_count[eid] = target_save_comm
        
        #expert_frequency_by_node = {
            #node_id: np.zeros(self.config.num_experts, dtype=np.int64)
            #for node_id in range(self.config.num_nodes)
        #}
        
        #for sample in samples:
            #node_id = sample_to_node.get(sample.sample_id)
            #node_experts = experts_by_node[node_id]
            #sample_dispatch_ids = np.array(sample.dispatch_ids).reshape(-1, topk)
            ## Select tokens only choose one local expert
            #solo_expert_token = sample_dispatch_ids[np.isin(sample_dispatch_ids, node_experts).sum(axis=-1)>=1]
            #expert_distributed = np.bincount(solo_expert_token.flatten(), minlength=self.config.num_experts)
            
            ## Count expert occurrences in this sample
            #expert_frequency_by_node[node_id] += expert_distributed
        
        ## Greedy assignment: iteratively assign experts to nodes
        #experts_by_node = {node_id: list() for node_id in range(self.config.num_nodes)}
        #assigned_experts = set()
        
        ## Create priority list: (frequency, node_id, expert_id)
        #priorities = []
        #for node_id in range(self.config.num_nodes):
            #for expert_id in range(self.config.num_experts):
                #freq = expert_frequency_by_node[node_id][expert_id]
                #priorities.append((freq, node_id, expert_id))
        
        ## Sort by frequency (descending)
        #priorities.sort(reverse=True, key=lambda x: x[0])
        
        ## Assign experts greedily
        #for freq, node_id, expert_id in priorities:
            ## Check if expert already assigned
            #if expert_id in assigned_experts:
                #continue
            
            ## Check if node already has enough experts
            #if len(experts_by_node[node_id]) >= experts_per_node:
                #continue
            
            ## Assign expert to node
            #experts_by_node[node_id].append(expert_id)
            #assigned_experts.add(expert_id)
            
            ## Check if all experts assigned
            #if len(assigned_experts) == self.config.num_experts:
                #break
        
        ## Verify all experts are assigned
        #if len(assigned_experts) < self.config.num_experts:
            #logger.warning(
                #f"Only {len(assigned_experts)}/{self.config.num_experts} experts assigned. "
                #"Assigning remaining experts to nodes with fewer experts."
            #)
            
            ## Assign remaining experts to balance load
            #unassigned_experts = set(range(self.config.num_experts)) - assigned_experts
            #for expert_id in unassigned_experts:
                ## Find node with fewest experts
                #min_node = min(experts_by_node.keys(), key=lambda n: len(experts_by_node[n]))
                #experts_by_node[min_node].add(expert_id)
        
        return experts_by_node
    
    def _update_clusters_from_assignment(
        self,
        samples: List[Sample],
        samples_by_node: Dict[int, int],
        experts_by_node: Dict[int, Set[int]]
    ):
        """
        Update cluster structure based on optimized assignment
        
        Args:
            samples: List of all samples
            samples_by_node: Optimized sample assignment {node_id: list of sample_id}
            experts_by_node: Optimized expert placement {node_id: list of expert_ids}
        """
        # Update clusters
        new_clusters = []
        for node_id in range(self.config.num_nodes):
            node_samples = samples_by_node[node_id]
            node_experts = experts_by_node[node_id]
            
            if not node_samples:
                logger.warning(f"Node {node_id} has no samples after optimization")
                continue
            
            # Calculate cluster center
            vectors = [s.expert_preference_vector for s in node_samples]
            center_vector = np.mean(vectors, axis=0)
            
            # Create new cluster
            cluster = Cluster(
                cluster_id=node_id,
                samples=node_samples,
                center_vector=center_vector,
                target_experts=node_experts,
                node_id=node_id
            )
            new_clusters.append(cluster)
        
        # Replace old clusters
        self.clusters = new_clusters
        logger.info(f"Updated {len(self.clusters)} clusters from optimized assignment")
    
    def _generate_results(self) -> OptimizationResult:
        """Generate optimization results"""
        if self.clusters is None:
            raise ValueError("Clustering results not initialized")
        
        # Create node data
        node_data_files = {}
        node_dispatch_files = {}
        node_scores_files = {}
        
        for cluster in self.clusters:
            if cluster.node_id is None:
                continue
            
            # Create node data object
            node_data = NodeData(
                node_id=cluster.node_id,
                samples=cluster.samples,
                experts=cluster.target_experts
            )
            
            # Save node data
            if not self.config.not_export_data:
                data_file, dispatch_file, scores_file = self.result_writer.save_node_data(
                    node_data, cluster.node_id, self.config.vocab_size
                )
            
                node_data_files[cluster.node_id] = data_file
                node_dispatch_files[cluster.node_id] = dispatch_file
                node_scores_files[cluster.node_id] = scores_file
            else:
                node_data_files[cluster.node_id] = None
                node_dispatch_files[cluster.node_id] = None
                node_scores_files[cluster.node_id] = None
        
        if self.config.export_analysis_report:
            # Create expert allocation list
            expert_assignments = []
            for cluster in self.clusters:
                if cluster.node_id is None:
                    continue
            
                # Allocate GPU for each expert
                for i, expert_id in enumerate(cluster.target_experts):
                    gpu_id = i % 8  # Simple round-robin allocation
                    expert_assignments.append({
                        'expert_id': expert_id,
                        'node_id': cluster.node_id,
                        'gpu_id': gpu_id,
                        'cluster_id': cluster.cluster_id
                    })
        
            # Save expert placement configuration
            expert_placement_path = self.result_writer.save_expert_placement(expert_assignments)
        
            # Save cluster information
            cluster_info_path = self.result_writer.save_cluster_info(
                self.clusters, self.config.to_dict(), self.sample_analyzer.pca.inverse_transform
            )
        
            # Calculate optimization statistics
            optimization_stats = self._compute_optimization_stats()
        
            # Save statistics
            stats_path = self.result_writer.save_optimization_stats(optimization_stats)
        
            # Create optimization result
            optimization_result = OptimizationResult(
                node_data_files=node_data_files,
                node_dispatch_files=node_dispatch_files,
                node_scores_files=node_scores_files,
                expert_placement_config=expert_placement_path,
                cluster_info=cluster_info_path,
                optimization_stats=optimization_stats,
                validation_report=""  # Fill later
            )
        
            return optimization_result
        return None

    def plan_micro_batch(self,
                         node_id_to_samples: Dict[int, List[Sample]],
                         batch_id: int = 0) -> Dict[str, any]:
        """
        Execute per-micro-batch assignment planning for one global micro-batch step.
        
        This method pools samples from each node, executes two-stage optimization:
        1. Node assignment (minimize cross-node communication)
        2. Expert-sample coordination (minimize intra-node traffic through coordinated placement)
        
        Args:
            node_id_to_samples: Mapping node_id -> ordered sample list for that node
            batch_id: Identifier of the micro batch
            
        Returns:
            Dictionary containing assignment results and plan file path
            
        Raises:
            ValueError: If clusters not initialized
        """
        if self.clusters is None:
            raise ValueError("Clusters not initialized; cannot plan micro batch")

        # Prepare experts per node from clustering output
        experts_by_node = {cluster.node_id: set(cluster.target_experts) 
                          for cluster in self.clusters if cluster.node_id is not None}

        # Initialize batch-time optimizers if not yet initialized
        if self.batch_node_assigner is None:
            self.batch_node_assigner = BatchNodeAssigner(
                num_nodes=self.config.num_nodes,
                experts_by_node=experts_by_node,
                topk=self.config.topk,
            )
        if self.expert_sample_coordinator is None:
            self.expert_sample_coordinator = ExpertSampleCoordinator(
                topk=self.config.topk, 
                num_experts=self.config.num_experts,
                num_nodes=self.config.num_nodes,
                micro_batch_size=self.config.training_config.micro_batch_size,
                experts_by_node = experts_by_node,
                config=self.config
            )
            # Initialize IndexedDataset builders for all GPUs
            self.expert_sample_coordinator.initialize_builders(self.data_paths.output_dir)

        # Pool samples for this global micro step
        micro_batch_size = self.config.training_config.micro_batch_size
        samples_per_node_per_batch = micro_batch_size * 8
        
        pooled_samples: List[Sample] = []
        next_indices: Dict[int, int] = {}
        
        for node_id, samples in node_id_to_samples.items():
            samples_to_take = min(len(samples), samples_per_node_per_batch)
            pooled_samples.extend(samples[:samples_to_take])
            next_indices[node_id] = samples_to_take

        # Execute two-stage optimization pipeline
        logger.debug(f"Executing assignment planning for batch {batch_id}")
        
        # Stage 1: Assign pooled samples to nodes (minimize cross-node communication)
        node_assignments = self.batch_node_assigner.assign(pooled_samples, micro_batch_size)
        logger.debug(f"Node assignments completed for batch {batch_id}")

        # Stage 2: Expert-sample coordination optimization (minimize intra-node traffic)
        gpu_assignments, expert_placements = self.expert_sample_coordinator.coordinate(
            node_assignments
        )
        logger.debug(f"Expert-sample coordination completed for batch {batch_id}")

        # Save GPU-level binary data for Megatron training
        gpu_data_files = self.expert_sample_coordinator.save_gpu_data(
            gpu_assignments, expert_placements, batch_id
        )
        logger.debug(f"GPU binary data added to builders for batch {batch_id}")

        if self.config.export_analysis_report:
            # Serialize plan to disk - use per-GPU JSON files instead of per-batch
            plan_dir = Path(self.data_paths.output_dir) / 'gpu_plans'
            plan_dir.mkdir(parents=True, exist_ok=True)
        
            # Save per-GPU batch information
            for (node_id, gpu_id), samples in gpu_assignments.items():
                gpu_plan_filename = f"{self.config.output_prefix}_node_{node_id}_gpu_{gpu_id}.json"
                gpu_plan_path = plan_dir / gpu_plan_filename
            
                # Load existing plan if it exists
                existing_plan = {}
                if gpu_plan_path.exists():
                    try:
                        with open(gpu_plan_path, 'r') as f:
                            existing_plan = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse existing plan file: {gpu_plan_path}")
                        existing_plan = {}
            
                # Initialize plan structure if not exists
                if 'gpu_info' not in existing_plan:
                    existing_plan['gpu_info'] = {
                        'node_id': node_id,
                        'gpu_id': gpu_id,
                        'gpu_data_files': gpu_data_files.get((node_id, gpu_id), {})
                    }
            
                if 'batches' not in existing_plan:
                    existing_plan['batches'] = {}
            
                # Add batch information
                batch_key = f"batch_{batch_id:06d}"
                existing_plan['batches'][batch_key] = {
                    'batch_id': int(batch_id),
                    'sample_ids': [(int(sample.cluster_id), int(sample.sample_id)) for sample in samples],
                    'expert_ids': expert_placements.get((node_id, gpu_id), [])
                }
            
                # Write updated plan
                with open(gpu_plan_path, 'w') as f:
                    json.dump(existing_plan, f, indent=2)
        
            logger.debug(f"GPU assignment plans updated for batch {batch_id}")
    
    def _save_results(self, optimization_result: OptimizationResult):
        """Save optimization results"""
        logger.info("Saving optimization results...")
        
        # Results already saved in _generate_results, here mainly for logging
        logger.info(f"Node data files: {len(optimization_result.node_data_files)}")
        logger.info(f"Node dispatch files: {len(optimization_result.node_dispatch_files)}")
        logger.info(f"Node scores files: {len([f for f in optimization_result.node_scores_files.values() if f is not None])}")
        logger.info(f"Expert placement configuration: {optimization_result.expert_placement_config}")
        logger.info(f"Cluster information: {optimization_result.cluster_info}")
        
        logger.info("Optimization results saving completed")
    
    def _generate_validation_report(self, optimization_result: OptimizationResult):
        """Generate validation report"""
        logger.info("Generating validation report...")
        
        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("EP Optimization Result Validation Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Basic information
        report_lines.append("1. Basic Information")
        report_lines.append(f"   Node count: {self.config.num_nodes}")
        report_lines.append(f"   Total experts: {self.config.num_experts}")
        report_lines.append(f"   Experts per GPU: {self.config.experts_per_gpu}")
        report_lines.append("")
        
        # Sample statistics
        if self.samples:
            report_lines.append("2. Sample Statistics")
            report_lines.append(f"   Total samples: {len(self.samples)}")
            report_lines.append(f"   Average tokens: {np.mean([s.token_count for s in self.samples]):.2f}")
            report_lines.append("")
        
        # Clustering statistics
        if self.clusters:
            report_lines.append("3. Clustering Statistics")
            for cluster in self.clusters:
                report_lines.append(f"   Cluster {cluster.cluster_id} (Node {cluster.node_id}):")
                report_lines.append(f"     Sample count: {cluster.sample_count}")
                report_lines.append(f"     Target experts: {sorted(cluster.target_experts)}")
                report_lines.append("")
        
        # Optimization statistics
        if optimization_result.optimization_stats:
            report_lines.append("4. Optimization Statistics")
            for key, value in optimization_result.optimization_stats.items():
                report_lines.append(f"   {key}: {value}")
            report_lines.append("")
        
        # File output
        report_lines.append("5. Output Files")
        report_lines.append(f"   Node data directory: {self.data_paths.node_data_dir}")
        report_lines.append(f"   Node dispatch directory: {self.data_paths.node_dispatch_dir}")
        report_lines.append(f"   Expert placement configuration: {optimization_result.expert_placement_config}")
        report_lines.append(f"   Cluster information: {optimization_result.cluster_info}")
        report_lines.append("")
        
        report_lines.append("=" * 50)
        report_lines.append("Validation Report Generation Completed")
        report_lines.append("=" * 50)
        
        # Save validation report
        validation_report = "\n".join(report_lines)
        validation_report_path = self.result_writer.save_validation_report(validation_report)
        
        # Update optimization result
        optimization_result.validation_report = validation_report_path
        
        logger.info("Validation report generation completed")

    def _compute_optimization_stats(self) -> Dict[str, float]:
        """Calculate optimization statistics"""
        if not self.clusters:
            return {}
        
        # Calculate various statistics
        total_samples = sum(c.sample_count for c in self.clusters)
        cluster_sizes = [c.sample_count for c in self.clusters]
        
        # Calculate load balancing metrics
        if cluster_sizes:
            load_balance_score = 1.0 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
        else:
            load_balance_score = 0.0
        
        # Calculate expert allocation metrics
        all_experts = set()
        for cluster in self.clusters:
            all_experts.update(cluster.target_experts)
        
        expert_coverage = len(all_experts) / self.config.num_experts
        
        # Calculate sample distribution metrics
        if self.samples:
            avg_tokens = np.mean([s.token_count for s in self.samples])
            token_std = np.std([s.token_count for s in self.samples])
            token_variation = token_std / avg_tokens if avg_tokens > 0 else 0.0
        else:
            avg_tokens = 0.0
            token_variation = 0.0
        
        # Calculate cross-node communication metrics
        cross_node_comm_stats = self._compute_cross_node_communication_stats()
        
        stats = {
            'total_samples': total_samples,
            'load_balance_score': load_balance_score,
            'expert_coverage': expert_coverage,
            'average_tokens_per_sample': float(avg_tokens),
            'token_variation_coefficient': float(token_variation),
            'cluster_count': len(self.clusters),
            'samples_per_cluster_mean': np.mean(cluster_sizes) if cluster_sizes else 0.0,
            'samples_per_cluster_std': np.std(cluster_sizes) if cluster_sizes else 0.0,
            **cross_node_comm_stats
        }
        
        return stats
    
    def _compute_cross_node_communication_stats(self) -> Dict[str, float]:
        """Calculate cross-node communication statistics for optimized clusters"""
        if not self.clusters or not self.samples:
            return {}
        
        # Calculate communication for optimized clusters
        optimized_cross_node_comm = 0
        optimized_a2a_cross_node_comm = 0
        max_cluster_cross_node_comm_ratio = 0
        max_cluster_a2a_cross_node_comm_ratio = 0
        total_tokens = 0

        # Calculate topk
        topk = len(self.clusters[0].samples[0].dispatch_ids) // (self.clusters[0].samples[0].token_count - 1)
        assert len(self.clusters[0].samples[0].dispatch_ids) % (self.clusters[0].samples[0].token_count - 1) == 0, "check to makse sure dispatch_ids is divisible of token_count"

        experts = [np.array(list(c.target_experts)) for c in self.clusters]
        optimized_cluster_comm = []
        
        for i, cluster in enumerate(self.clusters):
            non_local_experts = experts[:i] + experts[i+1:]
            cluster_comm = 0
            cluster_tokens = 0
            cluster_a2a_comm = 0
            
            for sample in cluster.samples:
                token_dispatch_ids = np.array(sample.dispatch_ids).reshape(-1, topk)
                for node_experts in non_local_experts:
                    inter_node_comm = np.any(np.isin(token_dispatch_ids, node_experts), axis=1).sum()
                    a2a_node_comm = np.isin(token_dispatch_ids, node_experts).sum()
                    cluster_comm += inter_node_comm
                    cluster_a2a_comm += a2a_node_comm
                    
                sample_tokens = len(token_dispatch_ids)
                cluster_tokens += sample_tokens
            optimized_cluster_comm.append({
                'cluster_id': i,
                'node_id': cluster.node_id,
                'cross_node_comm': int(cluster_comm),
                'a2a_cross_node_comm': int(cluster_a2a_comm),
                'totol_tokens': int(cluster_tokens),
                'cross_node_comm_ratio': float(cluster_comm / cluster_tokens),
                'a2a_cross_node_comm_ratio': float(cluster_a2a_comm / cluster_tokens),
            })
            optimized_cross_node_comm += cluster_comm
            optimized_a2a_cross_node_comm += cluster_a2a_comm
            total_tokens += cluster_tokens
            max_cluster_cross_node_comm_ratio = max(max_cluster_cross_node_comm_ratio, 
                                                    cluster_comm / cluster_tokens)
            max_cluster_a2a_cross_node_comm_ratio = max(max_cluster_a2a_cross_node_comm_ratio, 
                                                        cluster_a2a_comm / cluster_tokens)
            
        # Calculate communication ratios
        if total_tokens > 0:
            optimized_cross_node_ratio = optimized_cross_node_comm / total_tokens
            optimized_a2a_cross_node_ratio = optimized_a2a_cross_node_comm / total_tokens
        else:
            optimized_cross_node_ratio = 0.0

        logger.info(f"Average DeepEP communication ratio: {float(optimized_cross_node_ratio)}")
        logger.info(f"Average A2A Communication cost: {float(optimized_a2a_cross_node_ratio)}")
        logger.info(f"Max DeepEP communication cost: {float(max_cluster_cross_node_comm_ratio)}")
        logger.info(f"Max A2A communication cost: {float(max_cluster_a2a_cross_node_comm_ratio)}")
        
        return {
            'optimized_cross_node_communication': int(optimized_cross_node_comm),
            'optimized_cross_node_ratio': float(optimized_cross_node_ratio),
            'optimized_a2a_cross_node_ratio': float(optimized_a2a_cross_node_ratio),
            'max_cluster_cross_node_ratio': float(max_cluster_cross_node_comm_ratio),
            'max_a2a_cluster_cross_node_ratio': float(max_cluster_a2a_cross_node_comm_ratio),
            'total_tokens': int(total_tokens),
            'cluster_cross_node_communication': optimized_cluster_comm,
        }
    
    def get_optimization_summary(self) -> Dict[str, any]:
        """Get optimization summary"""
        summary = {
            'config': self.config.to_dict(),
            'samples_loaded': len(self.samples) if self.samples else 0,
            'clusters_created': len(self.clusters) if self.clusters else 0,
            'analysis_completed': self.analysis_results is not None,
            'clustering_completed': self.clusters is not None
        }
        
        if self.clusters:
            summary['cluster_info'] = [
                {
                    'cluster_id': c.cluster_id,
                    'node_id': c.node_id,
                    'sample_count': c.sample_count,
                    'target_experts_count': len(c.target_experts)
                }
                for c in self.clusters
            ]
        
        return summary
