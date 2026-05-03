"""
Expert-sample coordination optimizer for intra-node placement.

This module implements the mathematical optimization problem to find optimal 
expert-sample mapping relationships through coordinated sample-GPU and expert-GPU 
placement within each node.

The optimization problem:
- Maximizes: ∑(A_ij * z_ij) where z_ij = x_ik * Y_j(k % 8)
- Subject to:
  - ∑k x_ik = 1, ∀i (each sample assigned to exactly one GPU)
  - ∑k y_jk = 1, ∀j (each expert assigned to exactly one GPU)
  - Y_is = y_is + y_i(s+1) + y_i(s+6) + y_i(s+24), s ∈ {0,1,...,7} (DeepEP constraints)
  - Node-specific expert constraints

This coordinates sample and expert placement to minimize intra-node communication using PuLP.
"""

from typing import Dict, List, Tuple, Set
import numpy as np
import logging
import torch

from ..utils.data_structures import Sample
from ..core.config import OptimizationConfig

from pathlib import Path
from megatron.core.datasets import indexed_dataset

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_IS_AVAILABLE = True
except ImportError:
    import pulp
    GUROBI_IS_AVAILABLE = False


logger = logging.getLogger(__name__)


class ExpertSampleCoordinator:
    """
    Expert-sample coordination optimizer for intra-node placement.
    
    This class implements the complete mathematical optimization problem from GrouterEPSave.md
    to find optimal expert-sample mapping relationships through coordinated placement.
    It solves both sample-GPU and expert-GPU assignments simultaneously to minimize 
    intra-node communication using 0-1 integer programming.
    """

    def __init__(self, topk: int, num_experts: int, num_nodes: int, 
                 micro_batch_size: int, experts_by_node: Dict, config: OptimizationConfig):
        """
        Initialize expert-sample coordination optimizer.
        
        Args:
            topk: Number of top experts per token
            num_experts: Number of experts
            num_nodes: Number of nodes
            micro_batch_size: Micro batch size
            experts_by_node: Dictionary contains map between experts and nodes
        """
        self.topk = topk
        self.num_experts = num_experts
        self.num_nodes = num_nodes
        self.experts_by_node = experts_by_node
        self.config = config

        # pre compute
        self.token_file_type = indexed_dataset.DType.optimal_dtype(self.config.vocab_size)
        self.nodes_by_experts = {expert: node for node, experts in experts_by_node.items() for expert in experts}
        self.num_gpus = self.num_nodes * 8
        self.num_samples = self.num_gpus * micro_batch_size

        assert self.num_experts % self.num_gpus == 0, "Number of experts must be divisible of number of gpus."
        assert self.num_samples % self.num_gpus == 0, "Number of gpus must be divisible of number of gpus.."

        self.expert_per_gpu = self.num_experts // self.num_gpus
        self.sample_per_gpu = self.num_samples // self.num_gpus

        # IndexedDataset builders for each GPU (persistent across batches)
        self.tokenized_builders: Dict[Tuple[int, int], any] = {}
        self.dispatch_builders: Dict[Tuple[int, int], any] = {}
        self.expert_builders: Dict[Tuple[int, int], any] = {}

        # Megatron processed data builders
        self.labels_builders: Dict[Tuple[int, int], any] = {}

        # Store file paths for each GPU
        self.gpu_file_paths: Dict[Tuple[int, int], Dict[str, str]] = {}

        # Identify optimizer 
        if GUROBI_IS_AVAILABLE:
            self._solve_coordination_optimization = self._solve_coordination_optimization_gurobi
        else:
            self._solve_coordination_optimization = self._solve_coordination_optimization_pulp

        logger.debug(f"ExpertSampleCoordinator initialized with topk={topk}, num_experts={num_experts}")

    def initialize_builders(self, output_dir: str):
        """
        Initialize IndexedDataset builders for all GPUs.
        
        Args:
            output_dir: Output directory for binary files
        """
        
        output_path = Path(output_dir) / 'gpu_data'
        output_path.mkdir(parents=True, exist_ok=True)
        
        for node_id in range(self.num_nodes):
            for gpu_id in range(8):
                gpu_key = (node_id, gpu_id + node_id * 8)
                
                # Create GPU-specific directory
                gpu_dir = output_path / f'node_{node_id}' / f'gpu_{gpu_id}'
                gpu_dir.mkdir(parents=True, exist_ok=True)
                
                # Initialize builders
                tokenized_prefix = str(gpu_dir / 'tokens')
                dispatch_prefix = str(gpu_dir / 'dispatch')
                expert_prefix = str(gpu_dir / 'experts')
                labels_prefix = str(gpu_dir / 'labels')
                
                self.tokenized_builders[gpu_key] = indexed_dataset.IndexedDatasetBuilder(
                    f"{tokenized_prefix}.bin", dtype=self.token_file_type
                )
                self.dispatch_builders[gpu_key] = indexed_dataset.IndexedDatasetBuilder(
                    f"{dispatch_prefix}.bin", dtype=np.uint8
                )
                self.expert_builders[gpu_key] = indexed_dataset.IndexedDatasetBuilder(
                    f"{expert_prefix}.bin", dtype=np.uint8
                )
                self.labels_builders[gpu_key] = indexed_dataset.IndexedDatasetBuilder(
                    f"{labels_prefix}.bin", dtype=self.token_file_type
                )
                
                # Store file paths
                self.gpu_file_paths[gpu_key] = {
                    'tokens': tokenized_prefix,
                    'dispatch': dispatch_prefix,
                    'experts': expert_prefix,
                    'labels': labels_prefix,
                }
                
        logger.info(f"Initialized IndexedDataset builders for {self.num_nodes} nodes × 8 GPUs")

    def finalize_builders(self):
        """
        Finalize all IndexedDataset builders and create index files.
        """
        from pathlib import Path
        
        for gpu_key in self.tokenized_builders:
            # Finalize tokenized data
            tokenized_prefix = self.gpu_file_paths[gpu_key]['tokens']
            self.tokenized_builders[gpu_key].finalize(f"{tokenized_prefix}.idx")
            
            # Finalize dispatch data
            dispatch_prefix = self.gpu_file_paths[gpu_key]['dispatch']
            self.dispatch_builders[gpu_key].finalize(f"{dispatch_prefix}.idx")
            
            # Finalize expert data
            expert_prefix = self.gpu_file_paths[gpu_key]['experts']
            self.expert_builders[gpu_key].finalize(f"{expert_prefix}.idx")
            
            # Finalize Megatron processed data
            labels_prefix = self.gpu_file_paths[gpu_key]['labels']
            self.labels_builders[gpu_key].finalize(f"{labels_prefix}.idx")
            
        logger.info("Finalized all IndexedDataset builders")

    def coordinate(self, node_assignments: Dict[int, List[Sample]]):
        """
        Coordinate expert-sample mapping through optimization for intra-node placement.
        
        This method implements the complete optimization problem from GrouterEPSave.md:
        - Maximizes ∑(A_ij * z_ij) where z_ij = x_ik * Y_j(k % 8)
        - Subject to all mathematical constraints
        - Returns coordinated sample-GPU and expert-GPU assignments
        
        Args:
            node_assignments: node_id -> samples assigned to that node
            
        Returns:
            Tuple of (gpu_assignments, expert_placements) where:
            - gpu_assignments: (node_id, gpu_id) -> assigned samples
            - expert_placements: (node_id, gpu_id) -> assigned experts
        """
        gpu_assignments: Dict[Tuple[int, int], List[Sample]] = {}
        
        logger.debug("Starting expert-sample coordination for all nodes")

        samples = []
        for node_sample in node_assignments.values():
            samples.extend(node_sample)

        # Build expert preference matrix A_ij
        A_matrix = self._build_expert_preference_matrix(samples)

        # Solve the expert-sample coordination optimization problem
        sample_gpu_vars, expert_gpu_vars = self._solve_coordination_optimization(
            A_matrix,
        )
        
        # Extract results
        gpu_assignments = self._extract_sample_assignments(
            sample_gpu_vars, samples 
        )
        expert_placements = self._extract_expert_assignments(
            expert_gpu_vars, 
        )
        
        logger.debug(f"Expert-sample coordination completed for {len(node_assignments)} nodes")
        return gpu_assignments, expert_placements

    def save_gpu_data(self, 
                      gpu_assignments: Dict[Tuple[int, int], List[Sample]],
                      expert_placements: Dict[Tuple[int, int], List[int]],
                      batch_id: int) -> Dict[str, str]:
        """
        Save GPU-level data using persistent IndexedDataset builders.
        Data is accumulated in memory and written to builders directly.
        
        Args:
            gpu_assignments: (node_id, gpu_id) -> assigned samples
            expert_placements: (node_id, gpu_id) -> assigned experts
            batch_id: Batch identifier
            
        Returns:
            Dictionary mapping (node_id, gpu_id) -> file paths
        """
        import numpy as np
        
        saved_files = {}
        
        for (node_id, gpu_id), samples in gpu_assignments.items():
            if not samples:
                continue
                
            gpu_key = (node_id, gpu_id)
            
            # Prepare data for IndexedDataset
            tokenized_sequences = []
            dispatch_sequences = []
            labels_sequences = []
            
            tokenized_lengths = []
            dispatch_lengths = []
            labels_lengths = []
            
            for sample in samples:
                # Tokenized data
                token_ids = np.array(sample.token_ids, dtype=self.token_file_type)
                tokenized_sequences.extend(token_ids)
                tokenized_lengths.append(len(token_ids))
                
                # Dispatch data
                dispatch_ids = np.array(sample.dispatch_ids, dtype=np.uint8)
                dispatch_sequences.extend(dispatch_ids)
                dispatch_lengths.append(len(dispatch_ids))
                
                # Megatron processed data
                labels = np.array(sample.labels, dtype=self.token_file_type)
                labels_sequences.extend(labels)
                labels_lengths.append(len(labels))
            
            # Add data to persistent builders
            self.tokenized_builders[gpu_key].add_document(
                torch.tensor(tokenized_sequences),
                tokenized_lengths
            )
            
            self.dispatch_builders[gpu_key].add_document(
                torch.tensor(dispatch_sequences, dtype=torch.uint8),
                dispatch_lengths
            )
            
            self.labels_builders[gpu_key].add_document(
                torch.tensor(labels_sequences),
                labels_lengths
            )
            
            # Expert placement
            experts = expert_placements.get(gpu_key, [])
            self.expert_builders[gpu_key].add_document(
                torch.tensor(experts, dtype=torch.uint8),
                [len(experts)]
            )
            
            # Get file paths for return
            saved_files[gpu_key] = self.gpu_file_paths[gpu_key]
            
            logger.debug(f"Added GPU data for node_{node_id}/gpu_{gpu_id}: {len(samples)} samples, {len(experts)} experts")
        
        logger.debug(f"Added GPU data for batch {batch_id} to persistent builders")
        return saved_files

    def _build_expert_preference_matrix(self, samples: List[Sample]) -> np.ndarray:
        """
        Build expert preference matrix A_ij.
        
        A_ij represents how much sample i prefers expert j.
        This is the coefficient matrix in the objective function.
        
        Args:
            Samples: List of samples
            
        Returns:
            Matrix A_ij of shape (num_samples, num_experts)
        """

        A_matrix = np.zeros((self.num_samples, self.num_experts), dtype=np.float32)
        for i in range(self.num_samples):
            A_matrix[i,:] = np.bincount(samples[i].dispatch_ids, minlength=self.num_experts) / len(samples[i].dispatch_ids)
                
        return A_matrix

    def _solve_coordination_optimization_pulp(self, A_matrix: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Solve the expert-sample coordination 0-1 integer programming problem using PuLP.
        
        This implements the complete mathematical formulation from GrouterEPSave.md:
        - Variables: x_ik (sample i to GPU k), y_jk (expert j to GPU k)
        - Objective: Max ∑(A_ij * z_ij) where z_ij = x_ik * Y_j(k % 8)
        - Constraints: assignment constraints and DeepEP routing constraints
        
        Args:
            A_matrix: Expert preference matrix A_ij
            
        Returns:
            Tuple of (sample_gpu_vars, expert_gpu_vars) dictionaries
        """
        # Create PuLP problem
        prob = pulp.LpProblem(f"ExpertSampleCoordination", pulp.LpMaximize)
        
        # Decision variables
        # x_ik: sample i assigned to GPU k v (0-1 variables)
        x_vars = {}
        for i in range(self.num_samples):
            node_id = i // 8
            for k in range(8 * node_id, 8 * (node_id +1)):
                x_vars[(i, k)] = pulp.LpVariable(f"x_{i}_{k}", cat='Binary')
        
        # y_jk: expert j assigned to GPU k (0-1 variables)
        y_vars = {}
        for node_id, experts in self.experts_by_node.items():
            for j in experts:
                for k in range(8*node_id, 8*(node_id+1)):
                    y_vars[(j, k)] = pulp.LpVariable(f"y_{j}_{k}", cat='Binary')

        # z_ij: interaction variables (x_ik * Y_j(k % 8))
        z_vars = {}
        for i in range(self.num_samples):
            for j in range(self.num_experts):
                for k in range(8):
                    z_vars[(i, j, k)] = pulp.LpVariable(f"z_{i}_{j}_{k}", cat='Binary')
        
        # Objective function: Max ∑(A_ij * z_ij)
        objective_terms = []
        for i in range(self.num_samples):
            node_id = i // 8
            for j in range(self.num_experts):
                for k in range(8):
                    objective_terms.append(A_matrix[i, j] * z_vars[(i, j, k)])
        
        prob += pulp.lpSum(objective_terms), "Total_Expert_Preference"
        
        # Constraints
        
        # 1. Sample assignment constraints: ∑k x_ik = 1, ∀i
        for i in range(self.num_samples):
            node_id = i // 8
            prob += pulp.lpSum([x_vars[(i, k)] for k in range(8*node_id, 8*(node_id+1))]) == 1, f"Sample_{i}_Assignment"
        
        # 2. Expert assignment constraints: ∑k y_jk = 1, ∀j
        for node_id, experts in self.experts_by_node.items():
            for j in experts:
                prob += pulp.lpSum([y_vars[(j, k)] for k in range(8*node_id, 8*(node_id+1))]) == 1, f"Expert_{j}_Assignment"


        # 3. Experts and sample per GPU constraints: ∑j y_jk = 1, ∀k
        for n in range(self.num_nodes):
            for k in range(8*n, 8*(n+1)):
                sample_range = range(8 * self.sample_per_gpu * n, 8 * self.sample_per_gpu * (n + 1))
                expert_range = self.experts_by_node[n]
                prob += pulp.lpSum([y_vars[(j, k)] for j in expert_range]) == self.expert_per_gpu, f"Expert_GPU_{k}_Assignment"
                prob += pulp.lpSum([x_vars[(i, k)] for i in sample_range]) == self.sample_per_gpu, f"Sample_GPU_{k}_Assignment"

        # 4. Interaction constraints: z_ij = x_ik * Y_j(k % 8)
        # This is linearized as: z_ij ≤ x_ik, z_ij ≤ Y_j(k % 8), z_ij ≥ x_ik + Y_j(k % 8) - 1
        for i in range(self.num_samples):
            for j in range(self.num_experts):
                node_sample = i // 8
                node_expert = self.nodes_by_experts[j]
                for k in range(8):
                    k_expert = node_expert * 8 + k
                    k_sample = node_sample * 8 + k
                    prob += z_vars[(i, j, k)] <= y_vars[(j, k_expert)], f"Interaction1_{i}_{j}_{k}"
                    prob += z_vars[(i, j, k)] <= x_vars[(i, k_sample)], f"Interaction3_{i}_{j}_{k}"
                    prob += z_vars[(i, j, k)] >= y_vars[(j, k_expert)] + x_vars[(i, k_sample)] - 1, f"Interaction3_{i}_{j}_{k}"
        
        # Solve the problem
        logger.debug(f"Solving expert-sample coordination with {self.num_samples} samples and {self.num_experts} experts")
        prob.solve(pulp.PULP_CBC_CMD(msg=0))  # Use CBC solver, suppress output
        
        # Check solution status
        if prob.status != pulp.LpStatusOptimal:
            logger.warning(f"Optimization did not find optimal solution. Status: {pulp.LpStatus[prob.status]}")
            # Fall back to simple assignment
            return self._fallback_assignment()
        
        logger.debug(f"Optimal solution found")
        
        # Extract variable values
        sample_gpu_vars = {}
        expert_gpu_vars = {}
        
        for i in range(self.num_samples):
            node_id == i // 8
            for k in range(node_id * 8, (node_id + 1) * 8):
                if x_vars[(i, k)].varValue >= 0.5: # in case of floating point precision
                    sample_gpu_vars[i] = k
                    break
        
        for j in range(self.num_experts):
            node_id = self.nodes_by_experts[j]
            for k in range():
                if y_vars[(j, k)].varValue >= 0.5: # in case of floating point precision
                    expert_gpu_vars[j] = k
                    break
        
        return sample_gpu_vars, expert_gpu_vars
    
    def _solve_coordination_optimization_gurobi(self, A_matrix: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Solve the expert-sample coordination 0-1 integer programming problem using Gurobi.
        
        This implements the complete mathematical formulation from GrouterEPSave.md:
        - Variables: x_ik (sample i to GPU k), y_jk (expert j to GPU k)
        - Objective: Max ∑(A_ij * z_ij) where z_ij = x_ik * Y_j(k % 8)
        - Constraints: assignment constraints and DeepEP routing constraints
        
        Args:
            A_matrix: Expert preference matrix A_ij
            
        Returns:
            Tuple of (sample_gpu_vars, expert_gpu_vars) dictionaries
        """
        # Create Gurobi optimization model
        model = gp.Model(f"ExpertSampleCoordination")
        model.setParam("OutputFlag", 0)  # Enable log output
        model.setParam("LogToConsole", 0)  # Output logs to console
        model.setParam("MIPGap", 0.03)    # Allow 3% gap for convergence
        model.setParam("Heuristics", 0.8) # High-intensity heuristics for initial solution
        model.setParam("Cuts", 2)         # Aggressive cut generation
        model.setParam("BranchDir", 1)    # Prefer branching x=1
        model.setParam("Threads", 64)

        # Decision variables
        # x_ik: sample i assigned to GPU k (binary variables)
        x_vars = {}
        for i in range(self.num_samples):
            node_id = i // 8
            for k in range(8 * node_id, 8 * (node_id + 1)):
                x_vars[(i, k)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")

        # y_jk: expert j assigned to GPU k (binary variables)
        y_vars = {}
        for node_id, experts in self.experts_by_node.items():
            for j in experts:
                for k in range(8 * node_id, 8 * (node_id + 1)):
                    y_vars[(j, k)] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{k}")

        # z_ij: interaction variables (x_ik * Y_j(k % 8))
        z_vars = {}
        for i in range(self.num_samples):
            for j in range(self.num_experts):
                for k in range(8):
                    z_vars[(i, j, k)] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")

        # Update model to include variables
        model.update()

        # Objective function: Max ∑(A_ij * z_ij)
        objective_terms = []
        for i in range(self.num_samples):
            for j in range(self.num_experts):
                for k in range(8):
                    objective_terms.append(A_matrix[i, j] * z_vars[(i, j, k)])

        model.setObjective(gp.quicksum(objective_terms), GRB.MAXIMIZE)

        # Constraints

        # 1. Sample assignment: ∑k x_ik = 1, ∀i
        for i in range(self.num_samples):
            node_id = i // 8
            model.addConstr(
                gp.quicksum([x_vars[(i, k)] for k in range(8 * node_id, 8 * (node_id + 1))]) == 1,
                name=f"Sample_{i}_Assignment"
            )

        # 2. Expert assignment: ∑k y_jk = 1, ∀j
        for node_id, experts in self.experts_by_node.items():
            for j in experts:
                model.addConstr(
                    gp.quicksum([y_vars[(j, k)] for k in range(8 * node_id, 8 * (node_id + 1))]) == 1,
                    name=f"Expert_{j}_Assignment"
                )

        # 3. GPU capacity constraints: fixed expert and sample counts per GPU
        for n in range(self.num_nodes):
            for k in range(8 * n, 8 * (n + 1)):
                sample_range = range(8 * self.sample_per_gpu * n, 8 * self.sample_per_gpu * (n + 1))
                expert_range = self.experts_by_node[n]
        
                # Expert count per GPU
                model.addConstr(
                    gp.quicksum([y_vars[(j, k)] for j in expert_range]) == self.expert_per_gpu,
                    name=f"Expert_GPU_{k}_Assignment"
                )
        
                # Sample count per GPU
                model.addConstr(
                    gp.quicksum([x_vars[(i, k)] for i in sample_range]) == self.sample_per_gpu,
                    name=f"Sample_GPU_{k}_Assignment"
                )

        # 4. Interaction constraints: z_ij = x_ik * Y_j(k % 8)
        # Linearized as: z_ij ≤ x_ik, z_ij ≤ Y_j(k % 8), z_ij ≥ x_ik + Y_j(k % 8) - 1
        for i in range(self.num_samples):
            for j in range(self.num_experts):
                node_sample = i // 8
                node_expert = self.nodes_by_experts[j]
                for k in range(8):
                    k_expert = node_expert * 8 + k
                    k_sample = node_sample * 8 + k

                    model.addConstr(
                        z_vars[(i, j, k)] <= x_vars[(i, k_sample)],
                        name=f"Interaction1_{i}_{j}_{k}"
                    )

                    model.addConstr(
                        z_vars[(i, j, k)] <= y_vars[(j, k_expert)],
                        name=f"Interaction2_{i}_{j}_{k}"
                    )

                    model.addConstr(
                        z_vars[(i, j, k)] >= y_vars[(j, k_expert)] + x_vars[(i, k_sample)] - 1,
                        name=f"Interaction3_{i}_{j}_{k}"
                    )

        # Solve the model
        logger.debug(f"Solving expert-sample coordination with {self.num_samples} samples and {self.num_experts} experts")
        model.optimize()
        
        # Check solution status
        if model.status != GRB.OPTIMAL:
            logger.warning(f"Optimization did not find optimal solution. Status: {model.status}")
            # Fall back to simple assignment
            return self._fallback_assignment()
        
        logger.debug(f"Optimal solution found")
        
        # Extract variable values
        sample_gpu_vars = {}
        expert_gpu_vars = {}
        
        for i in range(self.num_samples):
            node_id = i // 8
            for k in range(node_id * 8, (node_id + 1) * 8):
                if x_vars[(i, k)].x > 0.5:  # in case of floating point precision
                    sample_gpu_vars[i] = k
                    break
        
        for j in range(self.num_experts):
            node_id = self.nodes_by_experts[j]
            for k in range(node_id * 8, (node_id + 1) * 8):
                if y_vars[(j, k)].x > 0.5:  # in case of floating point precision
                    expert_gpu_vars[j] = k
                    break
        
        return sample_gpu_vars, expert_gpu_vars


    def _fallback_assignment(self) -> Tuple[Dict, Dict]:
        """
        Fallback assignment when optimization fails.
        
        Directly assert False
        """
        assert False

        

    def _extract_sample_assignments(self, sample_gpu_vars: Dict[int, int], samples: List[Sample]):
        """
        Extract sample assignments from optimization variables.
        
        Args:
            sample_gpu_vars: sample_index -> gpu_id mapping
            Samples: List of samples
            
        Returns:
            gpu_assignments -> (Node, GPU) -> sample
        """
        gpu_assignments: Dict[Tuple[int, int], List[Sample]] = {(node_id, gpu_id+node_id*8):[] for node_id in range(self.num_nodes)
                                                                 for gpu_id in range(8) }
        
        for sample_id, chosen_gpu in sample_gpu_vars.items():
            node_id = sample_id // 8
            gpu_assignments[(node_id, chosen_gpu)].append(samples[sample_id])
        
        return gpu_assignments

    def _extract_expert_assignments(self, expert_gpu_vars: Dict[int, int]) -> Dict[int, List[int]]:
        """
        Extract expert assignments from optimization variables.
        
        Args:
            expert_gpu_vars: expert_index -> gpu_id mapping
            
        Returns:
            expert_placements -> (Node, GPU) -> expert
        """
        expert_placements: Dict[Tuple[int, int], List[int]] = {(node_id, gpu_id+node_id*8):[] for node_id in range(self.num_nodes)
                                                                 for gpu_id in range(8)}
        
        for expert_id, chosen_gpu in expert_gpu_vars.items():
            node_id = self.nodes_by_experts[expert_id]
            expert_placements[(node_id, chosen_gpu)].append(expert_id)
        
        return expert_placements
