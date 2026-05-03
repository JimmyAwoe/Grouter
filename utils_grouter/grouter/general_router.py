import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional, Union
import json
from .structure import (
    GrouterMLAConfig, 
    GrouterMHAConfig,
    GrouterRLConfig,
    GrouterRCConfig,
    GrouterMHA,
    GrouterMLA,
    GrouterRL,
    GrouterRC
)

class Grouter(nn.Module):
    def __init__(self, 
                 topk, 
                 structure_type, 
                 model_config_path, 
                 scoring_fun="softmax",
                 scaling_factor=2.5, 
                 mapping=None, 
                 target_num_experts=None, 
                 dynamic_act_threshold=2.5, 
                 output_logits=False, 
                 predispatch_mode=False):
        super().__init__()

        with open(model_config_path, 'r') as f:
            config = json.load(f)

        self.model_config_path = model_config_path
        self.num_experts = config['num_labels']
        self.top_k = topk
        self.structure_type = structure_type
        self.score_function = scoring_fun
        self.scaling_factor = scaling_factor

        # dynamic activation
        self.dynamic_act_threshold = dynamic_act_threshold

        # for distillation
        self.output_logits = output_logits

        # for predispatch
        self.predispatch_mode = predispatch_mode

        # Set to False defaultly
        self.bias_routing = False

        # convert experts number
        self.reconstruct_map = mapping
        self.target_num_experts = target_num_experts if target_num_experts else self.num_experts
        if int(self.target_num_experts) != int(self.num_experts):
            self.reconstruction = True
        else:
            self.reconstruction = False

        # setting bias for bias routing
        self.bias = nn.Parameter(torch.zeros(self.num_experts))

        self._create(config)

        self.register_load_state_dict_post_hook(self._post_load_process)
    
    def _create(self, config):
        if self.structure_type == 'mla':
            self.config = GrouterMLAConfig(**config)
            self.grouter_structure = GrouterMLA(config=self.config)
        elif self.structure_type == 'mha':
            self.config = GrouterMHAConfig(**config)
            self.grouter_structure = GrouterMHA(config=self.config)
        elif self.structure_type == 'rl':
            self.config = GrouterRLConfig(**config)
            self.grouter_structure = GrouterRL(config=self.config)
        elif self.structure_type == 'rc':
            self.config = GrouterRCConfig(**config)
            self.grouter_structure = GrouterRC(config=self.config)
        
    def _post_load_process(self, module, incompatible_keys):
        if self.reconstruction:
            self._apply_reconstruction()
    
    @torch.no_grad
    def _apply_reconstruction(self):
        """Apply shape convert to score module"""
        assert self.target_num_experts in self.reconstruct_map, f"Has not implement convertion to {self.target_num_experts} "
        mapping = {int(k): v for k, v in self.reconstruct_map[self.target_num_experts].items()}
        target_num_experts = len(mapping)
        
        self.reconstruct_matrix = torch.zeros(
            target_num_experts, 
            self.num_experts,
            dtype=self.grouter_structure.score.score.weight.dtype,
            device=self.grouter_structure.score.score.weight.device
        )
        
        for target_idx, source_indices in mapping.items():
            self.reconstruct_matrix[target_idx, source_indices] = 1.0
        
        new_weight = torch.matmul(
            self.reconstruct_matrix.detach(),
            self.grouter_structure.score.score.weight
        )
        self.num_experts = new_weight.shape[0]
        self.grouter_structure.score.score.weight = torch.nn.Parameter(new_weight)

        self.bias = torch.nn.Parameter(torch.matmul(
            self.reconstruct_matrix.detach(),
            self.bias
        ))


    def _routing(
        self,
        logits: torch.Tensor,
        grouter_top_indices: Union[torch.Tensor, tuple] = None,
    ):
        """Apply the top-k selection collaborated with grouter."""
        assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."

        if isinstance(grouter_top_indices, tuple):
            balanced_top_indices, grouter_top_indices = grouter_top_indices
        else:
            balanced_top_indices = grouter_top_indices

        if self.score_function == "softmax":
            scores = torch.gather(logits, dim=1, index=grouter_top_indices)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
        elif self.score_function == "sigmoid":
            scores = torch.sigmoid(logits.float()).type_as(logits)
            scores = torch.gather(scores, dim=1, index=grouter_top_indices).type_as(logits)
            probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        else:
            raise ValueError(f"Invalid score_function: {self.score_function}")

        if self.scaling_factor:
            probs = probs * self.scaling_factor

        topk_masked_gates = torch.zeros_like(logits).scatter(1, balanced_top_indices, probs)
        topk_map = torch.zeros_like(logits).int().scatter(1, balanced_top_indices, 1).bool()
        tokens_per_expert = topk_map.sum(dim=0)

        return topk_masked_gates, topk_map, tokens_per_expert

    def _process_score_output(self, logits):

        logits = logits.permute(1, 0, 2).contiguous()
        logits = logits.view(-1, int(self.target_num_experts))

        if self.bias_routing:
            # bias only used for balanced routing
            _, balanced_topk_idx = torch.topk(logits + self.bias, k=self.top_k, dim=-1, sorted=False)
            _, topk_idx = torch.topk(logits, k=self.top_k, dim=-1, sorted=False)
            topk_idx = (balanced_topk_idx, topk_idx)
        else:
            # select top-k experts
            _, topk_idx = torch.topk(logits, k=self.top_k, dim=-1, sorted=False)
        
        if self.output_logits:
            return topk_idx, logits
        
        if isinstance(topk_idx, tuple):
            topk_idx = tuple(idx.detach().clone() for idx in topk_idx)
            scores, routing_map, _ = self._routing(logits.detach().clone(), topk_idx)
        else:
            scores, routing_map, _ = self._routing(logits.detach().clone(), topk_idx.detach().clone())
        
        if self.predispatch_mode:
            if self.bias_routing:
                return scores[routing_map], topk_idx[0]
            else:
                return scores[routing_map], topk_idx

        if self.dynamic_act_threshold < self.scaling_factor:

            sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
            cumsum_scores = torch.cumsum(sorted_scores, dim=-1)
        
            threshold_mask = cumsum_scores <= self.dynamic_act_threshold
            # Ensure at least activate one experts
            threshold_mask[:, 0] = True
            threshold_mask[:, self.top_k:] = False
        
            routing_map_flat = torch.zeros_like(routing_map, dtype=torch.bool)
        
            routing_map = routing_map_flat.scatter(-1, sorted_indices, threshold_mask)
            scores = scores * routing_map

            sparsity = routing_map.sum() / len(routing_map)
            return scores, routing_map, sparsity

        return scores, routing_map
    
    def load_bias(self, bias_path: str):
        """Load bias to start bias routing"""
        # Use fp32 to enhance bias ability
        bias = torch.load(bias_path)
        self.bias.data = bias.to(dtype=self.bias.dtype, device=self.bias.device)
        self.bias_routing = True
    
    def forward(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        if self.structure_type == 'mla' or self.structure_type == 'mha':
            logits = self.grouter_structure(input_ids=input_ids, 
                                            attention_mask=attention_mask, 
                                            position_ids=position_ids).logits
        elif self.structure_type == 'rl' or self.structure_type == 'rc':
            logits = self.grouter_structure(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids)
        else:
            assert "Have not implement this type of grouter."

        return self._process_score_output(logits)
