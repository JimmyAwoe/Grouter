import torch
import torch.nn as nn
import torch.nn.functional as F
from .general_structure import (
    GrouterEmbedding,
    GrouterScore,
    GrouterRoPEEmbedding
)

class GrouterRL(nn.Module):
    """The residual linear structure for grouter"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layer = config.num_layer
        self.hidden_size = config.hidden_size
        self.layers = []
        self.middle_size = config.middle_size
        self.linear_bias = config.linear_bias

        if config.norm_fun == 'layernorm':
            self.norm_fun = nn.LayerNorm
        else:
            assert False, "Has not implemented this type of normalization function"

        # Define the activation function
        if config.act_fun == 'relu':
            self.act_fun = F.relu
        elif config.act_fun == 'gelu':
            self.act_fun = F.gelu
        elif config.act_fun == 'silu':
            self.act_fun = F.silu
        else:
            assert False, "Has not implemented this type of activation function"
        
        self._create()
        
        
    def _create(self):
        if self.config.no_pos_info:
            self.embed_tokens = GrouterEmbedding(self.config)
        else:
            self.embed_tokens = GrouterRoPEEmbedding(self.config)
        
        for idx in range(self.num_layer):
            if idx == 0:
                # First layer
                layer = nn.Sequential(
                    nn.Linear(self.hidden_size, self.middle_size, bias=self.linear_bias),
                    self.norm_fun(self.middle_size)
                )
            elif idx == self.num_layer - 1:
                # Last layer
                layer = nn.Sequential(
                    nn.Linear(self.middle_size, self.hidden_size, bias=self.linear_bias),
                    self.norm_fun(self.hidden_size)
                )
            else:
                # Middle layer
                layer = nn.Sequential(
                    nn.Linear(self.middle_size, self.middle_size, bias=self.linear_bias),
                    self.norm_fun(self.middle_size)
                )
                
            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)
        self.score = GrouterScore(self.config)

    def forward(self, input_ids, attention_mask, position_ids): 
        # Add attention_mask for compatible with other structure
        if self.config.no_pos_info:
            x = self.embed_tokens(input_ids)
        else:
            x = self.embed_tokens(input_ids, position_ids)

        for idx, layer in enumerate(self.layers):
            residual = x
            x = layer(x)
            x = self.act_fun(x)
            if x.shape == residual.shape:
                x = x + residual

        logits = self.score(x)
        return logits
            

        