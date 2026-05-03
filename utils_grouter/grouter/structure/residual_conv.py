import torch.nn as nn
import torch.nn.functional as F
import torch
from .general_structure import (
    GrouterEmbedding,
    GrouterRoPEEmbedding,
    GrouterScore
)

class GrouterRC(nn.Module):
    """The residual convolution structure for grouter"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.middle_size = config.middle_size
        self.num_layer = config.num_layer
        self.kernel_size = config.kernel_size

        self._create()
        
    
    def _create(self):
        if self.config.no_pos_info:
            self.embed_tokens = GrouterEmbedding(self.config)
        else:
            self.embed_tokens = GrouterRoPEEmbedding(self.config)
        
        # Create conv block
        self.layers = nn.ModuleList()
        in_channels = self.hidden_size
        for i in range(self.num_layer):
            if i == 0:
                out_channels = self.middle_size
            elif i == self.num_layer - 1:
                out_channels = self.hidden_size

            self.layers.append(ResidualConvBlock(
                in_channels, 
                out_channels,
                kernel_size=self.kernel_size
            ))
            in_channels = out_channels
        
        self.score = GrouterScore(self.config)
    
    def forward(self, input_ids, attention_mask=None, position_ids=None):
        """
        x: input, shape as (batch_size, sequence_length, hidden_size)
        attention_mask: 4d attention mask, shape as (batch_size, 1, seq_len, seq_len)
        """
        if self.config.no_pos_info:
            out = self.embed_tokens(input_ids)
        else:
            out = self.embed_tokens(input_ids, position_ids)

        for layer in self.layers:
            out = layer(out, attention_mask)
        
        logits = self.score(out)

        return logits 


class ResidualConvBlock(nn.Module):
    """The Convolution with residual connection """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2  # Keep sequence length
        
        self.conv1 = MaskedConv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = MaskedConv1d(
            out_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Tune the shape if in and out channel mismatch
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x, attention_mask=None):
        x = x.transpose(1, 2)  # (batch_size, hidden_size, sequence_length)
        out = self.conv1(x, attention_mask) 
        out = F.relu(self.bn1(out))
        
        out = self.conv2(out, attention_mask)
        out = self.bn2(out)
        # recover
        out = out.transpose(1, 2)
        
        shortcut = self.shortcut(x).transpose(1, 2)
        out += shortcut
        
        out = F.relu(out)
        return out


class MaskedConv1d(nn.Conv1d):
    def forward(self, x, attention_mask=None):
        batch_size, hidden_size, seq_len = x.shape
        
        
        # Use basic Conv
        out = super().forward(x)  # (batch_size, out_channels, sequence_length)
        
        # Use attention_mask to ensure not feel sentence not related
        if attention_mask is not None:
            kernel_size = self.kernel_size[0]
            padding = self.padding[0]
            
            mask = attention_mask.squeeze(1)  # (batch_size, seq_len, seq_len)
            
            valid_mask = torch.ones(batch_size, seq_len, device=x.device, dtype=x.dtype)
            
            for i in range(seq_len):
                start = max(0, i - padding)
                end = min(seq_len, i - padding + kernel_size)
                
                window_mask = mask[:, i, start:end].min(dim=1)[0]
                valid_mask[:, i] = window_mask
            
            out = out * valid_mask.unsqueeze(1)  # (batch_size, out_channels, seq_len)
        
        
        return out
