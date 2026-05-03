import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, Optional


def create_grouter_distillation_hook(logits_dict: Dict[str, torch.Tensor], 
                                   device: str):
    """
    Create hook function for Grouter distillation
    
    Args:
        logits_dict: Dictionary to store logits
        device: Device to use
    
    Returns:
        Hook function
    """
    def fetch_grouter_hook(name: str):
        """Fetch Grouter logits for distillation"""
        def hook(module, input, output):
            with torch.no_grad():
                # Get input hidden states
                hidden_state = input[0]

                # Compute gating logits
                logits = module.gating(hidden_state)
            
            logits_dict[name] = logits.to(device)
        return hook
    return fetch_grouter_hook


def register_grouter_hooks(model, logits_dict: Dict[str, torch.Tensor], 
                          device: str, moe_layer_start: int = 0, 
                          moe_layer_end: int = 1, transpose: bool = False):
    """
    Register hooks for MoE layer routers
    
    Args:
        model: GPT model
        logits_dict: Dictionary to store logits
        device: Device to use
        moe_layer_start: Start layer index for MoE layers
        moe_layer_end: End layer index for MoE layers
    """
    fetch_hook = create_grouter_distillation_hook(logits_dict, device)
    
    # Register hooks to MoE layer routers
    for layer_idx in range(moe_layer_start, moe_layer_end):
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
            # For core models
            if hasattr(model.decoder.layers[layer_idx], 'mlp'):
                if hasattr(model.decoder.layers[layer_idx].mlp, 'gate'):
                    model.decoder.layers[layer_idx].mlp.gate.register_forward_hook(
                        fetch_hook(f"gate_{layer_idx}")
                    )
                elif hasattr(model.decoder.layers[layer_idx].mlp, 'router'):
                    model.decoder.layers[layer_idx].mlp.router.register_forward_hook(
                        fetch_hook(f"gate_{layer_idx}")
                    )
        elif hasattr(model, 'layers'):
            # For legacy models
            if hasattr(model.layers[layer_idx], 'mlp'):
                if hasattr(model.layers[layer_idx].mlp, 'gate'):
                    model.layers[layer_idx].mlp.gate.register_forward_hook(
                        fetch_hook(f"gate_{layer_idx}")
                    )
                elif hasattr(model.layers[layer_idx].mlp, 'router'):
                    model.layers[layer_idx].mlp.router.register_forward_hook(
                        fetch_hook(f"gate_{layer_idx}")
                    )


def clear_grouter_hooks(model, moe_layer_start: int = 0, moe_layer_end: int = 1):
    """
    Clear Grouter hooks
    
    Args:
        model: GPT model
        moe_layer_start: Start layer index for MoE layers
        moe_layer_end: End layer index for MoE layers
    """
    for layer_idx in range(moe_layer_start, moe_layer_end):
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
            # For core models
            if hasattr(model.decoder.layers[layer_idx], 'mlp'):
                if hasattr(model.decoder.layers[layer_idx].mlp, 'gate'):
                    model.decoder.layers[layer_idx].mlp.gate._forward_hooks.clear()
                elif hasattr(model.decoder.layers[layer_idx].mlp, 'router'):
                    model.decoder.layers[layer_idx].mlp.router._forward_hooks.clear()
        elif hasattr(model, 'layers'):
            # For legacy models
            if hasattr(model.layers[layer_idx], 'mlp'):
                if hasattr(model.layers[layer_idx].mlp, 'gate'):
                    model.layers[layer_idx].mlp.gate._forward_hooks.clear()
                elif hasattr(model.layers[layer_idx].mlp, 'router'):
                    model.layers[layer_idx].mlp.router._forward_hooks.clear()
