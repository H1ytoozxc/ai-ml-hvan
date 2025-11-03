"""
FLOPs Counter
Improved FLOPs estimation for neural networks
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 32, 32)) -> float:
    """Estimate FLOPs for a model
    
    This is an improved estimation that handles common layer types.
    For more accurate counting, use libraries like fvcore or ptflops.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
    
    Returns:
        Estimated FLOPs in GFLOPs
    """
    
    flops = 0.0
    
    def hook_fn(module, input, output):
        nonlocal flops
        
        if isinstance(module, nn.Conv2d):
            # Conv2d FLOPs = output_size * kernel_ops
            # kernel_ops = kernel_h * kernel_w * in_channels * out_channels
            batch_size, in_channels, in_h, in_w = input[0].shape
            out_channels, _, kernel_h, kernel_w = module.weight.shape
            
            if isinstance(output, tuple):
                output = output[0]
            _, _, out_h, out_w = output.shape
            
            kernel_ops = kernel_h * kernel_w * in_channels * out_channels
            output_size = out_h * out_w
            flops_per_instance = kernel_ops * output_size
            
            # Add bias if present
            if module.bias is not None:
                flops_per_instance += out_channels * out_h * out_w
            
            flops += flops_per_instance * batch_size
        
        elif isinstance(module, nn.Linear):
            # Linear FLOPs = input_features * output_features
            batch_size = input[0].shape[0]
            in_features = module.in_features
            out_features = module.out_features
            
            flops_per_instance = in_features * out_features
            
            # Add bias if present
            if module.bias is not None:
                flops_per_instance += out_features
            
            flops += flops_per_instance * batch_size
        
        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm FLOPs = num_features * height * width * 2
            # (subtract mean + divide by std)
            batch_size = input[0].shape[0]
            num_features = module.num_features
            
            if isinstance(output, tuple):
                output = output[0]
            _, _, h, w = output.shape
            
            flops += batch_size * num_features * h * w * 2
        
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm FLOPs = normalized_shape * 2
            batch_size = input[0].shape[0]
            normalized_shape = module.normalized_shape
            
            if isinstance(normalized_shape, int):
                norm_size = normalized_shape
            else:
                norm_size = 1
                for dim in normalized_shape:
                    norm_size *= dim
            
            # Compute mean, variance, normalize
            flops += batch_size * norm_size * 4
        
        elif isinstance(module, nn.MultiheadAttention):
            # Attention FLOPs (simplified)
            # Q, K, V projections + attention computation
            batch_size = input[0].shape[0]
            seq_len = input[0].shape[1] if len(input[0].shape) > 2 else 1
            embed_dim = module.embed_dim
            num_heads = module.num_heads
            
            # QKV projections: 3 * (seq_len * embed_dim * embed_dim)
            qkv_flops = 3 * seq_len * embed_dim * embed_dim
            
            # Attention scores: (seq_len * seq_len * embed_dim)
            attn_flops = seq_len * seq_len * embed_dim
            
            # Output projection
            out_flops = seq_len * embed_dim * embed_dim
            
            flops += batch_size * (qkv_flops + attn_flops + out_flops)
        
        elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh)):
            # Activation functions: minimal FLOPs (count as 1 FLOP per element)
            if isinstance(input, tuple):
                input_tensor = input[0]
            else:
                input_tensor = input
            
            flops += input_tensor.numel()
        
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            # Adaptive pooling
            if isinstance(input, tuple):
                input_tensor = input[0]
            else:
                input_tensor = input
            
            flops += input_tensor.numel()
    
    # Register hooks
    hooks = []
    for module in model.modules():
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size).to(device)
        
        with torch.no_grad():
            model.eval()
            _ = model(dummy_input)
    except Exception as e:
        print(f"Warning: FLOPs estimation failed: {e}")
        # Fallback to parameter-based estimation
        flops = count_parameters(model) * 2.0 * 1e9
    
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    # Convert to GFLOPs
    gflops = flops / 1e9
    
    return gflops


def get_model_complexity(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 32, 32)) -> Dict[str, float]:
    """Get comprehensive model complexity metrics
    
    Returns:
        Dictionary with parameters, FLOPs, and other metrics
    """
    
    params = count_parameters(model)
    flops = estimate_flops(model, input_size)
    
    # Estimate memory usage (rough)
    param_memory_mb = params * 4 / (1024 ** 2)  # 4 bytes per float32
    
    return {
        "parameters": params,
        "parameters_M": params / 1e6,
        "flops": flops * 1e9,
        "flops_G": flops,
        "param_memory_mb": param_memory_mb
    }
