"""
Neural Architecture Building Blocks
Библиотека строительных блоков для генерации архитектур
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class Conv3x3Block(nn.Module):
    """Standard 3x3 convolution block"""
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, activation: str = "ReLU", 
                 normalization: str = "BatchNorm", dropout: float = 0.1):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                             stride=stride, padding=1, bias=False)
        
        # Normalization
        if normalization == "BatchNorm":
            self.norm = nn.BatchNorm2d(out_channels)
        elif normalization == "LayerNorm":
            self.norm = nn.GroupNorm(1, out_channels)
        elif normalization == "GroupNorm":
            self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        else:
            self.norm = nn.Identity()
        
        # Activation
        self.activation = self._get_activation(activation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def _get_activation(self, activation: str):
        activations = {
            "ReLU": nn.ReLU(inplace=True),
            "GELU": nn.GELU(),
            "Swish": nn.SiLU(),
            "Mish": nn.Mish(),
            "SiLU": nn.SiLU(),
            "Tanh": nn.Tanh()
        }
        return activations.get(activation, nn.ReLU(inplace=True))
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, activation: str = "ReLU",
                 normalization: str = "BatchNorm", dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = Conv3x3Block(in_channels, out_channels, stride, 
                                  activation, normalization, dropout)
        self.conv2 = Conv3x3Block(out_channels, out_channels, 1, 
                                  activation, normalization, 0)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if normalization == "BatchNorm" else nn.Identity()
            )
        else:
            self.shortcut = nn.Identity()
        
        self.activation = self.conv1._get_activation(activation)
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        out = self.activation(out)
        return out


class TransformerEncoderBlock(nn.Module):
    """Vision Transformer encoder block"""
    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 activation: str = "GELU"):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        
        act_layer = nn.GELU() if activation == "GELU" else nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x shape: (B, N, C) where N is sequence length
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MLPMixerBlock(nn.Module):
    """MLP-Mixer block"""
    def __init__(self, num_patches: int, embed_dim: int,
                 tokens_mlp_dim: int, channels_mlp_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.token_mixing = nn.Sequential(
            nn.Linear(num_patches, tokens_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tokens_mlp_dim, num_patches),
            nn.Dropout(dropout)
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.channel_mixing = nn.Sequential(
            nn.Linear(embed_dim, channels_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels_mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x shape: (B, N, C)
        x = x + self.token_mixing(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mixing(self.norm2(x))
        return x


class SpikingNeuronLayer(nn.Module):
    """Simplified Spiking Neural Network layer (LIF neuron)"""
    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 1.0, decay: float = 0.9):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.threshold = threshold
        self.decay = decay
        self.membrane_potential = None
    
    def forward(self, x):
        # Simplified SNN implementation
        batch_size = x.size(0)
        
        if self.membrane_potential is None or self.membrane_potential.size(0) != batch_size:
            self.membrane_potential = torch.zeros(batch_size, self.linear.out_features, 
                                                  device=x.device)
        
        current = self.linear(x)
        self.membrane_potential = self.decay * self.membrane_potential + current
        
        spikes = (self.membrane_potential >= self.threshold).float()
        self.membrane_potential = self.membrane_potential * (1 - spikes)
        
        return spikes
    
    def reset(self):
        self.membrane_potential = None


class GraphConvLayer(nn.Module):
    """Graph Convolution layer for processing structured data"""
    def __init__(self, in_features: int, out_features: int,
                 activation: str = "ReLU", dropout: float = 0.1):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU() if activation == "ReLU" else nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj_matrix=None):
        # x: node features (B, N, F)
        # adj_matrix: adjacency matrix (B, N, N) or None
        
        if adj_matrix is not None:
            # Graph convolution: A * X * W
            x = torch.bmm(adj_matrix, x)
        
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class HyperNetworkBlock(nn.Module):
    """HyperNetwork that generates weights for another network"""
    def __init__(self, hyper_input_dim: int, target_weight_shape: Tuple[int, ...],
                 hidden_dim: int = 128):
        super().__init__()
        
        self.target_weight_shape = target_weight_shape
        target_size = 1
        for dim in target_weight_shape:
            target_size *= dim
        
        self.hyper_net = nn.Sequential(
            nn.Linear(hyper_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_size)
        )
    
    def forward(self, hyper_input):
        # Generate weights based on hyper_input
        weights = self.hyper_net(hyper_input)
        weights = weights.view(self.target_weight_shape)
        return weights


class LearnedOptimizerBlock(nn.Module):
    """Meta-learned optimizer using LSTM"""
    def __init__(self, hidden_size: int = 20, num_layers: int = 2):
        super().__init__()
        
        # LSTM for learning optimization strategy
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, gradients, parameters):
        # Input: gradients and parameters
        # Output: update to apply
        
        # Combine gradient and parameter info
        inputs = torch.stack([gradients, parameters], dim=-1)
        
        lstm_out, _ = self.lstm(inputs.unsqueeze(0))
        update = self.output(lstm_out.squeeze(0))
        
        return update.squeeze(-1)


class DifferentiableAugmentation(nn.Module):
    """Learnable augmentation policy
    
    Simplified differentiable augmentation that learns to apply noise-based
    augmentations. For production use, consider:
    - AutoAugment/RandAugment integration
    - Differentiable image transformations
    - Policy gradient methods for discrete augmentations
    """
    def __init__(self, num_ops: int = 5, magnitude_range: Tuple[float, float] = (0.0, 1.0)):
        super().__init__()
        
        self.num_ops = num_ops
        # Learnable probabilities and magnitudes for augmentation operations
        self.probs = nn.Parameter(torch.ones(num_ops) * 0.5)
        self.magnitudes = nn.Parameter(torch.ones(num_ops) * 0.5)
        self.magnitude_range = magnitude_range
    
    def forward(self, x):
        """Apply learnable augmentations
        
        Current implementation uses Gaussian noise as a differentiable proxy.
        Can be extended with more sophisticated augmentation operations.
        """
        
        if not self.training:
            return x
        
        # Compute augmentation strengths
        probs = torch.sigmoid(self.probs)
        mags = torch.sigmoid(self.magnitudes) * \
               (self.magnitude_range[1] - self.magnitude_range[0]) + \
               self.magnitude_range[0]
        
        # Apply weighted noise augmentation
        for i in range(self.num_ops):
            # Use soft gating (differentiable)
            strength = probs[i] * mags[i]
            noise = torch.randn_like(x) * strength * 0.01
            x = x + noise
        
        return x


class GradientModificationBlock(nn.Module):
    """Block that modifies gradients during backpropagation"""
    def __init__(self, modification_type: str = "clip", threshold: float = 1.0):
        super().__init__()
        
        self.modification_type = modification_type
        self.threshold = threshold
    
    def forward(self, x):
        if self.modification_type == "clip":
            # Gradient clipping
            return GradientClip.apply(x, self.threshold)
        elif self.modification_type == "scale":
            # Gradient scaling
            return GradientScale.apply(x, self.threshold)
        else:
            return x


class GradientClip(torch.autograd.Function):
    """Custom autograd function for gradient clipping"""
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.threshold = threshold
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input = torch.clamp(grad_input, -ctx.threshold, ctx.threshold)
        return grad_input, None


class GradientScale(torch.autograd.Function):
    """Custom autograd function for gradient scaling"""
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def get_block_by_name(block_name: str, **kwargs):
    """Factory function to create blocks by name"""
    
    blocks = {
        "Conv3x3": Conv3x3Block,
        "ResidualBlock": ResidualBlock,
        "TransformerEncoderBlock": TransformerEncoderBlock,
        "MLP_Mixer_Block": MLPMixerBlock,
        "SpikingNeuronLayer": SpikingNeuronLayer,
        "GraphConv": GraphConvLayer,
        "HyperNetworkBlock": HyperNetworkBlock,
        "LearnedOptimizer": LearnedOptimizerBlock,
        "DifferentiableAugmentation": DifferentiableAugmentation,
        "GradientModificationBlock": GradientModificationBlock
    }
    
    if block_name not in blocks:
        raise ValueError(f"Unknown block type: {block_name}")
    
    return blocks[block_name](**kwargs)
