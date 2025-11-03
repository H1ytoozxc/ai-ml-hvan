"""
Architecture Generator
Генерация конкретных PyTorch моделей из геномов
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
import numpy as np

from .blocks import get_block_by_name
from .search_space import ArchitectureGenome


class DynamicArchitecture(nn.Module):
    """Dynamically generated neural architecture from genome"""
    
    def __init__(self, genome: ArchitectureGenome, num_classes: int = 100, 
                 input_channels: int = 3, input_size: int = 32):
        super().__init__()
        
        self.genome = genome
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_size = input_size
        
        # Build the network from genome
        self.network = self._build_network()
        
        # Build classifier head
        self.classifier = self._build_classifier()
        
        # Optional meta-learning components
        self.meta_components = self._build_meta_components()
    
    def _build_network(self) -> nn.ModuleDict:
        """Build network from genome blocks"""
        
        layers = nn.ModuleDict()
        self.layer_order = []  # Track order of layers
        current_channels = self.input_channels
        current_size = self.input_size
        is_sequence_model = False  # Track if we're using sequence-based blocks
        self.pos_embed = None  # Store positional embedding separately
        num_patches = 0  # Track number of patches for sequence models
        
        for i, block_config in enumerate(self.genome.blocks):
            block_type = block_config["type"]
            
            # Handle transition between different block types
            if block_type in ["TransformerEncoderBlock", "MLP_Mixer_Block"]:
                # Need to convert to sequence format if not already
                if not is_sequence_model:
                    # Skip if image is too small for patching
                    if current_size < 2:
                        print(f"Warning: Skipping {block_type} at layer {i} - image too small ({current_size}x{current_size})")
                        continue
                    
                    # Add patch embedding layer
                    patch_size = min(4 if current_size >= 16 else 2, current_size)
                    num_patches = (current_size // patch_size) ** 2
                    
                    # Skip if no patches
                    if num_patches == 0:
                        print(f"Warning: Skipping {block_type} at layer {i} - no patches")
                        continue
                    
                    embed_dim = block_config.get("embed_dim", 256)
                    
                    patch_embed = nn.Sequential(
                        nn.Conv2d(current_channels, embed_dim, 
                                 kernel_size=patch_size, stride=patch_size),
                        nn.Flatten(2),  # (B, C, H, W) -> (B, C, H*W)
                    )
                    layers["patch_embed"] = patch_embed
                    self.layer_order.append("patch_embed")
                    
                    # Add positional embedding
                    self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
                    
                    current_channels = embed_dim
                    current_size = num_patches  # Update size to num_patches
                    is_sequence_model = True
            
            elif block_type in ["Conv3x3", "ResidualBlock"]:
                # Need to convert from sequence to spatial if necessary
                if is_sequence_model:
                    # Add layer to reshape back to spatial
                    spatial_size = int(np.sqrt(current_size))
                    reshape_layer = ReshapeLayer((current_channels, spatial_size, spatial_size))
                    layer_name = f"reshape_to_spatial_{i}"
                    layers[layer_name] = reshape_layer
                    self.layer_order.append(layer_name)
                    current_size = spatial_size  # Update to spatial size
                    is_sequence_model = False
            
            # Create the actual block
            try:
                if block_type == "Conv3x3":
                    in_channels = current_channels
                    out_channels = block_config.get("out_channels", 128)
                    
                    block = get_block_by_name(
                        block_type,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=block_config.get("stride", 1),
                        activation=block_config.get("activation", "ReLU"),
                        normalization=block_config.get("normalization", "BatchNorm"),
                        dropout=block_config.get("dropout", 0.1)
                    )
                    current_channels = out_channels
                    if block_config.get("stride", 1) > 1:
                        current_size = current_size // block_config.get("stride", 1)
                
                elif block_type == "ResidualBlock":
                    in_channels = current_channels
                    out_channels = block_config.get("out_channels", current_channels)
                    
                    block = get_block_by_name(
                        block_type,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=block_config.get("stride", 1),
                        activation=block_config.get("activation", "ReLU"),
                        normalization=block_config.get("normalization", "BatchNorm"),
                        dropout=block_config.get("dropout", 0.1)
                    )
                    current_channels = out_channels
                    if block_config.get("stride", 1) > 1:
                        current_size = current_size // block_config.get("stride", 1)
                
                elif block_type == "TransformerEncoderBlock":
                    embed_dim = block_config.get("embed_dim", current_channels)
                    
                    # Add projection if dimensions don't match
                    if current_channels != embed_dim:
                        proj_layer = nn.Linear(current_channels, embed_dim)
                        proj_name = f"proj_to_transformer_{i}"
                        layers[proj_name] = proj_layer
                        self.layer_order.append(proj_name)
                        current_channels = embed_dim
                    
                    block = get_block_by_name(
                        block_type,
                        embed_dim=embed_dim,
                        num_heads=block_config.get("num_heads", 8),
                        mlp_ratio=block_config.get("mlp_ratio", 4.0),
                        dropout=block_config.get("dropout", 0.1),
                        activation=block_config.get("activation", "GELU")
                    )
                    current_channels = embed_dim
                
                elif block_type == "MLP_Mixer_Block":
                    # Use actual number of patches from current_size if in sequence mode
                    if is_sequence_model:
                        num_patches = current_size
                    else:
                        num_patches = block_config.get("num_patches", 64)
                    
                    embed_dim = block_config.get("embed_dim", current_channels)
                    
                    # Add projection if dimensions don't match
                    if current_channels != embed_dim:
                        proj_layer = nn.Linear(current_channels, embed_dim)
                        proj_name = f"proj_to_mixer_{i}"
                        layers[proj_name] = proj_layer
                        self.layer_order.append(proj_name)
                        current_channels = embed_dim
                    
                    block = get_block_by_name(
                        block_type,
                        num_patches=num_patches,
                        embed_dim=embed_dim,
                        tokens_mlp_dim=block_config.get("tokens_mlp_dim", 512),
                        channels_mlp_dim=block_config.get("channels_mlp_dim", 1024),
                        dropout=block_config.get("dropout", 0.1)
                    )
                    current_channels = embed_dim
                
                elif block_type == "SpikingNeuronLayer":
                    # Flatten if needed
                    if not is_sequence_model:
                        flatten_layer = nn.Flatten()
                        flatten_name = f"flatten_for_snn_{i}"
                        layers[flatten_name] = flatten_layer
                        self.layer_order.append(flatten_name)
                        in_features = current_channels * current_size * current_size
                    else:
                        in_features = current_channels
                    
                    out_features = block_config.get("out_features", 256)
                    
                    block = get_block_by_name(
                        block_type,
                        in_features=in_features,
                        out_features=out_features,
                        threshold=block_config.get("threshold", 1.0),
                        decay=block_config.get("decay", 0.9)
                    )
                    current_channels = out_features
                    current_size = 1  # After flatten
                    is_sequence_model = True
                
                elif block_type == "GraphConv":
                    # GraphConv needs flattened input
                    if not is_sequence_model:
                        flatten_layer = nn.Flatten()
                        flatten_name = f"flatten_for_graph_{i}"
                        layers[flatten_name] = flatten_layer
                        self.layer_order.append(flatten_name)
                        in_features = current_channels * current_size * current_size
                    else:
                        in_features = current_channels
                    
                    out_features = block_config.get("out_features", 128)
                    
                    block = get_block_by_name(
                        block_type,
                        in_features=in_features,
                        out_features=out_features,
                        activation=block_config.get("activation", "ReLU"),
                        dropout=block_config.get("dropout", 0.1)
                    )
                    current_channels = out_features
                    current_size = 1  # After flatten
                    is_sequence_model = True
                
                elif block_type == "HyperNetworkBlock":
                    # HyperNetworks are special - skip for now in main architecture
                    continue
                
                else:
                    continue
                
                layer_name = f"{block_type}_{i}"
                layers[layer_name] = block
                self.layer_order.append(layer_name)
            
            except Exception as e:
                print(f"Warning: Failed to create block {block_type} at layer {i}: {e}")
                continue
        
        self.is_sequence_model = is_sequence_model
        self.final_channels = current_channels
        self.final_size = current_size
        
        return layers
    
    def _build_classifier(self) -> nn.Module:
        """Build classification head"""
        
        if self.is_sequence_model:
            # For sequence models, use global average pooling
            classifier = nn.Sequential(
                nn.LayerNorm(self.final_channels),
                nn.Linear(self.final_channels, self.num_classes)
            )
        else:
            # For CNN models
            classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.final_channels, self.num_classes)
            )
        
        return classifier
    
    def _build_meta_components(self) -> nn.ModuleDict:
        """Build meta-learning components"""
        
        meta_dict = nn.ModuleDict()
        
        for meta_config in self.genome.meta_blocks:
            meta_type = meta_config.get("type")
            
            if meta_type == "DifferentiableAugmentation":
                meta_dict["diff_aug"] = get_block_by_name(meta_type)
            
            elif meta_type == "GradientModificationBlock":
                meta_dict["grad_mod"] = get_block_by_name(meta_type)
        
        return meta_dict
    
    def forward(self, x):
        """Forward pass"""
        
        # Apply differentiable augmentation if present
        if "diff_aug" in self.meta_components and self.training:
            x = self.meta_components["diff_aug"](x)
        
        # Process through network using layer_order
        for layer_name in self.layer_order:
            if layer_name == "patch_embed":
                x = self.network[layer_name](x)
                x = x.transpose(1, 2)  # (B, C, N) -> (B, N, C)
                if self.pos_embed is not None:
                    x = x + self.pos_embed
            else:
                x = self.network[layer_name](x)
        
        # Apply gradient modification if present
        if "grad_mod" in self.meta_components:
            x = self.meta_components["grad_mod"](x)
        
        # Handle different output formats
        if self.is_sequence_model and len(x.shape) == 3:
            # For sequence models, use mean pooling or first token
            x = x.mean(dim=1)  # Global average pooling over sequence
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_flops(self, input_size: tuple = None) -> float:
        """Estimate FLOPs for the model
        
        Uses improved FLOPs counter from src.utils.flops_counter
        For production use, consider fvcore or ptflops for exact counting.
        
        Returns:
            Estimated FLOPs in GFLOPs
        """
        try:
            from ..utils.flops_counter import estimate_flops
            if input_size is None:
                input_size = (1, self.input_channels, self.input_size, self.input_size)
            return estimate_flops(self, input_size)
        except Exception as e:
            # Fallback to parameter-based rough estimate
            return float(self.get_num_parameters()) * 2.0 / 1e9  # GFLOPs


class ReshapeLayer(nn.Module):
    """Helper layer to reshape tensors from sequence to spatial"""
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape  # (C, H, W)
    
    def forward(self, x):
        batch_size = x.size(0)
        C, H, W = self.target_shape
        
        # Handle different input shapes
        if len(x.shape) == 3:
            # x shape: (B, N, C) where N = H*W
            # Transpose from (B, N, C) to (B, C, N)
            x = x.transpose(1, 2)
            # Reshape to (B, C, H, W)
            return x.view(batch_size, C, H, W)
        elif len(x.shape) == 2:
            # x shape: (B, C) - already flattened
            # Reshape to (B, C, H, W)
            return x.view(batch_size, C, H, W)
        else:
            # Already spatial (B, C, H, W)
            return x


class ArchitectureBuilder:
    """Factory for building architectures from genomes"""
    
    @staticmethod
    def build(genome: ArchitectureGenome, num_classes: int = 100,
              input_channels: int = 3, input_size: int = 32) -> DynamicArchitecture:
        """Build a PyTorch model from genome"""
        
        model = DynamicArchitecture(
            genome=genome,
            num_classes=num_classes,
            input_channels=input_channels,
            input_size=input_size
        )
        
        return model
    
    @staticmethod
    def get_optimizer(model: nn.Module, genome: ArchitectureGenome) -> torch.optim.Optimizer:
        """Get optimizer from genome specification"""
        
        optimizer_name = genome.optimizer
        lr = genome.learning_rate
        
        if optimizer_name == "AdamW":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        elif optimizer_name == "LAMB":
            # Simplified LAMB (would need pytorch-lamb package)
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        elif optimizer_name == "SGD_Momentum":
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        
        elif optimizer_name == "RAdam":
            # Simplified RAdam
            return torch.optim.Adam(model.parameters(), lr=lr)
        
        elif optimizer_name == "Adafactor":
            # Simplified Adafactor
            return torch.optim.Adam(model.parameters(), lr=lr)
        
        else:
            return torch.optim.AdamW(model.parameters(), lr=lr)
    
    @staticmethod
    def get_scheduler(optimizer: torch.optim.Optimizer, 
                     genome: ArchitectureGenome,
                     num_epochs: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler from genome specification"""
        
        scheduler_name = genome.lr_scheduler
        
        if scheduler_name == "CosineAnnealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        elif scheduler_name == "OneCycleLR":
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=genome.learning_rate * 10, 
                total_steps=num_epochs, pct_start=0.3
            )
        
        elif scheduler_name == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
        
        elif scheduler_name == "ExponentialLR":
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
        else:
            return None
    
    @staticmethod
    def get_loss_function(genome: ArchitectureGenome) -> nn.Module:
        """Get loss function from genome specification"""
        
        loss_name = genome.loss_function
        
        if loss_name == "CrossEntropy":
            return nn.CrossEntropyLoss()
        
        elif loss_name == "FocalLoss":
            return FocalLoss(alpha=0.25, gamma=2.0)
        
        elif loss_name == "LabelSmoothing":
            return nn.CrossEntropyLoss(label_smoothing=0.1)
        
        elif loss_name == "CustomMetaLoss":
            return nn.CrossEntropyLoss()
        
        else:
            return nn.CrossEntropyLoss()


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
