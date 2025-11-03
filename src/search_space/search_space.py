"""
Search Space Definition
Определение пространства поиска для генерации архитектур
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class ArchitectureGenome:
    """Genome representing a neural architecture"""
    
    # Network structure
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Global hyperparameters
    optimizer: str = "AdamW"
    learning_rate: float = 1e-3
    lr_scheduler: str = "CosineAnnealing"
    loss_function: str = "CrossEntropy"
    batch_size: int = 128
    
    # Augmentation strategy
    augmentations: List[str] = field(default_factory=list)
    
    # Meta-learning components
    meta_blocks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Architecture metadata
    genome_id: Optional[str] = None
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    # Performance tracking
    metrics: Dict[str, float] = field(default_factory=dict)
    novelty_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary"""
        return {
            "blocks": self.blocks,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "lr_scheduler": self.lr_scheduler,
            "loss_function": self.loss_function,
            "batch_size": self.batch_size,
            "augmentations": self.augmentations,
            "meta_blocks": self.meta_blocks,
            "genome_id": self.genome_id,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "metrics": self.metrics,
            "novelty_score": self.novelty_score
        }
    
    def to_json(self) -> str:
        """Convert genome to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArchitectureGenome':
        """Create genome from dictionary"""
        return cls(**data)
    
    def clone(self) -> 'ArchitectureGenome':
        """Create a deep copy of the genome"""
        import copy
        return copy.deepcopy(self)


class ConditionalSearchSpace:
    """Conditional hierarchical search space"""
    
    def __init__(self, config):
        self.config = config
        self.conditionals = self._build_conditionals()
    
    def _build_conditionals(self) -> Dict[str, Dict[str, List[str]]]:
        """Build conditional constraints for the search space"""
        
        conditionals = {
            "TransformerEncoderBlock": {
                "normalizations": ["LayerNorm"],
                "activations": ["GELU", "ReLU"]
            },
            "Conv3x3": {
                "normalizations": ["BatchNorm", "GroupNorm"],
                "activations": ["ReLU", "Swish", "Mish"]
            },
            "ResidualBlock": {
                "normalizations": ["BatchNorm", "GroupNorm"],
                "activations": ["ReLU", "Swish"]
            },
            "MLP_Mixer_Block": {
                "normalizations": ["LayerNorm"],
                "activations": ["GELU"]
            }
        }
        
        return conditionals
    
    def get_valid_activations(self, block_type: str) -> List[str]:
        """Get valid activations for a given block type"""
        activations = self.config.get('activations', ['ReLU', 'GELU']) if isinstance(self.config, dict) else getattr(self.config, 'activations', ['ReLU', 'GELU'])
        if block_type in self.conditionals:
            return self.conditionals[block_type].get("activations", activations)
        return activations
    
    def get_valid_normalizations(self, block_type: str) -> List[str]:
        """Get valid normalizations for a given block type"""
        normalizations = self.config.get('normalizations', ['BatchNorm', 'LayerNorm']) if isinstance(self.config, dict) else getattr(self.config, 'normalizations', ['BatchNorm', 'LayerNorm'])
        if block_type in self.conditionals:
            return self.conditionals[block_type].get("normalizations", normalizations)
        return normalizations
    
    def sample_block_config(self, block_type: str) -> Dict[str, Any]:
        """Sample configuration for a specific block type"""
        
        dropout_range = self.config.get('dropout_range', [0.05, 0.3]) if isinstance(self.config, dict) else getattr(self.config, 'dropout_range', [0.05, 0.3])
        
        config = {
            "type": block_type,
            "activation": random.choice(self.get_valid_activations(block_type)),
            "normalization": random.choice(self.get_valid_normalizations(block_type)),
            "dropout": random.uniform(*dropout_range)
        }
        
        # Add block-specific parameters
        if block_type in ["Conv3x3", "ResidualBlock"]:
            config["out_channels"] = random.choice([64, 128, 256, 512])
            config["stride"] = random.choice([1, 2])
        
        elif block_type == "TransformerEncoderBlock":
            config["embed_dim"] = random.choice([128, 256, 512])
            config["num_heads"] = random.choice([4, 8, 16])
            config["mlp_ratio"] = random.uniform(2.0, 4.0)
        
        elif block_type == "MLP_Mixer_Block":
            config["num_patches"] = random.choice([49, 64, 196])  # 7x7, 8x8, 14x14
            config["embed_dim"] = random.choice([256, 512])
            config["tokens_mlp_dim"] = random.choice([256, 512])
            config["channels_mlp_dim"] = random.choice([512, 1024, 2048])
        
        elif block_type == "SpikingNeuronLayer":
            config["out_features"] = random.choice([128, 256, 512])
            config["threshold"] = random.uniform(0.5, 1.5)
            config["decay"] = random.uniform(0.8, 0.95)
        
        elif block_type == "GraphConv":
            config["out_features"] = random.choice([64, 128, 256])
        
        elif block_type == "HyperNetworkBlock":
            config["hyper_input_dim"] = random.choice([32, 64, 128])
            config["hidden_dim"] = random.choice([64, 128, 256])
        
        return config
    
    def sample_architecture(self, depth: Optional[int] = None) -> ArchitectureGenome:
        """Sample a random architecture from the search space"""
        
        if depth is None:
            # Safe access for both dict and object
            min_depth = self.config.get('min_depth', 3) if isinstance(self.config, dict) else getattr(self.config, 'min_depth', 3)
            max_depth = self.config.get('max_depth', 15) if isinstance(self.config, dict) else getattr(self.config, 'max_depth', 15)
            depth = random.randint(min_depth, max_depth)
        
        genome = ArchitectureGenome()
        
        # Sample blocks
        for i in range(depth):
            base_blocks = self.config.get('base_blocks', []) if isinstance(self.config, dict) else getattr(self.config, 'base_blocks', [])
            block_type = random.choice(base_blocks)
            block_config = self.sample_block_config(block_type)
            block_config["layer_id"] = i
            genome.blocks.append(block_config)
        
        # Sample global hyperparameters
        optimizers = self.config.get('optimizers', ['AdamW']) if isinstance(self.config, dict) else getattr(self.config, 'optimizers', ['AdamW'])
        genome.optimizer = random.choice(optimizers)
        
        # Use log-uniform sampling (compatible with numpy 2.x)
        lr_range = self.config.get('lr_range', [0.00001, 0.01]) if isinstance(self.config, dict) else getattr(self.config, 'lr_range', [0.00001, 0.01])
        log_min = np.log(lr_range[0])
        log_max = np.log(lr_range[1])
        genome.learning_rate = np.exp(np.random.uniform(log_min, log_max))
        
        lr_schedulers = self.config.get('lr_schedulers', ['CosineAnnealing']) if isinstance(self.config, dict) else getattr(self.config, 'lr_schedulers', ['CosineAnnealing'])
        genome.lr_scheduler = random.choice(lr_schedulers)
        
        loss_functions = self.config.get('loss_functions', ['CrossEntropy']) if isinstance(self.config, dict) else getattr(self.config, 'loss_functions', ['CrossEntropy'])
        genome.loss_function = random.choice(loss_functions)
        
        genome.batch_size = random.choice([32, 64, 128, 256])
        
        # Sample augmentations
        data_augmentations = self.config.get('data_augmentations', ['RandomCrop']) if isinstance(self.config, dict) else getattr(self.config, 'data_augmentations', ['RandomCrop'])
        num_augmentations = random.randint(2, min(len(data_augmentations), 5))
        genome.augmentations = random.sample(
            data_augmentations, 
            min(num_augmentations, len(data_augmentations))
        )
        
        # Optionally add meta-learning blocks
        meta_blocks = self.config.get('meta_blocks', []) if isinstance(self.config, dict) else getattr(self.config, 'meta_blocks', [])
        if meta_blocks and random.random() < 0.3:  # 30% chance if meta_blocks defined
            num_meta_blocks = random.randint(1, 3)
            for _ in range(num_meta_blocks):
                meta_block_type = random.choice(meta_blocks)
                meta_config = {
                    "type": meta_block_type,
                    "position": random.choice(["pre", "mid", "post"])  # where to insert
                }
                genome.meta_blocks.append(meta_config)
        
        # Generate genome ID
        import uuid
        genome.genome_id = str(uuid.uuid4())
        
        return genome
    
    def validate_architecture(self, genome: ArchitectureGenome) -> Tuple[bool, List[str]]:
        """Validate that an architecture is valid"""
        
        errors = []
        
        # Check depth constraints
        if len(genome.blocks) < self.config.min_depth:
            errors.append(f"Architecture too shallow: {len(genome.blocks)} < {self.config.min_depth}")
        
        if len(genome.blocks) > self.config.max_depth:
            errors.append(f"Architecture too deep: {len(genome.blocks)} > {self.config.max_depth}")
        
        # Check block validity
        for i, block in enumerate(genome.blocks):
            block_type = block.get("type")
            if block_type not in self.config.base_blocks:
                errors.append(f"Invalid block type at layer {i}: {block_type}")
            
            # Check conditional constraints
            activation = block.get("activation")
            valid_activations = self.get_valid_activations(block_type)
            if activation and activation not in valid_activations:
                errors.append(f"Invalid activation '{activation}' for block {block_type} at layer {i}")
            
            normalization = block.get("normalization")
            valid_normalizations = self.get_valid_normalizations(block_type)
            if normalization and normalization not in valid_normalizations:
                errors.append(f"Invalid normalization '{normalization}' for block {block_type} at layer {i}")
        
        # Check hyperparameters
        if genome.optimizer not in self.config.optimizers:
            errors.append(f"Invalid optimizer: {genome.optimizer}")
        
        if not (self.config.lr_range[0] <= genome.learning_rate <= self.config.lr_range[1]):
            errors.append(f"Learning rate out of range: {genome.learning_rate}")
        
        return len(errors) == 0, errors
    
    def get_architecture_complexity(self, genome: ArchitectureGenome) -> Dict[str, float]:
        """Estimate architecture complexity"""
        
        complexity = {
            "num_blocks": len(genome.blocks),
            "num_meta_blocks": len(genome.meta_blocks),
            "num_augmentations": len(genome.augmentations),
            "estimated_params": 0.0,
            "estimated_flops": 0.0
        }
        
        # Rough parameter estimation
        for block in genome.blocks:
            block_type = block.get("type")
            
            if block_type in ["Conv3x3", "ResidualBlock"]:
                channels = block.get("out_channels", 256)
                complexity["estimated_params"] += channels * channels * 9  # 3x3 conv
                complexity["estimated_flops"] += channels * channels * 9 * 32 * 32  # Assume 32x32 feature map
            
            elif block_type == "TransformerEncoderBlock":
                embed_dim = block.get("embed_dim", 256)
                complexity["estimated_params"] += embed_dim * embed_dim * 4  # QKV + output
                complexity["estimated_flops"] += embed_dim * embed_dim * 4 * 196  # Assume 14x14 patches
            
            elif block_type == "MLP_Mixer_Block":
                embed_dim = block.get("embed_dim", 256)
                tokens_mlp = block.get("tokens_mlp_dim", 512)
                channels_mlp = block.get("channels_mlp_dim", 1024)
                complexity["estimated_params"] += (embed_dim * tokens_mlp + embed_dim * channels_mlp)
        
        complexity["estimated_params"] /= 1e6  # Convert to millions
        complexity["estimated_flops"] /= 1e9  # Convert to GFLOPs
        
        return complexity


class ArchitectureEncoder:
    """Encode architectures to fixed-size vectors for novelty computation"""
    
    @staticmethod
    def encode_genome(genome: ArchitectureGenome, encoding_dim: int = 128) -> np.ndarray:
        """Encode genome to fixed-size vector"""
        
        # Create a feature vector
        features = []
        
        # Structural features
        features.append(len(genome.blocks))
        features.append(len(genome.meta_blocks))
        
        # Block type distribution
        block_types = [b.get("type") for b in genome.blocks]
        block_type_counts = {bt: block_types.count(bt) for bt in set(block_types)}
        
        # One-hot encoding for block types (simplified)
        all_block_types = ["Conv3x3", "ResidualBlock", "TransformerEncoderBlock", 
                          "MLP_Mixer_Block", "SpikingNeuronLayer", "GraphConv", "HyperNetworkBlock"]
        for bt in all_block_types:
            features.append(block_type_counts.get(bt, 0))
        
        # Hyperparameter features
        features.append(np.log(genome.learning_rate))
        features.append(genome.batch_size / 256.0)  # Normalize
        features.append(len(genome.augmentations) / 6.0)  # Normalize
        
        # Pad or truncate to encoding_dim
        features_array = np.array(features, dtype=np.float32)
        
        if len(features_array) < encoding_dim:
            features_array = np.pad(features_array, (0, encoding_dim - len(features_array)))
        else:
            features_array = features_array[:encoding_dim]
        
        return features_array
    
    @staticmethod
    def compute_distance(genome1: ArchitectureGenome, genome2: ArchitectureGenome) -> float:
        """Compute distance between two genomes"""
        
        enc1 = ArchitectureEncoder.encode_genome(genome1)
        enc2 = ArchitectureEncoder.encode_genome(genome2)
        
        # Euclidean distance
        distance = np.linalg.norm(enc1 - enc2)
        
        return float(distance)
