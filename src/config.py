"""
Project EvoArchitect v3 - Configuration Module
Центральная конфигурация для автономного ИИ-агента по эволюции архитектур
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import yaml


class StageType(Enum):
    PROXY_EVALUATION = "Stage1_Proxy_Evaluation"
    REFINEMENT_TRAINING = "Stage2_Refinement_Training"
    FULL_VALIDATION = "Stage3_Full_Validation"


class BlockType(Enum):
    CONV3X3 = "Conv3x3"
    RESIDUAL_BLOCK = "ResidualBlock"
    TRANSFORMER_ENCODER = "TransformerEncoderBlock"
    MLP_MIXER = "MLP_Mixer_Block"
    SPIKING_NEURON = "SpikingNeuronLayer"
    GRAPH_CONV = "GraphConv"
    HYPER_NETWORK = "HyperNetworkBlock"


class ActivationType(Enum):
    RELU = "ReLU"
    GELU = "GELU"
    SWISH = "Swish"
    MISH = "Mish"
    SILU = "SiLU"
    TANH = "Tanh"


class NormalizationType(Enum):
    BATCH_NORM = "BatchNorm"
    LAYER_NORM = "LayerNorm"
    GROUP_NORM = "GroupNorm"
    INSTANCE_NORM = "InstanceNorm"
    WEIGHT_NORM = "WeightNorm"


@dataclass
class DatasetConfig:
    name: str
    subset_percent: Optional[float] = None
    full: bool = False
    classes: Optional[int] = None
    images_per_class: Optional[int] = None
    strategy: str = "class_balanced_random_sample"


@dataclass
class MetricsConfig:
    metrics: List[str] = field(default_factory=list)
    track_memory: bool = True
    track_vram: bool = True
    track_flops: bool = True


@dataclass
class SurrogateModelConfig:
    enabled: bool = True
    type: str = "GraphNeuralNetwork"
    predict_target: List[str] = field(default_factory=lambda: ["final_accuracy", "convergence_speed"])
    hidden_dim: int = 128
    num_layers: int = 4


@dataclass
class NoveltyMetricConfig:
    type: str = "combined_arch_behavior_distance"
    weights: Dict[str, float] = field(default_factory=lambda: {"architectural": 0.4, "behavioral": 0.6})
    method: str = "graph_edit_distance + activation_profile_correlation"


@dataclass
class RobustnessMetricConfig:
    type: str = "corruption_benchmark"
    benchmark: str = "CIFAR-100-C"
    severities: List[int] = field(default_factory=lambda: [1, 3, 5])


@dataclass
class SelectionCriteriaConfig:
    strategy: str
    top_n: Optional[int] = None
    value: Optional[str] = None
    min_candidates: Optional[int] = None
    max_candidates: Optional[int] = None
    novelty_score_weight: float = 0.25
    diversity_weight: float = 0.35
    objectives: List[str] = field(default_factory=list)


@dataclass
class StageConfig:
    name: str
    description: str
    input_candidates: Any
    epochs: int
    datasets: List[DatasetConfig]
    metrics: MetricsConfig
    selection_criteria: SelectionCriteriaConfig
    surrogate_model: Optional[SurrogateModelConfig] = None
    weight_sharing: bool = False
    novelty_metric: Optional[NoveltyMetricConfig] = None
    robustness_metric: Optional[RobustnessMetricConfig] = None
    early_stopping: bool = False
    early_stopping_patience: int = 3
    use_augmentation_policy: Optional[str] = None


@dataclass
class SearchSpaceConfig:
    base_blocks: List[str] = field(default_factory=lambda: [
        "Conv3x3", "ResidualBlock", "TransformerEncoderBlock", 
        "MLP_Mixer_Block", "SpikingNeuronLayer", "GraphConv", "HyperNetworkBlock"
    ])
    activations: List[str] = field(default_factory=lambda: [
        "ReLU", "GELU", "Swish", "Mish", "SiLU", "Tanh"
    ])
    normalizations: List[str] = field(default_factory=lambda: [
        "BatchNorm", "LayerNorm", "GroupNorm", "InstanceNorm", "WeightNorm"
    ])
    dropout_range: tuple = (0.05, 0.5)
    optimizers: List[str] = field(default_factory=lambda: [
        "AdamW", "LAMB", "SGD_Momentum", "RAdam", "Adafactor"
    ])
    lr_range: tuple = (1e-5, 1e-2)
    lr_schedulers: List[str] = field(default_factory=lambda: [
        "CosineAnnealing", "OneCycleLR", "ReduceLROnPlateau", "ExponentialLR"
    ])
    loss_functions: List[str] = field(default_factory=lambda: [
        "CrossEntropy", "FocalLoss", "LabelSmoothing", "CustomMetaLoss"
    ])
    data_augmentations: List[str] = field(default_factory=lambda: [
        "RandomCrop", "Cutout", "Mixup", "RandAugment", "AutoAugment", "CutMix"
    ])
    meta_blocks: List[str] = field(default_factory=lambda: [
        "HyperNet", "LearnedOptimizer", "CurriculumLearning", 
        "DifferentiableAugmentation", "GradientModificationBlock"
    ])
    max_depth: int = 20
    min_depth: int = 3


@dataclass
class MetaOptimizationConfig:
    enabled: bool = True
    strategy: str = "Evolutionary_Strategy_with_RL_Controller"
    controller_type: str = "REINFORCE_Controller"
    target: str = "maximize_pareto_frontier_expansion"
    reward_signal: str = "performance_improvement_from_stage2 + novelty_gain"
    learning_rate: float = 1e-3
    hidden_dim: int = 256
    num_episodes: int = 100


@dataclass
class ComputeResourcesConfig:
    device: str = "cuda"
    num_workers_per_trial: int = 2
    num_parallel_trials: int = 2
    dynamic_resource_allocation: bool = True
    max_concurrent_trials: int = 4
    gpu_memory_limit_gb: Optional[float] = None


@dataclass
class RuntimeConfig:
    distributed_framework: str = "Ray_Tune_with_Ray_AIR"
    scheduler_type: str = "ASHA"
    max_t: int = 100
    grace_period: int = 5
    reduction_factor: int = 3
    compute_resources: ComputeResourcesConfig = field(default_factory=ComputeResourcesConfig)
    checkpoint_frequency: int = 5
    max_retries: int = 3


@dataclass
class LoggingConfig:
    dashboard_provider: str = "WeightsAndBiases"
    log_level: str = "INFO"
    log_metrics: str = "all"
    wandb_project: str = "evo-architect-v3"
    wandb_entity: Optional[str] = None
    save_top_n: int = 10
    artifact_save_path: str = "./evo_runs"


@dataclass
class EvoArchitectConfig:
    """Main configuration for Project EvoArchitect v3"""
    project_name: str = "Project_EvoArchitect_v3"
    initial_population_size: int = 3000
    
    # Pipeline stages
    stage1_config: StageConfig = None
    stage2_config: StageConfig = None
    stage3_config: StageConfig = None
    
    # Search space
    search_space: SearchSpaceConfig = field(default_factory=SearchSpaceConfig)
    
    # Meta-optimization
    meta_optimization: MetaOptimizationConfig = field(default_factory=MetaOptimizationConfig)
    
    # Runtime
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    
    # Logging
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Knowledge base
    knowledge_base_path: str = "./evo_runs/knowledge_base.db"
    
    # Random seed
    random_seed: int = 42
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'EvoArchitectConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EvoArchitectConfig':
        """Create configuration from dictionary"""
        # This is simplified - in production, you'd want more robust parsing
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        # Convert dataclass to dict recursively
        import dataclasses
        return dataclasses.asdict(self)


def get_default_config() -> EvoArchitectConfig:
    """Get default configuration for EvoArchitect v3"""
    
    # Stage 1 configuration
    stage1_config = StageConfig(
        name="Stage1_Proxy_Evaluation",
        description="Массовый скрининг 3000+ кандидатов на proxy-данных",
        input_candidates=3000,
        epochs=1,
        datasets=[DatasetConfig(name="CIFAR-10", subset_percent=10)],
        metrics=MetricsConfig(
            metrics=["proxy_accuracy", "proxy_loss", "estimated_FLOPs", "parameter_count"]
        ),
        surrogate_model=SurrogateModelConfig(),
        weight_sharing=True,
        selection_criteria=SelectionCriteriaConfig(
            strategy="dynamic_percentile",
            value="top_15_percent",
            min_candidates=250,
            max_candidates=600
        )
    )
    
    # Stage 2 configuration
    stage2_config = StageConfig(
        name="Stage2_Refinement_Training",
        description="Среднее обучение для выживших кандидатов",
        input_candidates="from_Stage1",
        epochs=20,
        datasets=[DatasetConfig(name="CIFAR-100", subset_percent=50)],
        metrics=MetricsConfig(
            metrics=["accuracy", "loss", "robustness_score", "novelty_score", "learning_curve_slope"]
        ),
        novelty_metric=NoveltyMetricConfig(),
        robustness_metric=RobustnessMetricConfig(),
        early_stopping=True,
        early_stopping_patience=3,
        selection_criteria=SelectionCriteriaConfig(
            strategy="pareto_frontier_selection",
            objectives=["accuracy", "novelty_score", "robustness_score", "learning_curve_slope"],
            top_n=100
        )
    )
    
    # Stage 3 configuration
    stage3_config = StageConfig(
        name="Stage3_Full_Validation",
        description="Полное обучение и кросс-валидация на нескольких бенчмарках",
        input_candidates="from_Stage2",
        epochs=100,
        datasets=[
            DatasetConfig(name="CIFAR-100", full=True),
            DatasetConfig(name="ImageNet_subset", classes=100, images_per_class=1000),
            DatasetConfig(name="SVHN", full=True),
            DatasetConfig(name="TinyImageNet", full=True)
        ],
        metrics=MetricsConfig(
            metrics=["accuracy", "robustness_score", "convergence_speed_epochs", 
                    "compute_efficiency", "novelty_score_combined", "generalization_gap"]
        ),
        use_augmentation_policy="AutoAugment + RandAugment",
        selection_criteria=SelectionCriteriaConfig(
            strategy="final_pareto_frontier_selection",
            objectives=["accuracy", "novelty_score_combined", "robustness_score", 
                       "compute_efficiency", "generalization_gap"],
            top_n=10
        )
    )
    
    return EvoArchitectConfig(
        stage1_config=stage1_config,
        stage2_config=stage2_config,
        stage3_config=stage3_config
    )
