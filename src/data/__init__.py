"""Data Loading Module"""

from .data_loaders import DatasetFactory, AugmentationPipeline

# HuggingFace loaders (optional)
try:
    from .huggingface_loaders import (
        HuggingFaceDatasetFactory,
        get_stage1_data,
        get_stage2_data,
        get_stage3_data
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("HuggingFace datasets not available. Install with: pip install datasets")

__all__ = [
    "DatasetFactory",
    "AugmentationPipeline",
]

if HF_AVAILABLE:
    __all__.extend([
        "HuggingFaceDatasetFactory",
        "get_stage1_data",
        "get_stage2_data",
        "get_stage3_data"
    ])
