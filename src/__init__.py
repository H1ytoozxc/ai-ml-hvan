"""Project EvoArchitect v3 - Autonomous AI Evolution Agent"""

__version__ = "3.0.0"
__author__ = "EvoMaster AI"

from .config import EvoArchitectConfig, get_default_config
from .orchestrator import EvoArchitectOrchestrator

__all__ = [
    "EvoArchitectConfig",
    "get_default_config",
    "EvoArchitectOrchestrator"
]
