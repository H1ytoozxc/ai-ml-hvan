"""
Main Entry Point for Project EvoArchitect v3
Точка входа для запуска автономного ИИ-агента
"""

import argparse
import torch
import random
import numpy as np
import os

from src.config import get_default_config, EvoArchitectConfig
from src.orchestrator import EvoArchitectOrchestrator
from src.data.data_loaders import DatasetFactory


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config_value(config, path, default=None):
    """Safely get nested config value supporting both dict and object access"""
    parts = path.split('.')
    current = config
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current

def set_config_value(config, path, value):
    """Safely set nested config value supporting both dict and object access"""
    parts = path.split('.')
    current = config
    for i, part in enumerate(parts[:-1]):
        if hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return
    
    last_part = parts[-1]
    if hasattr(current, last_part):
        setattr(current, last_part, value)
    elif isinstance(current, dict):
        current[last_part] = value

def main(args):
    """Main execution function"""
    
    # Load configuration
    if args.config:
        config = EvoArchitectConfig.from_yaml(args.config)
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    if args.population_size:
        set_config_value(config, 'initial_population_size', args.population_size)
    if args.device:
        set_config_value(config, 'runtime.compute_resources.device', args.device)
    if args.output_dir:
        set_config_value(config, 'logging.artifact_save_path', args.output_dir)
    
    # Set random seed
    set_seed(get_config_value(config, 'random_seed', 42))
    
    # Print configuration
    print("\n" + "="*80)
    print("PROJECT EVOARCHITECT V3")
    print("Autonomous AI-Evolutionary Agent for Neural Architecture Discovery")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Initial Population Size: {get_config_value(config, 'initial_population_size')}")
    print(f"  - Device: {get_config_value(config, 'runtime.compute_resources.device', 'cpu')}")
    print(f"  - Output Directory: {get_config_value(config, 'logging.artifact_save_path', './evo_runs')}")
    print(f"  - Random Seed: {get_config_value(config, 'random_seed', 42)}")
    print(f"  - Meta-Optimization: {'Enabled' if get_config_value(config, 'meta_optimization.enabled', False) else 'Disabled'}")
    print("="*80 + "\n")
    
    # Check CUDA availability
    device = get_config_value(config, 'runtime.compute_resources.device', 'cpu')
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        set_config_value(config, 'runtime.compute_resources.device', 'cpu')
    
    # Create output directory
    output_dir = get_config_value(config, 'logging.artifact_save_path', './evo_runs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    dataloaders = DatasetFactory.get_all_dataloaders(config)
    print("Datasets loaded successfully\n")
    
    # Create orchestrator
    orchestrator = EvoArchitectOrchestrator(config)
    
    # Run evolution pipeline
    try:
        final_models = orchestrator.run(dataloaders)
        
        print("\n" + "="*80)
        print("EVOLUTION COMPLETE!")
        print("="*80)
        print(f"\nFinal Results:")
        print(f"  - Number of final models: {len(final_models)}")
        
        if final_models:
            best_model = max(final_models, key=lambda g: g.metrics.get("val_accuracy", 0))
            print(f"  - Best accuracy: {best_model.metrics.get('val_accuracy', 0):.2f}%")
            print(f"  - Best novelty: {best_model.metrics.get('novelty_score_combined', 0):.4f}")
            print(f"  - Best robustness: {best_model.metrics.get('robustness_score', 0):.4f}")
            
            print(f"\nBest Model Architecture:")
            print(f"  - Genome ID: {best_model.genome_id}")
            print(f"  - Number of blocks: {len(best_model.blocks)}")
            print(f"  - Optimizer: {best_model.optimizer}")
            print(f"  - Learning rate: {best_model.learning_rate:.6f}")
            
        print(f"\nResults saved to: {config.logging.artifact_save_path}")
        print("="*80 + "\n")
        
        return final_models
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving current progress...")
        orchestrator.knowledge_base.close()
        print("Progress saved. Exiting.")
        return []
    
    except Exception as e:
        print(f"\nError during evolution: {e}")
        import traceback
        traceback.print_exc()
        orchestrator.knowledge_base.close()
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project EvoArchitect v3")
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="Initial population size (overrides config)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use for training (cuda or cpu)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results and artifacts"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with minimal population"
    )
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        print("\n*** QUICK TEST MODE ***")
        print("Running with minimal settings for testing...\n")
        args.population_size = 10
    
    main(args)
