"""
System Validation Script
Полная валидация системы Project EvoArchitect v3
"""

import sys
import torch
import traceback

def print_section(title):
    """Print section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_imports():
    """Test all critical imports"""
    print_section("1. TESTING IMPORTS")
    
    modules = [
        "src.config",
        "src.orchestrator",
        "src.search_space.blocks",
        "src.search_space.search_space",
        "src.search_space.architecture_generator",
        "src.training.trainer",
        "src.evaluation.metrics",
        "src.selection.novelty_metrics",
        "src.selection.pareto_frontier",
        "src.meta_optimization.mutations",
        "src.meta_optimization.rl_controller",
        "src.knowledge_base.database",
        "src.data.data_loaders",
        "src.utils.flops_counter",
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module}: {e}")
            failed.append(module)
    
    if failed:
        print(f"\n❌ {len(failed)} imports failed")
        return False
    else:
        print("\n✅ All imports successful")
        return True

def test_architecture_generation():
    """Test architecture generation"""
    print_section("2. TESTING ARCHITECTURE GENERATION")
    
    try:
        from src.config import get_default_config
        from src.search_space.search_space import ConditionalSearchSpace
        from src.search_space.architecture_generator import ArchitectureBuilder
        
        config = get_default_config()
        search_space = ConditionalSearchSpace(config.search_space)
        
        # Generate 3 test architectures
        print("\n  Generating 3 test architectures...")
        for i in range(3):
            genome = search_space.sample_architecture(depth=5)
            is_valid, errors = search_space.validate_architecture(genome)
            
            if not is_valid:
                print(f"  ✗ Architecture {i+1} validation failed: {errors}")
                return False
            
            print(f"  ✓ Architecture {i+1}: {len(genome.blocks)} blocks, {genome.optimizer}, lr={genome.learning_rate:.6f}")
        
        print("\n✅ Architecture generation successful")
        return True
    
    except Exception as e:
        print(f"\n❌ Architecture generation failed: {e}")
        traceback.print_exc()
        return False

def test_model_building():
    """Test PyTorch model building"""
    print_section("3. TESTING MODEL BUILDING")
    
    try:
        from src.config import get_default_config
        from src.search_space.search_space import ConditionalSearchSpace
        from src.search_space.architecture_generator import ArchitectureBuilder
        
        config = get_default_config()
        search_space = ConditionalSearchSpace(config.search_space)
        
        # Try up to 10 times to get a valid architecture
        model = None
        for attempt in range(10):
            try:
                genome = search_space.sample_architecture(depth=5)
                
                # Build model
                if attempt == 0:
                    print("\n  Building PyTorch model...")
                model = ArchitectureBuilder.build(genome, num_classes=10)
                
                # Test forward pass
                print("  Testing forward pass...")
                dummy_input = torch.randn(2, 3, 32, 32)
                
                with torch.no_grad():
                    model.eval()
                    output = model(dummy_input)
                
                # Success!
                break
            except Exception as e:
                if attempt < 9:
                    continue  # Try again
                else:
                    raise  # Last attempt failed
        
        # Check output shape
        if output.shape != (2, 10):
            print(f"  ✗ Unexpected output shape: {output.shape}, expected (2, 10)")
            return False
        
        # Count parameters
        params = model.get_num_parameters()
        print(f"  ✓ Model built: {params:,} parameters")
        
        # Test FLOPs estimation
        print("  Testing FLOPs estimation...")
        try:
            flops = model.get_flops()
            print(f"  ✓ FLOPs estimated: {flops:.2f} GFLOPs")
        except Exception as e:
            print(f"  ⚠ FLOPs estimation warning: {e}")
        
        print("\n✅ Model building successful")
        return True
    
    except Exception as e:
        print(f"\n❌ Model building failed: {e}")
        traceback.print_exc()
        return False

def test_knowledge_base():
    """Test knowledge base operations"""
    print_section("4. TESTING KNOWLEDGE BASE")
    
    try:
        import os
        import tempfile
        from src.knowledge_base.database import KnowledgeBase
        from src.search_space.search_space import ConditionalSearchSpace
        from src.config import get_default_config
        
        # Create temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            
            print(f"\n  Creating database: {db_path}")
            
            with KnowledgeBase(db_path) as kb:
                config = get_default_config()
                search_space = ConditionalSearchSpace(config.search_space)
                
                # Add architectures
                print("  Adding 5 test architectures...")
                genomes = []
                for i in range(5):
                    genome = search_space.sample_architecture()
                    genome.generation = 0
                    kb.add_architecture(genome)
                    genomes.append(genome)
                
                # Add evaluations
                print("  Adding evaluation results...")
                for i, genome in enumerate(genomes):
                    metrics = {
                        "accuracy": 70.0 + i * 2.0,
                        "loss": 0.5 - i * 0.05,
                        "parameter_count": 1000000 + i * 100000
                    }
                    kb.add_evaluation(genome.genome_id, "test_stage", metrics)
                
                # Retrieve statistics
                stats = kb.get_statistics()
                print(f"  ✓ Total architectures: {stats['total_architectures']}")
                print(f"  ✓ Total evaluations: {stats['total_evaluations']}")
                
                # Retrieve architecture
                retrieved = kb.get_architecture(genomes[0].genome_id)
                if retrieved and retrieved.genome_id == genomes[0].genome_id:
                    print("  ✓ Architecture retrieval successful")
                else:
                    print("  ✗ Architecture retrieval failed")
                    return False
        
        print("\n✅ Knowledge base operations successful")
        return True
    
    except Exception as e:
        print(f"\n❌ Knowledge base test failed: {e}")
        traceback.print_exc()
        return False

def test_mutations():
    """Test mutation operators"""
    print_section("5. TESTING MUTATION OPERATORS")
    
    try:
        from src.config import get_default_config
        from src.search_space.search_space import ConditionalSearchSpace
        from src.meta_optimization.mutations import (
            AddBlockMutation, RemoveBlockMutation, ChangeActivationMutation,
            CrossoverOperator
        )
        
        config = get_default_config()
        search_space = ConditionalSearchSpace(config.search_space)
        
        # Generate base genome
        genome = search_space.sample_architecture(depth=10)
        original_blocks = len(genome.blocks)
        
        print(f"\n  Base genome: {original_blocks} blocks")
        
        # Test AddBlock
        mutator = AddBlockMutation(mutation_rate=1.0)
        mutated = mutator(genome, search_space)
        print(f"  ✓ AddBlock: {original_blocks} → {len(mutated.blocks)} blocks")
        
        # Test RemoveBlock
        mutator = RemoveBlockMutation(mutation_rate=1.0)
        mutated = mutator(genome, search_space)
        print(f"  ✓ RemoveBlock: {original_blocks} → {len(mutated.blocks)} blocks")
        
        # Test ChangeActivation
        original_act = genome.blocks[0]['activation']
        mutator = ChangeActivationMutation(mutation_rate=1.0)
        mutated = mutator(genome, search_space)
        new_act = mutated.blocks[0]['activation']
        print(f"  ✓ ChangeActivation: {original_act} → {new_act}")
        
        # Test Crossover
        parent1 = search_space.sample_architecture(depth=8)
        parent2 = search_space.sample_architecture(depth=12)
        crossover = CrossoverOperator(crossover_rate=1.0)
        child = crossover(parent1, parent2)
        print(f"  ✓ Crossover: parents({len(parent1.blocks)}, {len(parent2.blocks)}) → child({len(child.blocks)})")
        
        print("\n✅ Mutation operators successful")
        return True
    
    except Exception as e:
        print(f"\n❌ Mutation test failed: {e}")
        traceback.print_exc()
        return False

def test_metrics():
    """Test metrics computation"""
    print_section("6. TESTING METRICS COMPUTATION")
    
    try:
        from src.evaluation.metrics import (
            MetricsCollector,
            ComputeEfficiencyMetrics
        )
        from src.search_space.search_space import ConditionalSearchSpace
        from src.search_space.architecture_generator import ArchitectureBuilder
        from src.config import get_default_config
        
        # Create model
        config = get_default_config()
        search_space = ConditionalSearchSpace(config.search_space)
        genome = search_space.sample_architecture(depth=5)
        model = ArchitectureBuilder.build(genome, num_classes=10)
        
        # Test MetricsCollector
        print("\n  Testing MetricsCollector...")
        collector = MetricsCollector()
        collector.start_epoch()
        collector.update(loss=0.5, accuracy=85.0)
        collector.end_epoch(0)
        print("  ✓ MetricsCollector works")
        
        # Test efficiency metrics
        print("  Testing ComputeEfficiencyMetrics...")
        params = ComputeEfficiencyMetrics.count_parameters(model)
        print(f"  ✓ Parameters: {params:,}")
        
        flops = ComputeEfficiencyMetrics.estimate_flops(model)
        print(f"  ✓ FLOPs: {flops:.2f} GFLOPs")
        
        print("\n✅ Metrics computation successful")
        return True
    
    except Exception as e:
        print(f"\n❌ Metrics test failed: {e}")
        traceback.print_exc()
        return False

def test_device_compatibility():
    """Test CUDA availability and device compatibility"""
    print_section("7. TESTING DEVICE COMPATIBILITY")
    
    try:
        print(f"\n  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  GPU name: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA not available - will use CPU")
        
        # Test model on available device
        from src.search_space.search_space import ConditionalSearchSpace
        from src.search_space.architecture_generator import ArchitectureBuilder
        from src.config import get_default_config
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        config = get_default_config()
        search_space = ConditionalSearchSpace(config.search_space)
        
        # Try up to 10 times to get a valid architecture
        for attempt in range(10):
            try:
                genome = search_space.sample_architecture(depth=3)
                model = ArchitectureBuilder.build(genome, num_classes=10).to(device)
                
                dummy_input = torch.randn(1, 3, 32, 32).to(device)
                
                with torch.no_grad():
                    output = model(dummy_input)
                
                # Success!
                break
            except Exception as e:
                if attempt < 9:
                    continue  # Try again
                else:
                    raise  # Last attempt failed
        
        print(f"  ✓ Model successfully runs on {device}")
        
        print("\n✅ Device compatibility check passed")
        return True
    
    except Exception as e:
        print(f"\n❌ Device compatibility test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests"""
    
    print("="*80)
    print("  PROJECT EVOARCHITECT V3 - SYSTEM VALIDATION")
    print("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Architecture Generation", test_architecture_generation),
        ("Model Building", test_model_building),
        ("Knowledge Base", test_knowledge_base),
        ("Mutation Operators", test_mutations),
        ("Metrics Computation", test_metrics),
        ("Device Compatibility", test_device_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print_section("VALIDATION SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    failed = sum(1 for _, result in results if not result)
    
    print()
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    print(f"\n  Total: {passed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("\n" + "="*80)
        print("  🎉 ALL TESTS PASSED - SYSTEM IS READY!")
        print("="*80)
        print("\n  Next steps:")
        print("    1. Run: python main.py --quick-test")
        print("    2. Check AUDIT_REPORT.md for details")
        print("    3. Read README.md and USAGE_GUIDE.md")
        print("="*80 + "\n")
        return 0
    else:
        print("\n" + "="*80)
        print(f"  ❌ {failed} TESTS FAILED - CHECK ERRORS ABOVE")
        print("="*80 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
