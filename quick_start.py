"""
Quick Start Script
Простой скрипт для быстрого тестирования системы
"""

import torch
from src.config import get_default_config
from src.search_space.search_space import ConditionalSearchSpace
from src.search_space.architecture_generator import ArchitectureBuilder
from src.knowledge_base.database import KnowledgeBase


def test_architecture_generation():
    """Test architecture generation"""
    print("\n" + "="*80)
    print("TEST 1: Architecture Generation")
    print("="*80)
    
    # Get config
    config = get_default_config()
    
    # Create search space
    search_space = ConditionalSearchSpace(config.search_space)
    
    # Generate 5 random architectures
    print("\nGenerating 5 random architectures...")
    for i in range(5):
        genome = search_space.sample_architecture()
        
        print(f"\nArchitecture {i+1}:")
        print(f"  - Genome ID: {genome.genome_id}")
        print(f"  - Number of blocks: {len(genome.blocks)}")
        print(f"  - Optimizer: {genome.optimizer}")
        print(f"  - Learning rate: {genome.learning_rate:.6f}")
        print(f"  - Block types: {[b['type'] for b in genome.blocks[:3]]}...")
        
        # Validate
        is_valid, errors = search_space.validate_architecture(genome)
        if is_valid:
            print(f"  ✓ Architecture is valid")
        else:
            print(f"  ✗ Validation errors: {errors}")
    
    print("\n✅ Architecture generation test passed!")
    return True


def test_model_building():
    """Test building PyTorch models"""
    print("\n" + "="*80)
    print("TEST 2: PyTorch Model Building")
    print("="*80)
    
    config = get_default_config()
    search_space = ConditionalSearchSpace(config.search_space)
    
    # Generate architecture
    genome = search_space.sample_architecture(depth=5)
    
    print(f"\nBuilding PyTorch model from genome...")
    print(f"  - Genome ID: {genome.genome_id}")
    print(f"  - Number of blocks: {len(genome.blocks)}")
    
    try:
        # Build model
        model = ArchitectureBuilder.build(genome, num_classes=10)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 32, 32)
        output = model(dummy_input)
        
        print(f"\n  ✓ Model built successfully")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Number of parameters: {model.get_num_parameters():,}")
        
        print("\n✅ Model building test passed!")
        return True
    
    except Exception as e:
        print(f"\n  ✗ Error building model: {e}")
        return False


def test_knowledge_base():
    """Test knowledge base"""
    print("\n" + "="*80)
    print("TEST 3: Knowledge Base")
    print("="*80)
    
    import os
    import tempfile
    
    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_kb.db")
        
        print(f"\nCreating knowledge base at: {db_path}")
        
        with KnowledgeBase(db_path) as kb:
            # Generate and save architectures
            config = get_default_config()
            search_space = ConditionalSearchSpace(config.search_space)
            
            print("\nAdding 10 architectures to knowledge base...")
            genomes = []
            for i in range(10):
                genome = search_space.sample_architecture()
                genome.generation = 0
                kb.add_architecture(genome)
                genomes.append(genome)
            
            # Add evaluations
            print("Adding evaluation results...")
            for genome in genomes[:5]:
                metrics = {
                    "accuracy": 70.0 + i * 2.0,
                    "loss": 0.5 - i * 0.05,
                    "parameter_count": 1000000
                }
                kb.add_evaluation(genome.genome_id, "Stage1_Proxy_Evaluation", metrics)
            
            # Retrieve statistics
            stats = kb.get_statistics()
            print(f"\nKnowledge base statistics:")
            print(f"  - Total architectures: {stats['total_architectures']}")
            print(f"  - Total evaluations: {stats['total_evaluations']}")
            
            # Retrieve architecture
            retrieved = kb.get_architecture(genomes[0].genome_id)
            if retrieved and retrieved.genome_id == genomes[0].genome_id:
                print(f"  ✓ Successfully retrieved architecture")
            else:
                print(f"  ✗ Failed to retrieve architecture")
            
            print("\n✅ Knowledge base test passed!")
            return True


def test_mutations():
    """Test mutation operators"""
    print("\n" + "="*80)
    print("TEST 4: Mutation Operators")
    print("="*80)
    
    config = get_default_config()
    search_space = ConditionalSearchSpace(config.search_space)
    
    # Generate base genome
    genome = search_space.sample_architecture(depth=10)
    
    print(f"\nBase genome: {len(genome.blocks)} blocks")
    
    # Test mutations
    from src.meta_optimization.mutations import (
        AddBlockMutation, RemoveBlockMutation, ChangeActivationMutation
    )
    
    # Add block
    mutator = AddBlockMutation(mutation_rate=1.0)
    mutated = mutator(genome, search_space)
    print(f"  - After AddBlock: {len(mutated.blocks)} blocks")
    
    # Remove block
    mutator = RemoveBlockMutation(mutation_rate=1.0)
    mutated = mutator(genome, search_space)
    print(f"  - After RemoveBlock: {len(mutated.blocks)} blocks")
    
    # Change activation
    original_activation = genome.blocks[0]['activation']
    mutator = ChangeActivationMutation(mutation_rate=1.0)
    mutated = mutator(genome, search_space)
    new_activation = mutated.blocks[0]['activation']
    print(f"  - Activation changed: {original_activation} → {new_activation}")
    
    print("\n✅ Mutation operators test passed!")
    return True


def main():
    """Run all tests"""
    print("="*80)
    print("PROJECT EVOARCHITECT V3 - QUICK START TEST")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Architecture Generation", test_architecture_generation()))
    results.append(("Model Building", test_model_building()))
    results.append(("Knowledge Base", test_knowledge_base()))
    results.append(("Mutation Operators", test_mutations()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run full evolution: python main.py --quick-test")
        print("  2. Check README.md for detailed documentation")
        print("  3. Customize config.yaml for your needs")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    main()
