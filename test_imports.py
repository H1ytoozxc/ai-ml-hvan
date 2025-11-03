"""
Test all imports and check for errors
Тест всех импортов и проверка ошибок
"""

import sys
import traceback

def test_import(module_name):
    """Test importing a module"""
    try:
        __import__(module_name)
        print(f"✓ {module_name}")
        return True
    except Exception as e:
        print(f"✗ {module_name}: {e}")
        traceback.print_exc()
        return False

print("="*80)
print("TESTING IMPORTS")
print("="*80 + "\n")

modules_to_test = [
    # Core
    "src.config",
    "src.orchestrator",
    
    # Search space
    "src.search_space.blocks",
    "src.search_space.search_space",
    "src.search_space.architecture_generator",
    
    # Training
    "src.training.trainer",
    
    # Evaluation
    "src.evaluation.metrics",
    
    # Selection
    "src.selection.novelty_metrics",
    "src.selection.pareto_frontier",
    
    # Meta-optimization
    "src.meta_optimization.mutations",
    "src.meta_optimization.rl_controller",
    
    # Knowledge base
    "src.knowledge_base.database",
    
    # Data
    "src.data.data_loaders",
]

results = []
for module in modules_to_test:
    results.append((module, test_import(module)))

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

passed = sum(1 for _, result in results if result)
failed = sum(1 for _, result in results if not result)

print(f"Passed: {passed}/{len(results)}")
print(f"Failed: {failed}/{len(results)}")

if failed == 0:
    print("\n✅ All imports successful!")
else:
    print(f"\n❌ {failed} imports failed")
    print("\nFailed modules:")
    for module, result in results:
        if not result:
            print(f"  - {module}")

sys.exit(0 if failed == 0 else 1)
