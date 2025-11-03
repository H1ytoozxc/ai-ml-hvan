"""
Analyze Training Results
Анализ результатов обучения
"""

import torch
import os

def analyze_checkpoint(checkpoint_path):
    """Анализ чекпоинта"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ File not found: {checkpoint_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"  CHECKPOINT ANALYSIS: {os.path.basename(checkpoint_path)}")
    print(f"{'='*80}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Print info
    print(f"\n📊 Training Info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Validation Accuracy: {checkpoint.get('val_acc', 0):.2f}%")
    print(f"  Validation Loss: {checkpoint.get('val_loss', 0):.4f}")
    
    # Model info
    model_state = checkpoint['model_state_dict']
    total_params = sum(p.numel() for p in model_state.values())
    print(f"\n🧠 Model Info:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Model Size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.1f} MB")
    
    # Layer info
    print(f"\n📦 Layers:")
    for name, param in list(model_state.items())[:10]:
        print(f"  {name}: {list(param.shape)}")
    if len(model_state) > 10:
        print(f"  ... and {len(model_state) - 10} more layers")
    
    print(f"\n{'='*80}\n")


def main():
    checkpoint_dir = "./checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Find all checkpoints
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    if not checkpoints:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"  TRAINING RESULTS ANALYSIS")
    print(f"{'='*80}")
    print(f"\nFound {len(checkpoints)} checkpoint(s):")
    for cp in checkpoints:
        print(f"  - {cp}")
    
    # Analyze each checkpoint
    for cp in sorted(checkpoints):
        analyze_checkpoint(os.path.join(checkpoint_dir, cp))
    
    # Compare checkpoints
    if len(checkpoints) > 1:
        print(f"{'='*80}")
        print(f"  COMPARISON")
        print(f"{'='*80}\n")
        
        results = []
        for cp in checkpoints:
            checkpoint = torch.load(os.path.join(checkpoint_dir, cp), map_location='cpu')
            results.append({
                'name': cp,
                'epoch': checkpoint.get('epoch', 0),
                'acc': checkpoint.get('val_acc', 0),
                'loss': checkpoint.get('val_loss', 0)
            })
        
        # Sort by accuracy
        results.sort(key=lambda x: x['acc'], reverse=True)
        
        print("Ranking by Validation Accuracy:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['name']:<30} Epoch {r['epoch']:>2} | Acc: {r['acc']:>6.2f}% | Loss: {r['loss']:.4f}")
        
        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
