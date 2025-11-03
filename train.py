"""
Simple Neural Network Training on HuggingFace Datasets
Простое обучение нейросети на датасетах из HuggingFace
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import time
import json
from datetime import datetime

# Fix for Python 3.14 + datasets compatibility
os.environ['HF_DATASETS_DISABLE_CACHING'] = '1'

from src.data.huggingface_loaders import HuggingFaceDatasetFactory


class SimpleCNN(nn.Module):
    """Простая CNN для CIFAR-100"""
    def __init__(self, num_classes=100):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, max_batches=None):
    """Обучение одной эпохи"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0
    
    start_time = time.time()
    
    try:
        pbar = tqdm(dataloader['train'], desc=f"Train Epoch {epoch}", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            if max_batches and batch_idx >= max_batches:
                break
                
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
    except Exception as e:
        print(f"\nError during training: {e}")
        if num_batches == 0:
            raise
    
    elapsed_time = time.time() - start_time
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = 100. * correct / total if total > 0 else 0
    
    return avg_loss, accuracy, elapsed_time


def validate(model, dataloader, criterion, device, max_batches=None):
    """Валидация"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0
    
    start_time = time.time()
    
    try:
        with torch.no_grad():
            pbar = tqdm(dataloader['test'], desc="Validation", leave=False)
            for batch_idx, (images, labels) in enumerate(pbar):
                if max_batches and batch_idx >= max_batches:
                    break
                    
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*correct/total:.2f}%'
                    })
    except Exception as e:
        print(f"\nError during validation: {e}")
        if num_batches == 0:
            raise
    
    elapsed_time = time.time() - start_time
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = 100. * correct / total if total > 0 else 0
    
    return avg_loss, accuracy, elapsed_time


def main():
    parser = argparse.ArgumentParser(description="Train CNN on HuggingFace CIFAR-100")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Save directory")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit training batches per epoch")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit validation batches")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
    
    print(f"\n{'='*80}")
    print(f"  TRAINING CNN ON CIFAR-100 (HuggingFace)")
    print(f"{'='*80}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    if args.max_train_batches:
        print(f"  Max Train Batches: {args.max_train_batches}")
    if args.max_val_batches:
        print(f"  Max Val Batches: {args.max_val_batches}")
    print(f"{'='*80}\n")
    
    # Load dataset
    print("Loading CIFAR-100...")
    print("Note: Using torchvision (Python 3.14 compatible)")
    
    from torchvision import datasets, transforms
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # No augmentation for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    print("Downloading dataset (first run only, ~160MB)...")
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False),
        'test': DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    }
    print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test\n")
    
    # Model
    model = SimpleCNN(num_classes=100).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Resume from checkpoint
    start_epoch = 1
    best_acc = 0
    training_history = []
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint.get('val_acc', 0)
            print(f"✓ Resumed from epoch {checkpoint['epoch']} (best acc: {best_acc:.2f}%)\n")
        else:
            print(f"⚠ Checkpoint not found: {args.resume}\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_acc, train_time = train_epoch(
            model, dataloaders, criterion, optimizer, device, epoch, 
            max_batches=args.max_train_batches
        )
        
        # Validate
        val_loss, val_acc, val_time = validate(
            model, dataloaders, criterion, device,
            max_batches=args.max_val_batches
        )
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Time: {train_time:.1f}s")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Time: {val_time:.1f}s")
        print(f"  Epoch Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0],
            'time': epoch_time
        })
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (acc: {best_acc:.2f}%)")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint")
    
    # Save training history
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"  TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"  Best Validation Accuracy: {best_acc:.2f}%")
    print(f"  Models saved in: {args.save_dir}")
    print(f"  Training history: {history_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
