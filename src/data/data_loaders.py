"""
Data Loading Utilities
Загрузка датасетов для обучения и валидации
"""

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Dict


class DatasetFactory:
    """Factory for creating dataset loaders"""
    
    @staticmethod
    def get_cifar10(batch_size: int = 128,
                   subset_percent: float = None,
                   num_workers: int = 2) -> Dict[str, DataLoader]:
        """Get CIFAR-10 dataloaders"""
        
        # Transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # Download datasets
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        # Subset if needed
        if subset_percent is not None and subset_percent < 100:
            num_train = int(len(trainset) * (subset_percent / 100.0))
            indices = DatasetFactory._class_balanced_sample(trainset, num_train)
            trainset = Subset(trainset, indices)
        
        # Create dataloaders
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return {
            "train": train_loader,
            "val": test_loader,
            "test": test_loader
        }
    
    @staticmethod
    def get_cifar100(batch_size: int = 128,
                    subset_percent: float = None,
                    num_workers: int = 2) -> Dict[str, DataLoader]:
        """Get CIFAR-100 dataloaders"""
        
        # Transforms with augmentation
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        # Download datasets
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train
        )
        
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        # Subset if needed
        if subset_percent is not None and subset_percent < 100:
            num_train = int(len(trainset) * (subset_percent / 100.0))
            indices = DatasetFactory._class_balanced_sample(trainset, num_train)
            trainset = Subset(trainset, indices)
        
        # Create dataloaders
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return {
            "train": train_loader,
            "val": test_loader,
            "test": test_loader
        }
    
    @staticmethod
    def get_svhn(batch_size: int = 128,
                num_workers: int = 2) -> Dict[str, DataLoader]:
        """Get SVHN dataloaders"""
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        trainset = torchvision.datasets.SVHN(
            root='./data', split='train', download=True, transform=transform
        )
        
        testset = torchvision.datasets.SVHN(
            root='./data', split='test', download=True, transform=transform
        )
        
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return {
            "train": train_loader,
            "val": test_loader,
            "test": test_loader
        }
    
    @staticmethod
    def _class_balanced_sample(dataset, num_samples: int) -> np.ndarray:
        """Sample indices in a class-balanced way"""
        
        # Get labels
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
        else:
            # Fallback: random sampling
            return np.random.choice(len(dataset), num_samples, replace=False)
        
        num_classes = len(np.unique(labels))
        samples_per_class = num_samples // num_classes
        
        indices = []
        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            selected = np.random.choice(
                class_indices,
                min(samples_per_class, len(class_indices)),
                replace=False
            )
            indices.extend(selected.tolist())
        
        return np.array(indices)
    
    @staticmethod
    def get_all_dataloaders(config) -> Dict[str, Dict[str, DataLoader]]:
        """Get all dataloaders needed for the pipeline"""
        
        dataloaders = {}
        
        # CIFAR-10 for Stage 1 (proxy)
        dataloaders["CIFAR-10"] = DatasetFactory.get_cifar10(
            batch_size=128,
            subset_percent=10,  # 10% for fast proxy evaluation
            num_workers=2
        )
        
        # CIFAR-100 for Stage 2 and 3
        dataloaders["CIFAR-100"] = DatasetFactory.get_cifar100(
            batch_size=128,
            subset_percent=50,  # 50% for Stage 2
            num_workers=2
        )
        
        # Full CIFAR-100 for Stage 3
        dataloaders["CIFAR-100-Full"] = DatasetFactory.get_cifar100(
            batch_size=128,
            subset_percent=None,  # Full dataset
            num_workers=2
        )
        
        # SVHN for cross-validation
        dataloaders["SVHN"] = DatasetFactory.get_svhn(
            batch_size=128,
            num_workers=2
        )
        
        return dataloaders


class AugmentationPipeline:
    """Advanced augmentation pipelines"""
    
    @staticmethod
    def get_autoaugment():
        """Get AutoAugment policy"""
        return transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)
    
    @staticmethod
    def get_randaugment(n: int = 2, m: int = 9):
        """Get RandAugment policy"""
        return transforms.RandAugment(num_ops=n, magnitude=m)
    
    @staticmethod
    def get_combined_augmentation():
        """Get combined augmentation pipeline"""
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            transforms.RandomErasing(p=0.5)
        ])
