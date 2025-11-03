"""
HuggingFace Dataset Loaders
Загрузка датасетов из HuggingFace Hub
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Dict, Optional
import numpy as np

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: 'datasets' library not installed. Install with: pip install datasets")


class HuggingFaceDatasetWrapper(Dataset):
    """Wrapper for HuggingFace datasets to work with PyTorch DataLoader"""
    
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get image and label
        if 'image' in item:
            image = item['image']
        elif 'img' in item:
            image = item['img']
        else:
            raise KeyError(f"No image key found in dataset. Keys: {item.keys()}")
        
        if 'label' in item:
            label = item['label']
        elif 'fine_label' in item:
            label = item['fine_label']
        else:
            raise KeyError(f"No label key found in dataset. Keys: {item.keys()}")
        
        # Convert to PIL if needed
        if not hasattr(image, 'convert'):
            from PIL import Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


class HuggingFaceDatasetFactory:
    """Factory for loading datasets from HuggingFace"""
    
    @staticmethod
    def get_cifar100(batch_size: int = 128, subset_size: Optional[int] = None,
                     streaming: bool = True) -> Dict[str, DataLoader]:
        """Load CIFAR-100 from HuggingFace
        
        Args:
            batch_size: Batch size for DataLoader
            subset_size: Limit dataset to this size (for Stage 1 proxy)
            streaming: Use streaming mode for large datasets
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("Install datasets: pip install datasets")
        
        # Transformations
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                               std=[0.2675, 0.2565, 0.2761])
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                               std=[0.2675, 0.2565, 0.2761])
        ])
        
        # Load dataset
        print(f"Loading CIFAR-100 from HuggingFace...")
        if streaming:
            dataset = load_dataset("uoft-cs/cifar100", streaming=True)
            # Take subset if specified
            if subset_size:
                train_dataset = dataset['train'].take(subset_size)
                test_dataset = dataset['test'].take(subset_size // 10)
        else:
            dataset = load_dataset("uoft-cs/cifar100")
            train_dataset = dataset['train']
            test_dataset = dataset['test']
            
            # Subset if specified
            if subset_size:
                indices = np.random.choice(len(train_dataset), 
                                         min(subset_size, len(train_dataset)), 
                                         replace=False)
                train_dataset = train_dataset.select(indices)
        
        # Wrap datasets
        train_wrapped = HuggingFaceDatasetWrapper(train_dataset, transform_train)
        test_wrapped = HuggingFaceDatasetWrapper(test_dataset, transform_test)
        
        # Create DataLoaders
        train_loader = DataLoader(train_wrapped, batch_size=batch_size, 
                                 shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_wrapped, batch_size=batch_size, 
                                shuffle=False, num_workers=2, pin_memory=True)
        
        return {
            "train": train_loader,
            "val": test_loader,
            "num_classes": 100
        }
    
    @staticmethod
    def get_tiny_imagenet(batch_size: int = 128, subset_size: Optional[int] = None) -> Dict[str, DataLoader]:
        """Load Tiny ImageNet from HuggingFace
        
        Args:
            batch_size: Batch size for DataLoader
            subset_size: Limit dataset to this size
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("Install datasets: pip install datasets")
        
        # Transformations for 64x64 images
        transform_train = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load dataset
        print(f"Loading Tiny ImageNet from HuggingFace...")
        dataset = load_dataset("zh-plus/tiny-imagenet")
        
        train_dataset = dataset['train']
        test_dataset = dataset['valid']
        
        # Subset if specified
        if subset_size:
            indices = np.random.choice(len(train_dataset), 
                                     min(subset_size, len(train_dataset)), 
                                     replace=False)
            train_dataset = train_dataset.select(indices)
        
        # Wrap datasets
        train_wrapped = HuggingFaceDatasetWrapper(train_dataset, transform_train)
        test_wrapped = HuggingFaceDatasetWrapper(test_dataset, transform_test)
        
        # Create DataLoaders
        train_loader = DataLoader(train_wrapped, batch_size=batch_size, 
                                 shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_wrapped, batch_size=batch_size, 
                                shuffle=False, num_workers=2, pin_memory=True)
        
        return {
            "train": train_loader,
            "val": test_loader,
            "num_classes": 200
        }
    
    @staticmethod
    def get_cifar100_c(batch_size: int = 128, corruption_type: str = "gaussian_noise",
                       severity: int = 3) -> DataLoader:
        """Load CIFAR-100-C (corrupted) for robustness testing
        
        Args:
            batch_size: Batch size for DataLoader
            corruption_type: Type of corruption
            severity: Severity level (1-5)
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("Install datasets: pip install datasets")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                               std=[0.2675, 0.2565, 0.2761])
        ])
        
        # Load corrupted dataset
        print(f"Loading CIFAR-100-C ({corruption_type}, severity={severity})...")
        dataset = load_dataset("randall-lab/cifar100-c", 
                             corruption_type=corruption_type,
                             severity=severity)
        
        test_dataset = dataset['test']
        test_wrapped = HuggingFaceDatasetWrapper(test_dataset, transform)
        
        test_loader = DataLoader(test_wrapped, batch_size=batch_size, 
                                shuffle=False, num_workers=2)
        
        return test_loader


# Export functions
def get_stage1_data(batch_size: int = 128) -> Dict[str, DataLoader]:
    """Get data for Stage 1 (Proxy Evaluation)
    
    Uses 10% of CIFAR-100 for fast screening
    STREAMING MODE: Данные загружаются по мере необходимости, не сохраняются на диск
    """
    return HuggingFaceDatasetFactory.get_cifar100(
        batch_size=batch_size,
        subset_size=5000,  # 10% of 50k
        streaming=True  # Streaming - не скачивает на диск!
    )


def get_stage2_data(batch_size: int = 128) -> Dict[str, DataLoader]:
    """Get data for Stage 2 (Refinement Training)
    
    Uses 50% of Tiny ImageNet for mid-level training
    """
    return HuggingFaceDatasetFactory.get_tiny_imagenet(
        batch_size=batch_size,
        subset_size=50000  # 50% of 100k
    )


def get_stage3_data(batch_size: int = 128) -> Dict[str, DataLoader]:
    """Get data for Stage 3 (Full Validation)
    
    Uses full CIFAR-100 for final validation
    STREAMING MODE: Данные загружаются по мере необходимости
    """
    return HuggingFaceDatasetFactory.get_cifar100(
        batch_size=batch_size,
        subset_size=None,  # Full dataset
        streaming=True  # Streaming - не скачивает на диск!
    )
