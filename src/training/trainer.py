"""
Training Pipeline
Обучение и оценка архитектур
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from ..search_space.search_space import ArchitectureGenome
from ..search_space.architecture_generator import ArchitectureBuilder
from ..evaluation.metrics import MetricsCollector, compute_all_metrics


class ArchitectureTrainer:
    """Train and evaluate a single architecture"""
    
    def __init__(self, 
                 genome: ArchitectureGenome,
                 num_classes: int = 100,
                 device: str = "cuda"):
        
        self.genome = genome
        self.num_classes = num_classes
        self.device = device
        
        # Build model
        self.model = ArchitectureBuilder.build(
            genome, 
            num_classes=num_classes
        ).to(device)
        
        # Get optimizer and scheduler
        self.optimizer = ArchitectureBuilder.get_optimizer(self.model, genome)
        self.scheduler = None
        self.loss_fn = ArchitectureBuilder.get_loss_function(genome)
        
        # Metrics
        self.metrics_collector = MetricsCollector()
        
        # History
        self.train_history = {
            "loss": [],
            "accuracy": []
        }
        self.val_history = {
            "loss": [],
            "accuracy": []
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        self.metrics_collector.start_epoch()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0:
                self.metrics_collector.update(
                    batch_loss=loss.item(),
                    batch_accuracy=100. * correct / total
                )
        
        # Epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        
        self.train_history["loss"].append(epoch_loss)
        self.train_history["accuracy"].append(epoch_accuracy)
        
        self.metrics_collector.end_epoch(epoch)
        
        return {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = total_loss / len(val_loader)
        val_accuracy = 100. * correct / total
        
        self.val_history["loss"].append(val_loss)
        self.val_history["accuracy"].append(val_accuracy)
        
        return {
            "loss": val_loss,
            "accuracy": val_accuracy
        }
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              early_stopping: bool = False,
              patience: int = 5) -> Dict[str, Any]:
        """Full training loop"""
        
        # Initialize scheduler
        self.scheduler = ArchitectureBuilder.get_scheduler(
            self.optimizer, self.genome, num_epochs
        )
        
        best_val_accuracy = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
            
            # Early stopping
            if early_stopping:
                if val_metrics["accuracy"] > best_val_accuracy:
                    best_val_accuracy = val_metrics["accuracy"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Compute final metrics
        final_metrics = compute_all_metrics(
            self.model,
            self.train_history,
            self.val_history,
            self.device
        )
        
        return final_metrics
    
    def get_model(self) -> nn.Module:
        """Get trained model"""
        return self.model


class FastProxyEvaluator:
    """Fast proxy evaluation for Stage 1"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def evaluate(self,
                genome: ArchitectureGenome,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_classes: int = 10) -> Dict[str, float]:
        """Quick evaluation on proxy dataset"""
        
        try:
            # Build model
            model = ArchitectureBuilder.build(
                genome,
                num_classes=num_classes
            ).to(self.device)
            
            optimizer = ArchitectureBuilder.get_optimizer(model, genome)
            loss_fn = ArchitectureBuilder.get_loss_function(genome)
            
            # Single epoch training
            model.train()
            total_loss = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            val_loss = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            proxy_accuracy = 100. * correct / total if total > 0 else 0
            proxy_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
            
            # Estimate complexity
            parameter_count = sum(p.numel() for p in model.parameters())
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
            return {
                "proxy_accuracy": proxy_accuracy,
                "proxy_loss": proxy_loss,
                "parameter_count": parameter_count,
                "estimated_FLOPs": parameter_count * 2.0 / 1e9  # Rough estimate
            }
        
        except Exception as e:
            print(f"Error evaluating genome {genome.genome_id}: {e}")
            return {
                "proxy_accuracy": 0,
                "proxy_loss": float('inf'),
                "parameter_count": 0,
                "estimated_FLOPs": 0
            }


class WeightSharingEvaluator:
    """Evaluator using weight sharing (one-shot supernet)
    
    Note: This is a simplified evaluator. For production use of weight sharing,
    consider implementing proper supernet mechanisms like:
    - ENAS-style parameter sharing
    - DARTS-style differentiable architecture search
    - Single-path one-shot NAS
    """
    
    def __init__(self, supernet: nn.Module, device: str = "cuda"):
        self.supernet = supernet
        self.device = device
    
    def evaluate(self,
                genome: ArchitectureGenome,
                val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate architecture using shared weights
        
        Current implementation: Direct evaluation without weight sharing.
        For true weight sharing, integrate with specialized NAS frameworks.
        """
        
        self.supernet.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.supernet(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total if total > 0 else 0
        
        return {
            "proxy_accuracy": accuracy,
            "parameter_count": sum(p.numel() for p in self.supernet.parameters())
        }


class SurrogateModelPredictor:
    """Predict final performance using surrogate model"""
    
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = None
        self.is_trained = False
    
    def build_model(self, input_dim: int):
        """Build GNN surrogate model"""
        
        # Simplified - would use proper GNN in production
        layers = []
        current_dim = input_dim
        
        for _ in range(self.num_layers):
            layers.append(nn.Linear(current_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            current_dim = self.hidden_dim
        
        # Output: predict final accuracy and convergence speed
        layers.append(nn.Linear(current_dim, 2))
        
        self.model = nn.Sequential(*layers)
    
    def train_surrogate(self, 
                       training_data: List[Tuple[np.ndarray, np.ndarray]],
                       num_epochs: int = 50):
        """Train surrogate model on historical data"""
        
        if not training_data:
            return
        
        X = torch.FloatTensor([x for x, _ in training_data])
        y = torch.FloatTensor([y for _, y in training_data])
        
        if self.model is None:
            self.build_model(X.shape[1])
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            predictions = self.model(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
        
        self.is_trained = True
    
    def predict(self, genome_encoding: np.ndarray) -> Dict[str, float]:
        """Predict final performance"""
        
        if not self.is_trained or self.model is None:
            return {"predicted_accuracy": 0, "predicted_convergence_speed": 0}
        
        self.model.eval()
        
        with torch.no_grad():
            X = torch.FloatTensor(genome_encoding).unsqueeze(0)
            predictions = self.model(X).squeeze()
        
        return {
            "predicted_accuracy": predictions[0].item(),
            "predicted_convergence_speed": predictions[1].item()
        }
