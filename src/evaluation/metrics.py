"""
Metrics Collection and Computation
Сбор и вычисление метрик для оценки архитектур
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
import time


class MetricsCollector:
    """Collect and compute various metrics during training"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.epoch_metrics = {}
        self.start_time = None
        self.epoch_start_time = None
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.epoch_metrics.clear()
    
    def start_timer(self):
        """Start timing"""
        self.start_time = time.time()
    
    def start_epoch(self):
        """Mark start of epoch"""
        self.epoch_start_time = time.time()
    
    def update(self, **kwargs):
        """Update metrics with new values"""
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)
    
    def end_epoch(self, epoch: int):
        """Compute epoch-level metrics"""
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        self.epoch_metrics[epoch] = {
            "epoch_time": epoch_time,
        }
        
        # Compute averages for this epoch
        for key, values in self.metrics.items():
            if values:
                self.epoch_metrics[epoch][f"avg_{key}"] = np.mean(values)
    
    def get_metric(self, metric_name: str, aggregation: str = "last") -> float:
        """Get specific metric value"""
        if metric_name not in self.metrics:
            return 0.0
        
        values = self.metrics[metric_name]
        if not values:
            return 0.0
        
        if aggregation == "last":
            return values[-1]
        elif aggregation == "mean":
            return np.mean(values)
        elif aggregation == "max":
            return np.max(values)
        elif aggregation == "min":
            return np.min(values)
        else:
            return values[-1]
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics"""
        summary = {}
        
        for key, values in self.metrics.items():
            if values:
                summary[f"{key}_mean"] = np.mean(values)
                summary[f"{key}_std"] = np.std(values)
                summary[f"{key}_last"] = values[-1]
        
        return summary


class RobustnessEvaluator:
    """Evaluate model robustness to corruptions"""
    
    def __init__(self, corruption_types: List[str] = None, 
                 severities: List[int] = None):
        self.corruption_types = corruption_types or [
            "gaussian_noise", "shot_noise", "impulse_noise",
            "defocus_blur", "motion_blur", "zoom_blur",
            "brightness", "contrast", "jpeg_compression"
        ]
        self.severities = severities or [1, 3, 5]
    
    def apply_corruption(self, images: torch.Tensor, 
                        corruption_type: str, 
                        severity: int) -> torch.Tensor:
        """Apply corruption to images"""
        
        corrupted = images.clone()
        
        if corruption_type == "gaussian_noise":
            noise = torch.randn_like(images) * (0.01 * severity)
            corrupted = images + noise
        
        elif corruption_type == "shot_noise":
            noise = torch.randn_like(images) * torch.sqrt(torch.abs(images)) * (0.01 * severity)
            corrupted = images + noise
        
        elif corruption_type == "impulse_noise":
            mask = torch.rand_like(images) < (0.01 * severity)
            impulse = torch.randint_like(images, 0, 2, dtype=images.dtype)
            corrupted = torch.where(mask, impulse.float(), images)
        
        elif corruption_type == "brightness":
            factor = 1.0 + (0.1 * severity)
            corrupted = images * factor
        
        elif corruption_type == "contrast":
            mean = images.mean(dim=[2, 3], keepdim=True)
            factor = 1.0 + (0.1 * severity)
            corrupted = (images - mean) * factor + mean
        
        # Add more corruption types as needed
        
        # Clamp to valid range
        corrupted = torch.clamp(corrupted, 0.0, 1.0)
        
        return corrupted
    
    def evaluate(self, model: nn.Module, dataloader, device: str = "cuda") -> Dict[str, float]:
        """Evaluate model on corrupted data"""
        
        model.eval()
        robustness_scores = {}
        
        with torch.no_grad():
            for corruption_type in self.corruption_types:
                corruption_accuracies = []
                
                for severity in self.severities:
                    correct = 0
                    total = 0
                    
                    for images, labels in dataloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        # Apply corruption
                        corrupted_images = self.apply_corruption(images, corruption_type, severity)
                        
                        # Evaluate
                        outputs = model(corrupted_images)
                        _, predicted = outputs.max(1)
                        
                        correct += predicted.eq(labels).sum().item()
                        total += labels.size(0)
                    
                    accuracy = correct / total if total > 0 else 0
                    corruption_accuracies.append(accuracy)
                
                # Average across severities
                robustness_scores[f"robustness_{corruption_type}"] = np.mean(corruption_accuracies)
        
        # Overall robustness score
        robustness_scores["robustness_score"] = np.mean(list(robustness_scores.values()))
        
        return robustness_scores


class ComputeEfficiencyMetrics:
    """Compute efficiency metrics (FLOPs, params, latency)"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def estimate_flops(model: nn.Module, input_size: tuple = (1, 3, 32, 32)) -> float:
        """Estimate FLOPs using improved counter"""
        try:
            from ..utils.flops_counter import estimate_flops
            return estimate_flops(model, input_size)
        except Exception as e:
            # Fallback to simple parameter-based estimation
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return params * 2.0 / 1e9  # GFLOPs
    
    @staticmethod
    def measure_latency(model: nn.Module, input_size: tuple = (1, 3, 32, 32),
                       device: str = "cuda", num_iterations: int = 100) -> float:
        """Measure inference latency"""
        
        model.eval()
        model = model.to(device)
        
        dummy_input = torch.randn(input_size).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Measure
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if device == "cuda" else None
        end_time = time.time()
        
        avg_latency_ms = (end_time - start_time) / num_iterations * 1000
        
        return avg_latency_ms
    
    @staticmethod
    def measure_memory_usage(model: nn.Module, input_size: tuple = (1, 3, 32, 32),
                            device: str = "cuda") -> Dict[str, float]:
        """Measure memory usage"""
        
        if device != "cuda":
            return {"memory_allocated_mb": 0, "memory_reserved_mb": 0}
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model = model.to(device)
        dummy_input = torch.randn(input_size).to(device)
        
        # Forward pass
        with torch.no_grad():
            _ = model(dummy_input)
        
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**2  # MB
        
        return {
            "memory_allocated_mb": memory_allocated,
            "memory_reserved_mb": memory_reserved
        }


class LearningCurveAnalyzer:
    """Analyze learning curves for convergence properties"""
    
    @staticmethod
    def compute_convergence_speed(train_losses: List[float], 
                                 threshold: float = 0.1) -> int:
        """Compute epochs to convergence"""
        
        if not train_losses or len(train_losses) < 2:
            return len(train_losses)
        
        # Find when loss improvement becomes small
        for i in range(1, len(train_losses)):
            improvement = abs(train_losses[i] - train_losses[i-1])
            if improvement < threshold:
                return i
        
        return len(train_losses)
    
    @staticmethod
    def compute_learning_curve_slope(accuracies: List[float], 
                                    window_size: int = 5) -> float:
        """Compute slope of learning curve"""
        
        if len(accuracies) < window_size:
            return 0.0
        
        # Use linear regression on last window_size points
        recent_accuracies = accuracies[-window_size:]
        x = np.arange(len(recent_accuracies))
        
        # Linear regression
        slope = np.polyfit(x, recent_accuracies, 1)[0]
        
        return float(slope)
    
    @staticmethod
    def detect_overfitting(train_accuracies: List[float],
                          val_accuracies: List[float]) -> float:
        """Compute generalization gap"""
        
        if not train_accuracies or not val_accuracies:
            return 0.0
        
        # Use last values
        train_acc = train_accuracies[-1]
        val_acc = val_accuracies[-1]
        
        gap = max(0, train_acc - val_acc)
        
        return gap


def compute_all_metrics(model: nn.Module, 
                       train_history: Dict[str, List[float]],
                       val_history: Dict[str, List[float]],
                       device: str = "cuda") -> Dict[str, float]:
    """Compute comprehensive metrics for a model"""
    
    metrics = {}
    
    # Performance metrics
    if "accuracy" in train_history:
        metrics["train_accuracy"] = train_history["accuracy"][-1] if train_history["accuracy"] else 0
    if "accuracy" in val_history:
        metrics["val_accuracy"] = val_history["accuracy"][-1] if val_history["accuracy"] else 0
    if "loss" in train_history:
        metrics["train_loss"] = train_history["loss"][-1] if train_history["loss"] else 0
    if "loss" in val_history:
        metrics["val_loss"] = val_history["loss"][-1] if val_history["loss"] else 0
    
    # Compute efficiency metrics
    metrics["parameter_count"] = ComputeEfficiencyMetrics.count_parameters(model)
    metrics["estimated_flops"] = ComputeEfficiencyMetrics.estimate_flops(model)
    
    try:
        metrics["inference_latency_ms"] = ComputeEfficiencyMetrics.measure_latency(model, device=device)
        mem_metrics = ComputeEfficiencyMetrics.measure_memory_usage(model, device=device)
        metrics.update(mem_metrics)
    except:
        metrics["inference_latency_ms"] = 0
        metrics["memory_allocated_mb"] = 0
    
    # Learning curve analysis
    if "accuracy" in val_history and len(val_history["accuracy"]) > 0:
        metrics["learning_curve_slope"] = LearningCurveAnalyzer.compute_learning_curve_slope(
            val_history["accuracy"]
        )
    
    if "loss" in train_history and len(train_history["loss"]) > 0:
        metrics["convergence_speed_epochs"] = LearningCurveAnalyzer.compute_convergence_speed(
            train_history["loss"]
        )
    
    if "accuracy" in train_history and "accuracy" in val_history:
        metrics["generalization_gap"] = LearningCurveAnalyzer.detect_overfitting(
            train_history["accuracy"], val_history["accuracy"]
        )
    
    # Compute efficiency score
    if metrics["parameter_count"] > 0 and "val_accuracy" in metrics:
        metrics["compute_efficiency"] = metrics["val_accuracy"] / (metrics["parameter_count"] / 1e6)
    else:
        metrics["compute_efficiency"] = 0
    
    return metrics
