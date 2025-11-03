"""
Novelty Metrics
Вычисление новизны архитектур на основе структуры и поведения
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import networkx as nx

from ..search_space.search_space import ArchitectureGenome, ArchitectureEncoder


class ArchitecturalNoveltyComputer:
    """Compute novelty based on architectural structure"""
    
    @staticmethod
    def genome_to_graph(genome: ArchitectureGenome) -> nx.DiGraph:
        """Convert genome to directed graph"""
        
        G = nx.DiGraph()
        
        # Add nodes for each block
        for i, block in enumerate(genome.blocks):
            G.add_node(i, 
                      block_type=block.get("type"),
                      activation=block.get("activation"),
                      normalization=block.get("normalization"))
        
        # Add edges (sequential connections)
        for i in range(len(genome.blocks) - 1):
            G.add_edge(i, i + 1)
        
        return G
    
    @staticmethod
    def graph_edit_distance(genome1: ArchitectureGenome, 
                           genome2: ArchitectureGenome) -> float:
        """Compute graph edit distance between two genomes"""
        
        # Simplified graph edit distance based on structure
        g1_blocks = [b.get("type") for b in genome1.blocks]
        g2_blocks = [b.get("type") for b in genome2.blocks]
        
        # Levenshtein distance on block sequences
        m, n = len(g1_blocks), len(g2_blocks)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if g1_blocks[i-1] == g2_blocks[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        # Normalize by max length
        max_len = max(m, n)
        normalized_distance = dp[m][n] / max_len if max_len > 0 else 0
        
        return normalized_distance
    
    @staticmethod
    def compute_diversity_score(genome: ArchitectureGenome, 
                               population: List[ArchitectureGenome],
                               k_nearest: int = 15) -> float:
        """Compute diversity score relative to population"""
        
        if not population:
            return 1.0
        
        # Compute distances to all other genomes
        distances = []
        for other_genome in population:
            if genome.genome_id != other_genome.genome_id:
                dist = ArchitecturalNoveltyComputer.graph_edit_distance(genome, other_genome)
                distances.append(dist)
        
        if not distances:
            return 1.0
        
        # Average distance to k nearest neighbors
        distances.sort()
        k_distances = distances[:min(k_nearest, len(distances))]
        diversity = np.mean(k_distances)
        
        return float(diversity)


class BehavioralNoveltyComputer:
    """Compute novelty based on model behavior"""
    
    def __init__(self, reference_dataset=None):
        self.reference_dataset = reference_dataset
        self.activation_profiles = {}
    
    def extract_activation_profile(self, model: nn.Module, 
                                   dataloader, 
                                   device: str = "cuda",
                                   num_batches: int = 10) -> np.ndarray:
        """Extract activation profile from model"""
        
        model.eval()
        activations = []
        
        # Hook to capture intermediate activations
        activation_dict = {}
        
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activation_dict[name] = output.detach()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.MultiheadAttention)):
                hooks.append(module.register_forward_hook(get_activation(name)))
        
        # Run inference
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                images = images.to(device)
                _ = model(images)
                
                # Collect activations
                batch_activations = []
                for name in sorted(activation_dict.keys()):
                    act = activation_dict[name]
                    # Compute statistics
                    if len(act.shape) >= 2:
                        mean = act.mean().item()
                        std = act.std().item()
                        batch_activations.extend([mean, std])
                
                if batch_activations:
                    activations.append(batch_activations)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        if not activations:
            return np.zeros(10)
        
        # Average across batches and pad/truncate to fixed size
        profile = np.mean(activations, axis=0)
        
        # Pad or truncate to fixed size
        target_size = 128
        if len(profile) < target_size:
            profile = np.pad(profile, (0, target_size - len(profile)))
        else:
            profile = profile[:target_size]
        
        return profile
    
    def compute_activation_distance(self, profile1: np.ndarray, 
                                   profile2: np.ndarray) -> float:
        """Compute distance between activation profiles"""
        
        # Cosine distance
        dot_product = np.dot(profile1, profile2)
        norm1 = np.linalg.norm(profile1)
        norm2 = np.linalg.norm(profile2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        cosine_dist = 1 - cosine_sim
        
        return float(cosine_dist)
    
    def compute_behavioral_novelty(self, model: nn.Module,
                                   genome: ArchitectureGenome,
                                   population_profiles: Dict[str, np.ndarray],
                                   dataloader,
                                   device: str = "cuda",
                                   k_nearest: int = 15) -> float:
        """Compute behavioral novelty score"""
        
        # Extract activation profile for this model
        profile = self.extract_activation_profile(model, dataloader, device)
        
        # Store for future comparisons
        self.activation_profiles[genome.genome_id] = profile
        
        if not population_profiles:
            return 1.0
        
        # Compute distances to population
        distances = []
        for other_id, other_profile in population_profiles.items():
            if other_id != genome.genome_id:
                dist = self.compute_activation_distance(profile, other_profile)
                distances.append(dist)
        
        if not distances:
            return 1.0
        
        # Average distance to k nearest neighbors
        distances.sort()
        k_distances = distances[:min(k_nearest, len(distances))]
        novelty = np.mean(k_distances)
        
        return float(novelty)


class CombinedNoveltyMetric:
    """Combined architectural and behavioral novelty"""
    
    def __init__(self, architectural_weight: float = 0.4,
                 behavioral_weight: float = 0.6):
        self.architectural_weight = architectural_weight
        self.behavioral_weight = behavioral_weight
        
        self.architectural_computer = ArchitecturalNoveltyComputer()
        self.behavioral_computer = BehavioralNoveltyComputer()
    
    def compute_novelty(self, 
                       model: nn.Module,
                       genome: ArchitectureGenome,
                       population: List[ArchitectureGenome],
                       population_profiles: Dict[str, np.ndarray],
                       dataloader=None,
                       device: str = "cuda") -> Dict[str, float]:
        """Compute combined novelty score"""
        
        # Architectural novelty
        arch_novelty = self.architectural_computer.compute_diversity_score(
            genome, population
        )
        
        # Behavioral novelty (if dataloader provided)
        if dataloader is not None:
            behav_novelty = self.behavioral_computer.compute_behavioral_novelty(
                model, genome, population_profiles, dataloader, device
            )
        else:
            behav_novelty = 0.0
        
        # Combined score
        combined_novelty = (self.architectural_weight * arch_novelty + 
                          self.behavioral_weight * behav_novelty)
        
        return {
            "architectural_novelty": arch_novelty,
            "behavioral_novelty": behav_novelty,
            "combined_novelty": combined_novelty,
            "novelty_score": combined_novelty  # Alias
        }
    
    def get_activation_profiles(self) -> Dict[str, np.ndarray]:
        """Get all stored activation profiles"""
        return self.behavioral_computer.activation_profiles


class NoveltyArchive:
    """Archive of novel solutions for novelty search"""
    
    def __init__(self, archive_size: int = 100, 
                 novelty_threshold: float = 0.1):
        self.archive_size = archive_size
        self.novelty_threshold = novelty_threshold
        self.archive: List[Tuple[ArchitectureGenome, float]] = []
    
    def add(self, genome: ArchitectureGenome, novelty_score: float):
        """Add genome to archive if novel enough"""
        
        if novelty_score >= self.novelty_threshold:
            self.archive.append((genome, novelty_score))
            
            # Sort by novelty score (descending)
            self.archive.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only top entries
            if len(self.archive) > self.archive_size:
                self.archive = self.archive[:self.archive_size]
    
    def get_archive(self) -> List[ArchitectureGenome]:
        """Get archived genomes"""
        return [genome for genome, _ in self.archive]
    
    def get_average_novelty(self) -> float:
        """Get average novelty in archive"""
        if not self.archive:
            return 0.0
        return np.mean([score for _, score in self.archive])


def compute_population_diversity(population: List[ArchitectureGenome]) -> float:
    """Compute overall diversity of a population"""
    
    if len(population) < 2:
        return 0.0
    
    # Encode all genomes
    encodings = [ArchitectureEncoder.encode_genome(g) for g in population]
    
    # Compute pairwise distances
    distances = []
    for i in range(len(encodings)):
        for j in range(i + 1, len(encodings)):
            dist = np.linalg.norm(encodings[i] - encodings[j])
            distances.append(dist)
    
    # Average distance
    diversity = np.mean(distances) if distances else 0.0
    
    return float(diversity)
