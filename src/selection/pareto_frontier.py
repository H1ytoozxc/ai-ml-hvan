"""
Pareto Frontier Selection
Многокритериальный отбор на основе фронта Парето
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass

from ..search_space.search_space import ArchitectureGenome


@dataclass
class ParetoSolution:
    """Solution on Pareto frontier"""
    genome: ArchitectureGenome
    objectives: Dict[str, float]
    dominance_count: int = 0
    crowding_distance: float = 0.0


class ParetoFrontierSelector:
    """Select solutions based on Pareto optimality"""
    
    def __init__(self, objectives: List[str], 
                 maximize: List[bool] = None):
        """
        Args:
            objectives: List of objective names
            maximize: List of bools indicating whether to maximize each objective
                     (default: True for all)
        """
        self.objectives = objectives
        self.maximize = maximize if maximize else [True] * len(objectives)
    
    def dominates(self, solution1: Dict[str, float], 
                  solution2: Dict[str, float]) -> bool:
        """Check if solution1 dominates solution2"""
        
        better_in_at_least_one = False
        
        for obj, maximize in zip(self.objectives, self.maximize):
            val1 = solution1.get(obj, 0)
            val2 = solution2.get(obj, 0)
            
            if maximize:
                if val1 < val2:
                    return False
                if val1 > val2:
                    better_in_at_least_one = True
            else:
                if val1 > val2:
                    return False
                if val1 < val2:
                    better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def compute_pareto_fronts(self, 
                             population: List[Tuple[ArchitectureGenome, Dict[str, float]]]
                             ) -> List[List[ParetoSolution]]:
        """Compute Pareto fronts (NSGA-II style)"""
        
        solutions = [ParetoSolution(genome, objectives) 
                    for genome, objectives in population]
        
        fronts = []
        remaining = solutions.copy()
        
        while remaining:
            current_front = []
            
            # Find non-dominated solutions in remaining set
            for solution in remaining:
                is_dominated = False
                for other in remaining:
                    if solution != other and self.dominates(other.objectives, solution.objectives):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    current_front.append(solution)
            
            # Remove current front from remaining
            for solution in current_front:
                remaining.remove(solution)
            
            # Compute crowding distance for this front
            if len(current_front) > 0:
                self._compute_crowding_distance(current_front)
                fronts.append(current_front)
        
        return fronts
    
    def _compute_crowding_distance(self, front: List[ParetoSolution]):
        """Compute crowding distance for solutions in a front"""
        
        n = len(front)
        
        if n <= 2:
            for solution in front:
                solution.crowding_distance = float('inf')
            return
        
        # Initialize distances
        for solution in front:
            solution.crowding_distance = 0.0
        
        # For each objective
        for obj_idx, obj_name in enumerate(self.objectives):
            # Sort by this objective
            front.sort(key=lambda s: s.objectives.get(obj_name, 0))
            
            # Boundary points get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Compute range
            obj_min = front[0].objectives.get(obj_name, 0)
            obj_max = front[-1].objectives.get(obj_name, 0)
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # Compute crowding distance for intermediate points
            for i in range(1, n - 1):
                if front[i].crowding_distance != float('inf'):
                    prev_val = front[i-1].objectives.get(obj_name, 0)
                    next_val = front[i+1].objectives.get(obj_name, 0)
                    front[i].crowding_distance += (next_val - prev_val) / obj_range
    
    def select_top_n(self, 
                     population: List[Tuple[ArchitectureGenome, Dict[str, float]]],
                     n: int) -> List[ArchitectureGenome]:
        """Select top n solutions based on Pareto optimality and crowding distance"""
        
        fronts = self.compute_pareto_fronts(population)
        
        selected = []
        
        for front in fronts:
            if len(selected) + len(front) <= n:
                # Add entire front
                selected.extend([sol.genome for sol in front])
            else:
                # Sort by crowding distance and add remaining
                front.sort(key=lambda s: s.crowding_distance, reverse=True)
                remaining = n - len(selected)
                selected.extend([sol.genome for sol in front[:remaining]])
                break
        
        return selected
    
    def get_pareto_frontier(self,
                           population: List[Tuple[ArchitectureGenome, Dict[str, float]]]
                           ) -> List[ArchitectureGenome]:
        """Get first Pareto front only"""
        
        fronts = self.compute_pareto_fronts(population)
        
        if fronts:
            return [sol.genome for sol in fronts[0]]
        else:
            return []
    
    def visualize_pareto_frontier(self, 
                                 population: List[Tuple[ArchitectureGenome, Dict[str, float]]],
                                 save_path: str = "pareto_frontier.png",
                                 obj_x: str = None,
                                 obj_y: str = None):
        """Visualize Pareto frontier (2D or 3D)"""
        
        if len(self.objectives) < 2:
            print("Need at least 2 objectives to visualize")
            return
        
        # Use first two objectives if not specified
        obj_x = obj_x or self.objectives[0]
        obj_y = obj_y or self.objectives[1]
        
        fronts = self.compute_pareto_fronts(population)
        
        plt.figure(figsize=(10, 8))
        
        # Plot each front with different colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(fronts)))
        
        for i, (front, color) in enumerate(zip(fronts, colors)):
            x_vals = [sol.objectives.get(obj_x, 0) for sol in front]
            y_vals = [sol.objectives.get(obj_y, 0) for sol in front]
            
            label = f"Front {i+1}" if i == 0 else f"Front {i+1}"
            plt.scatter(x_vals, y_vals, c=[color], label=label, s=100, alpha=0.7)
        
        plt.xlabel(obj_x, fontsize=12)
        plt.ylabel(obj_y, fontsize=12)
        plt.title("Pareto Frontier Visualization", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()


class DynamicPercentileSelector:
    """Dynamic percentile-based selection"""
    
    def __init__(self, percentile: float = 15.0,
                 min_candidates: int = 250,
                 max_candidates: int = 600,
                 novelty_weight: float = 0.25,
                 diversity_weight: float = 0.35):
        self.percentile = percentile
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates
        self.novelty_weight = novelty_weight
        self.diversity_weight = diversity_weight
    
    def select(self, 
              population: List[Tuple[ArchitectureGenome, Dict[str, float]]],
              primary_metric: str = "proxy_accuracy") -> List[ArchitectureGenome]:
        """Select candidates based on composite score"""
        
        # Compute composite scores
        scored_population = []
        
        for genome, metrics in population:
            # Primary metric
            primary_score = metrics.get(primary_metric, 0)
            
            # Novelty score
            novelty_score = metrics.get("novelty_score", 0)
            
            # Diversity contribution (could be computed)
            diversity_score = metrics.get("diversity_score", 0)
            
            # Composite score
            composite = (primary_score * (1 - self.novelty_weight - self.diversity_weight) +
                        novelty_score * self.novelty_weight +
                        diversity_score * self.diversity_weight)
            
            scored_population.append((genome, composite))
        
        # Sort by composite score
        scored_population.sort(key=lambda x: x[1], reverse=True)
        
        # Determine number to select
        num_to_select = int(len(population) * (self.percentile / 100.0))
        num_to_select = max(self.min_candidates, min(self.max_candidates, num_to_select))
        
        # Select top candidates
        selected = [genome for genome, _ in scored_population[:num_to_select]]
        
        return selected


class HybridSelector:
    """Hybrid selection combining multiple strategies"""
    
    def __init__(self, strategies: List[str] = None):
        self.strategies = strategies or ["pareto", "elite", "novelty"]
    
    def select(self,
              population: List[Tuple[ArchitectureGenome, Dict[str, float]]],
              n: int,
              objectives: List[str]) -> List[ArchitectureGenome]:
        """Hybrid selection"""
        
        selected = []
        n_per_strategy = n // len(self.strategies)
        
        for strategy in self.strategies:
            if strategy == "pareto":
                # Pareto-based selection
                selector = ParetoFrontierSelector(objectives)
                strategy_selected = selector.select_top_n(population, n_per_strategy)
            
            elif strategy == "elite":
                # Elite selection based on first objective
                sorted_pop = sorted(population, 
                                  key=lambda x: x[1].get(objectives[0], 0),
                                  reverse=True)
                strategy_selected = [genome for genome, _ in sorted_pop[:n_per_strategy]]
            
            elif strategy == "novelty":
                # Novelty-based selection
                sorted_pop = sorted(population,
                                  key=lambda x: x[1].get("novelty_score", 0),
                                  reverse=True)
                strategy_selected = [genome for genome, _ in sorted_pop[:n_per_strategy]]
            
            else:
                strategy_selected = []
            
            selected.extend(strategy_selected)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_selected = []
        for genome in selected:
            if genome.genome_id not in seen:
                seen.add(genome.genome_id)
                unique_selected.append(genome)
        
        # If we need more, add from remaining population
        if len(unique_selected) < n:
            remaining_genomes = [g for g, _ in population if g.genome_id not in seen]
            unique_selected.extend(remaining_genomes[:n - len(unique_selected)])
        
        return unique_selected[:n]


def compute_hypervolume(front: List[Dict[str, float]],
                       objectives: List[str],
                       reference_point: Dict[str, float]) -> float:
    """Compute hypervolume indicator (simplified)"""
    
    if not front:
        return 0.0
    
    # This is a simplified 2D hypervolume computation
    # For production, use pygmo or similar library
    
    if len(objectives) != 2:
        return 0.0
    
    obj1, obj2 = objectives
    
    # Sort by first objective
    sorted_front = sorted(front, key=lambda x: x.get(obj1, 0))
    
    hypervolume = 0.0
    ref1 = reference_point.get(obj1, 0)
    ref2 = reference_point.get(obj2, 0)
    
    for i, point in enumerate(sorted_front):
        val1 = point.get(obj1, 0)
        val2 = point.get(obj2, 0)
        
        # Width
        if i == 0:
            width = ref1 - val1
        else:
            width = sorted_front[i-1].get(obj1, 0) - val1
        
        # Height
        height = val2 - ref2
        
        if width > 0 and height > 0:
            hypervolume += width * height
    
    return hypervolume
