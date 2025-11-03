"""
Main Orchestrator
Основной оркестратор для управления всем процессом эволюции
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Any, Tuple
import os
from tqdm import tqdm
import wandb

from .config import EvoArchitectConfig
from .search_space.search_space import ConditionalSearchSpace, ArchitectureGenome
from .search_space.architecture_generator import ArchitectureBuilder
from .training.trainer import FastProxyEvaluator, ArchitectureTrainer
from .evaluation.metrics import RobustnessEvaluator, compute_all_metrics
from .selection.novelty_metrics import CombinedNoveltyMetric
from .selection.pareto_frontier import (
    ParetoFrontierSelector, DynamicPercentileSelector
)
from .meta_optimization.mutations import (
    get_mutation_operator, CrossoverOperator, ALL_MUTATIONS
)
from .meta_optimization.rl_controller import REINFORCEController
from .knowledge_base.database import KnowledgeBase


def get_config_value(config, path, default=None):
    """Safely get nested config value supporting both dict and object access"""
    parts = path.split('.')
    current = config
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


class EvoArchitectOrchestrator:
    """Main orchestrator for autonomous architecture evolution"""
    
    def __init__(self, config: EvoArchitectConfig):
        self.config = config
        
        # Initialize components
        search_space_config = get_config_value(config, 'search_space', {})
        self.search_space = ConditionalSearchSpace(search_space_config)
        
        kb_path = get_config_value(config, 'knowledge_base_path', './evo_runs/knowledge_base.db')
        self.knowledge_base = KnowledgeBase(kb_path)
        
        # Novelty metric
        self.novelty_metric = CombinedNoveltyMetric(
            architectural_weight=0.4,
            behavioral_weight=0.6
        )
        
        # Meta-optimization controller
        meta_enabled = get_config_value(config, 'meta_optimization.enabled', False)
        if meta_enabled:
            self.rl_controller = REINFORCEController(
                state_dim=64,
                action_dim=len(ALL_MUTATIONS),
                hidden_dim=get_config_value(config, 'meta_optimization.hidden_dim', 256),
                learning_rate=get_config_value(config, 'meta_optimization.learning_rate', 0.001)
            )
        else:
            self.rl_controller = None
        
        # Population
        self.current_population: List[ArchitectureGenome] = []
        self.population_metrics: Dict[str, Dict[str, float]] = {}
        
        # Activation profiles for behavioral novelty
        self.activation_profiles: Dict[str, np.ndarray] = {}
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # WandB logging
        self.use_wandb = False
        dashboard_provider = get_config_value(config, 'logging.dashboard_provider', None)
        if dashboard_provider == "WeightsAndBiases":
            try:
                wandb_project = get_config_value(config, 'logging.wandb_project', 'evo-architect')
                wandb_entity = get_config_value(config, 'logging.wandb_entity', None)
                
                # Convert config to dict if possible
                config_dict = config.to_dict() if hasattr(config, 'to_dict') else {}
                
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    config=config_dict
                )
                self.use_wandb = True
            except Exception as e:
                print(f"Warning: Failed to initialize WandB: {e}")
                print("Continuing without WandB logging...")
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to WandB if enabled"""
        if self.use_wandb:
            try:
                wandb.log(metrics)
            except Exception as e:
                print(f"Warning: Failed to log metrics to WandB: {e}")
    
    def initialize_population(self) -> List[ArchitectureGenome]:
        """Generate initial population"""
        
        print(f"Generating initial population of {self.config.initial_population_size} architectures...")
        
        population = []
        for i in range(self.config.initial_population_size):
            genome = self.search_space.sample_architecture()
            genome.generation = 0
            population.append(genome)
            
            # Save to knowledge base
            self.knowledge_base.add_architecture(genome)
        
        return population
    
    def stage1_proxy_evaluation(self, 
                                population: List[ArchitectureGenome],
                                train_loader: DataLoader,
                                val_loader: DataLoader) -> List[ArchitectureGenome]:
        """Stage 1: Proxy evaluation on subset of data"""
        
        print("\n" + "="*80)
        print("STAGE 1: Proxy Evaluation")
        print("="*80)
        
        evaluator = FastProxyEvaluator(device=self.device)
        
        # Evaluate all candidates
        evaluated_population = []
        
        for genome in tqdm(population, desc="Evaluating candidates"):
            metrics = evaluator.evaluate(
                genome,
                train_loader,
                val_loader,
                num_classes=10  # CIFAR-10 for proxy
            )
            
            # Compute novelty
            novelty_metrics = self.novelty_metric.compute_novelty(
                model=None,  # Skip behavioral for Stage 1
                genome=genome,
                population=self.current_population,
                population_profiles={},
                dataloader=None
            )
            
            # Combine metrics
            combined_metrics = {**metrics, **novelty_metrics}
            
            # Store
            genome.metrics = combined_metrics
            self.population_metrics[genome.genome_id] = combined_metrics
            
            # Save to knowledge base
            self.knowledge_base.add_evaluation(
                genome.genome_id,
                "Stage1_Proxy_Evaluation",
                combined_metrics
            )
            self.knowledge_base.add_novelty_score(
                genome.genome_id,
                novelty_metrics
            )
            
            evaluated_population.append((genome, combined_metrics))
        
        # Select top candidates
        selector = DynamicPercentileSelector(
            percentile=15.0,
            min_candidates=250,
            max_candidates=600,
            novelty_weight=0.25,
            diversity_weight=0.35
        )
        
        selected = selector.select(evaluated_population, primary_metric="proxy_accuracy")
        
        print(f"Selected {len(selected)} candidates from Stage 1")
        
        # Log to WandB
        self._log_metrics({
            "stage1/num_evaluated": len(population),
            "stage1/num_selected": len(selected),
            "stage1/mean_accuracy": np.mean([m["proxy_accuracy"] for _, m in evaluated_population]),
            "stage1/max_accuracy": np.max([m["proxy_accuracy"] for _, m in evaluated_population])
        })
        
        return selected
    
    def stage2_refinement_training(self,
                                  population: List[ArchitectureGenome],
                                  train_loader: DataLoader,
                                  val_loader: DataLoader) -> List[ArchitectureGenome]:
        """Stage 2: Medium training on larger dataset"""
        
        print("\n" + "="*80)
        print("STAGE 2: Refinement Training")
        print("="*80)
        
        evaluated_population = []
        
        for genome in tqdm(population, desc="Training candidates"):
            try:
                # Train
                trainer = ArchitectureTrainer(
                    genome,
                    num_classes=100,  # CIFAR-100
                    device=self.device
                )
                
                metrics = trainer.train(
                    train_loader,
                    val_loader,
                    num_epochs=self.config.stage2_config.epochs,
                    early_stopping=self.config.stage2_config.early_stopping,
                    patience=self.config.stage2_config.early_stopping_patience
                )
                
                # Compute robustness
                robustness_eval = RobustnessEvaluator(severities=[1, 3, 5])
                robustness_metrics = robustness_eval.evaluate(
                    trainer.get_model(),
                    val_loader,
                    device=self.device
                )
                
                # Compute novelty (including behavioral)
                novelty_metrics = self.novelty_metric.compute_novelty(
                    model=trainer.get_model(),
                    genome=genome,
                    population=self.current_population,
                    population_profiles=self.activation_profiles,
                    dataloader=val_loader,
                    device=self.device
                )
                
                # Update activation profiles
                self.activation_profiles.update(
                    self.novelty_metric.get_activation_profiles()
                )
                
                # Combine metrics
                combined_metrics = {**metrics, **robustness_metrics, **novelty_metrics}
                
                # Store
                genome.metrics = combined_metrics
                self.population_metrics[genome.genome_id] = combined_metrics
                
                # Save to knowledge base
                self.knowledge_base.add_evaluation(
                    genome.genome_id,
                    "Stage2_Refinement_Training",
                    combined_metrics
                )
                self.knowledge_base.add_novelty_score(
                    genome.genome_id,
                    novelty_metrics,
                    self.activation_profiles.get(genome.genome_id)
                )
                
                evaluated_population.append((genome, combined_metrics))
            
            except Exception as e:
                print(f"Error training genome {genome.genome_id}: {e}")
                continue
        
        # Pareto frontier selection
        objectives = ["val_accuracy", "novelty_score", "robustness_score", "learning_curve_slope"]
        selector = ParetoFrontierSelector(objectives)
        
        selected = selector.select_top_n(evaluated_population, n=100)
        
        # Save Pareto front
        pareto_objectives = [
            {obj: genome.metrics.get(obj, 0) for obj in objectives}
            for genome in selected
        ]
        self.knowledge_base.save_pareto_front(
            generation=1,
            stage="Stage2_Refinement_Training",
            genome_ids=[g.genome_id for g in selected],
            objectives=pareto_objectives
        )
        
        # Visualize Pareto frontier
        selector.visualize_pareto_frontier(
            evaluated_population,
            save_path=os.path.join(self.config.logging.artifact_save_path, "pareto_stage2.png"),
            obj_x="val_accuracy",
            obj_y="novelty_score"
        )
        
        print(f"Selected {len(selected)} candidates from Stage 2 (Pareto frontier)")
        
        # Log to WandB
        self._log_metrics({
            "stage2/num_evaluated": len(population),
            "stage2/num_selected": len(selected),
            "stage2/mean_accuracy": np.mean([m["val_accuracy"] for _, m in evaluated_population]),
            "stage2/max_accuracy": np.max([m["val_accuracy"] for _, m in evaluated_population]),
            "stage2/mean_novelty": np.mean([m["novelty_score"] for _, m in evaluated_population])
        })
        
        return selected
    
    def stage3_full_validation(self,
                              population: List[ArchitectureGenome],
                              dataloaders: Dict[str, Tuple[DataLoader, DataLoader]]) -> List[ArchitectureGenome]:
        """Stage 3: Full training and cross-dataset validation"""
        
        print("\n" + "="*80)
        print("STAGE 3: Full Validation")
        print("="*80)
        
        evaluated_population = []
        
        for genome in tqdm(population, desc="Full validation"):
            try:
                # Train on primary dataset (CIFAR-100)
                train_loader, val_loader = dataloaders["CIFAR-100"]
                
                trainer = ArchitectureTrainer(
                    genome,
                    num_classes=100,
                    device=self.device
                )
                
                metrics = trainer.train(
                    train_loader,
                    val_loader,
                    num_epochs=self.config.stage3_config.epochs,
                    early_stopping=False
                )
                
                # Cross-validate on other datasets
                for dataset_name, (_, test_loader) in dataloaders.items():
                    if dataset_name != "CIFAR-100":
                        test_metrics = trainer.validate(test_loader)
                        metrics[f"{dataset_name}_accuracy"] = test_metrics["accuracy"]
                
                # Robustness
                robustness_eval = RobustnessEvaluator(severities=[1, 3, 5])
                robustness_metrics = robustness_eval.evaluate(
                    trainer.get_model(),
                    val_loader,
                    device=self.device
                )
                
                # Novelty
                novelty_metrics = self.novelty_metric.compute_novelty(
                    model=trainer.get_model(),
                    genome=genome,
                    population=self.current_population,
                    population_profiles=self.activation_profiles,
                    dataloader=val_loader,
                    device=self.device
                )
                
                # Combine
                combined_metrics = {**metrics, **robustness_metrics, **novelty_metrics}
                combined_metrics["novelty_score_combined"] = novelty_metrics["combined_novelty"]
                
                # Store
                genome.metrics = combined_metrics
                self.population_metrics[genome.genome_id] = combined_metrics
                
                # Save to knowledge base
                self.knowledge_base.add_evaluation(
                    genome.genome_id,
                    "Stage3_Full_Validation",
                    combined_metrics
                )
                
                evaluated_population.append((genome, combined_metrics))
            
            except Exception as e:
                print(f"Error in full validation for {genome.genome_id}: {e}")
                continue
        
        # Final Pareto selection
        objectives = ["val_accuracy", "novelty_score_combined", "robustness_score", 
                     "compute_efficiency", "generalization_gap"]
        selector = ParetoFrontierSelector(objectives)
        
        final_top10 = selector.select_top_n(evaluated_population, n=10)
        
        # Save final Pareto front
        pareto_objectives = [
            {obj: genome.metrics.get(obj, 0) for obj in objectives}
            for genome in final_top10
        ]
        self.knowledge_base.save_pareto_front(
            generation=1,
            stage="Stage3_Full_Validation",
            genome_ids=[g.genome_id for g in final_top10],
            objectives=pareto_objectives
        )
        
        # Visualize
        selector.visualize_pareto_frontier(
            evaluated_population,
            save_path=os.path.join(self.config.logging.artifact_save_path, "pareto_final.png"),
            obj_x="val_accuracy",
            obj_y="novelty_score_combined"
        )
        
        print(f"\nFinal top-{len(final_top10)} models selected")
        
        # Log to WandB
        self._log_metrics({
            "stage3/num_evaluated": len(population),
            "stage3/final_top_n": len(final_top10),
            "stage3/best_accuracy": np.max([m["val_accuracy"] for _, m in evaluated_population]),
            "stage3/mean_novelty": np.mean([m["novelty_score_combined"] for _, m in evaluated_population])
        })
        
        return final_top10
    
    def evolve_population(self, 
                         population: List[ArchitectureGenome],
                         generation: int) -> List[ArchitectureGenome]:
        """Evolve population using mutations and crossover"""
        
        new_population = []
        
        # Keep elite
        elite_size = max(10, len(population) // 10)
        sorted_pop = sorted(population, 
                          key=lambda g: g.metrics.get("val_accuracy", 0),
                          reverse=True)
        new_population.extend(sorted_pop[:elite_size])
        
        # Generate offspring
        crossover_op = CrossoverOperator(crossover_rate=0.7)
        
        while len(new_population) < self.config.initial_population_size:
            # Select parents (tournament selection)
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Crossover
            child = crossover_op(parent1, parent2)
            child.generation = generation
            
            # Mutation (adaptive based on RL controller)
            if self.rl_controller:
                # Use RL controller to select mutation
                state = self._encode_population_state(population, generation)
                action, log_prob = self.rl_controller.select_action(state)
                
                mutation_name = list(ALL_MUTATIONS.keys())[action]
                mutation_op = get_mutation_operator(mutation_name, mutation_rate=0.3)
                
                child = mutation_op(child, self.search_space)
            else:
                # Random mutation
                mutation_name = np.random.choice(list(ALL_MUTATIONS.keys()))
                mutation_op = get_mutation_operator(mutation_name)
                child = mutation_op(child, self.search_space)
            
            # Validate and add
            is_valid, errors = self.search_space.validate_architecture(child)
            if is_valid:
                new_population.append(child)
                self.knowledge_base.add_architecture(child)
        
        return new_population[:self.config.initial_population_size]
    
    def _tournament_selection(self, population: List[ArchitectureGenome],
                             tournament_size: int = 3) -> ArchitectureGenome:
        """Tournament selection"""
        
        tournament = np.random.choice(population, size=tournament_size, replace=False)
        return max(tournament, key=lambda g: g.metrics.get("val_accuracy", 0))
    
    def _encode_population_state(self, population: List[ArchitectureGenome],
                                generation: int) -> torch.Tensor:
        """Encode population state for RL controller"""
        
        if not population:
            return torch.zeros(64)
        
        accuracies = [g.metrics.get("val_accuracy", 0) for g in population]
        novelties = [g.metrics.get("novelty_score", 0) for g in population]
        
        stats = {
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "max_accuracy": np.max(accuracies),
            "mean_novelty": np.mean(novelties),
            "diversity": np.std(novelties),
            "mean_complexity": np.mean([len(g.blocks) for g in population]),
            "pareto_size": len(population),
            "recent_improvement": 0.0
        }
        
        return self.rl_controller.encode_state(stats, generation, 10)
    
    def run(self, dataloaders: Dict[str, Any]):
        """Run complete evolution pipeline"""
        
        print("="*80)
        print("EvoArchitect v3: Autonomous Architecture Evolution")
        print("="*80)
        
        # Initialize population
        self.current_population = self.initialize_population()
        
        # Stage 1: Proxy Evaluation
        stage1_survivors = self.stage1_proxy_evaluation(
            self.current_population,
            dataloaders["CIFAR-10"]["train"],
            dataloaders["CIFAR-10"]["val"]
        )
        
        # Stage 2: Refinement Training
        stage2_survivors = self.stage2_refinement_training(
            stage1_survivors,
            dataloaders["CIFAR-100"]["train"],
            dataloaders["CIFAR-100"]["val"]
        )
        
        # Stage 3: Full Validation
        final_models = self.stage3_full_validation(
            stage2_survivors,
            dataloaders
        )
        
        # Export results
        self.export_results(final_models)
        
        print("\n" + "="*80)
        print("Evolution Complete!")
        print("="*80)
        
        return final_models
    
    def export_results(self, final_models: List[ArchitectureGenome]):
        """Export final results"""
        
        output_dir = self.config.logging.artifact_save_path
        os.makedirs(output_dir, exist_ok=True)
        
        # Export top models
        self.knowledge_base.export_top_models(output_dir, top_n=10)
        
        # Export summary
        summary = {
            "num_final_models": len(final_models),
            "best_accuracy": max([m.metrics.get("val_accuracy", 0) for m in final_models]),
            "mean_novelty": np.mean([m.metrics.get("novelty_score_combined", 0) for m in final_models]),
            "knowledge_base_stats": self.knowledge_base.get_statistics()
        }
        
        import json
        with open(os.path.join(output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults exported to {output_dir}")
