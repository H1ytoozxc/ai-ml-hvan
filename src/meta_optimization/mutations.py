"""
Mutation Operations
Операции мутации для эволюции архитектур
"""

import random
import numpy as np
from typing import List, Dict, Any
import copy

from ..search_space.search_space import ArchitectureGenome, ConditionalSearchSpace


class MutationOperator:
    """Base class for mutation operations"""
    
    def __init__(self, mutation_rate: float = 0.3):
        self.mutation_rate = mutation_rate
    
    def __call__(self, genome: ArchitectureGenome, 
                 search_space: ConditionalSearchSpace) -> ArchitectureGenome:
        """Apply mutation"""
        raise NotImplementedError


class AddBlockMutation(MutationOperator):
    """Add a new block to the architecture"""
    
    def __call__(self, genome: ArchitectureGenome,
                 search_space: ConditionalSearchSpace) -> ArchitectureGenome:
        
        mutated = genome.clone()
        
        if random.random() < self.mutation_rate:
            # Choose random position
            if mutated.blocks:
                position = random.randint(0, len(mutated.blocks))
            else:
                position = 0
            
            # Sample new block
            block_type = random.choice(search_space.config.base_blocks)
            new_block = search_space.sample_block_config(block_type)
            new_block["layer_id"] = position
            
            # Insert block
            mutated.blocks.insert(position, new_block)
            
            # Update layer IDs
            for i in range(position + 1, len(mutated.blocks)):
                mutated.blocks[i]["layer_id"] = i
        
        return mutated


class RemoveBlockMutation(MutationOperator):
    """Remove a block from the architecture"""
    
    def __call__(self, genome: ArchitectureGenome,
                 search_space: ConditionalSearchSpace) -> ArchitectureGenome:
        
        mutated = genome.clone()
        
        if (random.random() < self.mutation_rate and 
            len(mutated.blocks) > search_space.config.min_depth):
            
            # Choose random block to remove
            position = random.randint(0, len(mutated.blocks) - 1)
            mutated.blocks.pop(position)
            
            # Update layer IDs
            for i in range(position, len(mutated.blocks)):
                mutated.blocks[i]["layer_id"] = i
        
        return mutated


class ChangeActivationMutation(MutationOperator):
    """Change activation function in a block"""
    
    def __call__(self, genome: ArchitectureGenome,
                 search_space: ConditionalSearchSpace) -> ArchitectureGenome:
        
        mutated = genome.clone()
        
        if random.random() < self.mutation_rate and mutated.blocks:
            # Choose random block
            block_idx = random.randint(0, len(mutated.blocks) - 1)
            block = mutated.blocks[block_idx]
            
            # Get valid activations for this block type
            block_type = block.get("type")
            valid_activations = search_space.get_valid_activations(block_type)
            
            if valid_activations:
                block["activation"] = random.choice(valid_activations)
        
        return mutated


class AdjustDropoutMutation(MutationOperator):
    """Adjust dropout rate in a block"""
    
    def __call__(self, genome: ArchitectureGenome,
                 search_space: ConditionalSearchSpace) -> ArchitectureGenome:
        
        mutated = genome.clone()
        
        if random.random() < self.mutation_rate and mutated.blocks:
            # Choose random block
            block_idx = random.randint(0, len(mutated.blocks) - 1)
            block = mutated.blocks[block_idx]
            
            # Adjust dropout with small perturbation
            if "dropout" in block:
                current_dropout = block["dropout"]
                delta = random.gauss(0, 0.05)
                new_dropout = np.clip(current_dropout + delta, 
                                     search_space.config.dropout_range[0],
                                     search_space.config.dropout_range[1])
                block["dropout"] = new_dropout
        
        return mutated


class SwapOptimizerMutation(MutationOperator):
    """Change optimizer"""
    
    def __call__(self, genome: ArchitectureGenome,
                 search_space: ConditionalSearchSpace) -> ArchitectureGenome:
        
        mutated = genome.clone()
        
        if random.random() < self.mutation_rate:
            mutated.optimizer = random.choice(search_space.config.optimizers)
        
        return mutated


class AdjustLRScheduleMutation(MutationOperator):
    """Change learning rate schedule"""
    
    def __call__(self, genome: ArchitectureGenome,
                 search_space: ConditionalSearchSpace) -> ArchitectureGenome:
        
        mutated = genome.clone()
        
        if random.random() < self.mutation_rate:
            mutated.lr_scheduler = random.choice(search_space.config.lr_schedulers)
            
            # Also potentially adjust learning rate
            if random.random() < 0.5:
                delta = random.gauss(0, 0.2)
                log_lr = np.log(mutated.learning_rate)
                log_lr += delta
                mutated.learning_rate = np.clip(
                    np.exp(log_lr),
                    search_space.config.lr_range[0],
                    search_space.config.lr_range[1]
                )
        
        return mutated


class MutateBlockTypeMutation(MutationOperator):
    """Change the type of a block"""
    
    def __call__(self, genome: ArchitectureGenome,
                 search_space: ConditionalSearchSpace) -> ArchitectureGenome:
        
        mutated = genome.clone()
        
        if random.random() < self.mutation_rate and mutated.blocks:
            # Choose random block
            block_idx = random.randint(0, len(mutated.blocks) - 1)
            
            # Sample new block type
            new_block_type = random.choice(search_space.config.base_blocks)
            new_block = search_space.sample_block_config(new_block_type)
            new_block["layer_id"] = block_idx
            
            # Replace block
            mutated.blocks[block_idx] = new_block
        
        return mutated


class MetaBlockInsertMutation(MutationOperator):
    """Insert a meta-learning block"""
    
    def __call__(self, genome: ArchitectureGenome,
                 search_space: ConditionalSearchSpace) -> ArchitectureGenome:
        
        mutated = genome.clone()
        
        if random.random() < self.mutation_rate:
            # Choose meta block type
            meta_type = random.choice(search_space.config.meta_blocks)
            
            meta_config = {
                "type": meta_type,
                "position": random.choice(["pre", "mid", "post"])
            }
            
            mutated.meta_blocks.append(meta_config)
        
        return mutated


class AugmentStrategyMutation(MutationOperator):
    """Mutate data augmentation strategy"""
    
    def __call__(self, genome: ArchitectureGenome,
                 search_space: ConditionalSearchSpace) -> ArchitectureGenome:
        
        mutated = genome.clone()
        
        if random.random() < self.mutation_rate:
            # Add or remove augmentation
            if random.random() < 0.5 and mutated.augmentations:
                # Remove random augmentation
                aug_to_remove = random.choice(mutated.augmentations)
                mutated.augmentations.remove(aug_to_remove)
            else:
                # Add random augmentation
                available_augs = [aug for aug in search_space.config.data_augmentations
                                 if aug not in mutated.augmentations]
                if available_augs:
                    mutated.augmentations.append(random.choice(available_augs))
        
        return mutated


class CrossoverOperator:
    """Crossover between two genomes"""
    
    def __init__(self, crossover_rate: float = 0.7):
        self.crossover_rate = crossover_rate
    
    def __call__(self, parent1: ArchitectureGenome,
                 parent2: ArchitectureGenome) -> ArchitectureGenome:
        """Perform crossover between two parents"""
        
        if random.random() > self.crossover_rate:
            # No crossover, return copy of parent1
            return parent1.clone()
        
        child = ArchitectureGenome()
        
        # Crossover blocks (single-point or uniform)
        if random.random() < 0.5:
            # Single-point crossover
            min_len = min(len(parent1.blocks), len(parent2.blocks))
            if min_len > 0:
                crossover_point = random.randint(0, min_len)
                child.blocks = (parent1.blocks[:crossover_point] + 
                              parent2.blocks[crossover_point:])
            else:
                child.blocks = parent1.blocks.copy()
        else:
            # Uniform crossover
            max_len = max(len(parent1.blocks), len(parent2.blocks))
            for i in range(max_len):
                if i < len(parent1.blocks) and i < len(parent2.blocks):
                    block = parent1.blocks[i] if random.random() < 0.5 else parent2.blocks[i]
                    child.blocks.append(copy.deepcopy(block))
                elif i < len(parent1.blocks):
                    child.blocks.append(copy.deepcopy(parent1.blocks[i]))
                else:
                    child.blocks.append(copy.deepcopy(parent2.blocks[i]))
        
        # Update layer IDs
        for i, block in enumerate(child.blocks):
            block["layer_id"] = i
        
        # Inherit hyperparameters randomly
        child.optimizer = random.choice([parent1.optimizer, parent2.optimizer])
        child.learning_rate = random.choice([parent1.learning_rate, parent2.learning_rate])
        child.lr_scheduler = random.choice([parent1.lr_scheduler, parent2.lr_scheduler])
        child.loss_function = random.choice([parent1.loss_function, parent2.loss_function])
        child.batch_size = random.choice([parent1.batch_size, parent2.batch_size])
        
        # Combine augmentations
        child.augmentations = list(set(parent1.augmentations + parent2.augmentations))
        
        # Combine meta blocks
        child.meta_blocks = parent1.meta_blocks.copy()
        if random.random() < 0.5:
            child.meta_blocks.extend(parent2.meta_blocks)
        
        # Set parent IDs
        child.parent_ids = [parent1.genome_id, parent2.genome_id]
        
        # Generate new genome ID
        import uuid
        child.genome_id = str(uuid.uuid4())
        
        return child


class MutationScheduler:
    """Schedule mutation rates over time"""
    
    def __init__(self, initial_rate: float = 0.3,
                 final_rate: float = 0.1,
                 decay_type: str = "linear"):
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.decay_type = decay_type
    
    def get_rate(self, generation: int, max_generations: int) -> float:
        """Get mutation rate for current generation"""
        
        progress = generation / max_generations if max_generations > 0 else 0
        
        if self.decay_type == "linear":
            rate = self.initial_rate - (self.initial_rate - self.final_rate) * progress
        elif self.decay_type == "exponential":
            rate = self.final_rate + (self.initial_rate - self.final_rate) * np.exp(-5 * progress)
        else:
            rate = self.initial_rate
        
        return rate


ALL_MUTATIONS = {
    "add_block": AddBlockMutation,
    "remove_block": RemoveBlockMutation,
    "change_activation": ChangeActivationMutation,
    "adjust_dropout": AdjustDropoutMutation,
    "swap_optimizer": SwapOptimizerMutation,
    "adjust_lr_schedule": AdjustLRScheduleMutation,
    "mutate_block_type": MutateBlockTypeMutation,
    "meta_block_insert": MetaBlockInsertMutation,
    "augment_strategy_mutate": AugmentStrategyMutation
}


def get_mutation_operator(mutation_name: str, mutation_rate: float = 0.3) -> MutationOperator:
    """Factory function to get mutation operator by name"""
    
    if mutation_name in ALL_MUTATIONS:
        return ALL_MUTATIONS[mutation_name](mutation_rate=mutation_rate)
    else:
        raise ValueError(f"Unknown mutation: {mutation_name}")
