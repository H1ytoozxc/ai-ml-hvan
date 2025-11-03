"""
RL Controller for Meta-Optimization
REINFORCE контроллер для управления стратегией поиска
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import deque


class REINFORCEController(nn.Module):
    """REINFORCE-based controller for search strategy"""
    
    def __init__(self, 
                 state_dim: int = 64,
                 action_dim: int = 9,  # Number of mutation types
                 hidden_dim: int = 256,
                 learning_rate: float = 1e-3):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value network (for baseline)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # Statistics
        self.episode_returns = deque(maxlen=100)
        self.best_return = -float('inf')
    
    def encode_state(self, 
                     population_stats: Dict[str, float],
                     generation: int,
                     max_generations: int) -> torch.Tensor:
        """Encode current search state to vector"""
        
        state_features = []
        
        # Progress
        state_features.append(generation / max_generations)
        
        # Population statistics
        state_features.append(population_stats.get("mean_accuracy", 0))
        state_features.append(population_stats.get("std_accuracy", 0))
        state_features.append(population_stats.get("max_accuracy", 0))
        state_features.append(population_stats.get("mean_novelty", 0))
        state_features.append(population_stats.get("diversity", 0))
        state_features.append(population_stats.get("mean_complexity", 0))
        
        # Pareto frontier size
        state_features.append(population_stats.get("pareto_size", 0) / 100.0)
        
        # Recent improvement
        state_features.append(population_stats.get("recent_improvement", 0))
        
        # Pad to state_dim
        state_array = np.array(state_features, dtype=np.float32)
        if len(state_array) < self.state_dim:
            state_array = np.pad(state_array, (0, self.state_dim - len(state_array)))
        else:
            state_array = state_array[:self.state_dim]
        
        return torch.FloatTensor(state_array)
    
    def select_action(self, state: torch.Tensor, 
                     deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """Select mutation strategy action"""
        
        with torch.no_grad() if deterministic else torch.enable_grad():
            action_probs = self.policy_net(state)
        
        if deterministic:
            action = action_probs.argmax().item()
            log_prob = torch.log(action_probs[action])
        else:
            # Sample from distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = action.item()
        
        return action, log_prob
    
    def store_transition(self, state: torch.Tensor, 
                        action: int, 
                        log_prob: torch.Tensor,
                        reward: float):
        """Store transition in episode buffer"""
        
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def compute_returns(self, gamma: float = 0.99) -> List[float]:
        """Compute discounted returns"""
        
        returns = []
        G = 0
        
        for reward in reversed(self.episode_rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        return returns
    
    def update(self, gamma: float = 0.99) -> Dict[str, float]:
        """Update policy using REINFORCE"""
        
        if not self.episode_states:
            return {"policy_loss": 0, "value_loss": 0}
        
        # Compute returns
        returns = self.compute_returns(gamma)
        returns = torch.FloatTensor(returns)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Stack states
        states = torch.stack(self.episode_states)
        
        # Compute values for baseline
        values = self.value_net(states).squeeze()
        
        # Compute advantages
        advantages = returns - values.detach()
        
        # Compute policy loss
        policy_probs = self.policy_net(states)
        dist = torch.distributions.Categorical(policy_probs)
        
        log_probs = torch.stack([
            dist.log_prob(torch.tensor(action))
            for action in self.episode_actions
        ])
        
        policy_loss = -(log_probs * advantages).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(values, returns)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        # Statistics
        episode_return = sum(self.episode_rewards)
        self.episode_returns.append(episode_return)
        self.best_return = max(self.best_return, episode_return)
        
        # Clear episode buffer
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "episode_return": episode_return,
            "mean_return_100": np.mean(self.episode_returns) if self.episode_returns else 0
        }
    
    def get_action_distribution(self, state: torch.Tensor) -> np.ndarray:
        """Get action probability distribution"""
        
        with torch.no_grad():
            action_probs = self.policy_net(state)
        
        return action_probs.cpu().numpy()
    
    def save(self, path: str):
        """Save controller state"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_return': self.best_return,
            'episode_returns': list(self.episode_returns)
        }, path)
    
    def load(self, path: str):
        """Load controller state"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_return = checkpoint['best_return']
        self.episode_returns = deque(checkpoint['episode_returns'], maxlen=100)


class EvolutionaryStrategyController:
    """Alternative controller using Evolutionary Strategy (ES)"""
    
    def __init__(self, 
                 state_dim: int = 64,
                 action_dim: int = 9,
                 population_size: int = 50,
                 sigma: float = 0.1,
                 learning_rate: float = 0.01):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        
        # Initialize policy parameters (simple linear policy)
        self.weights = np.random.randn(state_dim, action_dim) * 0.1
        self.bias = np.zeros(action_dim)
    
    def select_action(self, state: np.ndarray, 
                     deterministic: bool = False) -> int:
        """Select action using current policy"""
        
        # Compute logits
        logits = np.dot(state, self.weights) + self.bias
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        if deterministic:
            return np.argmax(probs)
        else:
            return np.random.choice(self.action_dim, p=probs)
    
    def update(self, fitness_function, state: np.ndarray) -> Dict[str, float]:
        """Update using ES"""
        
        # Generate perturbations
        perturbations = []
        fitness_scores = []
        
        for _ in range(self.population_size):
            # Sample perturbation
            noise_w = np.random.randn(*self.weights.shape)
            noise_b = np.random.randn(*self.bias.shape)
            
            # Perturbed parameters
            perturbed_weights = self.weights + self.sigma * noise_w
            perturbed_bias = self.bias + self.sigma * noise_b
            
            # Evaluate fitness
            fitness = fitness_function(perturbed_weights, perturbed_bias, state)
            
            perturbations.append((noise_w, noise_b))
            fitness_scores.append(fitness)
        
        # Normalize fitness scores
        fitness_scores = np.array(fitness_scores)
        fitness_scores = (fitness_scores - np.mean(fitness_scores)) / (np.std(fitness_scores) + 1e-8)
        
        # Update parameters
        grad_w = np.zeros_like(self.weights)
        grad_b = np.zeros_like(self.bias)
        
        for (noise_w, noise_b), fitness in zip(perturbations, fitness_scores):
            grad_w += fitness * noise_w
            grad_b += fitness * noise_b
        
        grad_w /= (self.population_size * self.sigma)
        grad_b /= (self.population_size * self.sigma)
        
        self.weights += self.learning_rate * grad_w
        self.bias += self.learning_rate * grad_b
        
        return {
            "mean_fitness": np.mean(fitness_scores),
            "max_fitness": np.max(fitness_scores)
        }


class AdaptiveSearchController:
    """Adaptive controller that switches between exploration and exploitation"""
    
    def __init__(self, exploration_weight: float = 0.5):
        self.exploration_weight = exploration_weight
        self.history = {
            "novelty_improvements": deque(maxlen=10),
            "accuracy_improvements": deque(maxlen=10)
        }
    
    def update_history(self, novelty_gain: float, accuracy_gain: float):
        """Update improvement history"""
        self.history["novelty_improvements"].append(novelty_gain)
        self.history["accuracy_improvements"].append(accuracy_gain)
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get weights for different mutation strategies"""
        
        # Compute recent trends
        novelty_trend = np.mean(self.history["novelty_improvements"]) if self.history["novelty_improvements"] else 0
        accuracy_trend = np.mean(self.history["accuracy_improvements"]) if self.history["accuracy_improvements"] else 0
        
        # Adaptive weighting
        if novelty_trend > accuracy_trend:
            # Favor exploration
            weights = {
                "add_block": 0.15,
                "remove_block": 0.10,
                "mutate_block_type": 0.20,
                "meta_block_insert": 0.15,
                "change_activation": 0.10,
                "adjust_dropout": 0.10,
                "swap_optimizer": 0.10,
                "adjust_lr_schedule": 0.05,
                "augment_strategy_mutate": 0.05
            }
        else:
            # Favor exploitation (fine-tuning)
            weights = {
                "add_block": 0.05,
                "remove_block": 0.05,
                "mutate_block_type": 0.05,
                "meta_block_insert": 0.05,
                "change_activation": 0.20,
                "adjust_dropout": 0.20,
                "swap_optimizer": 0.15,
                "adjust_lr_schedule": 0.15,
                "augment_strategy_mutate": 0.10
            }
        
        return weights
    
    def select_mutation(self, available_mutations: List[str]) -> str:
        """Select mutation based on adaptive strategy"""
        
        weights = self.get_strategy_weights()
        
        # Filter to available mutations
        mutation_probs = [weights.get(m, 0.1) for m in available_mutations]
        mutation_probs = np.array(mutation_probs)
        mutation_probs /= mutation_probs.sum()
        
        selected = np.random.choice(available_mutations, p=mutation_probs)
        
        return selected


def create_controller(controller_type: str = "REINFORCE", **kwargs):
    """Factory function to create controllers"""
    
    if controller_type == "REINFORCE":
        return REINFORCEController(**kwargs)
    elif controller_type == "ES":
        return EvolutionaryStrategyController(**kwargs)
    elif controller_type == "Adaptive":
        return AdaptiveSearchController(**kwargs)
    else:
        return REINFORCEController(**kwargs)
