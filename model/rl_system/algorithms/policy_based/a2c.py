"""
Advantage Actor-Critic (A2C) Algorithm

A synchronous version of the Asynchronous Advantage Actor-Critic (A3C) algorithm.
A2C uses a critic to estimate the value function and an actor to select actions,
training both with the advantage function.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
from collections import deque

from ...core.base_agents import BaseRLAgent, AgentConfig


class A2CConfig(AgentConfig):
    """Configuration for A2C agent."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_steps: int = 5
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class A2CNetwork(nn.Module):
    """Neural network for A2C with shared features."""
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: list = [64, 64]):
        super().__init__()
        
        # Shared feature layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.shared_features = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Linear(prev_dim, action_dim)
        
        # Critic head (value function)
        self.critic = nn.Linear(prev_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # Initialize actor head with smaller weights
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy logits and value."""
        features = self.shared_features(x)
        policy_logits = self.actor(features)
        value = self.critic(features)
        return policy_logits, value


class A2CAgent(BaseRLAgent):
    """Advantage Actor-Critic agent."""
    
    def __init__(self, observation_space, action_space, config: A2CConfig = None):
        super().__init__(observation_space, action_space, config or A2CConfig())
        
        # Network setup
        if hasattr(observation_space, 'shape'):
            input_dim = observation_space.shape[0] if len(observation_space.shape) == 1 else observation_space.shape
        else:
            input_dim = observation_space
        
        if hasattr(action_space, 'n'):
            self.action_dim = action_space.n
        else:
            self.action_dim = action_space
        
        if isinstance(input_dim, (list, tuple)):
            input_dim = input_dim[0]
        
        self.network = A2CNetwork(input_dim, self.action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        
        # Training state
        self.rollout_buffer = deque(maxlen=self.config.n_steps)
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Current episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        print(f"A2C Agent initialized with input_dim={input_dim}, action_dim={self.action_dim}")
    
    def act(self, observation, training=True):
        """Select action using the current policy."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, value = self.network(obs_tensor)
            
        if training:
            # Sample from policy distribution
            probs = F.softmax(policy_logits, dim=-1)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
            
            return {
                'action': action.item(),
                'log_prob': action_log_prob.item(),
                'value': value.item(),
                'entropy': action_dist.entropy().item()
            }
        else:
            # Greedy action selection
            action = torch.argmax(policy_logits, dim=-1)
            return {'action': action.item()}
    
    def learn(self, experiences):
        """Update the agent using collected experiences."""
        if len(experiences) < self.config.n_steps:
            return {}
        
        # Prepare batch data
        observations = torch.FloatTensor([exp['observation'] for exp in experiences])
        actions = torch.LongTensor([exp['action'] for exp in experiences])
        rewards = torch.FloatTensor([exp['reward'] for exp in experiences])
        next_observations = torch.FloatTensor([exp['next_observation'] for exp in experiences])
        dones = torch.BoolTensor([exp['done'] for exp in experiences])
        log_probs = torch.FloatTensor([exp.get('log_prob', 0) for exp in experiences])
        
        # Get current policy and values
        policy_logits, values = self.network(observations)
        values = values.squeeze()
        
        # Get next values for bootstrapping
        with torch.no_grad():
            _, next_values = self.network(next_observations)
            next_values = next_values.squeeze()
        
        # Compute returns and advantages using GAE
        returns, advantages = self._compute_gae(rewards, values, next_values, dones)
        
        # Normalize advantages
        if self.config.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute losses
        policy_loss = self._compute_policy_loss(policy_logits, actions, advantages, log_probs)
        value_loss = self._compute_value_loss(values, returns)
        entropy_loss = self._compute_entropy_loss(policy_logits)
        
        # Total loss
        total_loss = (policy_loss + 
                     self.config.value_loss_coef * value_loss - 
                     self.config.entropy_coef * entropy_loss)
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        self.training_step += 1
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_return': returns.mean().item()
        }
    
    def _compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation."""
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        gae = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step].float()
                next_value = next_values[step]
            else:
                next_non_terminal = 1.0 - dones[step].float()
                next_value = values[step + 1]
            
            delta = rewards[step] + self.config.gamma * next_value * next_non_terminal - values[step]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages[step] = gae
            returns[step] = gae + values[step]
        
        return returns, advantages
    
    def _compute_policy_loss(self, policy_logits, actions, advantages, old_log_probs):
        """Compute policy loss."""
        log_probs = F.log_softmax(policy_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        policy_loss = -(action_log_probs * advantages).mean()
        return policy_loss
    
    def _compute_value_loss(self, values, returns):
        """Compute value function loss."""
        return F.mse_loss(values, returns)
    
    def _compute_entropy_loss(self, policy_logits):
        """Compute entropy loss for exploration."""
        probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return entropy
    
    def save_model(self, filepath: str):
        """Save the agent's model."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'training_step': self.training_step
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load the agent's model."""
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        metrics = super().get_metrics()
        
        metrics.update({
            'mean_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'current_episode_reward': self.current_episode_reward,
            'current_episode_length': self.current_episode_length
        })
        
        return metrics
