"""
Policy Gradient Algorithms Implementation.

This module contains implementations of policy gradient methods including
REINFORCE, Actor-Critic variants (A2C), and related algorithms.
"""

import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import util.loggers as loggers

from ...core.base_agents import (ActorCriticAgent, AgentConfig, EpisodeBuffer,
                                 PolicyBasedAgent)

logger = loggers.setup_loggers()
rl_logger = logger['rl']


class PolicyNetwork(nn.Module):
    """Policy network for discrete actions."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 activation: str = 'relu',
                 dropout: float = 0.0):
        super(PolicyNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer with softmax for action probabilities
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softmax(dim=-1))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get action probabilities."""
        return self.network(x)


class ValueNetwork(nn.Module):
    """Value network for state value estimation."""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 activation: str = 'relu',
                 dropout: float = 0.0):
        super(ValueNetwork, self).__init__()

        # Build network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output single value
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get state value."""
        return self.network(x)


@dataclass
class PolicyGradientConfig(AgentConfig):
    """Configuration for policy gradient agents."""
    learning_rate: float = 0.001
    gamma: float = 0.99
    policy_lr: float = 0.001
    value_lr: float = 0.005
    hidden_dims: List[int] = None
    activation: str = 'relu'
    dropout: float = 0.0
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    use_gae: bool = True
    gae_lambda: float = 0.95

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]


class REINFORCEAgent(PolicyBasedAgent):
    """REINFORCE algorithm with baseline."""

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 config: PolicyGradientConfig = None):

        self.config = config or PolicyGradientConfig()
        self.num_actions = num_actions
        super().__init__(state_dim, num_actions, self.config, "REINFORCEAgent")

    def _initialize_agent(self) -> None:
        """Initialize REINFORCE agent components."""
        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')

        # Policy network
        self.policy_network = PolicyNetwork(
            self.state_dim,
            self.num_actions,
            self.config.hidden_dims,
            self.config.activation,
            self.config.dropout
        ).to(self.device)

        # Value network (baseline)
        self.value_network = ValueNetwork(
            self.state_dim,
            self.config.hidden_dims,
            self.config.activation,
            self.config.dropout
        ).to(self.device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.config.policy_lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.config.value_lr)

        # Episode buffer
        self.episode_buffer = EpisodeBuffer()

        # Training metrics
        self.policy_loss_history = deque(maxlen=1000)
        self.value_loss_history = deque(maxlen=1000)
        self.entropy_history = deque(maxlen=1000)

        rl_logger.info(f"Initialized REINFORCE agent with device: {self.device}")

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using policy network."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)

        if training:
            # Sample action from probability distribution
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Store for episode buffer
            self._last_log_prob = log_prob.item()

            return action.item()
        else:
            # Take most probable action
            return action_probs.argmax().item()

    def update(self,
               state: np.ndarray,
               action: int,
               reward: float,
               next_state: np.ndarray,
               done: bool) -> Dict[str, float]:
        """Store experience and update at episode end."""

        # Get value estimate for baseline
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.value_network(state_tensor).item()

        # Store experience
        self.episode_buffer.store(
            state=state,
            action=action,
            reward=reward,
            log_prob=getattr(self, '_last_log_prob', 0.0),
            value=value,
            done=done
        )

        if done:
            return self._update_networks()

        return {}

    def _update_networks(self) -> Dict[str, float]:
        """Update policy and value networks using REINFORCE."""
        if len(self.episode_buffer) == 0:
            return {}

        # Get episode data
        episode_data = self.episode_buffer.get_episode_data()
        states = episode_data['states']
        actions = episode_data['actions']
        rewards = episode_data['rewards']
        values = episode_data['values']

        # Calculate returns
        returns = self._calculate_returns(rewards)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.FloatTensor(values).to(self.device)

        # Calculate advantages (returns - baseline)
        advantages = returns - values

        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update value network
        current_values = self.value_network(states).squeeze()
        value_loss = F.mse_loss(current_values, returns)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.config.max_grad_norm)
        self.value_optimizer.step()

        # Update policy network
        action_probs = self.policy_network(states)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(new_log_probs * advantages.detach()).mean()
        entropy_loss = -self.config.entropy_coefficient * entropy
        total_policy_loss = policy_loss + entropy_loss

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.config.max_grad_norm)
        self.policy_optimizer.step()

        # Record metrics
        self.policy_loss_history.append(policy_loss.item())
        self.value_loss_history.append(value_loss.item())
        self.entropy_history.append(entropy.item())

        # Clear episode buffer
        self.episode_buffer.clear()

        self.training_step += 1

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'avg_return': returns.mean().item(),
            'avg_advantage': advantages.mean().item()
        }

    def _calculate_returns(self, rewards: np.ndarray) -> np.ndarray:
        """Calculate discounted returns."""
        returns = np.zeros_like(rewards)
        running_return = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.config.gamma * running_return
            returns[t] = running_return

        return returns

    def _get_agent_state(self) -> Dict[str, Any]:
        """Get agent state for checkpointing."""
        return {
            'policy_network_state': self.policy_network.state_dict(),
            'value_network_state': self.value_network.state_dict(),
            'policy_optimizer_state': self.policy_optimizer.state_dict(),
            'value_optimizer_state': self.value_optimizer.state_dict(),
            'policy_loss_history': list(self.policy_loss_history),
            'value_loss_history': list(self.value_loss_history),
            'entropy_history': list(self.entropy_history)
        }

    def _restore_agent_state(self, state: Dict[str, Any]) -> None:
        """Restore agent state from checkpoint."""
        self.policy_network.load_state_dict(state['policy_network_state'])
        self.value_network.load_state_dict(state['value_network_state'])
        self.policy_optimizer.load_state_dict(state['policy_optimizer_state'])
        self.value_optimizer.load_state_dict(state['value_optimizer_state'])
        self.policy_loss_history = deque(state.get('policy_loss_history', []), maxlen=1000)
        self.value_loss_history = deque(state.get('value_loss_history', []), maxlen=1000)
        self.entropy_history = deque(state.get('entropy_history', []), maxlen=1000)

    def _save_model_weights(self, filepath: str) -> None:
        """Save model weights."""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict()
        }, f"{filepath}.pth")

    def _load_model_weights(self, filepath: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(f"{filepath}.pth", map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.value_network.load_state_dict(checkpoint['value_network'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])


class A2CAgent(ActorCriticAgent):
    """Advantage Actor-Critic (A2C) agent."""

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 config: PolicyGradientConfig = None):

        self.config = config or PolicyGradientConfig()
        self.num_actions = num_actions
        super().__init__(state_dim, num_actions, self.config, "A2CAgent")

    def _initialize_agent(self) -> None:
        """Initialize A2C agent components."""
        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')

        # Shared network for both policy and value
        self.shared_network = self._build_shared_network()
        self.policy_head = nn.Linear(self.config.hidden_dims[-1], self.num_actions)
        self.value_head = nn.Linear(self.config.hidden_dims[-1], 1)

        # Move to device
        self.shared_network = self.shared_network.to(self.device)
        self.policy_head = self.policy_head.to(self.device)
        self.value_head = self.value_head.to(self.device)

        # Combined optimizer
        all_params = list(self.shared_network.parameters()) + \
                    list(self.policy_head.parameters()) + \
                    list(self.value_head.parameters())
        self.optimizer = optim.Adam(all_params, lr=self.config.learning_rate)

        # Episode buffer
        self.episode_buffer = EpisodeBuffer()

        # Training metrics
        self.loss_history = deque(maxlen=1000)
        self.entropy_history = deque(maxlen=1000)

        rl_logger.info(f"Initialized A2C agent with device: {self.device}")

    def _build_shared_network(self) -> nn.Module:
        """Build shared feature extraction network."""
        layers = []
        prev_dim = self.state_dim

        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            prev_dim = hidden_dim

        return nn.Sequential(*layers)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using actor-critic policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.shared_network(state_tensor)
            action_logits = self.policy_head(features)
            action_probs = F.softmax(action_logits, dim=-1)

        if training:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            self._last_log_prob = log_prob.item()
            return action.item()
        else:
            return action_probs.argmax().item()

    def update(self,
               state: np.ndarray,
               action: int,
               reward: float,
               next_state: np.ndarray,
               done: bool) -> Dict[str, float]:
        """Store experience and update networks."""

        # Get value estimate
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.shared_network(state_tensor)
            value = self.value_head(features).item()

        # Store experience
        self.episode_buffer.store(
            state=state,
            action=action,
            reward=reward,
            log_prob=getattr(self, '_last_log_prob', 0.0),
            value=value,
            done=done
        )

        if done:
            return self._update_networks()

        return {}

    def _update_networks(self) -> Dict[str, float]:
        """Update actor-critic networks."""
        if len(self.episode_buffer) == 0:
            return {}

        # Get episode data
        episode_data = self.episode_buffer.get_episode_data()
        states = episode_data['states']
        actions = episode_data['actions']
        rewards = episode_data['rewards']
        values = episode_data['values']

        # Calculate returns and advantages
        returns = self._calculate_returns(rewards)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        old_values = torch.FloatTensor(values).to(self.device)

        # Forward pass
        features = self.shared_network(states)
        action_logits = self.policy_head(features)
        values_pred = self.value_head(features).squeeze()

        # Calculate advantages
        advantages = returns - old_values
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(log_probs * advantages.detach()).mean()

        # Value loss
        value_loss = F.mse_loss(values_pred, returns)

        # Combined loss
        total_loss = policy_loss + self.config.value_coefficient * value_loss - self.config.entropy_coefficient * entropy

        # Update networks
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.shared_network.parameters()) +
            list(self.policy_head.parameters()) +
            list(self.value_head.parameters()),
            self.config.max_grad_norm
        )
        self.optimizer.step()

        # Record metrics
        self.loss_history.append(total_loss.item())
        self.entropy_history.append(entropy.item())

        # Clear buffer
        self.episode_buffer.clear()

        self.training_step += 1

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'avg_return': returns.mean().item(),
            'avg_advantage': advantages.mean().item()
        }

    def _calculate_returns(self, rewards: np.ndarray) -> np.ndarray:
        """Calculate discounted returns."""
        returns = np.zeros_like(rewards)
        running_return = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.config.gamma * running_return
            returns[t] = running_return

        return returns

    def _get_agent_state(self) -> Dict[str, Any]:
        """Get agent state for checkpointing."""
        return {
            'shared_network_state': self.shared_network.state_dict(),
            'policy_head_state': self.policy_head.state_dict(),
            'value_head_state': self.value_head.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss_history': list(self.loss_history),
            'entropy_history': list(self.entropy_history)
        }

    def _restore_agent_state(self, state: Dict[str, Any]) -> None:
        """Restore agent state from checkpoint."""
        self.shared_network.load_state_dict(state['shared_network_state'])
        self.policy_head.load_state_dict(state['policy_head_state'])
        self.value_head.load_state_dict(state['value_head_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.loss_history = deque(state.get('loss_history', []), maxlen=1000)
        self.entropy_history = deque(state.get('entropy_history', []), maxlen=1000)

    def _save_model_weights(self, filepath: str) -> None:
        """Save model weights."""
        torch.save({
            'shared_network': self.shared_network.state_dict(),
            'policy_head': self.policy_head.state_dict(),
            'value_head': self.value_head.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f"{filepath}.pth")

    def _load_model_weights(self, filepath: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(f"{filepath}.pth", map_location=self.device)
        self.shared_network.load_state_dict(checkpoint['shared_network'])
        self.policy_head.load_state_dict(checkpoint['policy_head'])
        self.value_head.load_state_dict(checkpoint['value_head'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


@dataclass
class EnhancedPolicyGradientConfig(PolicyGradientConfig):
    """Enhanced Policy Gradient configuration with optimized hyperparameters for trading."""
    learning_rate: float = 0.003  # Optimized for trading
    gamma: float = 0.99
    hidden_dims: List[int] = None
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    baseline_coeff: float = 0.5

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]  # Professional layer sizes


class EnhancedPolicyGradientNetwork(nn.Module):
    """
    Enhanced Policy Gradient Network with optimized architecture for trading.

    Features modern deep learning best practices with proper initialization,
    dropout regularization, and professional layer sizing.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: list = [256, 256]):
        super(EnhancedPolicyGradientNetwork, self).__init__()

        # Build network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        # Output layer with softmax
        layers.extend([
            nn.Linear(prev_dim, action_dim),
            nn.Softmax(dim=-1)
        ])

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Enhanced weight initialization with modern techniques."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to get action probabilities."""
        return self.network(state)


class EnhancedPolicyGradientAgent(PolicyBasedAgent):
    """
    Enhanced Policy Gradient Agent with optimized hyperparameters for trading applications.

    Improvements from TF implementation:
    - Enhanced return calculation
    - Better baseline integration
    - Professional gradient handling
    - Improved exploration strategies
    """

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 config: EnhancedPolicyGradientConfig = None):

        if config is None:
            config = EnhancedPolicyGradientConfig()

        super().__init__(state_dim, num_actions, config, "EnhancedPolicyGradient")

        # Episode storage
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.log_prob_memory = []

        # Performance tracking
        self.episode_rewards = []
        self.policy_losses = []
        self.baseline_losses = []

        rl_logger.info("Initialized Enhanced Policy Gradient agent")

    def _initialize_agent(self) -> None:
        """Initialize enhanced components."""
        config = self.config

        # Policy network
        self.policy_network = EnhancedPolicyGradientNetwork(
            self.state_dim,
            self.num_actions,
            config.hidden_dims
        )

        # Baseline network (for variance reduction)
        self.baseline_network = ValueNetwork(
            self.state_dim,
            config.hidden_dims,
            'relu',
            0.0
        )

        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=config.learning_rate
        )
        self.baseline_optimizer = optim.Adam(
            self.baseline_network.parameters(),
            lr=config.learning_rate
        )

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Enhanced action selection with modern patterns."""
        state_tensor = torch.FloatTensor(state.reshape(1, -1))

        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)

        if training:
            # Sample from distribution
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Store transition
            self.state_memory.append(state)
            self.action_memory.append(action.item())
            self.log_prob_memory.append(log_prob.item())

            return action.item()
        else:
            # Greedy action selection
            return action_probs.argmax().item()

    def store_reward(self, reward: float) -> None:
        """Store reward for current step."""
        self.reward_memory.append(reward)

    def learn_episode(self) -> Dict[str, float]:
        """
        Learn from complete episode using enhanced REINFORCE.

        Returns:
            Dictionary containing training metrics
        """
        if len(self.reward_memory) == 0:
            return {}

        # Convert to tensors
        states = torch.FloatTensor(self.state_memory)
        rewards = torch.FloatTensor(self.reward_memory)
        log_probs = torch.FloatTensor(self.log_prob_memory)

        # Calculate returns (enhanced method)
        returns = self._calculate_returns_enhanced(rewards)

        # Baseline values
        with torch.no_grad():
            baseline_values = self.baseline_network(states).squeeze()

        # Calculate advantages
        advantages = returns - baseline_values

        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update baseline network
        current_values = self.baseline_network(states).squeeze()
        baseline_loss = F.mse_loss(current_values, returns)

        self.baseline_optimizer.zero_grad()
        baseline_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.baseline_network.parameters(),
            self.config.max_grad_norm
        )
        self.baseline_optimizer.step()

        # Update policy network
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Add entropy for exploration
        action_probs = self.policy_network(states)
        dist = Categorical(action_probs)
        entropy = dist.entropy().mean()
        entropy_loss = -self.config.entropy_coefficient * entropy

        total_policy_loss = policy_loss + entropy_loss

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            self.config.max_grad_norm
        )
        self.policy_optimizer.step()

        # Track performance
        episode_reward = sum(self.reward_memory)
        self.episode_rewards.append(episode_reward)
        self.policy_losses.append(policy_loss.item())
        self.baseline_losses.append(baseline_loss.item())

        # Clear episode memory
        self._clear_episode_memory()

        self.training_step += 1

        return {
            'policy_loss': policy_loss.item(),
            'baseline_loss': baseline_loss.item(),
            'entropy': entropy.item(),
            'episode_reward': episode_reward,
            'episode_length': len(rewards),
            'avg_return': returns.mean().item()
        }

    def _calculate_returns_enhanced(self, rewards: torch.Tensor) -> torch.Tensor:
        """Calculate discounted returns using enhanced method."""
        returns = torch.zeros_like(rewards)
        running_return = 0.0

        # Calculate returns backwards (more numerically stable)
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.config.gamma * running_return
            returns[t] = running_return

        return returns

    def _clear_episode_memory(self):
        """Clear episode memory."""
        self.state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()
        self.log_prob_memory.clear()

    def get_training_metrics(self) -> Dict[str, float]:
        """Get enhanced training metrics."""
        base_metrics = super().get_training_metrics()

        enhanced_metrics = {
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            'avg_policy_loss': np.mean(self.policy_losses[-100:]) if self.policy_losses else 0.0,
            'avg_baseline_loss': np.mean(self.baseline_losses[-100:]) if self.baseline_losses else 0.0,
            'total_episodes': len(self.episode_rewards),
            'episode_buffer_size': len(self.reward_memory)
        }

        base_metrics.update(enhanced_metrics)
        return base_metrics

    def save_model(self, filepath: str) -> None:
        """Save enhanced model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'baseline_network_state_dict': self.baseline_network.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'baseline_optimizer_state_dict': self.baseline_optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step
        }, filepath)

        rl_logger.info(f"Enhanced Policy Gradient model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load enhanced model."""
        checkpoint = torch.load(filepath, map_location='cpu')

        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.baseline_network.load_state_dict(checkpoint['baseline_network_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.baseline_optimizer.load_state_dict(checkpoint['baseline_optimizer_state_dict'])
        self.training_step = checkpoint['training_step']

        rl_logger.info(f"Enhanced Policy Gradient model loaded from {filepath}")


def create_policy_gradient_agent(agent_type: str,
                                state_dim: int,
                                num_actions: int,
                                config: PolicyGradientConfig = None) -> PolicyBasedAgent:
    """
    Factory function to create policy gradient agents.

    Args:
        agent_type: Type of agent ('reinforce', 'a2c', 'enhanced')
        state_dim: State space dimension
        num_actions: Number of actions
        config: Agent configuration

    Returns:
        Policy gradient agent instance
    """
    agent_type = agent_type.lower()

    if agent_type == 'reinforce':
        return REINFORCEAgent(state_dim, num_actions, config)
    elif agent_type == 'a2c':
        return A2CAgent(state_dim, num_actions, config)
    elif agent_type == 'enhanced' or agent_type == 'enhanced_pg':
        # Use Enhanced config if no config provided
        if config is None or not isinstance(config, EnhancedPolicyGradientConfig):
            enhanced_config = EnhancedPolicyGradientConfig()
            if config is not None:
                # Copy over any provided parameters
                for key, value in config.__dict__.items():
                    if hasattr(enhanced_config, key):
                        setattr(enhanced_config, key, value)
            config = enhanced_config
        return EnhancedPolicyGradientAgent(state_dim, num_actions, config)
    else:
        raise ValueError(f"Unknown policy gradient agent type: {agent_type}. "
                        f"Available types: reinforce, a2c, enhanced")
