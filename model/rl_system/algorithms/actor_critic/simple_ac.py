"""
Simple Actor-Critic Implementation.

This module implements a straightforward Actor-Critic algorithm that learns
both a policy (actor) and value function (critic) in a single network.
This is simpler than A3C and useful for single-threaded training scenarios.
"""

import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

import util.loggers as loggers

from ...core.base_agents import ActorCriticAgent, AgentConfig

logger = loggers.setup_loggers()
rl_logger = logger['rl']


@dataclass
class SimpleACConfig(AgentConfig):
    """Configuration for Simple Actor-Critic agent."""
    learning_rate: float = 0.0003
    gamma: float = 0.99
    hidden_dims: list = None
    continuous: bool = False
    action_std: float = 0.5  # For continuous actions
    entropy_coeff: float = 0.01  # Entropy regularization
    value_coeff: float = 0.5  # Value loss coefficient
    max_grad_norm: float = 0.5  # Gradient clipping

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [1024, 512]


class SimpleActorCriticNetwork(nn.Module):
    """
    Simple Actor-Critic network with shared features.

    The network shares the initial layers between actor and critic,
    which helps with sample efficiency and stability.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: list = [1024, 512],
                 continuous: bool = False):
        super(SimpleActorCriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous

        # Shared feature layers
        shared_layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.shared_network = nn.Sequential(*shared_layers)

        # Value head (critic)
        self.value_head = nn.Linear(prev_dim, 1)

        # Policy head (actor)
        if continuous:
            # For continuous actions, output mean and log_std
            self.action_mean = nn.Linear(prev_dim, action_dim)
            self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # For discrete actions, output action probabilities
            self.action_probs = nn.Sequential(
                nn.Linear(prev_dim, action_dim),
                nn.Softmax(dim=-1)
            )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor):
        """
        Forward pass through the network.

        Returns:
            value: State value estimate
            action_dist: Action distribution (Categorical or Normal)
        """
        # Shared features
        features = self.shared_network(state)

        # Value estimate
        value = self.value_head(features)

        # Action distribution
        if self.continuous:
            action_mean = self.action_mean(features)
            action_std = torch.exp(self.action_log_std.expand_as(action_mean))
            action_dist = Normal(action_mean, action_std)
        else:
            action_probs = self.action_probs(features)
            action_dist = Categorical(action_probs)

        return value, action_dist


class SimpleActorCriticAgent(ActorCriticAgent):
    """
    Simple Actor-Critic Agent.

    This implements a basic actor-critic algorithm that learns both policy
    and value function simultaneously. It's simpler than A3C but still effective
    for many reinforcement learning tasks.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: SimpleACConfig,
                 continuous: bool = False):
        """
        Initialize Simple Actor-Critic agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Agent configuration
            continuous: Whether action space is continuous
        """
        super().__init__(state_dim, action_dim, config, "SimpleAC")

        self.continuous = continuous
        self.current_action = None

        # Storage for episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        self.episode_values = []

        # Performance tracking
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_values = []

        rl_logger.info("Initialized Simple Actor-Critic agent")
        rl_logger.info(f"Continuous actions: {continuous}")

    def _initialize_agent(self) -> None:
        """Initialize Simple AC-specific components."""
        config = self.config

        # Initialize network
        self.network = SimpleActorCriticNetwork(
            self.state_dim,
            self.action_dim,
            config.hidden_dims,
            self.continuous
        )

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate
        )

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using current policy.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state.reshape(1, -1))

        with torch.no_grad():
            value, action_dist = self.network(state_tensor)

        if training:
            # Sample action from distribution
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            # Store for learning
            if len(self.episode_states) == 0 or not torch.equal(state_tensor, self.episode_states[-1]):
                self.episode_states.append(state_tensor)
                self.episode_values.append(value)
                self.episode_actions.append(action)
                self.episode_log_probs.append(log_prob)
        else:
            # Use mean action during evaluation
            if self.continuous:
                action = action_dist.mean
            else:
                action = torch.argmax(action_dist.probs, dim=-1)

        self.current_action = action

        if self.continuous:
            return action.cpu().numpy().flatten()
        else:
            return action.cpu().numpy().flatten()[0]

    def store_reward(self, reward: float) -> None:
        """Store reward for current step."""
        self.episode_rewards.append(reward)

    def update_on_episode_end(self, final_state: np.ndarray = None, done: bool = True) -> Dict[str, float]:
        """
        Update the network at the end of an episode.

        Args:
            final_state: Final state of episode (for bootstrapping)
            done: Whether episode terminated naturally

        Returns:
            Dictionary containing training metrics
        """
        if len(self.episode_rewards) == 0:
            return {}

        # Convert episode data to tensors
        states = torch.cat(self.episode_states, dim=0)
        log_probs = torch.cat(self.episode_log_probs, dim=0)
        values = torch.cat(self.episode_values, dim=0).squeeze()
        rewards = torch.FloatTensor(self.episode_rewards)

        # Compute returns and advantages
        returns = self._compute_returns(rewards, values, final_state, done)
        advantages = returns - values

        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute losses
        value_loss = F.mse_loss(values, returns)

        # Actor loss (policy gradient with advantage)
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Entropy loss for exploration
        _, action_dist = self.network(states)
        entropy = action_dist.entropy().mean()
        entropy_loss = -self.config.entropy_coeff * entropy

        # Total loss
        total_loss = (
            actor_loss +
            self.config.value_coeff * value_loss +
            entropy_loss
        )

        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.config.max_grad_norm
        )

        self.optimizer.step()

        # Store metrics
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(value_loss.item())
        self.entropy_values.append(entropy.item())

        # Clear episode data
        self._clear_episode_data()

        self.training_step += 1

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': value_loss.item(),
            'entropy': entropy.item(),
            'avg_return': returns.mean().item(),
            'episode_length': len(rewards)
        }

    def _compute_returns(self, rewards, values, final_state=None, done=True):
        """Compute discounted returns."""
        returns = torch.zeros_like(rewards)

        # Bootstrap value for the final state if episode didn't terminate
        if not done and final_state is not None:
            final_state_tensor = torch.FloatTensor(final_state.reshape(1, -1))
            with torch.no_grad():
                final_value, _ = self.network(final_state_tensor)
            next_value = final_value.item()
        else:
            next_value = 0.0

        # Compute returns backwards
        for t in reversed(range(len(rewards))):
            next_value = rewards[t] + self.config.gamma * next_value
            returns[t] = next_value

        return returns

    def _clear_episode_data(self):
        """Clear stored episode data."""
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_log_probs.clear()
        self.episode_values.clear()

    def save_model(self, filepath: str) -> None:
        """Save the agent's model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'continuous': self.continuous
        }, filepath)

        rl_logger.info(f"Simple AC model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load the agent's model."""
        checkpoint = torch.load(filepath, map_location='cpu')

        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.continuous = checkpoint['continuous']

        rl_logger.info(f"Simple AC model loaded from {filepath}")

    def get_training_metrics(self) -> Dict[str, float]:
        """Get comprehensive training metrics."""
        base_metrics = super().get_training_metrics()

        ac_metrics = {
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0.0,
            'avg_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0.0,
            'avg_entropy': np.mean(self.entropy_values[-100:]) if self.entropy_values else 0.0,
            'episode_buffer_size': len(self.episode_rewards),
            'continuous_actions': self.continuous
        }

        base_metrics.update(ac_metrics)
        return base_metrics

    def set_eval_mode(self) -> None:
        """Set network to evaluation mode."""
        self.network.eval()
        self.is_training = False

    def set_train_mode(self) -> None:
        """Set network to training mode."""
        self.network.train()
        self.is_training = True
        self.is_training = True
