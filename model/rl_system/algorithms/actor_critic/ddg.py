"""
Deep Deterministic Gradient (DDG) Implementation.

This module implements the Deep Deterministic Policy Gradient (DDPG) algorithm,
also known as DDG, which is an actor-critic method for continuous control tasks.
DDG combines the benefits of value-based and policy-based methods.
"""

import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import util.loggers as loggers

from ...core.base_agents import ActorCriticAgent, AgentConfig, ReplayBuffer

logger = loggers.setup_loggers()
rl_logger = logger['rl']


@dataclass
class DDGConfig(AgentConfig):
    """Configuration for DDG agent."""
    actor_lr: float = 0.001
    critic_lr: float = 0.002
    tau: float = 0.005  # Soft update rate
    noise_std: float = 0.1  # Exploration noise
    noise_clip: float = 0.5
    actor_hidden_dims: list = None
    critic_hidden_dims: list = None
    target_update_freq: int = 1
    warmup_steps: int = 1000

    def __post_init__(self):
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [400, 300]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [400, 300]


class DDGActor(nn.Module):
    """Actor network for DDG algorithm."""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: list = [400, 300],
                 max_action: float = 1.0):
        super(DDGActor, self).__init__()

        self.max_action = max_action

        # Build network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        # Output layer with tanh activation
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actor network."""
        action = self.network(state)
        return self.max_action * action


class DDGCritic(nn.Module):
    """Critic network for DDG algorithm."""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: list = [400, 300]):
        super(DDGCritic, self).__init__()

        # Build network layers
        layers = []
        prev_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        # Output layer (Q-value)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic network."""
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for temporally correlated noise."""

    def __init__(self,
                 action_dim: int,
                 mu: float = 0.0,
                 theta: float = 0.15,
                 sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        """Reset the noise process."""
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self) -> np.ndarray:
        """Sample noise from the Ornstein-Uhlenbeck process."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class DDGAgent(ActorCriticAgent):
    """
    Deep Deterministic Gradient (DDG/DDPG) Agent.

    DDG is an actor-critic method that uses deterministic policies and learns
    Q-functions to guide policy improvement. It's particularly effective for
    continuous control tasks like position sizing in trading.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: DDGConfig,
                 max_action: float = 1.0,
                 min_action: float = -1.0):
        """
        Initialize DDG agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: DDG configuration
            max_action: Maximum action value
            min_action: Minimum action value
        """
        super().__init__(state_dim, action_dim, config, "DDG")

        self.max_action = max_action
        self.min_action = min_action

        # Initialize noise process
        self.noise = OrnsteinUhlenbeckNoise(action_dim)

        rl_logger.info(f"Initialized DDG agent with action range [{min_action}, {max_action}]")

    def _initialize_agent(self) -> None:
        """Initialize DDG-specific components."""
        config = self.config

        # Initialize networks
        self.actor = DDGActor(
            self.state_dim,
            self.action_dim,
            config.actor_hidden_dims,
            self.max_action
        )

        self.critic = DDGCritic(
            self.state_dim,
            self.action_dim,
            config.critic_hidden_dims
        )

        # Initialize target networks
        self.target_actor = DDGActor(
            self.state_dim,
            self.action_dim,
            config.actor_hidden_dims,
            self.max_action
        )

        self.target_critic = DDGCritic(
            self.state_dim,
            self.action_dim,
            config.critic_hidden_dims
        )

        # Copy weights to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.memory_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim
        )

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using the current policy.

        Args:
            state: Current state
            training: Whether in training mode (adds noise)

        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state.reshape(1, -1))

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()

        if training and self.training_step > self.config.warmup_steps:
            # Add exploration noise
            noise = self.noise.sample() * self.config.noise_std
            action = action + noise

        # Clip action to valid range
        action = np.clip(action, self.min_action, self.max_action)

        return action

    def store_transition(self,
                        state: np.ndarray,
                        action: np.ndarray,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """
        Update the agent's networks.

        Returns:
            Dictionary containing training metrics
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {}

        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)

        # Update critic
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, target_actions)
            target_q = rewards + (self.config.gamma * target_q * ~dones)

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        if self.training_step % self.config.target_update_freq == 0:
            self._soft_update_targets()

        self.training_step += 1

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': current_q.mean().item()
        }

    def _soft_update_targets(self) -> None:
        """Soft update target networks."""
        tau = self.config.tau

        # Update target actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Update target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_model(self, filepath: str) -> None:
        """Save the agent's model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step
        }, filepath)

        rl_logger.info(f"DDG model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load the agent's model."""
        checkpoint = torch.load(filepath, map_location='cpu')

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_step = checkpoint['training_step']

        rl_logger.info(f"DDG model loaded from {filepath}")

    def reset_noise(self) -> None:
        """Reset the exploration noise process."""
        self.noise.reset()

    def set_eval_mode(self) -> None:
        """Set networks to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
        self.is_training = False

    def set_train_mode(self) -> None:
        """Set networks to training mode."""
        self.actor.train()
        self.critic.train()
        self.is_training = True
        self.is_training = True
