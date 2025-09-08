"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) Implementation.

This module implements the TD3 algorithm, an improved version of DDPG that addresses
the overestimation bias in actor-critic methods through:
- Twin critics that take the minimum Q-value
- Delayed policy updates
- Target policy smoothing with noise
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
class TD3Config(AgentConfig):
    """Configuration for TD3 agent."""
    actor_lr: float = 0.0001  # Optimized for trading
    critic_lr: float = 0.0001
    tau: float = 0.005  # Soft update rate
    gamma: float = 0.99
    exploration_noise: float = 0.1  # Exploration noise std
    policy_noise: float = 0.2  # Target policy smoothing noise
    noise_clip: float = 0.5  # Noise clipping range
    policy_update_delay: int = 2  # Delayed policy updates
    warmup_steps: int = 1000
    actor_hidden_dims: list = None
    critic_hidden_dims: list = None
    l2_reg: float = 1e-6
    dropout: float = 0.1

    def __post_init__(self):
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [400, 300]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [400, 300]


class TD3Actor(nn.Module):
    """Actor network for TD3 algorithm."""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: list = [400, 300],
                 max_action: float = 1.0,
                 dropout: float = 0.1):
        super(TD3Actor, self).__init__()

        self.max_action = max_action

        # Build network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Remove the last dropout layer
        if layers:
            layers = layers[:-1]

        # Output layer with tanh activation
        layers.extend([
            nn.Linear(prev_dim, action_dim),
            nn.Tanh()
        ])

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actor network."""
        action = self.network(state)
        return self.max_action * action


class TD3Critic(nn.Module):
    """Critic network for TD3 algorithm."""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: list = [400, 300],
                 dropout: float = 0.1):
        super(TD3Critic, self).__init__()

        # Build network layers
        layers = []
        prev_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Remove the last dropout layer
        if layers:
            layers = layers[:-1]

        # Output layer (Q-value)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic network."""
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class TD3Agent(ActorCriticAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent.

    TD3 addresses the overestimation bias in DDPG through:
    1. Twin critics - uses two critic networks and takes the minimum Q-value
    2. Delayed policy updates - updates the actor less frequently than critics
    3. Target policy smoothing - adds noise to target actions
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: TD3Config,
                 max_action: float = 1.0,
                 min_action: float = -1.0):
        """
        Initialize TD3 agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: TD3 configuration
            max_action: Maximum action value
            min_action: Minimum action value
        """
        super().__init__(state_dim, action_dim, config, "TD3")

        self.max_action = max_action
        self.min_action = min_action

        # Training counters
        self.total_steps = 0
        self.policy_update_counter = 0

        # Performance tracking
        self.actor_losses = []
        self.critic1_losses = []
        self.critic2_losses = []

        rl_logger.info(f"Initialized TD3 agent with action range [{min_action}, {max_action}]")

    def _initialize_agent(self) -> None:
        """Initialize TD3-specific components."""
        config = self.config

        # Initialize actor network
        self.actor = TD3Actor(
            self.state_dim,
            self.action_dim,
            config.actor_hidden_dims,
            self.max_action,
            config.dropout
        )

        # Initialize twin critics
        self.critic1 = TD3Critic(
            self.state_dim,
            self.action_dim,
            config.critic_hidden_dims,
            config.dropout
        )

        self.critic2 = TD3Critic(
            self.state_dim,
            self.action_dim,
            config.critic_hidden_dims,
            config.dropout
        )

        # Initialize target networks
        self.target_actor = TD3Actor(
            self.state_dim,
            self.action_dim,
            config.actor_hidden_dims,
            self.max_action,
            config.dropout
        )

        self.target_critic1 = TD3Critic(
            self.state_dim,
            self.action_dim,
            config.critic_hidden_dims,
            config.dropout
        )

        self.target_critic2 = TD3Critic(
            self.state_dim,
            self.action_dim,
            config.critic_hidden_dims,
            config.dropout
        )

        # Copy weights to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Initialize optimizers with weight decay for regularization
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=config.actor_lr,
            weight_decay=config.l2_reg
        )
        self.critic1_optimizer = optim.AdamW(
            self.critic1.parameters(),
            lr=config.critic_lr,
            weight_decay=config.l2_reg
        )
        self.critic2_optimizer = optim.AdamW(
            self.critic2.parameters(),
            lr=config.critic_lr,
            weight_decay=config.l2_reg
        )

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

        if training and self.total_steps > self.config.warmup_steps:
            # Add exploration noise
            noise = np.random.normal(0, self.config.exploration_noise, size=action.shape)
            action = action + noise
        elif training:
            # Random actions during warmup
            action = np.random.uniform(self.min_action, self.max_action, size=action.shape)

        # Clip action to valid range
        action = np.clip(action, self.min_action, self.max_action)

        self.total_steps += 1
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

        # Update critics
        critic1_loss, critic2_loss = self._update_critics(states, actions, rewards, next_states, dones)

        metrics = {
            'critic1_loss': critic1_loss,
            'critic2_loss': critic2_loss
        }

        # Delayed policy updates
        if self.training_step % self.config.policy_update_delay == 0:
            actor_loss = self._update_actor(states)
            metrics['actor_loss'] = actor_loss

            # Soft update target networks
            self._soft_update_targets()
            metrics['target_update'] = 1.0

            self.policy_update_counter += 1

        self.training_step += 1
        return metrics

    def _update_critics(self, states, actions, rewards, next_states, dones) -> tuple:
        """Update critic networks."""
        with torch.no_grad():
            # Target policy smoothing: add noise to target actions
            target_actions = self.target_actor(next_states)
            noise = torch.clamp(
                torch.randn_like(target_actions) * self.config.policy_noise,
                -self.config.noise_clip,
                self.config.noise_clip
            )
            target_actions = torch.clamp(
                target_actions + noise,
                self.min_action,
                self.max_action
            )

            # Compute target Q-values (take minimum for overestimation bias reduction)
            target_q1 = self.target_critic1(next_states, target_actions)
            target_q2 = self.target_critic2(next_states, target_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (self.config.gamma * target_q * ~dones)

        # Update critic 1
        current_q1 = self.critic1(states, actions)
        critic1_loss = F.mse_loss(current_q1, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        self.critic1_optimizer.step()

        # Update critic 2
        current_q2 = self.critic2(states, actions)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic2_optimizer.step()

        # Store losses for tracking
        self.critic1_losses.append(critic1_loss.item())
        self.critic2_losses.append(critic2_loss.item())

        return critic1_loss.item(), critic2_loss.item()

    def _update_actor(self, states) -> float:
        """Update actor network."""
        # Compute actor loss
        actor_actions = self.actor(states)
        actor_loss = -self.critic1(states, actor_actions).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Store loss for tracking
        self.actor_losses.append(actor_loss.item())

        return actor_loss.item()

    def _soft_update_targets(self) -> None:
        """Soft update target networks."""
        tau = self.config.tau

        # Update target actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Update target critics
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_model(self, filepath: str) -> None:
        """Save the agent's model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'total_steps': self.total_steps,
            'policy_update_counter': self.policy_update_counter
        }, filepath)

        rl_logger.info(f"TD3 model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load the agent's model."""
        checkpoint = torch.load(filepath, map_location='cpu')

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.total_steps = checkpoint['total_steps']
        self.policy_update_counter = checkpoint['policy_update_counter']

        rl_logger.info(f"TD3 model loaded from {filepath}")

    def get_training_metrics(self) -> Dict[str, float]:
        """Get comprehensive training metrics."""
        base_metrics = super().get_training_metrics()

        td3_metrics = {
            'total_steps': self.total_steps,
            'policy_updates': self.policy_update_counter,
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0.0,
            'avg_critic1_loss': np.mean(self.critic1_losses[-100:]) if self.critic1_losses else 0.0,
            'avg_critic2_loss': np.mean(self.critic2_losses[-100:]) if self.critic2_losses else 0.0,
            'replay_buffer_size': len(self.replay_buffer),
            'warmup_progress': min(1.0, self.total_steps / self.config.warmup_steps),
            'exploration_noise': self.config.exploration_noise
        }

        base_metrics.update(td3_metrics)
        return base_metrics

    def set_eval_mode(self) -> None:
        """Set networks to evaluation mode."""
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.is_training = False

    def set_train_mode(self) -> None:
        """Set networks to training mode."""
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self.is_training = True
        self.is_training = True
