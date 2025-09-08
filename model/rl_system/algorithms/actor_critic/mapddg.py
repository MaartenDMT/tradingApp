"""
Multi-Agent Policy Deep Deterministic Gradient (MAPDDG) Implementation.

This module implements MAPDDG, a multi-agent extension of DDPG that uses:
- Centralized training with decentralized execution
- A centralized critic that sees global state and all agents' actions
- Individual actors for each agent that only see local observations
- Shared experience replay across all agents
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

from ...core.base_agents import AgentConfig

logger = loggers.setup_loggers()
rl_logger = logger['rl']


@dataclass
class MAPDDGConfig(AgentConfig):
    """Configuration for MAPDDG multi-agent system."""
    actor_lr: float = 0.001
    critic_lr: float = 0.002
    tau: float = 0.005  # Soft update rate
    exploration_noise: float = 0.1
    num_agents: int = 2
    actor_hidden_dims: list = None
    critic_hidden_dims: list = None
    l2_reg: float = 1e-6
    dropout: float = 0.1

    def __post_init__(self):
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [128, 128]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [256, 256]


class MAPDDGActor(nn.Module):
    """Actor network for individual agents in MAPDDG."""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: list = [128, 128],
                 max_action: float = 1.0,
                 dropout: float = 0.1):
        super(MAPDDGActor, self).__init__()

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

        # Remove the last dropout
        if layers:
            layers = layers[:-1]

        # Output layer
        layers.extend([
            nn.Linear(prev_dim, action_dim),
            nn.Tanh()
        ])

        self.network = nn.Sequential(*layers)
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


class MAPDDGCentralizedCritic(nn.Module):
    """Centralized critic that sees global state and all agents' actions."""

    def __init__(self,
                 global_state_dim: int,
                 global_action_dim: int,
                 num_agents: int,
                 hidden_dims: list = [256, 256],
                 dropout: float = 0.1):
        super(MAPDDGCentralizedCritic, self).__init__()

        self.num_agents = num_agents

        # Build network layers
        layers = []
        prev_dim = global_state_dim + global_action_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Remove the last dropout
        if layers:
            layers = layers[:-1]

        # Output layer (one Q-value per agent)
        layers.append(nn.Linear(prev_dim, num_agents))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, global_state: torch.Tensor, global_action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the centralized critic."""
        x = torch.cat([global_state, global_action], dim=1)
        return self.network(x)


class MultiAgentReplayBuffer:
    """Replay buffer for multi-agent experiences."""

    def __init__(self,
                 capacity: int,
                 num_agents: int,
                 state_dim: int,
                 action_dim: int):
        self.capacity = capacity
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.position = 0
        self.size = 0

        # Initialize memory arrays
        self.states = np.zeros((capacity, num_agents, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, num_agents), dtype=np.float32)
        self.next_states = np.zeros((capacity, num_agents, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, num_agents), dtype=bool)

    def add(self, states, actions, rewards, next_states, dones):
        """Add a multi-agent experience to the buffer."""
        self.states[self.position] = states
        self.actions[self.position] = actions
        self.rewards[self.position] = rewards
        self.next_states[self.position] = next_states
        self.dones[self.position] = dones

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a batch of multi-agent experiences."""
        indices = np.random.choice(self.size, batch_size, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size


class MAPDDGAgent:
    """
    Multi-Agent Policy Deep Deterministic Gradient (MAPDDG) Agent.

    Implements centralized training with decentralized execution:
    - Centralized critic sees global state and all agents' actions
    - Individual actors for each agent see only local observations
    - Shared experience replay across all agents
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: MAPDDGConfig,
                 max_action: float = 1.0,
                 min_action: float = -1.0):
        """
        Initialize MAPDDG multi-agent system.

        Args:
            state_dim: Dimension of individual agent's state space
            action_dim: Dimension of individual agent's action space
            config: MAPDDG configuration
            max_action: Maximum action value
            min_action: Minimum action value
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = config.num_agents
        self.config = config
        self.max_action = max_action
        self.min_action = min_action

        # Global dimensions for centralized critic
        self.global_state_dim = state_dim * config.num_agents
        self.global_action_dim = action_dim * config.num_agents

        # Training state
        self.training_step = 0
        self.is_training = True

        # Performance tracking
        self.critic_losses = []
        self.actor_losses = [[] for _ in range(config.num_agents)]

        self._initialize_networks()

        rl_logger.info(f"Initialized MAPDDG with {config.num_agents} agents")
        rl_logger.info(f"Individual state dim: {state_dim}, action dim: {action_dim}")
        rl_logger.info(f"Global state dim: {self.global_state_dim}, action dim: {self.global_action_dim}")

    def _initialize_networks(self):
        """Initialize all networks and optimizers."""
        config = self.config

        # Initialize individual actors for each agent
        self.actors = nn.ModuleList([
            MAPDDGActor(
                self.state_dim,
                self.action_dim,
                config.actor_hidden_dims,
                self.max_action,
                config.dropout
            ) for _ in range(self.num_agents)
        ])

        # Initialize target actors
        self.target_actors = nn.ModuleList([
            MAPDDGActor(
                self.state_dim,
                self.action_dim,
                config.actor_hidden_dims,
                self.max_action,
                config.dropout
            ) for _ in range(self.num_agents)
        ])

        # Copy weights to target actors
        for i in range(self.num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())

        # Initialize centralized critic
        self.critic = MAPDDGCentralizedCritic(
            self.global_state_dim,
            self.global_action_dim,
            self.num_agents,
            config.critic_hidden_dims,
            config.dropout
        )

        # Initialize target critic
        self.target_critic = MAPDDGCentralizedCritic(
            self.global_state_dim,
            self.global_action_dim,
            self.num_agents,
            config.critic_hidden_dims,
            config.dropout
        )

        # Copy weights to target critic
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Initialize optimizers
        self.actor_optimizers = [
            optim.AdamW(actor.parameters(), lr=config.actor_lr, weight_decay=config.l2_reg)
            for actor in self.actors
        ]

        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(),
            lr=config.critic_lr,
            weight_decay=config.l2_reg
        )

        # Initialize multi-agent replay buffer
        self.replay_buffer = MultiAgentReplayBuffer(
            config.memory_size,
            self.num_agents,
            self.state_dim,
            self.action_dim
        )

    def select_actions(self, states: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select actions for all agents.

        Args:
            states: Array of shape (num_agents, state_dim)
            training: Whether in training mode (adds noise)

        Returns:
            Array of actions of shape (num_agents, action_dim)
        """
        actions = []

        for i, (actor, state) in enumerate(zip(self.actors, states)):
            state_tensor = torch.FloatTensor(state.reshape(1, -1))

            with torch.no_grad():
                action = actor(state_tensor).cpu().data.numpy().flatten()

            if training:
                # Add exploration noise
                noise = np.random.normal(0, self.config.exploration_noise, size=action.shape)
                action = action + noise

            # Clip action to valid range
            action = np.clip(action, self.min_action, self.max_action)
            actions.append(action)

        return np.array(actions)

    def store_transition(self,
                        states: np.ndarray,
                        actions: np.ndarray,
                        rewards: np.ndarray,
                        next_states: np.ndarray,
                        dones: np.ndarray) -> None:
        """Store multi-agent transition in replay buffer."""
        self.replay_buffer.add(states, actions, rewards, next_states, dones)

    def update(self) -> Dict[str, float]:
        """
        Update all networks using centralized training.

        Returns:
            Dictionary containing training metrics
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {}

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)

        # Convert to tensors
        # states: (batch_size, num_agents, state_dim)
        # actions: (batch_size, num_agents, action_dim)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)  # (batch_size, num_agents)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        # Flatten for centralized critic
        batch_size = states.shape[0]
        global_states = states.view(batch_size, -1)  # (batch_size, global_state_dim)
        global_actions = actions.view(batch_size, -1)  # (batch_size, global_action_dim)
        global_next_states = next_states.view(batch_size, -1)

        # Update centralized critic
        critic_loss = self._update_critic(global_states, global_actions, rewards, global_next_states, dones, next_states)

        # Update individual actors
        actor_losses = self._update_actors(states, actions, global_states)

        # Soft update target networks
        self._soft_update_targets()

        self.training_step += 1

        metrics = {
            'critic_loss': critic_loss,
            'avg_actor_loss': np.mean(actor_losses)
        }

        # Add individual actor losses
        for i, loss in enumerate(actor_losses):
            metrics[f'actor_{i}_loss'] = loss

        return metrics

    def _update_critic(self, global_states, global_actions, rewards, global_next_states, dones, next_states):
        """Update the centralized critic."""
        with torch.no_grad():
            # Compute target actions using target actors
            target_actions = []
            for i, target_actor in enumerate(self.target_actors):
                agent_next_states = next_states[:, i, :]  # (batch_size, state_dim)
                target_action = target_actor(agent_next_states)
                target_actions.append(target_action)

            # Concatenate target actions
            target_global_actions = torch.cat(target_actions, dim=1)  # (batch_size, global_action_dim)

            # Compute target Q-values
            target_q_values = self.target_critic(global_next_states, target_global_actions)

            # Compute targets for each agent
            target_values = rewards + (self.config.gamma * target_q_values * ~dones)

        # Current Q-values
        current_q_values = self.critic(global_states, global_actions)

        # Critic loss
        critic_loss = F.mse_loss(current_q_values, target_values)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Store loss for tracking
        loss_value = critic_loss.item()
        self.critic_losses.append(loss_value)

        return loss_value

    def _update_actors(self, states, actions, global_states):
        """Update individual actor networks."""
        actor_losses = []

        for i, (actor, optimizer) in enumerate(zip(self.actors, self.actor_optimizers)):
            # Get agent's individual states
            agent_states = states[:, i, :]  # (batch_size, state_dim)

            # Compute new actions for this agent
            new_actions = actor(agent_states)

            # Create modified global actions (replace agent i's actions with new actions)
            modified_global_actions = actions.view(actions.shape[0], -1).clone()
            start_idx = i * self.action_dim
            end_idx = (i + 1) * self.action_dim
            modified_global_actions[:, start_idx:end_idx] = new_actions

            # Compute actor loss using centralized critic
            q_values = self.critic(global_states, modified_global_actions)
            actor_loss = -q_values[:, i].mean()  # Use Q-value for agent i

            # Update actor
            optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optimizer.step()

            # Store loss for tracking
            loss_value = actor_loss.item()
            self.actor_losses[i].append(loss_value)
            actor_losses.append(loss_value)

        return actor_losses

    def _soft_update_targets(self):
        """Soft update target networks."""
        tau = self.config.tau

        # Update target actors
        for target_actor, actor in zip(self.target_actors, self.actors):
            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Update target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_model(self, filepath: str) -> None:
        """Save all models."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        save_dict = {
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step
        }

        # Save individual actors
        for i in range(self.num_agents):
            save_dict[f'actor_{i}_state_dict'] = self.actors[i].state_dict()
            save_dict[f'target_actor_{i}_state_dict'] = self.target_actors[i].state_dict()
            save_dict[f'actor_{i}_optimizer_state_dict'] = self.actor_optimizers[i].state_dict()

        torch.save(save_dict, filepath)
        rl_logger.info(f"MAPDDG model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load all models."""
        checkpoint = torch.load(filepath, map_location='cpu')

        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_step = checkpoint['training_step']

        # Load individual actors
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint[f'actor_{i}_state_dict'])
            self.target_actors[i].load_state_dict(checkpoint[f'target_actor_{i}_state_dict'])
            self.actor_optimizers[i].load_state_dict(checkpoint[f'actor_{i}_optimizer_state_dict'])

        rl_logger.info(f"MAPDDG model loaded from {filepath}")

    def get_training_metrics(self) -> Dict[str, float]:
        """Get comprehensive training metrics."""
        metrics = {
            'training_step': self.training_step,
            'num_agents': self.num_agents,
            'replay_buffer_size': len(self.replay_buffer),
            'avg_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0.0
        }

        # Add individual actor metrics
        for i in range(self.num_agents):
            if self.actor_losses[i]:
                metrics[f'avg_actor_{i}_loss'] = np.mean(self.actor_losses[i][-100:])
            else:
                metrics[f'avg_actor_{i}_loss'] = 0.0

        return metrics

    def set_eval_mode(self) -> None:
        """Set all networks to evaluation mode."""
        for actor in self.actors:
            actor.eval()
        self.critic.eval()
        self.is_training = False

    def set_train_mode(self) -> None:
        """Set all networks to training mode."""
        for actor in self.actors:
            actor.train()
        self.critic.train()
        self.is_training = True
        self.is_training = True
