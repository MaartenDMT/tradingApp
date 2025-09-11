"""
Deep Deterministic Policy Gradient (DDPG) Algorithm

DDPG is an actor-critic algorithm for continuous action spaces.
It uses deterministic policy gradients and target networks for stability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
from collections import deque
import random

from ...core.base_agents import BaseRLAgent, AgentConfig


class DDPGConfig(AgentConfig):
    """Configuration for DDPG agent."""
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005  # Soft update rate
    buffer_size: int = 100000
    batch_size: int = 64
    noise_std: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1
    policy_delay: int = 2
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class Actor(nn.Module):
    """Actor network for DDPG."""
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0, hidden_dims: list = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        self.max_action = max_action
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.max_action * self.network(state)


class Critic(nn.Module):
    """Critic network for DDPG."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        # Q1 network
        layers = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.q_network = nn.Sequential(*layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        sa = torch.cat([state, action], dim=-1)
        return self.q_network(sa)


class ReplayBuffer:
    """Experience replay buffer for DDPG."""
    
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.FloatTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch]).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def size(self):
        """Get buffer size."""
        return len(self.buffer)


class DDPGAgent(BaseRLAgent):
    """Deep Deterministic Policy Gradient agent."""
    
    def __init__(self, observation_space, action_space, config: DDPGConfig = None):
        super().__init__(observation_space, action_space, config or DDPGConfig())
        
        # Environment setup
        if hasattr(observation_space, 'shape'):
            self.state_dim = observation_space.shape[0] if len(observation_space.shape) == 1 else observation_space.shape
        else:
            self.state_dim = observation_space
        
        if hasattr(action_space, 'shape'):
            self.action_dim = action_space.shape[0] if len(action_space.shape) == 1 else action_space.shape
        else:
            self.action_dim = action_space
        
        if isinstance(self.state_dim, (list, tuple)):
            self.state_dim = self.state_dim[0]
        if isinstance(self.action_dim, (list, tuple)):
            self.action_dim = self.action_dim[0]
        
        # Networks
        self.actor = Actor(self.state_dim, self.action_dim)
        self.actor_target = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)
        
        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        
        # Training state
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        print(f"DDPG Agent initialized with state_dim={self.state_dim}, action_dim={self.action_dim}")
    
    def act(self, observation, training=True):
        """Select action using the current policy."""
        state = torch.FloatTensor(observation).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        if training:
            # Add exploration noise
            noise = np.random.normal(0, self.config.exploration_noise, size=self.action_dim)
            action = action + noise
            action = np.clip(action, -1, 1)  # Assuming normalized action space
        
        return {'action': action}
    
    def learn(self, experiences):
        """Update the agent using collected experiences."""
        # Add experiences to replay buffer
        for exp in experiences:
            self.replay_buffer.add(
                exp['observation'],
                exp['action'],
                exp['reward'],
                exp['next_observation'],
                exp['done']
            )
        
        # Only train if we have enough experiences
        if self.replay_buffer.size() < self.config.batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        
        # Update critic
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)
        
        # Update actor (delayed)
        actor_loss = None
        if self.training_step % self.config.policy_delay == 0:
            actor_loss = self._update_actor(states)
            self._soft_update_targets()
        
        self.training_step += 1
        self.total_steps += len(experiences)
        
        metrics = {
            'critic_loss': critic_loss,
            'total_steps': self.total_steps,
            'buffer_size': self.replay_buffer.size()
        }
        
        if actor_loss is not None:
            metrics['actor_loss'] = actor_loss
        
        return metrics
    
    def _update_critic(self, states, actions, rewards, next_states, dones):
        """Update critic network."""
        with torch.no_grad():
            # Get target actions
            next_actions = self.actor_target(next_states)
            
            # Add noise to target actions for regularization
            noise = torch.randn_like(next_actions) * self.config.noise_std
            noise = torch.clamp(noise, -self.config.noise_clip, self.config.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1, 1)
            
            # Compute target Q value
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones.float()) * self.config.gamma * target_q
        
        # Current Q value
        current_q = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(self, states):
        """Update actor network."""
        # Actor loss
        actions = self.actor(states)
        actor_loss = -self.critic(states, actions).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _soft_update_targets(self):
        """Soft update target networks."""
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
    
    def save_model(self, filepath: str):
        """Save the agent's model."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config.__dict__,
            'training_step': self.training_step,
            'total_steps': self.total_steps
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load the agent's model."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        metrics = super().get_metrics()
        
        metrics.update({
            'total_steps': self.total_steps,
            'buffer_size': self.replay_buffer.size(),
            'mean_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0
        })
        
        return metrics
