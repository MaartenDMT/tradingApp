"""
Simple Planning Agent

A basic model-based RL agent that learns a model of the environment
and uses it for planning.
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


class PlanningConfig(AgentConfig):
    """Configuration for planning agent."""
    learning_rate: float = 1e-3
    gamma: float = 0.99
    planning_steps: int = 10
    model_buffer_size: int = 10000
    batch_size: int = 32
    planning_horizon: int = 5
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class EnvironmentModel(nn.Module):
    """Simple neural network model of the environment."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Model predicts next state and reward
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 1)  # next_state + reward
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next state and reward."""
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        # Convert action to one-hot if discrete
        if action.dtype == torch.long:
            action_onehot = torch.zeros(action.size(0), self.action_dim)
            action_onehot.scatter_(1, action.unsqueeze(1), 1)
            action = action_onehot
        
        input_tensor = torch.cat([state, action], dim=-1)
        output = self.model(input_tensor)
        
        next_state = output[..., :-1]
        reward = output[..., -1:]
        
        return next_state, reward


class QNetwork(nn.Module):
    """Q-network for value estimation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.q_network(state)


class SimplePlanningAgent(BaseRLAgent):
    """Simple model-based RL agent with planning."""
    
    def __init__(self, observation_space, action_space, config: PlanningConfig = None):
        super().__init__(observation_space, action_space, config or PlanningConfig())
        
        # Environment setup
        if hasattr(observation_space, 'shape'):
            self.state_dim = observation_space.shape[0] if len(observation_space.shape) == 1 else observation_space.shape
        else:
            self.state_dim = observation_space
        
        if hasattr(action_space, 'n'):
            self.action_dim = action_space.n
        else:
            self.action_dim = action_space
        
        if isinstance(self.state_dim, (list, tuple)):
            self.state_dim = self.state_dim[0]
        
        # Networks
        self.model = EnvironmentModel(self.state_dim, self.action_dim)
        self.q_network = QNetwork(self.state_dim, self.action_dim)
        
        # Optimizers
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        
        # Experience buffer
        self.model_buffer = deque(maxlen=self.config.model_buffer_size)
        
        # Training state
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        print(f"SimplePlanning Agent initialized with state_dim={self.state_dim}, action_dim={self.action_dim}")
    
    def act(self, observation, training=True):
        """Select action using epsilon-greedy policy."""
        state = torch.FloatTensor(observation).unsqueeze(0)
        
        if training and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                action = q_values.argmax().item()
        
        return {'action': action}
    
    def learn(self, experiences):
        """Update the agent using collected experiences."""
        # Add experiences to buffer
        for exp in experiences:
            self.model_buffer.append(exp)
        
        if len(self.model_buffer) < self.config.batch_size:
            return {}
        
        # Update model
        model_loss = self._update_model()
        
        # Update Q-network using both real and simulated experiences
        q_loss = self._update_q_network(experiences)
        
        # Planning with learned model
        planning_loss = self._planning_update()
        
        # Update exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_step += 1
        
        return {
            'model_loss': model_loss,
            'q_loss': q_loss,
            'planning_loss': planning_loss,
            'epsilon': self.epsilon
        }
    
    def _update_model(self):
        """Update environment model."""
        # Sample batch
        batch = random.sample(self.model_buffer, min(self.config.batch_size, len(self.model_buffer)))
        
        states = torch.FloatTensor([exp['observation'] for exp in batch])
        actions = torch.LongTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_states = torch.FloatTensor([exp['next_observation'] for exp in batch])
        
        # Predict next state and reward
        pred_next_states, pred_rewards = self.model(states, actions)
        
        # Compute losses
        state_loss = F.mse_loss(pred_next_states, next_states)
        reward_loss = F.mse_loss(pred_rewards.squeeze(), rewards)
        
        total_loss = state_loss + reward_loss
        
        # Update model
        self.model_optimizer.zero_grad()
        total_loss.backward()
        self.model_optimizer.step()
        
        return total_loss.item()
    
    def _update_q_network(self, experiences):
        """Update Q-network using real experiences."""
        if len(experiences) == 0:
            return 0.0
        
        states = torch.FloatTensor([exp['observation'] for exp in experiences])
        actions = torch.LongTensor([exp['action'] for exp in experiences])
        rewards = torch.FloatTensor([exp['reward'] for exp in experiences])
        next_states = torch.FloatTensor([exp['next_observation'] for exp in experiences])
        dones = torch.BoolTensor([exp['done'] for exp in experiences])
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.q_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones.float()) * self.config.gamma * next_q
        
        # Q loss
        q_loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Update Q-network
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        return q_loss.item()
    
    def _planning_update(self):
        """Perform planning updates using learned model."""
        if len(self.model_buffer) < self.config.batch_size:
            return 0.0
        
        total_loss = 0.0
        
        for _ in range(self.config.planning_steps):
            # Sample random starting states from buffer
            start_experiences = random.sample(self.model_buffer, min(self.config.batch_size, len(self.model_buffer)))
            start_states = torch.FloatTensor([exp['observation'] for exp in start_experiences])
            
            # Simulate trajectories
            simulated_q_loss = self._simulate_trajectory(start_states)
            total_loss += simulated_q_loss
        
        return total_loss / self.config.planning_steps
    
    def _simulate_trajectory(self, start_states):
        """Simulate trajectory using learned model."""
        current_states = start_states
        total_loss = 0.0
        
        for step in range(self.config.planning_horizon):
            # Select actions using current policy
            with torch.no_grad():
                q_values = self.q_network(current_states)
                actions = q_values.argmax(dim=1)
            
            # Predict next states and rewards using model
            with torch.no_grad():
                next_states, rewards = self.model(current_states, actions)
            
            # Update Q-network using simulated experience
            current_q = self.q_network(current_states).gather(1, actions.unsqueeze(1))
            
            with torch.no_grad():
                next_q = self.q_network(next_states).max(1)[0]
                target_q = rewards.squeeze() + self.config.gamma * next_q
            
            # Q loss for this step
            step_loss = F.mse_loss(current_q.squeeze(), target_q)
            
            # Update Q-network
            self.q_optimizer.zero_grad()
            step_loss.backward()
            self.q_optimizer.step()
            
            total_loss += step_loss.item()
            current_states = next_states.detach()
        
        return total_loss / self.config.planning_horizon
    
    def save_model(self, filepath: str):
        """Save the agent's model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'q_network_state_dict': self.q_network.state_dict(),
            'model_optimizer_state_dict': self.model_optimizer.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'config': self.config.__dict__,
            'training_step': self.training_step,
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load the agent's model."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        self.epsilon = checkpoint.get('epsilon', 1.0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        metrics = super().get_metrics()
        
        metrics.update({
            'epsilon': self.epsilon,
            'buffer_size': len(self.model_buffer)
        })
        
        return metrics


# Alias for factory
MCTSAgent = SimplePlanningAgent
