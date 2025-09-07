"""
DQN Family Algorithms Implementation.

This module contains implementations of the Deep Q-Network (DQN) family of algorithms,
including vanilla DQN, Double DQN, Dueling DQN, and Rainbow DQN variants.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import util.loggers as loggers

from ..core.base_agents import (
    AgentConfig,
    EpsilonGreedyExploration,
    ReplayBuffer,
    ValueBasedAgent,
)

logger = loggers.setup_loggers()
rl_logger = logger['rl']


class DQNNetwork(nn.Module):
    """Deep Q-Network with configurable architecture."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: list = [256, 256],
                 activation: str = 'relu',
                 dropout: float = 0.0):
        super(DQNNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build network layers
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

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN Network with separate value and advantage streams."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: list = [256, 256],
                 activation: str = 'relu',
                 dropout: float = 0.0):
        super(DuelingDQNNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Shared feature layers
        shared_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims[:-1]:  # All but last layer
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))

            if activation == 'relu':
                shared_layers.append(nn.ReLU())
            elif activation == 'tanh':
                shared_layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                shared_layers.append(nn.LeakyReLU())

            if dropout > 0:
                shared_layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.shared_network = nn.Sequential(*shared_layers)

        # Value stream
        value_dim = hidden_dims[-1] if hidden_dims else prev_dim
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, value_dim),
            nn.ReLU(),
            nn.Linear(value_dim, 1)
        )

        # Advantage stream
        advantage_dim = hidden_dims[-1] if hidden_dims else prev_dim
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, advantage_dim),
            nn.ReLU(),
            nn.Linear(advantage_dim, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in [self.shared_network, self.value_stream, self.advantage_stream]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dueling architecture."""
        shared_features = self.shared_network(x)

        value = self.value_stream(shared_features)
        advantage = self.advantage_stream(shared_features)

        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


@dataclass
class DQNConfig(AgentConfig):
    """Configuration for DQN agents."""
    learning_rate: float = 0.001
    gamma: float = 0.99
    batch_size: int = 32
    memory_size: int = 50000
    target_update_frequency: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    hidden_dims: list = None
    activation: str = 'relu'
    dropout: float = 0.0
    double_dqn: bool = False
    dueling_dqn: bool = False
    prioritized_replay: bool = False
    noisy_networks: bool = False

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]


class BaseDQNAgent(ValueBasedAgent):
    """Base class for DQN family agents."""

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 config: DQNConfig,
                 name: str = "BaseDQNAgent"):

        self.config = config
        super().__init__(state_dim, num_actions, config, name)

    def _initialize_agent(self) -> None:
        """Initialize DQN-specific components."""
        # Set device
        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')

        # Initialize networks
        if self.config.dueling_dqn:
            self.q_network = DuelingDQNNetwork(
                self.state_dim,
                self.num_actions,
                self.config.hidden_dims,
                self.config.activation,
                self.config.dropout
            ).to(self.device)

            self.target_network = DuelingDQNNetwork(
                self.state_dim,
                self.num_actions,
                self.config.hidden_dims,
                self.config.activation,
                self.config.dropout
            ).to(self.device)
        else:
            self.q_network = DQNNetwork(
                self.state_dim,
                self.num_actions,
                self.config.hidden_dims,
                self.config.activation,
                self.config.dropout
            ).to(self.device)

            self.target_network = DQNNetwork(
                self.state_dim,
                self.num_actions,
                self.config.hidden_dims,
                self.config.activation,
                self.config.dropout
            ).to(self.device)

        # Initialize target network with main network weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            self.config.memory_size,
            self.state_dim,
            1,  # Action dimension for discrete actions
            self.config.random_seed
        )

        # Initialize exploration strategy
        self.exploration_strategy = EpsilonGreedyExploration(
            self.config.epsilon_start,
            self.config.epsilon_end,
            self.config.epsilon_decay
        )

        # Training metrics
        self.loss_history = deque(maxlen=1000)
        self.q_value_history = deque(maxlen=1000)

        rl_logger.info(f"Initialized {self.name} with device: {self.device}")

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if not training:
            # Greedy action during evaluation
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return int(q_values.argmax().item())

        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            q_values_np = q_values.cpu().numpy().flatten()

        # Record Q-values for monitoring
        self.q_value_history.append(np.mean(q_values_np))

        # Select action using exploration strategy
        action = self.exploration_strategy.select_action(q_values_np, self.training_step)

        return action

    def update(self,
               state: np.ndarray,
               action: int,
               reward: float,
               next_state: np.ndarray,
               done: bool) -> Dict[str, float]:
        """Update the agent's Q-network."""
        # Store experience in replay buffer
        self.replay_buffer.store(state, action, reward, next_state, done)

        metrics = {}

        # Update if we have enough experiences
        if self.replay_buffer.can_sample(self.config.batch_size):
            loss = self._update_network()
            metrics['loss'] = loss

            # Update exploration
            self.exploration_strategy.update(self.training_step)
            metrics['epsilon'] = self.exploration_strategy.get_epsilon()

            # Update target network
            if self.training_step % self.config.target_update_frequency == 0:
                self._update_target_network()
                metrics['target_update'] = 1.0

        self.training_step += 1
        return metrics

    def _update_network(self) -> float:
        """Update the Q-network using sampled experiences."""
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions.flatten()).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states).max(1)[0]

            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Record loss
        loss_value = loss.item()
        self.loss_history.append(loss_value)

        return loss_value

    def _update_target_network(self) -> None:
        """Update target network with main network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        rl_logger.debug("Target network updated")

    def _get_agent_state(self) -> Dict[str, Any]:
        """Get agent-specific state for checkpointing."""
        return {
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.exploration_strategy.get_epsilon(),
            'loss_history': list(self.loss_history),
            'q_value_history': list(self.q_value_history)
        }

    def _restore_agent_state(self, state: Dict[str, Any]) -> None:
        """Restore agent-specific state from checkpoint."""
        self.q_network.load_state_dict(state['q_network_state'])
        self.target_network.load_state_dict(state['target_network_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.exploration_strategy.epsilon = state['epsilon']
        self.loss_history = deque(state.get('loss_history', []), maxlen=1000)
        self.q_value_history = deque(state.get('q_value_history', []), maxlen=1000)

    def _save_model_weights(self, filepath: str) -> None:
        """Save model weights."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f"{filepath}.pth")

    def _load_model_weights(self, filepath: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(f"{filepath}.pth", map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics."""
        base_metrics = super().get_training_metrics()

        dqn_metrics = {
            'epsilon': self.exploration_strategy.get_epsilon(),
            'replay_buffer_size': self.replay_buffer.size,
            'avg_loss': np.mean(self.loss_history) if self.loss_history else 0.0,
            'avg_q_value': np.mean(self.q_value_history) if self.q_value_history else 0.0,
            'memory_utilization': self.replay_buffer.size / self.replay_buffer.capacity
        }

        base_metrics.update(dqn_metrics)
        return base_metrics


class DQNAgent(BaseDQNAgent):
    """Standard Deep Q-Network agent."""

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 config: DQNConfig = None):

        if config is None:
            config = DQNConfig()

        super().__init__(state_dim, num_actions, config, "DQNAgent")


class DoubleDQNAgent(BaseDQNAgent):
    """Double Deep Q-Network agent."""

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 config: DQNConfig = None):

        if config is None:
            config = DQNConfig()

        config.double_dqn = True
        super().__init__(state_dim, num_actions, config, "DoubleDQNAgent")


class DuelingDQNAgent(BaseDQNAgent):
    """Dueling Deep Q-Network agent."""

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 config: DQNConfig = None):

        if config is None:
            config = DQNConfig()

        config.dueling_dqn = True
        super().__init__(state_dim, num_actions, config, "DuelingDQNAgent")


class DuelingDoubleDQNAgent(BaseDQNAgent):
    """Dueling Double Deep Q-Network agent."""

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 config: DQNConfig = None):

        if config is None:
            config = DQNConfig()

        config.double_dqn = True
        config.dueling_dqn = True
        super().__init__(state_dim, num_actions, config, "DuelingDoubleDQNAgent")


class RainbowDQNAgent(BaseDQNAgent):
    """Rainbow DQN agent with multiple enhancements."""

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 config: DQNConfig = None):

        if config is None:
            config = DQNConfig()

        # Enable Rainbow features
        config.double_dqn = True
        config.dueling_dqn = True
        config.prioritized_replay = True

        super().__init__(state_dim, num_actions, config, "RainbowDQNAgent")

        rl_logger.info("Initialized Rainbow DQN with Double DQN and Dueling architecture")


def create_dqn_agent(agent_type: str,
                     state_dim: int,
                     num_actions: int,
                     config: DQNConfig = None) -> BaseDQNAgent:
    """
    Factory function to create DQN agents.

    Args:
        agent_type: Type of DQN agent ('dqn', 'double_dqn', 'dueling_dqn', 'rainbow')
        state_dim: State space dimension
        num_actions: Number of actions
        config: Agent configuration

    Returns:
        DQN agent instance
    """
    agent_type = agent_type.lower()

    if agent_type == 'dqn':
        return DQNAgent(state_dim, num_actions, config)
    elif agent_type == 'double_dqn':
        return DoubleDQNAgent(state_dim, num_actions, config)
    elif agent_type == 'dueling_dqn':
        return DuelingDQNAgent(state_dim, num_actions, config)
    elif agent_type == 'dueling_double_dqn':
        return DuelingDoubleDQNAgent(state_dim, num_actions, config)
    elif agent_type == 'rainbow':
        return RainbowDQNAgent(state_dim, num_actions, config)
    else:
        raise ValueError(f"Unknown DQN agent type: {agent_type}")
