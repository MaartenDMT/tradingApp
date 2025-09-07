"""
Modern RL Agent Implementation
Optimized using PyTorch RL and Stable Baselines3 best practices

Key improvements:
- Standardized agent interface (select_action, update, save/load)
- Modern training loop patterns
- Callback system for monitoring and evaluation
- Performance optimizations
- Better error handling and logging
"""

import logging
import os
import random
import warnings
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Suppress TensorFlow warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

MODEL_PATH = 'data/saved_model'


class BaseAgent(ABC):
    """
    Standardized agent interface following PyTorch RL patterns.

    This abstract base class defines the common interface that all RL agents
    should implement, ensuring consistency across different algorithm implementations.
    """

    def __init__(self, state_size: int, action_size: int, device: str = "auto"):
        """
        Initialize the agent.

        Args:
            state_size: The size of the state space
            action_size: The size of the action space
            device: Device to run computations on ("auto", "cpu", "cuda")
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = self._get_device(device)

        # Performance tracking
        self.episode_rewards: List[float] = []
        self.episode_losses: List[float] = []
        self.step_count: int = 0
        self.episode_count: int = 0

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_device(self, device: str) -> torch.device:
        """Automatically select best available device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action based on the current state.

        Args:
            state: The current state observation

        Returns:
            The selected action
        """
        pass

    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> Optional[float]:
        """
        Update the agent's policy based on experience.

        Args:
            state: The current state
            action: The action taken
            reward: The reward received
            next_state: The next state
            done: Whether the episode has terminated

        Returns:
            Optional loss value for monitoring
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the agent's model to a file."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load the agent's model from a file."""
        pass


class ModernDQNAgent(BaseAgent):
    """
    Modern DQN implementation with PyTorch RL optimizations.

    Features:
    - Standardized interface
    - Efficient memory management
    - Modern network architectures
    - Comprehensive callback system
    - Performance monitoring
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        memory_size: int = 10000,
        target_update_freq: int = 100,
        device: str = "auto",
        model_type: str = "standard",
        hidden_units: int = 64,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(state_size, action_size, device)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.model_type = model_type

        # Memory buffer
        self.memory = deque(maxlen=memory_size)

        # Neural networks
        self.q_network = self._build_network(model_type).to(self.device)
        self.target_network = self._build_network(model_type).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Initialize target network
        self.update_target_network()

        # Callbacks and monitoring
        self.callbacks: List[Callback] = []
        self.metrics: Dict[str, List[float]] = {
            'rewards': [],
            'losses': [],
            'epsilon_values': [],
            'q_values': []
        }

        self.logger.info(f"ModernDQNAgent initialized with {model_type} network")

    def _build_network(self, model_type: str) -> nn.Module:
        """Build neural network based on specified architecture."""
        networks = {
            "standard": StandardDQN,
            "dense": DenseDQN,
            "transformer": TransformerDQN,
            "resnet": ResNetDQN,
            "conv1d": Conv1DDQN
        }

        if model_type not in networks:
            raise ValueError(f"Unknown model type: {model_type}")

        return networks[model_type](
            self.state_size,
            self.action_size,
            self.hidden_units,
            self.dropout
        )

    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)

        # Track Q-values for monitoring
        self.metrics['q_values'].append(q_values.max().item())

        return q_values.argmax().item()

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> Optional[float]:
        """Update the agent using DQN learning algorithm."""
        # Store experience
        self.memory.append((state, action, reward, next_state, done))

        # Only train if we have enough experiences
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        loss = self._train_step(batch)

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        self.step_count += 1
        self.metrics['losses'].append(loss)
        self.metrics['epsilon_values'].append(self.epsilon)

        return loss

    def _train_step(self, batch: List[Tuple]) -> float:
        """Perform a single training step."""
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filepath: str) -> None:
        """Save agent state and model."""
        save_dict = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'metrics': self.metrics,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'model_type': self.model_type,
                'hidden_units': self.hidden_units,
                'dropout': self.dropout
            }
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(save_dict, filepath)
        self.logger.info(f"Agent saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load agent state and model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved agent found at {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.metrics = checkpoint['metrics']

        self.logger.info(f"Agent loaded from {filepath}")

    def add_callback(self, callback: 'Callback') -> None:
        """Add a callback for training monitoring."""
        self.callbacks.append(callback)

    def train(self, env, episodes: int, eval_freq: int = 100) -> Dict[str, List[float]]:
        """
        Train the agent using modern training patterns.

        Args:
            env: The environment to train on
            episodes: Number of episodes to train for
            eval_freq: Frequency of evaluation episodes

        Returns:
            Training metrics
        """
        episode_rewards = []

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_loss = 0
            step_count = 0

            # Episode start callbacks
            for callback in self.callbacks:
                callback.on_episode_start(episode, state)

            while not env.is_episode_done():
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)

                loss = self.update(state, action, reward, next_state, done)
                if loss is not None:
                    episode_loss += loss
                    step_count += 1

                episode_reward += reward
                state = next_state

                # Step callbacks
                for callback in self.callbacks:
                    callback.on_step(state, action, reward, next_state, done)

            episode_rewards.append(episode_reward)
            avg_loss = episode_loss / max(step_count, 1)

            # Episode end callbacks
            for callback in self.callbacks:
                callback.on_episode_end(episode, episode_reward, avg_loss)

            # Periodic evaluation
            if episode % eval_freq == 0 and episode > 0:
                eval_reward = self.evaluate(env, num_episodes=5)
                self.logger.info(f"Episode {episode}: Avg Reward = {np.mean(episode_rewards[-100:]):.2f}, "
                               f"Eval Reward = {eval_reward:.2f}, Epsilon = {self.epsilon:.3f}")

        return {
            'episode_rewards': episode_rewards,
            'metrics': self.metrics
        }

    def evaluate(self, env, num_episodes: int = 5) -> float:
        """Evaluate agent performance without exploration."""
        original_epsilon = self.epsilon
        self.epsilon = 0.0  # No exploration during evaluation

        eval_rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0

            while not env.is_episode_done():
                action = self.select_action(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward

            eval_rewards.append(episode_reward)

        self.epsilon = original_epsilon
        return np.mean(eval_rewards)


# Neural Network Architectures
class StandardDQN(nn.Module):
    """Standard fully connected DQN."""

    def __init__(self, state_size: int, action_size: int, hidden_units: int = 64, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, action_size)
        )

    def forward(self, x):
        return self.network(x)


class DenseDQN(nn.Module):
    """Deeper fully connected network."""

    def __init__(self, state_size: int, action_size: int, hidden_units: int = 64, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_units * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_units * 2, hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units // 2),
            nn.Linear(hidden_units // 2, action_size)
        )

    def forward(self, x):
        return self.network(x)


class TransformerDQN(nn.Module):
    """Transformer-based DQN for sequential data."""

    def __init__(self, state_size: int, action_size: int, hidden_units: int = 64, dropout: float = 0.1):
        super().__init__()
        self.state_size = state_size
        self.hidden_units = hidden_units

        self.embedding = nn.Linear(state_size, hidden_units)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_units,
                nhead=4,
                dim_feedforward=hidden_units * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        self.output = nn.Linear(hidden_units, action_size)

    def forward(self, x):
        # Reshape for transformer if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.output(x)


class ResNetDQN(nn.Module):
    """ResNet-style DQN with skip connections."""

    def __init__(self, state_size: int, action_size: int, hidden_units: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_layer = nn.Linear(state_size, hidden_units)
        self.res_block1 = ResidualBlock(hidden_units, dropout)
        self.res_block2 = ResidualBlock(hidden_units, dropout)
        self.output_layer = nn.Linear(hidden_units, action_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.output_layer(x)


class ResidualBlock(nn.Module):
    """Residual block for ResNet-style architectures."""

    def __init__(self, hidden_units: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_units, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.bn2 = nn.BatchNorm1d(hidden_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        out += residual
        return F.relu(out)


class Conv1DDQN(nn.Module):
    """1D Convolutional DQN for temporal patterns."""

    def __init__(self, state_size: int, action_size: int, hidden_units: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, action_size)
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Callback System for Monitoring and Evaluation
class Callback(ABC):
    """Base class for training callbacks."""

    def on_episode_start(self, episode: int, state: np.ndarray) -> None:
        pass

    def on_step(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool) -> None:
        pass

    def on_episode_end(self, episode: int, episode_reward: float,
                       episode_loss: float) -> None:
        pass


class EvaluationCallback(Callback):
    """Callback for periodic evaluation."""

    def __init__(self, eval_env, eval_freq: int = 100, num_eval_episodes: int = 5):
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.eval_rewards = []

    def on_episode_end(self, episode: int, episode_reward: float, episode_loss: float) -> None:
        if episode % self.eval_freq == 0 and episode > 0:
            # Evaluation logic would go here
            pass


class CheckpointCallback(Callback):
    """Callback for saving model checkpoints."""

    def __init__(self, agent: BaseAgent, save_path: str, save_freq: int = 500):
        self.agent = agent
        self.save_path = save_path
        self.save_freq = save_freq
        self.best_reward = float('-inf')

    def on_episode_end(self, episode: int, episode_reward: float, episode_loss: float) -> None:
        if episode % self.save_freq == 0:
            self.agent.save(f"{self.save_path}/checkpoint_episode_{episode}.pt")

        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.agent.save(f"{self.save_path}/best_model.pt")


class LoggingCallback(Callback):
    """Callback for logging training progress."""

    def __init__(self, log_freq: int = 10):
        self.log_freq = log_freq
        self.recent_rewards = deque(maxlen=100)
        self.logger = logging.getLogger("TrainingLogger")

    def on_episode_end(self, episode: int, episode_reward: float, episode_loss: float) -> None:
        self.recent_rewards.append(episode_reward)

        if episode % self.log_freq == 0:
            avg_reward = np.mean(self.recent_rewards)
            self.logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                           f"Episode Reward = {episode_reward:.2f}, Loss = {episode_loss:.4f}")
