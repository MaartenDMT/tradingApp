"""
Modern MYN Agent Implementation
Enhanced version of the custom MYN agent using modern RL patterns and PyTorch optimizations

Key improvements:
- Converted from TensorFlow to PyTorch for better performance
- Standardized agent interface compatible with modern_agent.py
- Advanced transformer architectures
- Efficient memory management
- Modern training patterns
"""

import os
import random
import warnings
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.reinforcement.agents.agent_utils import is_new_record, save_to_csv
from model.reinforcement.agents.modern_agent import BaseAgent

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

MODEL_PATH = 'data/saved_model/myn_agents'


class ModernMYNAgent(BaseAgent):
    """
    Modern MYN (My Neural) Agent with PyTorch and advanced features.

    Features:
    - Multiple neural network architectures
    - Efficient PyTorch implementation
    - Advanced transformer models
    - Standardized interface
    - Performance optimizations
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        model_type: str = "transformer",
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        memory_size: int = 10000,
        hidden_units: int = 128,
        dropout: float = 0.1,
        device: str = "auto",
        **kwargs
    ):
        super().__init__(state_size, action_size, device)

        # Hyperparameters
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.dropout = dropout

        # Memory and tracking
        self.memory = deque(maxlen=memory_size)
        self.state_history: List[np.ndarray] = []
        self.reward_history: List[float] = []
        self.action_history: List[int] = []

        # Model and optimizer
        self.model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Performance tracking
        self.best_reward = float('-inf')
        self.model_count = 1

        self.logger.info(f"ModernMYNAgent initialized with {model_type} architecture")

    def _build_model(self) -> nn.Module:
        """Build neural network based on specified architecture."""
        models = {
            "standard": StandardMYNNet,
            "dense": DenseMYNNet,
            "transformer": TransformerMYNNet,
            "resnet": ResNetMYNNet,
            "conv1d": Conv1DMYNNet,
            "lstm": LSTMMYNNet,
            "advanced_transformer": AdvancedTransformerMYNNet
        }

        if self.model_type not in models:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return models[self.model_type](
            self.state_size,
            self.action_size,
            self.hidden_units,
            self.dropout
        )

    def select_action(self, state: np.ndarray, method: str = "epsilon_greedy") -> int:
        """
        Select action using various strategies.

        Args:
            state: Current state observation
            method: Action selection method ("epsilon_greedy", "softmax", "greedy")

        Returns:
            Selected action
        """
        if method == "epsilon_greedy" and np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_tensor)

        if method == "softmax":
            # Softmax action selection
            action_probs = F.softmax(q_values / 0.1, dim=-1)  # Temperature = 0.1
            action = torch.multinomial(action_probs, 1).item()
        else:
            # Greedy action selection
            action = q_values.argmax().item()

        return action

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> Optional[float]:
        """Update the agent using DQN learning with experience replay."""
        # Store experience
        self.memory.append((state, action, reward, next_state, done))
        self.state_history.append(state)
        self.reward_history.append(reward)
        self.action_history.append(action)

        # Only train if we have enough experiences
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch and train
        loss = self._replay_experience()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def _replay_experience(self) -> float:
        """Experience replay training step."""
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # Next Q-values
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def save(self, filepath: str) -> None:
        """Save agent state and model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'model_type': self.model_type,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'hidden_units': self.hidden_units,
                'dropout': self.dropout
            },
            'performance_data': {
                'best_reward': self.best_reward,
                'recent_rewards': self.reward_history[-100:] if len(self.reward_history) > 100 else self.reward_history
            }
        }

        torch.save(save_dict, filepath)
        self.logger.info(f"MYN Agent saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load agent state and model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved agent found at {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.best_reward = checkpoint['performance_data']['best_reward']

        self.logger.info(f"MYN Agent loaded from {filepath}")

    def train_episodes(self, env, episodes: int, eval_freq: int = 100) -> Dict[str, List[float]]:
        """
        Train the agent for specified number of episodes.

        Args:
            env: Training environment
            episodes: Number of training episodes
            eval_freq: Frequency of evaluation

        Returns:
            Training metrics
        """
        episode_rewards = []
        episode_losses = []

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_loss = 0
            step_count = 0

            while not env.is_episode_done():
                action = self.select_action(state, "epsilon_greedy")
                next_state, reward, done, _ = env.step(action)

                loss = self.update(state, action, reward, next_state, done)
                if loss is not None:
                    episode_loss += loss
                    step_count += 1

                episode_reward += reward
                state = next_state

            episode_rewards.append(episode_reward)
            avg_loss = episode_loss / max(step_count, 1)
            episode_losses.append(avg_loss)

            # Check for best performance
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self._save_best_model(env)

            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                self.logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                               f"Epsilon = {self.epsilon:.3f}, Loss = {avg_loss:.4f}")

            # Periodic evaluation
            if episode % eval_freq == 0 and episode > 0:
                eval_reward = self._evaluate(env, num_episodes=5)
                self.logger.info(f"Episode {episode}: Evaluation Reward = {eval_reward:.2f}")

        return {
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses
        }

    def _evaluate(self, env, num_episodes: int = 5) -> float:
        """Evaluate agent performance without exploration."""
        original_epsilon = self.epsilon
        self.epsilon = 0.0  # No exploration during evaluation

        eval_rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0

            while not env.is_episode_done():
                action = self.select_action(state, "greedy")
                state, reward, done, _ = env.step(action)
                episode_reward += reward

            eval_rewards.append(episode_reward)

        self.epsilon = original_epsilon
        return np.mean(eval_rewards)

    def _save_best_model(self, env):
        """Save model when achieving best performance."""
        if hasattr(env, 'accuracy'):
            should_save = is_new_record(self.best_reward, env.accuracy, self.model_type)
            if should_save:
                model_params = self._get_model_parameters(env)
                save_to_csv(self.best_reward, env.accuracy, model_params)

                filename = f"{MODEL_PATH}/best_{self.model_type}_{self.model_count}.pt"
                self.save(filename)

    def _get_model_parameters(self, env) -> Dict[str, Any]:
        """Get model parameters for logging."""
        return {
            "model_name": self.model_type,
            "agent_type": "ModernMYNAgent",
            "gamma": self.gamma,
            "hidden_units": self.hidden_units,
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "dropout": self.dropout,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "best_reward": self.best_reward,
            "env_accuracy": getattr(env, 'accuracy', 'N/A')
        }


# Neural Network Architectures for MYN Agent
class StandardMYNNet(nn.Module):
    """Standard fully connected network."""

    def __init__(self, state_size: int, action_size: int, hidden_units: int = 128, dropout: float = 0.1):
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


class DenseMYNNet(nn.Module):
    """Dense network with multiple layers."""

    def __init__(self, state_size: int, action_size: int, hidden_units: int = 128, dropout: float = 0.1):
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


class TransformerMYNNet(nn.Module):
    """Transformer-based network for sequential patterns."""

    def __init__(self, state_size: int, action_size: int, hidden_units: int = 128, dropout: float = 0.1):
        super().__init__()
        self.state_size = state_size
        self.hidden_units = hidden_units

        self.embedding = nn.Linear(state_size, hidden_units)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, hidden_units))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_units,
                nhead=8,
                dim_feedforward=hidden_units * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=3
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units // 2, action_size)
        )

    def forward(self, x):
        # Reshape if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Embedding and positional encoding
        x = self.embedding(x)
        x = x + self.positional_encoding.expand(x.size(0), -1, -1)

        # Transformer processing
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling

        return self.output(x)


class AdvancedTransformerMYNNet(nn.Module):
    """Advanced transformer with multi-head attention and residual connections."""

    def __init__(self, state_size: int, action_size: int, hidden_units: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden_units = hidden_units

        # Input projection
        self.input_projection = nn.Linear(state_size, hidden_units)

        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_units, num_heads=8, dropout=dropout, batch_first=True)
            for _ in range(3)
        ])

        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_units, hidden_units * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_units * 2, hidden_units)
            )
            for _ in range(3)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_units) for _ in range(6)
        ])

        # Output layers
        self.output = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, action_size)
        )

    def forward(self, x):
        # Reshape and project input
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = self.input_projection(x)

        # Apply transformer layers with residual connections
        for i, (attn, ffn) in enumerate(zip(self.attention_layers, self.ffn_layers)):
            # Multi-head attention with residual connection
            attn_out, _ = attn(x, x, x)
            x = self.layer_norms[i * 2](x + attn_out)

            # Feed-forward with residual connection
            ffn_out = ffn(x)
            x = self.layer_norms[i * 2 + 1](x + ffn_out)

        # Global average pooling and output
        x = x.mean(dim=1)
        return self.output(x)


class ResNetMYNNet(nn.Module):
    """ResNet-style network with skip connections."""

    def __init__(self, state_size: int, action_size: int, hidden_units: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_layer = nn.Linear(state_size, hidden_units)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_units, dropout) for _ in range(3)
        ])

        self.output_layer = nn.Linear(hidden_units, action_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        for res_block in self.res_blocks:
            x = res_block(x)

        return self.output_layer(x)


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""

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


class Conv1DMYNNet(nn.Module):
    """1D Convolutional network for temporal patterns."""

    def __init__(self, state_size: int, action_size: int, hidden_units: int = 128, dropout: float = 0.1):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, action_size)
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc_layers(x)


class LSTMMYNNet(nn.Module):
    """LSTM-based network for sequential data."""

    def __init__(self, state_size: int, action_size: int, hidden_units: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden_units = hidden_units

        self.lstm = nn.LSTM(
            input_size=state_size,
            hidden_size=hidden_units,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units // 2, action_size)
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        lstm_out, _ = self.lstm(x)
        # Use the last output of the sequence
        last_output = lstm_out[:, -1, :]
        return self.output(last_output)
