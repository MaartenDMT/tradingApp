"""
Core Base Classes for Reinforcement Learning Agents.

This module provides the fundamental base classes and interfaces for all
reinforcement learning agents in the trading system. All agent implementations
should inherit from these base classes to ensure consistency and compatibility.
"""

import abc
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

import util.loggers as loggers

logger = loggers.setup_loggers()
rl_logger = logger['rl']


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    episode: int
    step: int
    reward: float
    loss: Optional[float] = None
    epsilon: Optional[float] = None
    learning_rate: Optional[float] = None
    additional_metrics: Optional[Dict[str, Any]] = None


@dataclass
class AgentConfig:
    """Base configuration for RL agents."""
    learning_rate: float = 0.001
    gamma: float = 0.99
    batch_size: int = 32
    memory_size: int = 10000
    random_seed: Optional[int] = None
    save_frequency: int = 1000
    device: str = 'cpu'


class BaseRLAgent(abc.ABC):
    """
    Abstract base class for all reinforcement learning agents.

    This class defines the common interface that all RL agents must implement,
    providing a consistent API for training, prediction, and model management.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: AgentConfig,
                 name: str = "BaseRLAgent"):
        """
        Initialize the base RL agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Agent configuration
            name: Agent name for identification
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.name = name

        # Training state
        self.training_step = 0
        self.episode_count = 0
        self.total_reward = 0.0
        self.is_training = True

        # Metrics tracking
        self.training_metrics = []
        self.episode_rewards = []

        # Set random seed if provided
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        # Initialize agent-specific components
        self._initialize_agent()

        rl_logger.info(f"Initialized {name} agent with state_dim={state_dim}, action_dim={action_dim}")

    @abc.abstractmethod
    def _initialize_agent(self) -> None:
        """Initialize agent-specific components (networks, optimizers, etc.)."""
        pass

    @abc.abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> Union[int, float, np.ndarray]:
        """
        Select an action given the current state.

        Args:
            state: Current environment state
            training: Whether the agent is in training mode

        Returns:
            Selected action
        """
        pass

    @abc.abstractmethod
    def update(self,
               state: np.ndarray,
               action: Union[int, float, np.ndarray],
               reward: float,
               next_state: np.ndarray,
               done: bool) -> Dict[str, float]:
        """
        Update the agent's policy based on the observed transition.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is finished

        Returns:
            Dictionary of training metrics
        """
        pass

    def train_mode(self) -> None:
        """Set agent to training mode."""
        self.is_training = True

    def eval_mode(self) -> None:
        """Set agent to evaluation mode."""
        self.is_training = False

    def reset_episode(self) -> None:
        """Reset agent state for a new episode."""
        self.episode_count += 1
        self.total_reward = 0.0

    def add_reward(self, reward: float) -> None:
        """Add reward to the current episode total."""
        self.total_reward += reward

    def end_episode(self) -> None:
        """Mark the end of an episode and record metrics."""
        self.episode_rewards.append(self.total_reward)
        rl_logger.debug(f"Episode {self.episode_count} finished with reward: {self.total_reward:.2f}")

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics."""
        if not self.episode_rewards:
            return {}

        return {
            'total_episodes': self.episode_count,
            'training_steps': self.training_step,
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'last_reward': self.episode_rewards[-1],
            'recent_mean_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
        }

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save agent checkpoint.

        Args:
            filepath: Path to save the checkpoint
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save agent state
            checkpoint = {
                'name': self.name,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'config': asdict(self.config),
                'training_step': self.training_step,
                'episode_count': self.episode_count,
                'episode_rewards': self.episode_rewards,
                'training_metrics': self.training_metrics,
                'timestamp': datetime.now().isoformat()
            }

            # Save agent-specific state
            agent_state = self._get_agent_state()
            checkpoint.update(agent_state)

            with open(f"{filepath}.json", 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)

            # Save model weights if available
            self._save_model_weights(f"{filepath}_weights")

            rl_logger.info(f"Checkpoint saved: {filepath}")

        except Exception as e:
            rl_logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load agent checkpoint.

        Args:
            filepath: Path to load the checkpoint from
        """
        try:
            with open(f"{filepath}.json", 'r') as f:
                checkpoint = json.load(f)

            # Restore agent state
            self.training_step = checkpoint['training_step']
            self.episode_count = checkpoint['episode_count']
            self.episode_rewards = checkpoint['episode_rewards']
            self.training_metrics = checkpoint['training_metrics']

            # Restore agent-specific state
            self._restore_agent_state(checkpoint)

            # Load model weights if available
            weights_path = f"{filepath}_weights"
            if os.path.exists(f"{weights_path}.pkl"):
                self._load_model_weights(weights_path)

            rl_logger.info(f"Checkpoint loaded: {filepath}")

        except Exception as e:
            rl_logger.error(f"Failed to load checkpoint: {e}")
            raise

    @abc.abstractmethod
    def _get_agent_state(self) -> Dict[str, Any]:
        """Get agent-specific state for checkpointing."""
        pass

    @abc.abstractmethod
    def _restore_agent_state(self, state: Dict[str, Any]) -> None:
        """Restore agent-specific state from checkpoint."""
        pass

    @abc.abstractmethod
    def _save_model_weights(self, filepath: str) -> None:
        """Save model weights."""
        pass

    @abc.abstractmethod
    def _load_model_weights(self, filepath: str) -> None:
        """Load model weights."""
        pass


class ValueBasedAgent(BaseRLAgent):
    """Base class for value-based RL agents (DQN, Q-Learning, etc.)."""

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 config: AgentConfig,
                 name: str = "ValueBasedAgent"):
        self.num_actions = num_actions
        super().__init__(state_dim, num_actions, config, name)


class PolicyBasedAgent(BaseRLAgent):
    """Base class for policy-based RL agents (REINFORCE, A2C, etc.)."""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: AgentConfig,
                 name: str = "PolicyBasedAgent"):
        super().__init__(state_dim, action_dim, config, name)


class ActorCriticAgent(BaseRLAgent):
    """Base class for actor-critic RL agents (A3C, PPO, SAC, etc.)."""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: AgentConfig,
                 name: str = "ActorCriticAgent"):
        super().__init__(state_dim, action_dim, config, name)


class ReplayBuffer:
    """
    Professional replay buffer for experience replay.

    Efficiently stores and samples experiences for off-policy learning algorithms.
    """

    def __init__(self,
                 capacity: int,
                 state_dim: int,
                 action_dim: int,
                 seed: Optional[int] = None):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer capacity
            state_dim: State dimension
            action_dim: Action dimension
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Pre-allocate arrays for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        self.size = 0
        self.ptr = 0

        if seed is not None:
            np.random.seed(seed)

    def store(self,
              state: np.ndarray,
              action: Union[int, float, np.ndarray],
              reward: float,
              next_state: np.ndarray,
              done: bool) -> None:
        """Store experience in buffer."""
        self.states[self.ptr] = state
        if isinstance(action, (int, float)):
            self.actions[self.ptr] = [action] if self.action_dim == 1 else action
        else:
            self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample random batch of experiences."""
        if self.size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} experiences, only have {self.size}")

        indices = np.random.choice(self.size, batch_size, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences to sample."""
        return self.size >= batch_size

    def clear(self) -> None:
        """Clear the buffer."""
        self.size = 0
        self.ptr = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'capacity': self.capacity,
            'size': self.size,
            'utilization': self.size / self.capacity,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }


class EpisodeBuffer:
    """
    Episode buffer for on-policy algorithms.

    Stores full episodes for algorithms like REINFORCE, A2C, PPO.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def store(self,
              state: np.ndarray,
              action: Union[int, float, np.ndarray],
              reward: float,
              log_prob: Optional[float] = None,
              value: Optional[float] = None,
              done: bool = False) -> None:
        """Store step information."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if log_prob is not None:
            self.log_probs.append(log_prob)
        if value is not None:
            self.values.append(value)
        self.dones.append(done)

    def get_episode_data(self) -> Dict[str, np.ndarray]:
        """Get episode data as numpy arrays."""
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'log_probs': np.array(self.log_probs) if self.log_probs else None,
            'values': np.array(self.values) if self.values else None,
            'dones': np.array(self.dones)
        }

    def clear(self) -> None:
        """Clear the episode buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)


class ExplorationStrategy(abc.ABC):
    """Base class for exploration strategies."""

    @abc.abstractmethod
    def select_action(self, q_values: np.ndarray, step: int) -> int:
        """Select action based on exploration strategy."""
        pass

    @abc.abstractmethod
    def update(self, step: int) -> None:
        """Update exploration parameters."""
        pass


class EpsilonGreedyExploration(ExplorationStrategy):
    """Epsilon-greedy exploration strategy."""

    def __init__(self,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start

    def select_action(self, q_values: np.ndarray, step: int) -> int:
        """Select action using epsilon-greedy strategy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(len(q_values))
        return np.argmax(q_values)

    def update(self, step: int) -> None:
        """Update epsilon value."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        return self.epsilon
