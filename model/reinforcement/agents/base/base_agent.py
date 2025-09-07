"""
Base Agent Classes for Reinforcement Learning Trading.

Provides abstract base classes and common functionality for all RL agents
in the enhanced trading system with professional patterns.
"""

import abc
import json
import pickle
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

import util.loggers as loggers

logger = loggers.setup_loggers()
agent_logger = logger['agent']


class BaseAgent(abc.ABC):
    """
    Abstract base class for all reinforcement learning agents.

    Provides common interface and functionality that all agents should implement.
    Follows professional design patterns for RL agent development.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 name: str = "BaseAgent",
                 config: Optional[Dict] = None):
        """
        Initialize base agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            name: Agent name for logging and saving
            config: Optional configuration dictionary
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = name
        self.config = config or {}

        # Training state
        self.training_step = 0
        self.episode_count = 0
        self.total_reward = 0.0
        self.training_history = []

        # Agent metadata
        self.created_at = datetime.now()
        self.agent_id = f"{name}_{self.created_at.strftime('%Y%m%d_%H%M%S')}"

        agent_logger.info(f"Initialized {self.name} agent with ID: {self.agent_id}")

    @abc.abstractmethod
    def act(self, state: np.ndarray, **kwargs) -> Union[int, float, np.ndarray]:
        """
        Select action given current state.

        Args:
            state: Current environment state
            **kwargs: Additional arguments for action selection

        Returns:
            Selected action
        """
        pass

    @abc.abstractmethod
    def learn(self, experiences: Tuple, **kwargs) -> Dict[str, float]:
        """
        Learn from experiences/transitions.

        Args:
            experiences: Tuple of (state, action, reward, next_state, done)
            **kwargs: Additional learning parameters

        Returns:
            Dictionary of learning metrics (loss, etc.)
        """
        pass

    @abc.abstractmethod
    def save(self, filepath: str) -> None:
        """
        Save agent state to file.

        Args:
            filepath: Path to save agent
        """
        pass

    @abc.abstractmethod
    def load(self, filepath: str) -> None:
        """
        Load agent state from file.

        Args:
            filepath: Path to load agent from
        """
        pass

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.episode_count += 1
        self.total_reward = 0.0
        agent_logger.debug(f"Agent reset for episode {self.episode_count}")

    def update_training_step(self) -> None:
        """Increment training step counter."""
        self.training_step += 1

    def add_reward(self, reward: float) -> None:
        """Add reward to episode total."""
        self.total_reward += reward

    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.

        Returns:
            Dictionary of agent statistics
        """
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'total_reward': self.total_reward,
            'created_at': self.created_at.isoformat(),
            'config': self.config
        }

    def save_training_history(self, filepath: str) -> None:
        """
        Save training history to file.

        Args:
            filepath: Path to save training history
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            agent_logger.info(f"Training history saved to {filepath}")
        except Exception as e:
            agent_logger.error(f"Failed to save training history: {e}")

    def load_training_history(self, filepath: str) -> None:
        """
        Load training history from file.

        Args:
            filepath: Path to load training history from
        """
        try:
            with open(filepath, 'r') as f:
                self.training_history = json.load(f)
            agent_logger.info(f"Training history loaded from {filepath}")
        except Exception as e:
            agent_logger.error(f"Failed to load training history: {e}")


class DiscreteAgent(BaseAgent):
    """
    Base class for agents with discrete action spaces.

    Provides common functionality for discrete action agents like DQN, DDQN.
    """

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 name: str = "DiscreteAgent",
                 config: Optional[Dict] = None):
        """
        Initialize discrete action agent.

        Args:
            state_dim: Dimension of state space
            num_actions: Number of discrete actions
            name: Agent name
            config: Optional configuration
        """
        super().__init__(state_dim, num_actions, name, config)
        self.num_actions = num_actions

        # Exploration parameters
        self.epsilon = config.get('epsilon_start', 1.0) if config else 1.0
        self.epsilon_min = config.get('epsilon_min', 0.01) if config else 0.01
        self.epsilon_decay = config.get('epsilon_decay', 0.995) if config else 0.995

        agent_logger.info(f"Initialized discrete agent with {num_actions} actions")

    def update_epsilon(self) -> None:
        """Update epsilon for epsilon-greedy exploration."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_epsilon_stats(self) -> Dict[str, float]:
        """Get epsilon exploration statistics."""
        return {
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay
        }


class ContinuousAgent(BaseAgent):
    """
    Base class for agents with continuous action spaces.

    Provides common functionality for continuous action agents like TD3, SAC.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 action_bounds: Tuple[float, float] = (-1.0, 1.0),
                 name: str = "ContinuousAgent",
                 config: Optional[Dict] = None):
        """
        Initialize continuous action agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_bounds: Tuple of (min_action, max_action)
            name: Agent name
            config: Optional configuration
        """
        super().__init__(state_dim, action_dim, name, config)
        self.action_bounds = action_bounds

        # Noise parameters for exploration
        self.noise_std = config.get('noise_std', 0.1) if config else 0.1
        self.noise_clip = config.get('noise_clip', 0.5) if config else 0.5

        agent_logger.info(f"Initialized continuous agent with bounds {action_bounds}")

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """
        Clip action to valid bounds.

        Args:
            action: Raw action from network

        Returns:
            Clipped action within bounds
        """
        return np.clip(action, self.action_bounds[0], self.action_bounds[1])

    def add_noise(self, action: np.ndarray, noise_std: Optional[float] = None) -> np.ndarray:
        """
        Add exploration noise to action.

        Args:
            action: Base action
            noise_std: Noise standard deviation (optional)

        Returns:
            Action with added noise
        """
        if noise_std is None:
            noise_std = self.noise_std

        noise = np.random.normal(0, noise_std, size=action.shape)
        noise = np.clip(noise, -self.noise_clip, self.noise_clip)

        return self.clip_action(action + noise)

    def get_noise_stats(self) -> Dict[str, float]:
        """Get noise exploration statistics."""
        return {
            'noise_std': self.noise_std,
            'noise_clip': self.noise_clip,
            'action_bounds': self.action_bounds
        }


class ReplayBuffer:
    """
    Professional replay buffer implementation for experience replay.

    Provides efficient storage and sampling of experiences with
    professional memory management and statistics tracking.
    """

    def __init__(self,
                 capacity: int,
                 state_dim: int,
                 action_dim: int,
                 seed: Optional[int] = None):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
            state_dim: State dimension
            action_dim: Action dimension
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize storage arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        # Buffer management
        self.size = 0
        self.index = 0

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        agent_logger.info(f"Initialized replay buffer with capacity {capacity}")

    def add(self,
            state: np.ndarray,
            action: Union[int, float, np.ndarray],
            reward: float,
            next_state: np.ndarray,
            done: bool) -> None:
        """
        Add experience to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        # Store experience
        self.states[self.index] = state
        if isinstance(action, (int, float)):
            # Convert scalar action to array
            self.actions[self.index] = [action] if self.action_dim == 1 else action
        else:
            self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done

        # Update pointers
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample random batch of experiences.

        Args:
            batch_size: Size of batch to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough experiences to sample {batch_size} (have {self.size})")

        # Random sampling
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
        self.index = 0
        agent_logger.info("Replay buffer cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'capacity': self.capacity,
            'size': self.size,
            'utilization': self.size / self.capacity,
            'index': self.index,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }

    def save(self, filepath: str) -> None:
        """
        Save buffer to file.

        Args:
            filepath: Path to save buffer
        """
        try:
            buffer_data = {
                'states': self.states[:self.size],
                'actions': self.actions[:self.size],
                'rewards': self.rewards[:self.size],
                'next_states': self.next_states[:self.size],
                'dones': self.dones[:self.size],
                'capacity': self.capacity,
                'size': self.size,
                'index': self.index,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim
            }

            with open(filepath, 'wb') as f:
                pickle.dump(buffer_data, f)

            agent_logger.info(f"Replay buffer saved to {filepath}")

        except Exception as e:
            agent_logger.error(f"Failed to save replay buffer: {e}")
            raise

    def load(self, filepath: str) -> None:
        """
        Load buffer from file.

        Args:
            filepath: Path to load buffer from
        """
        try:
            with open(filepath, 'rb') as f:
                buffer_data = pickle.load(f)

            # Restore buffer state
            self.capacity = buffer_data['capacity']
            self.size = buffer_data['size']
            self.index = buffer_data['index']
            self.state_dim = buffer_data['state_dim']
            self.action_dim = buffer_data['action_dim']

            # Restore arrays
            self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
            self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
            self.rewards = np.zeros(self.capacity, dtype=np.float32)
            self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
            self.dones = np.zeros(self.capacity, dtype=bool)

            # Fill with saved data
            self.states[:self.size] = buffer_data['states']
            self.actions[:self.size] = buffer_data['actions']
            self.rewards[:self.size] = buffer_data['rewards']
            self.next_states[:self.size] = buffer_data['next_states']
            self.dones[:self.size] = buffer_data['dones']

            agent_logger.info(f"Replay buffer loaded from {filepath}")

        except Exception as e:
            agent_logger.error(f"Failed to load replay buffer: {e}")
            raise
