"""
Base Agent Class

Abstract base class for all RL agents in the system.
Provides common interface and functionality for all algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

class BaseAgent(ABC):
    """Abstract base class for all RL agents."""
    
    def __init__(self, observation_space, action_space, config: Dict[str, Any]):
        """
        Initialize the base agent.
        
        Args:
            observation_space: The observation space of the environment
            action_space: The action space of the environment
            config: Configuration dictionary for the agent
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        
        # Common attributes
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.device = config.get('device', 'cpu')
        
        # Training state
        self.training = True
        self.total_steps = 0
        self.episode_count = 0
        
        # Metrics tracking
        self.metrics = {
            'total_reward': 0.0,
            'episode_rewards': [],
            'losses': [],
            'exploration_rate': 1.0
        }
        
        print(f"✅ Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def act(self, observation: np.ndarray) -> Union[int, np.ndarray]:
        """
        Select an action given an observation.
        
        Args:
            observation: The current observation
            
        Returns:
            The selected action
        """
        pass
    
    @abstractmethod
    def learn(self, experiences: List[Tuple]) -> Dict[str, float]:
        """
        Learn from a batch of experiences.
        
        Args:
            experiences: List of (state, action, reward, next_state, done) tuples
            
        Returns:
            Dictionary of learning metrics (e.g., loss values)
        """
        pass
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience in memory (if applicable).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Default implementation - override in subclasses that use replay memory
        pass
    
    def update_metrics(self, reward: float, metrics: Optional[Dict[str, float]] = None):
        """
        Update agent metrics.
        
        Args:
            reward: Reward received in the current step
            metrics: Additional metrics to update
        """
        self.metrics['total_reward'] += reward
        
        if metrics:
            for key, value in metrics.items():
                if key in self.metrics:
                    if isinstance(self.metrics[key], list):
                        self.metrics[key].append(value)
                    else:
                        self.metrics[key] = value
                else:
                    self.metrics[key] = value
    
    def end_episode(self, episode_reward: float):
        """
        Called at the end of each episode.
        
        Args:
            episode_reward: Total reward for the episode
        """
        self.episode_count += 1
        self.metrics['episode_rewards'].append(episode_reward)
        self.metrics['total_reward'] = 0.0  # Reset for next episode
    
    def set_training_mode(self, training: bool):
        """
        Set the agent to training or evaluation mode.
        
        Args:
            training: Whether the agent should be in training mode
        """
        self.training = training
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current agent metrics.
        
        Returns:
            Dictionary of current metrics
        """
        return self.metrics.copy()
    
    def save_model(self, filepath: str):
        """
        Save the agent's model to file.
        
        Args:
            filepath: Path to save the model
        """
        # Default implementation - override in subclasses
        print(f"⚠️  Model saving not implemented for {self.__class__.__name__}")
    
    def load_model(self, filepath: str):
        """
        Load the agent's model from file.
        
        Args:
            filepath: Path to load the model from
        """
        # Default implementation - override in subclasses
        print(f"⚠️  Model loading not implemented for {self.__class__.__name__}")
    
    def reset(self):
        """Reset the agent's internal state."""
        self.total_steps = 0
        self.metrics = {
            'total_reward': 0.0,
            'episode_rewards': [],
            'losses': [],
            'exploration_rate': 1.0
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get the agent's configuration."""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Update the agent's configuration.
        
        Args:
            new_config: New configuration parameters
        """
        self.config.update(new_config)
        
        # Update common attributes
        self.learning_rate = self.config.get('learning_rate', self.learning_rate)
        self.gamma = self.config.get('gamma', self.gamma)
        self.device = self.config.get('device', self.device)


class DummyAgent(BaseAgent):
    """
    Dummy agent implementation for testing and fallback purposes.
    """
    
    def __init__(self, observation_space, action_space, config: Dict[str, Any]):
        super().__init__(observation_space, action_space, config)
        self.action_size = getattr(action_space, 'n', action_space.shape[0] if hasattr(action_space, 'shape') else 1)
    
    def act(self, observation: np.ndarray) -> Union[int, np.ndarray]:
        """Select a random action."""
        if hasattr(self.action_space, 'sample'):
            return self.action_space.sample()
        else:
            # Fallback for simple action spaces
            return np.random.randint(0, self.action_size)
    
    def learn(self, experiences: List[Tuple]) -> Dict[str, float]:
        """Dummy learning - returns zero loss."""
        return {'loss': 0.0}


# Simple space classes for compatibility
class DiscreteSpace:
    """Simple discrete action space."""
    
    def __init__(self, n: int):
        self.n = n
    
    def sample(self):
        return np.random.randint(0, self.n)


class BoxSpace:
    """Simple continuous action space."""
    
    def __init__(self, low, high, shape):
        self.low = np.array(low)
        self.high = np.array(high)
        self.shape = shape if isinstance(shape, tuple) else (shape,)
    
    def sample(self):
        return np.random.uniform(self.low, self.high, self.shape)
