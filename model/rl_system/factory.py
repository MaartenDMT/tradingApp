"""
RL Algorithm Factory

Central factory for creating and managing RL algorithms.
Provides a unified interface for algorithm instantiation and registration.
"""

from typing import Dict, Type, Any, Optional
import importlib
from pathlib import Path

class RLAlgorithmFactory:
    """Factory class for creating RL algorithms."""
    
    _algorithms: Dict[str, Type] = {}
    _initialized = False
    
    @classmethod
    def initialize(cls):
        """Initialize the factory with available algorithms."""
        if cls._initialized:
            return
        
        try:
            # Import and register algorithms based on actual structure
            cls._register_value_based_algorithms()
            cls._register_policy_based_algorithms()
            cls._register_actor_critic_algorithms()
            cls._register_exploration_algorithms()
            
            cls._initialized = True
            print(f"✅ Initialized RL Algorithm Factory with {len(cls._algorithms)} algorithms")
            
        except Exception as e:
            print(f"⚠️  Warning: Some RL algorithms may not be available: {e}")
            cls._initialized = True
    
    @classmethod
    def _register_value_based_algorithms(cls):
        """Register value-based algorithms."""
        try:
            # DQN Family
            from model.rl_system.algorithms.value_based.dqn_family import (
                DQNAgent, DoubleDQNAgent, DuelingDQNAgent, RainbowDQNAgent
            )
            cls._algorithms['dqn'] = DQNAgent
            cls._algorithms['double_dqn'] = DoubleDQNAgent
            cls._algorithms['dueling_dqn'] = DuelingDQNAgent
            cls._algorithms['rainbow_dqn'] = RainbowDQNAgent
            
            # Tabular Methods
            from model.rl_system.algorithms.value_based.tabular_methods import (
                QLearningAgent, SARSAAgent, MonteCarloAgent, ExpectedSARSAAgent
            )
            cls._algorithms['q_learning'] = QLearningAgent
            cls._algorithms['sarsa'] = SARSAAgent
            cls._algorithms['monte_carlo'] = MonteCarloAgent
            cls._algorithms['expected_sarsa'] = ExpectedSARSAAgent
            
        except ImportError as e:
            print(f"⚠️  Some value-based algorithms not available: {e}")
    
    @classmethod
    def _register_policy_based_algorithms(cls):
        """Register policy-based algorithms."""
        try:
            from model.rl_system.algorithms.policy_based.policy_gradients import (
                REINFORCEAgent, A2CAgent, PPOAgent
            )
            cls._algorithms['reinforce'] = REINFORCEAgent
            cls._algorithms['a2c'] = A2CAgent
            cls._algorithms['ppo'] = PPOAgent
            
        except ImportError as e:
            print(f"⚠️  Some policy-based algorithms not available: {e}")
    
    @classmethod
    def _register_actor_critic_algorithms(cls):
        """Register actor-critic algorithms."""
        try:
            from model.rl_system.algorithms.actor_critic.continuous_control import (
                DDPGAgent, TD3Agent, SACAgent
            )
            cls._algorithms['ddpg'] = DDPGAgent
            cls._algorithms['td3'] = TD3Agent
            cls._algorithms['sac'] = SACAgent
            
        except ImportError as e:
            print(f"⚠️  Some actor-critic algorithms not available: {e}")
    
    @classmethod
    def _register_exploration_algorithms(cls):
        """Register exploration algorithms."""
        try:
            from model.rl_system.algorithms.exploration import CuriosityDrivenAgent
            cls._algorithms['curiosity_driven'] = CuriosityDrivenAgent
            cls._algorithms['icm'] = CuriosityDrivenAgent  # Alias
            
        except ImportError as e:
            print(f"⚠️  Exploration algorithms not available: {e}")
    
    @classmethod
    def create_algorithm(cls, algorithm_type: str, observation_space, action_space, config=None):
        """Create an algorithm instance."""
        if not cls._initialized:
            cls.initialize()
        
        algorithm_type = algorithm_type.lower()
        
        if algorithm_type not in cls._algorithms:
            available = list(cls._algorithms.keys())
            raise ValueError(f"Unknown algorithm type: {algorithm_type}. Available: {available}")
        
        algorithm_class = cls._algorithms[algorithm_type]
        
        try:
            if config is not None:
                return algorithm_class(observation_space, action_space, config)
            else:
                return algorithm_class(observation_space, action_space)
        except Exception as e:
            print(f"Error creating {algorithm_type} agent: {e}")
            raise
    
    @classmethod
    def get_available_algorithms(cls):
        """Get list of available algorithms."""
        if not cls._initialized:
            cls.initialize()
        return list(cls._algorithms.keys())
    
    @classmethod
    def register_algorithm(cls, name: str, algorithm_class: Type):
        """Register a custom algorithm."""
        cls._algorithms[name.lower()] = algorithm_class
        print(f"Registered custom algorithm: {name}")
    
    @classmethod
    def get_algorithm_info(cls, algorithm_type: str = None):
        """Get information about algorithms."""
        if not cls._initialized:
            cls.initialize()
        
        if algorithm_type:
            algorithm_type = algorithm_type.lower()
            if algorithm_type in cls._algorithms:
                return {
                    'name': algorithm_type,
                    'class': cls._algorithms[algorithm_type],
                    'module': cls._algorithms[algorithm_type].__module__
                }
            else:
                return None
        else:
            return {
                name: {
                    'class': algorithm_class,
                    'module': algorithm_class.__module__
                }
            }

# Global factory instance
factory = RLAlgorithmFactory()

# Convenience functions
def create_algorithm(algorithm_type: str, observation_space, action_space, config=None):
    """Create an algorithm instance using the global factory."""
    return factory.create_algorithm(algorithm_type, observation_space, action_space, config)

def get_available_algorithms():
    """Get list of available algorithms."""
    return factory.get_available_algorithms()

def register_algorithm(name: str, algorithm_class):
    """Register a custom algorithm."""
    return factory.register_algorithm(name, algorithm_class)

def get_algorithm_info(algorithm_type: str = None):
    """Get algorithm information."""
    return factory.get_algorithm_info(algorithm_type)
    
    @classmethod
    def register_algorithm(cls, name: str, algorithm_class: Type):
        """Register a custom algorithm."""
        cls._algorithms[name] = algorithm_class
        print(f"✅ Registered custom algorithm: {name}")
    
    @classmethod
    def create_algorithm(cls, algorithm_name: str, observation_space, action_space, config: Optional[Dict[str, Any]] = None):
        """Create an algorithm instance."""
        if not cls._initialized:
            cls.initialize()
        
        if algorithm_name not in cls._algorithms:
            available = list(cls._algorithms.keys())
            raise ValueError(f"Algorithm '{algorithm_name}' not found. Available: {available}")
        
        algorithm_class = cls._algorithms[algorithm_name]
        
        # Default config if none provided
        if config is None:
            config = cls.get_default_config(algorithm_name)
        
        try:
            return algorithm_class(observation_space, action_space, config)
        except Exception as e:
            raise RuntimeError(f"Failed to create {algorithm_name} algorithm: {e}")
    
    @classmethod
    def get_available_algorithms(cls) -> list:
        """Get list of available algorithm names."""
        if not cls._initialized:
            cls.initialize()
        return list(cls._algorithms.keys())
    
    @classmethod
    def get_algorithm_info(cls, algorithm_name: str) -> Dict[str, Any]:
        """Get information about an algorithm."""
        if not cls._initialized:
            cls.initialize()
        
        if algorithm_name not in cls._algorithms:
            raise ValueError(f"Algorithm '{algorithm_name}' not found")
        
        algorithm_class = cls._algorithms[algorithm_name]
        
        return {
            'name': algorithm_name,
            'class': algorithm_class.__name__,
            'module': algorithm_class.__module__,
            'doc': algorithm_class.__doc__ or "No documentation available",
            'type': cls._get_algorithm_type(algorithm_name)
        }
    
    @classmethod
    def _get_algorithm_type(cls, algorithm_name: str) -> str:
        """Get the type category of an algorithm."""
        value_based = ['dqn', 'double_dqn', 'dueling_dqn', 'q_learning', 'sarsa']
        policy_based = ['a2c', 'ppo', 'trpo']
        actor_critic = ['ddpg', 'td3', 'sac']
        model_based = ['mcts']
        
        if algorithm_name in value_based:
            return 'value_based'
        elif algorithm_name in policy_based:
            return 'policy_based'
        elif algorithm_name in actor_critic:
            return 'actor_critic'
        elif algorithm_name in model_based:
            return 'model_based'
        else:
            return 'custom'
    
    @classmethod
    def get_default_config(cls, algorithm_name: str) -> Dict[str, Any]:
        """Get default configuration for an algorithm."""
        
        # Common default values
        common_config = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'memory_size': 10000,
            'exploration_rate': 1.0,
            'exploration_decay': 0.995,
            'exploration_min': 0.01,
            'gamma': 0.99,
            'update_frequency': 4,
            'device': 'cuda' if cls._cuda_available() else 'cpu'
        }
        
        # Algorithm-specific configurations
        algorithm_configs = {
            'dqn': {
                **common_config,
                'target_update_frequency': 1000,
                'double_q': False
            },
            'double_dqn': {
                **common_config,
                'target_update_frequency': 1000,
                'double_q': True
            },
            'dueling_dqn': {
                **common_config,
                'target_update_frequency': 1000,
                'dueling': True
            },
            'a2c': {
                **common_config,
                'value_loss_coef': 0.5,
                'entropy_coef': 0.01,
                'n_steps': 5
            },
            'ppo': {
                **common_config,
                'clip_ratio': 0.2,
                'value_loss_coef': 0.5,
                'entropy_coef': 0.01,
                'n_epochs': 4,
                'n_steps': 2048
            },
            'ddpg': {
                **common_config,
                'tau': 0.005,
                'noise_std': 0.1,
                'actor_lr': 0.001,
                'critic_lr': 0.002
            },
            'td3': {
                **common_config,
                'tau': 0.005,
                'noise_std': 0.1,
                'target_noise': 0.2,
                'noise_clip': 0.5,
                'policy_delay': 2
            },
            'sac': {
                **common_config,
                'tau': 0.005,
                'alpha': 0.2,
                'auto_entropy_tuning': True,
                'target_entropy': None
            }
        }
        
        return algorithm_configs.get(algorithm_name, common_config)
    
    @classmethod
    def _cuda_available(cls) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    @classmethod
    def get_algorithm_categories(cls) -> Dict[str, list]:
        """Get algorithms grouped by category."""
        if not cls._initialized:
            cls.initialize()
        
        categories = {
            'value_based': [],
            'policy_based': [],
            'actor_critic': [],
            'model_based': [],
            'custom': []
        }
        
        for algorithm_name in cls._algorithms.keys():
            category = cls._get_algorithm_type(algorithm_name)
            categories[category].append(algorithm_name)
        
        return categories

# Initialize the factory when module is imported
RLAlgorithmFactory.initialize()
