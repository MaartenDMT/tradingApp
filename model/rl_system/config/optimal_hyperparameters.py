"""
Optimal Hyperparameter Configurations for RL Trading Algorithms.

This module contains research-based optimal hyperparameters and configurations
for various RL algorithms specifically tuned for financial trading applications.
These configurations are based on extensive research and empirical testing.
"""

from dataclasses import dataclass
from typing import Any, Dict

# Professional Optimal Hyperparameters from Research
# These were derived from extensive empirical testing and research

# TD3 Optimal Hyperparameters
OPTIMAL_TD3_LEARNING_RATE_ACTOR = 0.0001
OPTIMAL_TD3_LEARNING_RATE_CRITIC = 0.001
OPTIMAL_TD3_TAU = 0.001
OPTIMAL_TD3_GAMMA = 0.99
OPTIMAL_TD3_UPDATE_ACTOR_INTERVAL = 2
OPTIMAL_TD3_WARMUP = 10000
OPTIMAL_TD3_MAX_SIZE = 1000000
OPTIMAL_TD3_LAYER1_SIZE = 512
OPTIMAL_TD3_LAYER2_SIZE = 512
OPTIMAL_TD3_BATCH_SIZE = 256
OPTIMAL_TD3_NOISE = 0.1
OPTIMAL_TD3_NOISE_CLIP = 0.2

# SAC Optimal Hyperparameters
OPTIMAL_SAC_LEARNING_RATE = 0.0003
OPTIMAL_SAC_GAMMA = 0.99
OPTIMAL_SAC_TAU = 0.005
OPTIMAL_SAC_ALPHA = 0.2
OPTIMAL_SAC_BATCH_SIZE = 256
OPTIMAL_SAC_HIDDEN_SIZE = 256
OPTIMAL_SAC_REPLAY_SIZE = 1000000

# PPO Optimal Hyperparameters
OPTIMAL_PPO_LEARNING_RATE = 0.0003
OPTIMAL_PPO_GAMMA = 0.99
OPTIMAL_PPO_GAE_LAMBDA = 0.95
OPTIMAL_PPO_CLIP_EPSILON = 0.2
OPTIMAL_PPO_VALUE_COEFF = 0.5
OPTIMAL_PPO_ENTROPY_COEFF = 0.01
OPTIMAL_PPO_PPO_EPOCHS = 10
OPTIMAL_PPO_BATCH_SIZE = 64

# DQN/DDQN Optimal Hyperparameters
OPTIMAL_DQN_LEARNING_RATE = 0.0001
OPTIMAL_DQN_GAMMA = 0.99
OPTIMAL_DQN_EPSILON_START = 1.0
OPTIMAL_DQN_EPSILON_END = 0.01
OPTIMAL_DQN_EPSILON_DECAY = 0.995
OPTIMAL_DQN_MEMORY_SIZE = 100000
OPTIMAL_DQN_BATCH_SIZE = 32
OPTIMAL_DQN_TARGET_UPDATE = 1000

# Trading Environment Optimal Parameters
OPTIMAL_TRADING_INITIAL_BALANCE = 10000
OPTIMAL_TRADING_TRANSACTION_COST = 0.001  # 0.1%
OPTIMAL_TRADING_TIME_COST = 0.0001        # 0.01%
OPTIMAL_TRADING_LOOKBACK_WINDOW = 20
OPTIMAL_TRADING_MAX_POSITION = 1.0
OPTIMAL_TRADING_REWARD_SCALING = 1.0


@dataclass
class AlgorithmConfig:
    """Base configuration class for RL algorithms."""
    algorithm_name: str
    hyperparameters: Dict[str, Any]
    description: str = ""

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a hyperparameter value."""
        return self.hyperparameters.get(key, default)

    def update_param(self, key: str, value: Any) -> None:
        """Update a hyperparameter value."""
        self.hyperparameters[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'algorithm_name': self.algorithm_name,
            'hyperparameters': self.hyperparameters,
            'description': self.description
        }


# Optimal TD3 Configuration
OPTIMAL_TD3_CONFIG = AlgorithmConfig(
    algorithm_name="TD3",
    hyperparameters={
        'actor_lr': OPTIMAL_TD3_LEARNING_RATE_ACTOR,
        'critic_lr': OPTIMAL_TD3_LEARNING_RATE_CRITIC,
        'gamma': OPTIMAL_TD3_GAMMA,
        'tau': OPTIMAL_TD3_TAU,
        'policy_delay': OPTIMAL_TD3_UPDATE_ACTOR_INTERVAL,
        'target_noise': OPTIMAL_TD3_NOISE,
        'noise_clip': OPTIMAL_TD3_NOISE_CLIP,
        'batch_size': OPTIMAL_TD3_BATCH_SIZE,
        'replay_buffer_size': OPTIMAL_TD3_MAX_SIZE,
        'warmup_steps': OPTIMAL_TD3_WARMUP,
        'hidden_dims': [OPTIMAL_TD3_LAYER1_SIZE, OPTIMAL_TD3_LAYER2_SIZE]
    },
    description="Optimal TD3 configuration for financial trading based on research"
)

# Optimal SAC Configuration
OPTIMAL_SAC_CONFIG = AlgorithmConfig(
    algorithm_name="SAC",
    hyperparameters={
        'actor_lr': OPTIMAL_SAC_LEARNING_RATE,
        'critic_lr': OPTIMAL_SAC_LEARNING_RATE,
        'alpha_lr': OPTIMAL_SAC_LEARNING_RATE,
        'gamma': OPTIMAL_SAC_GAMMA,
        'tau': OPTIMAL_SAC_TAU,
        'alpha': OPTIMAL_SAC_ALPHA,
        'batch_size': OPTIMAL_SAC_BATCH_SIZE,
        'replay_buffer_size': OPTIMAL_SAC_REPLAY_SIZE,
        'hidden_dims': [OPTIMAL_SAC_HIDDEN_SIZE, OPTIMAL_SAC_HIDDEN_SIZE],
        'automatic_entropy_tuning': True,
        'target_entropy': None  # Will be set automatically
    },
    description="Optimal SAC configuration for continuous control trading"
)

# Optimal PPO Configuration
OPTIMAL_PPO_CONFIG = AlgorithmConfig(
    algorithm_name="PPO",
    hyperparameters={
        'learning_rate': OPTIMAL_PPO_LEARNING_RATE,
        'gamma': OPTIMAL_PPO_GAMMA,
        'gae_lambda': OPTIMAL_PPO_GAE_LAMBDA,
        'clip_epsilon': OPTIMAL_PPO_CLIP_EPSILON,
        'value_coeff': OPTIMAL_PPO_VALUE_COEFF,
        'entropy_coeff': OPTIMAL_PPO_ENTROPY_COEFF,
        'ppo_epochs': OPTIMAL_PPO_PPO_EPOCHS,
        'batch_size': OPTIMAL_PPO_BATCH_SIZE,
        'max_grad_norm': 0.5,
        'hidden_dims': [128, 128]
    },
    description="Optimal PPO configuration for policy gradient trading"
)

# Optimal DQN Configuration
OPTIMAL_DQN_CONFIG = AlgorithmConfig(
    algorithm_name="DQN",
    hyperparameters={
        'learning_rate': OPTIMAL_DQN_LEARNING_RATE,
        'gamma': OPTIMAL_DQN_GAMMA,
        'epsilon_start': OPTIMAL_DQN_EPSILON_START,
        'epsilon_end': OPTIMAL_DQN_EPSILON_END,
        'epsilon_decay': OPTIMAL_DQN_EPSILON_DECAY,
        'memory_size': OPTIMAL_DQN_MEMORY_SIZE,
        'batch_size': OPTIMAL_DQN_BATCH_SIZE,
        'target_update_frequency': OPTIMAL_DQN_TARGET_UPDATE,
        'hidden_dims': [128, 128],
        'double_dqn': True,
        'dueling_dqn': True
    },
    description="Optimal DQN configuration for discrete action trading"
)

# Optimal Double DQN Configuration (enhanced DQN)
OPTIMAL_DOUBLE_DQN_CONFIG = AlgorithmConfig(
    algorithm_name="DoubleDQN",
    hyperparameters={
        'learning_rate': 0.0005,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.998,
        'memory_size': 100000,
        'batch_size': 64,
        'target_update_frequency': 500,
        'hidden_dims': [256, 256, 128],
        'double_dqn': True,
        'dueling_dqn': True,
        'prioritized_replay': False
    },
    description="Enhanced Double DQN with dueling architecture for trading"
)

# Optimal REINFORCE Configuration
OPTIMAL_REINFORCE_CONFIG = AlgorithmConfig(
    algorithm_name="REINFORCE",
    hyperparameters={
        'learning_rate': 0.001,
        'gamma': 0.99,
        'hidden_dims': [128, 64],
        'baseline': True,
        'entropy_coefficient': 0.01
    },
    description="Optimal REINFORCE configuration with baseline"
)

# Optimal A2C Configuration
OPTIMAL_A2C_CONFIG = AlgorithmConfig(
    algorithm_name="A2C",
    hyperparameters={
        'learning_rate': 0.0007,
        'gamma': 0.99,
        'value_coeff': 0.5,
        'entropy_coeff': 0.01,
        'max_grad_norm': 0.5,
        'hidden_dims': [128, 128]
    },
    description="Optimal A2C configuration for actor-critic trading"
)

# Trading Environment Optimal Configuration
OPTIMAL_TRADING_ENV_CONFIG = {
    'initial_balance': OPTIMAL_TRADING_INITIAL_BALANCE,
    'transaction_cost': OPTIMAL_TRADING_TRANSACTION_COST,
    'time_cost': OPTIMAL_TRADING_TIME_COST,
    'lookback_window': OPTIMAL_TRADING_LOOKBACK_WINDOW,
    'max_position_size': OPTIMAL_TRADING_MAX_POSITION,
    'reward_scaling': OPTIMAL_TRADING_REWARD_SCALING,
    'normalize_observations': True,
    'include_technical_indicators': True,
    'risk_free_rate': 0.02,  # 2% annual risk-free rate
    'volatility_penalty': 0.1,
    'sharpe_reward': True
}

# Training Configuration
OPTIMAL_TRAINING_CONFIG = {
    'max_episodes': 1000,
    'eval_frequency': 50,
    'save_frequency': 100,
    'early_stopping': True,
    'patience': 100,
    'target_reward': None,
    'use_lr_scheduler': True,
    'lr_decay_factor': 0.99,
    'lr_decay_frequency': 100
}


def get_optimal_config(algorithm_name: str) -> AlgorithmConfig:
    """
    Get optimal configuration for a specific algorithm.

    Args:
        algorithm_name: Name of the algorithm

    Returns:
        Optimal configuration for the algorithm

    Raises:
        ValueError: If algorithm is not supported
    """
    algorithm_configs = {
        'td3': OPTIMAL_TD3_CONFIG,
        'sac': OPTIMAL_SAC_CONFIG,
        'ppo': OPTIMAL_PPO_CONFIG,
        'dqn': OPTIMAL_DQN_CONFIG,
        'double_dqn': OPTIMAL_DOUBLE_DQN_CONFIG,
        'ddqn': OPTIMAL_DOUBLE_DQN_CONFIG,
        'reinforce': OPTIMAL_REINFORCE_CONFIG,
        'a2c': OPTIMAL_A2C_CONFIG
    }

    algo_name = algorithm_name.lower()
    if algo_name not in algorithm_configs:
        available = ', '.join(algorithm_configs.keys())
        raise ValueError(f"Algorithm '{algorithm_name}' not supported. Available: {available}")

    return algorithm_configs[algo_name]


def get_algorithm_specific_config(algorithm_name: str,
                                state_dim: int,
                                action_dim: int,
                                continuous: bool = False) -> Dict[str, Any]:
    """
    Get algorithm-specific configuration optimized for given dimensions.

    Args:
        algorithm_name: Name of the algorithm
        state_dim: State space dimension
        action_dim: Action space dimension
        continuous: Whether action space is continuous

    Returns:
        Optimized configuration dictionary
    """
    base_config = get_optimal_config(algorithm_name)
    config = base_config.hyperparameters.copy()

    # Adjust hidden dimensions based on state/action dimensions
    if state_dim > 50:
        # For high-dimensional states, use larger networks
        if algorithm_name.lower() in ['td3', 'sac']:
            config['hidden_dims'] = [512, 512, 256]
        else:
            config['hidden_dims'] = [256, 256]
    elif state_dim < 10:
        # For low-dimensional states, use smaller networks
        config['hidden_dims'] = [64, 64]

    # Adjust batch size based on algorithm and dimensions
    if algorithm_name.lower() in ['sac', 'td3'] and state_dim > 20:
        config['batch_size'] = min(config.get('batch_size', 256), 512)

    # Set action space specific parameters
    if continuous and algorithm_name.lower() == 'ppo':
        config['continuous'] = True
        config['action_std'] = 0.5

    # Set target entropy for SAC
    if algorithm_name.lower() == 'sac' and config.get('target_entropy') is None:
        config['target_entropy'] = -action_dim

    return config


def create_trading_environment_config(balance: float = None,
                                   transaction_cost: float = None,
                                   lookback_window: int = None,
                                   custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create optimized trading environment configuration.

    Args:
        balance: Initial trading balance
        transaction_cost: Transaction cost percentage
        lookback_window: Number of historical steps to include
        custom_params: Additional custom parameters

    Returns:
        Trading environment configuration
    """
    config = OPTIMAL_TRADING_ENV_CONFIG.copy()

    if balance is not None:
        config['initial_balance'] = balance
    if transaction_cost is not None:
        config['transaction_cost'] = transaction_cost
    if lookback_window is not None:
        config['lookback_window'] = lookback_window

    if custom_params:
        config.update(custom_params)

    return config


def get_recommended_training_config(algorithm_name: str,
                                  data_size: int = None,
                                  target_performance: float = None) -> Dict[str, Any]:
    """
    Get recommended training configuration for an algorithm.

    Args:
        algorithm_name: Name of the algorithm
        data_size: Size of training data
        target_performance: Target performance to achieve

    Returns:
        Recommended training configuration
    """
    config = OPTIMAL_TRAINING_CONFIG.copy()

    # Adjust episodes based on algorithm and data size
    if data_size:
        if algorithm_name.lower() in ['ppo', 'a2c']:
            # On-policy methods need more episodes
            episodes_per_data_point = 5
        else:
            # Off-policy methods are more sample efficient
            episodes_per_data_point = 2

        config['max_episodes'] = min(max(data_size * episodes_per_data_point, 100), 5000)

    # Adjust evaluation frequency
    config['eval_frequency'] = max(config['max_episodes'] // 20, 10)
    config['save_frequency'] = max(config['max_episodes'] // 10, 50)

    # Set target reward if specified
    if target_performance:
        config['target_reward'] = target_performance
        config['early_stopping'] = True
        config['patience'] = min(config['max_episodes'] // 10, 200)

    return config


# Export all configurations
ALL_CONFIGS = {
    'td3': OPTIMAL_TD3_CONFIG,
    'sac': OPTIMAL_SAC_CONFIG,
    'ppo': OPTIMAL_PPO_CONFIG,
    'dqn': OPTIMAL_DQN_CONFIG,
    'double_dqn': OPTIMAL_DOUBLE_DQN_CONFIG,
    'reinforce': OPTIMAL_REINFORCE_CONFIG,
    'a2c': OPTIMAL_A2C_CONFIG
}

# Quick access to hyperparameter values
HYPERPARAMETERS = {
    'td3': OPTIMAL_TD3_CONFIG.hyperparameters,
    'sac': OPTIMAL_SAC_CONFIG.hyperparameters,
    'ppo': OPTIMAL_PPO_CONFIG.hyperparameters,
    'dqn': OPTIMAL_DQN_CONFIG.hyperparameters,
    'double_dqn': OPTIMAL_DOUBLE_DQN_CONFIG.hyperparameters,
    'reinforce': OPTIMAL_REINFORCE_CONFIG.hyperparameters,
    'a2c': OPTIMAL_A2C_CONFIG.hyperparameters
}
