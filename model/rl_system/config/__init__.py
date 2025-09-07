"""
Configuration Package for RL Trading System.

This package contains optimal hyperparameters and configurations
for various reinforcement learning algorithms.
"""

from .optimal_hyperparameters import (  # Algorithm Configurations; Environment Configuration; Configuration Classes; Utility Functions; Data Exports
    ALL_CONFIGS,
    HYPERPARAMETERS,
    OPTIMAL_A2C_CONFIG,
    OPTIMAL_DOUBLE_DQN_CONFIG,
    OPTIMAL_DQN_CONFIG,
    OPTIMAL_PPO_CONFIG,
    OPTIMAL_REINFORCE_CONFIG,
    OPTIMAL_SAC_CONFIG,
    OPTIMAL_TD3_CONFIG,
    OPTIMAL_TRADING_ENV_CONFIG,
    OPTIMAL_TRAINING_CONFIG,
    AlgorithmConfig,
    create_trading_environment_config,
    get_algorithm_specific_config,
    get_optimal_config,
    get_recommended_training_config,
)

__all__ = [
    # Configurations
    'OPTIMAL_TD3_CONFIG',
    'OPTIMAL_SAC_CONFIG',
    'OPTIMAL_PPO_CONFIG',
    'OPTIMAL_DQN_CONFIG',
    'OPTIMAL_DOUBLE_DQN_CONFIG',
    'OPTIMAL_REINFORCE_CONFIG',
    'OPTIMAL_A2C_CONFIG',
    'OPTIMAL_TRADING_ENV_CONFIG',
    'OPTIMAL_TRAINING_CONFIG',

    # Classes
    'AlgorithmConfig',

    # Functions
    'get_optimal_config',
    'get_algorithm_specific_config',
    'create_trading_environment_config',
    'get_recommended_training_config',

    # Data
    'ALL_CONFIGS',
    'HYPERPARAMETERS'
]
