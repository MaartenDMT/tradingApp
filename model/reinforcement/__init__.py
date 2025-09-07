"""
Reinforcement Learning Module for Trading

This module contains all reinforcement learning components including:
- Trading environments
- RL agents and models
- Training utilities
- Visualization tools
"""

from .environments import (DataManager, RewardCalculator, TradingEngine,
                           TradingEnvironment)

__all__ = [
    'TradingEnvironment',
    'TradingEngine',
    'DataManager',
    'RewardCalculator'
]

# Version and metadata
__version__ = "2.0.0"
__author__ = "Trading RL Team"
__description__ = "Unified Trading Reinforcement Learning Environment"

# Configuration defaults
DEFAULT_CONFIG = {
    'symbol': 'BTC',
    'initial_balance': 10000.0,
    'trading_mode': 'spot',
    'leverage': 1.0,
    'transaction_costs': 0.001,
    'min_accuracy': 0.6,
    'max_episode_steps': 1000,
    'features': [
        'close', 'volume', 'rsi14', 'rsi40', 'ema_200', 'dots', 'l_wave',
        'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'atr', 'adx'
    ]
}

def create_environment(config: dict = None, **kwargs):
    """
    Factory function to create a trading environment with default settings.

    Args:
        config: Configuration dictionary
        **kwargs: Additional parameters to override defaults

    Returns:
        TradingEnvironment instance
    """
    # Merge default config with provided config
    env_config = DEFAULT_CONFIG.copy()
    if config:
        env_config.update(config)
    env_config.update(kwargs)

    return TradingEnvironment(**env_config)

def create_multi_agent_environment(num_agents=2, config=None, **kwargs):
    """
    Factory function to create multi-agent trading environments.

    Args:
        num_agents: Number of agents in the environment
        config: Configuration dictionary
        **kwargs: Additional parameters

    Returns:
        TradingEnvironment instance configured for multi-agent use
    """
    # Merge default config with provided config
    env_config = DEFAULT_CONFIG.copy()
    if config:
        env_config.update(config)
    env_config.update(kwargs)

    # For now, return a standard environment - can be extended for true multi-agent later
    return TradingEnvironment(**env_config)

# Additional convenience imports and utilities can be added here as needed
    # For now, return a standard environment - can be extended for true multi-agent later
    return TradingEnvironment(**env_config)

# Additional convenience imports and utilities can be added here as needed
