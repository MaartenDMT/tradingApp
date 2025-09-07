"""
Comprehensive RL System for Trading Applications.

This package provides a complete reinforcement learning system specifically
designed for trading applications. It includes implementations of all major
RL algorithms, a sophisticated trading environment, comprehensive training
framework, and extensive testing capabilities.

Main Components:
- Core: Base classes and fundamental components
- Algorithms: All major RL algorithm families
- Environments: Trading environment with realistic features
- Training: Advanced training framework with monitoring
- Integration: Unified system interface
- Tests: Comprehensive testing suite

Quick Start:
    from model.rl_system.integration.rl_system import RLSystemManager

    # Create system manager
    system = RLSystemManager()

    # Create agent and environment
    agent = system.create_agent('dqn', state_dim=50, action_dim=5)
    environment = system.create_environment(data)

    # Train agent
    results = system.train_agent(agent, environment)
"""

# Version information
__version__ = "1.0.0"
__author__ = "Trading RL System"

from .algorithms.actor_critic.continuous_control import (
                                                         ActorCriticConfig,
                                                         DDPGAgent,
                                                         TD3Agent,
)
from .algorithms.policy_based.policy_gradients import (
                                                         A2CAgent,
                                                         PolicyGradientConfig,
                                                         REINFORCEAgent,
)

# Algorithm imports
from .algorithms.value_based.dqn_family import (
                                                         DoubleDQNAgent,
                                                         DQNAgent,
                                                         DQNConfig,
                                                         DuelingDQNAgent,
                                                         RainbowDQNAgent,
)
from .algorithms.value_based.tabular_methods import (
                                                         MonteCarloAgent,
                                                         QLearningAgent,
                                                         SARSAAgent,
                                                         TabularConfig,
)
from .core.base_agents import (
                                                         ActorCriticAgent,
                                                         AgentConfig,
                                                         BaseRLAgent,
                                                         PolicyBasedAgent,
                                                         ValueBasedAgent,
)
from .environments.trading_env import TradingAction, TradingConfig, TradingEnvironment

# Import main components for easy access
from .integration.rl_system import (
                                                         RLSystemManager,
                                                         compare_algorithms,
                                                         create_rl_system,
                                                         quick_experiment,
)

# Test framework
from .tests.test_suite import run_all_tests, run_benchmarks
from .training.trainer import RLTrainer, TrainingConfig, train_agent

__all__ = [
    # Main system
    'RLSystemManager',
    'create_rl_system',
    'quick_experiment',
    'compare_algorithms',

    # Core components
    'BaseRLAgent',
    'ValueBasedAgent',
    'PolicyBasedAgent',
    'ActorCriticAgent',
    'AgentConfig',

    # Environment
    'TradingEnvironment',
    'TradingConfig',
    'TradingAction',

    # Training
    'RLTrainer',
    'TrainingConfig',
    'train_agent',

    # DQN family
    'DQNAgent',
    'DoubleDQNAgent',
    'DuelingDQNAgent',
    'RainbowDQNAgent',
    'DQNConfig',

    # Tabular methods
    'QLearningAgent',
    'SARSAAgent',
    'MonteCarloAgent',
    'TabularConfig',

    # Policy gradients
    'REINFORCEAgent',
    'A2CAgent',
    'PolicyGradientConfig',

    # Actor-critic
    'DDPGAgent',
    'TD3Agent',
    'ActorCriticConfig',

    # Testing
    'run_all_tests',
    'run_benchmarks'
]
