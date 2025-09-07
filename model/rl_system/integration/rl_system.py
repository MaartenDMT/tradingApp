"""
Comprehensive RL System Integration Module.

This module provides a unified interface for creating, training, and evaluating
all types of RL agents with the trading environment. It serves as the main
entry point for the entire RL system.
"""

import json
import os
import warnings
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import util.loggers as loggers

from ..algorithms.actor_critic.continuous_control import (
    ActorCriticConfig,
    DDPGAgent,
    TD3Agent,
)
from ..algorithms.policy_based.policy_gradients import (
    A2CAgent,
    PolicyGradientConfig,
    REINFORCEAgent,
)

# Algorithms
from ..algorithms.value_based.dqn_family import (
    DoubleDQNAgent,
    DQNAgent,
    DQNConfig,
    DuelingDQNAgent,
    RainbowDQNAgent,
)
from ..algorithms.value_based.tabular_methods import (
    MonteCarloAgent,
    QLearningAgent,
    SARSAAgent,
    TabularConfig,
)

# Core components
from ..core.base_agents import BaseRLAgent
from ..environments.trading_env import TradingConfig, TradingEnvironment

# Training framework
from ..training.trainer import RLTrainer, TrainingConfig

logger = loggers.setup_loggers()
rl_logger = logger['rl']

warnings.filterwarnings('ignore', category=FutureWarning)


class RLSystemManager:
    """
    Comprehensive manager for the entire RL system.

    This class provides a unified interface for:
    - Creating agents and environments
    - Training and evaluation
    - Model management and persistence
    - Performance analysis and comparison
    """

    def __init__(self, base_dir: str = "rl_experiments"):
        """
        Initialize the RL system manager.

        Args:
            base_dir: Base directory for all RL experiments
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        # Registry of available agents
        self.agent_registry = {
            # Value-based agents
            'dqn': (DQNAgent, DQNConfig),
            'double_dqn': (DoubleDQNAgent, DQNConfig),
            'dueling_dqn': (DuelingDQNAgent, DQNConfig),
            'rainbow_dqn': (RainbowDQNAgent, DQNConfig),

            # Tabular agents
            'q_learning': (QLearningAgent, TabularConfig),
            'sarsa': (SARSAAgent, TabularConfig),
            'monte_carlo': (MonteCarloAgent, TabularConfig),

            # Policy gradient agents
            'reinforce': (REINFORCEAgent, PolicyGradientConfig),
            'a2c': (A2CAgent, PolicyGradientConfig),

            # Actor-critic agents
            'ddpg': (DDPGAgent, ActorCriticConfig),
            'td3': (TD3Agent, ActorCriticConfig),
        }

        rl_logger.info(f"RL System Manager initialized with {len(self.agent_registry)} agent types")

    def create_agent(self,
                    agent_type: str,
                    state_dim: int,
                    action_dim: int,
                    config: Optional[Dict[str, Any]] = None) -> BaseRLAgent:
        """
        Create an RL agent of the specified type.

        Args:
            agent_type: Type of agent to create
            state_dim: State space dimension
            action_dim: Action space dimension
            config: Agent configuration parameters

        Returns:
            Initialized RL agent
        """
        agent_type = agent_type.lower()

        if agent_type not in self.agent_registry:
            available_types = list(self.agent_registry.keys())
            raise ValueError(f"Unknown agent type '{agent_type}'. Available types: {available_types}")

        agent_class, config_class = self.agent_registry[agent_type]

        # Create configuration
        if config is None:
            agent_config = config_class()
        else:
            agent_config = config_class(**config)

        # Create agent
        if agent_type in ['q_learning', 'sarsa', 'monte_carlo']:
            # Tabular agents need discrete action space
            agent = agent_class(state_dim, action_dim, agent_config)
        elif agent_type in ['dqn', 'double_dqn', 'dueling_dqn', 'rainbow_dqn']:
            # DQN agents need discrete action space
            agent = agent_class(state_dim, action_dim, agent_config)
        elif agent_type in ['reinforce', 'a2c']:
            # Policy gradient agents for discrete actions
            agent = agent_class(state_dim, action_dim, agent_config)
        elif agent_type in ['ddpg', 'td3']:
            # Actor-critic agents for continuous actions
            agent = agent_class(state_dim, action_dim, agent_config)
        else:
            agent = agent_class(state_dim, action_dim, agent_config)

        rl_logger.info(f"Created {agent_type} agent with state_dim={state_dim}, action_dim={action_dim}")
        return agent

    def create_environment(self,
                          data: pd.DataFrame,
                          config: Optional[Dict[str, Any]] = None) -> TradingEnvironment:
        """
        Create a trading environment.

        Args:
            data: Market data
            config: Environment configuration

        Returns:
            Trading environment
        """
        if config is None:
            env_config = TradingConfig()
        else:
            env_config = TradingConfig(**config)

        environment = TradingEnvironment(data, env_config)

        rl_logger.info(f"Created trading environment with {len(data)} data points")
        return environment

    def train_agent(self,
                   agent: BaseRLAgent,
                   environment: TradingEnvironment,
                   config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train an RL agent in the given environment.

        Args:
            agent: RL agent to train
            environment: Training environment
            config: Training configuration

        Returns:
            Training results and metrics
        """
        if config is None:
            training_config = TrainingConfig()
        else:
            training_config = TrainingConfig(**config)

        # Set up experiment directory
        experiment_dir = os.path.join(
            self.base_dir,
            f"{agent.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(experiment_dir, exist_ok=True)
        training_config.checkpoint_dir = experiment_dir

        # Train the agent
        trainer = RLTrainer(agent, environment, training_config)
        results = trainer.train()

        # Save training plots
        trainer.plot_training_progress()

        rl_logger.info(f"Training completed for {agent.name}")
        return results

    def evaluate_agent(self,
                      agent: BaseRLAgent,
                      environment: TradingEnvironment,
                      num_episodes: int = 10,
                      render: bool = False) -> Dict[str, Any]:
        """
        Evaluate an agent's performance.

        Args:
            agent: RL agent to evaluate
            environment: Environment for evaluation
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes

        Returns:
            Evaluation metrics
        """
        agent.eval_mode()

        episode_rewards = []
        episode_lengths = []
        portfolio_values = []

        for episode in range(num_episodes):
            state = environment.reset()
            total_reward = 0
            episode_length = 0

            for step in range(1000):  # Max steps per episode
                action = agent.select_action(state, training=False)
                next_state, reward, done, info = environment.step(action)

                total_reward += reward
                episode_length += 1
                state = next_state

                if render and episode == 0:  # Render only first episode
                    environment.render()

                if done:
                    break

            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)

            # Get final portfolio value
            performance = environment.get_performance_summary()
            if performance:
                portfolio_values.append(performance['final_value'])

        agent.train_mode()

        # Calculate evaluation metrics
        evaluation_metrics = {
            'num_episodes': num_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'episode_rewards': episode_rewards
        }

        if portfolio_values:
            initial_value = environment.initial_balance
            total_returns = [(pv - initial_value) / initial_value for pv in portfolio_values]

            evaluation_metrics.update({
                'mean_portfolio_value': np.mean(portfolio_values),
                'mean_total_return': np.mean(total_returns),
                'std_total_return': np.std(total_returns),
                'sharpe_ratio': np.mean(total_returns) / (np.std(total_returns) + 1e-8) * np.sqrt(252)
            })

        rl_logger.info(f"Evaluation completed: Mean reward = {evaluation_metrics['mean_reward']:.2f}")
        return evaluation_metrics

    def compare_agents(self,
                      agents: Dict[str, BaseRLAgent],
                      environment: TradingEnvironment,
                      num_episodes: int = 10) -> pd.DataFrame:
        """
        Compare multiple agents' performance.

        Args:
            agents: Dictionary of agent_name -> agent
            environment: Environment for evaluation
            num_episodes: Number of evaluation episodes per agent

        Returns:
            Comparison results as DataFrame
        """
        results = []

        for agent_name, agent in agents.items():
            rl_logger.info(f"Evaluating {agent_name}...")
            metrics = self.evaluate_agent(agent, environment, num_episodes)

            result = {
                'agent_name': agent_name,
                'agent_type': agent.name,
                **metrics
            }
            results.append(result)

        comparison_df = pd.DataFrame(results)

        # Save comparison results
        comparison_path = os.path.join(self.base_dir, f"agent_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        comparison_df.to_csv(comparison_path, index=False)

        rl_logger.info(f"Agent comparison saved to {comparison_path}")
        return comparison_df

    def save_agent(self,
                  agent: BaseRLAgent,
                  filepath: str,
                  include_config: bool = True) -> None:
        """
        Save agent to file.

        Args:
            agent: Agent to save
            filepath: Path to save the agent
            include_config: Whether to include configuration
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save agent checkpoint
        agent.save_checkpoint(filepath)

        # Save additional metadata
        if include_config:
            metadata = {
                'agent_name': agent.name,
                'agent_type': type(agent).__name__,
                'state_dim': agent.state_dim,
                'action_dim': agent.action_dim,
                'training_step': agent.training_step,
                'episode_count': agent.episode_count,
                'timestamp': datetime.now().isoformat()
            }

            metadata_path = f"{filepath}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        rl_logger.info(f"Agent saved to {filepath}")

    def load_agent(self,
                  filepath: str,
                  agent_type: Optional[str] = None,
                  state_dim: Optional[int] = None,
                  action_dim: Optional[int] = None) -> BaseRLAgent:
        """
        Load agent from file.

        Args:
            filepath: Path to load the agent from
            agent_type: Type of agent (if not in metadata)
            state_dim: State dimension (if not in metadata)
            action_dim: Action dimension (if not in metadata)

        Returns:
            Loaded agent
        """
        # Try to load metadata first
        metadata_path = f"{filepath}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            agent_type = agent_type or metadata.get('agent_type')
            state_dim = state_dim or metadata.get('state_dim')
            action_dim = action_dim or metadata.get('action_dim')

        if not all([agent_type, state_dim, action_dim]):
            raise ValueError("Agent type, state_dim, and action_dim must be provided or available in metadata")

        # Map class name to agent type
        class_to_type = {
            'DQNAgent': 'dqn',
            'DoubleDQNAgent': 'double_dqn',
            'DuelingDQNAgent': 'dueling_dqn',
            'RainbowDQNAgent': 'rainbow_dqn',
            'QLearningAgent': 'q_learning',
            'SARSAAgent': 'sarsa',
            'MonteCarloAgent': 'monte_carlo',
            'REINFORCEAgent': 'reinforce',
            'A2CAgent': 'a2c',
            'DDPGAgent': 'ddpg',
            'TD3Agent': 'td3'
        }

        if agent_type in class_to_type.values():
            # Agent type is already correct
            pass
        elif agent_type in class_to_type:
            # Convert class name to agent type
            agent_type = class_to_type[agent_type]
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Create agent
        agent = self.create_agent(agent_type, state_dim, action_dim)

        # Load checkpoint
        agent.load_checkpoint(filepath)

        rl_logger.info(f"Agent loaded from {filepath}")
        return agent

    def run_experiment(self,
                      experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a complete RL experiment.

        Args:
            experiment_config: Complete experiment configuration

        Returns:
            Experiment results
        """
        # Extract configurations
        agent_config = experiment_config.get('agent', {})
        env_config = experiment_config.get('environment', {})
        training_config = experiment_config.get('training', {})

        # Load data
        data_path = experiment_config.get('data_path')
        if data_path and os.path.exists(data_path):
            data = pd.read_csv(data_path)
        else:
            # Generate dummy data for testing
            data = self._generate_dummy_data()

        # Create environment
        environment = self.create_environment(data, env_config)

        # Create agent
        agent_type = agent_config.get('type', 'dqn')
        state_dim = environment.get_observation_space_size()
        action_dim = environment.get_action_space_size()

        agent = self.create_agent(agent_type, state_dim, action_dim, agent_config.get('config'))

        # Train agent
        training_results = self.train_agent(agent, environment, training_config)

        # Evaluate agent
        evaluation_results = self.evaluate_agent(agent, environment)

        # Combine results
        experiment_results = {
            'experiment_config': experiment_config,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'agent_info': {
                'name': agent.name,
                'type': agent_type,
                'state_dim': state_dim,
                'action_dim': action_dim
            }
        }

        # Save experiment results
        results_path = os.path.join(
            self.base_dir,
            f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_path, 'w') as f:
            json.dump(experiment_results, f, indent=2, default=str)

        rl_logger.info(f"Experiment completed and saved to {results_path}")
        return experiment_results

    def _generate_dummy_data(self, length: int = 1000) -> pd.DataFrame:
        """Generate dummy market data for testing."""
        np.random.seed(42)

        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, length)
        prices = 100 * np.exp(np.cumsum(returns))

        # Generate OHLCV data
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, length)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, length))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, length))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, length)
        })

        # Ensure high >= low
        data['high'] = np.maximum(data['high'], data['low'])

        return data

    def get_available_agents(self) -> Dict[str, str]:
        """Get list of available agent types."""
        return {agent_type: agent_class.__name__
                for agent_type, (agent_class, _) in self.agent_registry.items()}


# Convenience functions
def create_rl_system(base_dir: str = "rl_experiments") -> RLSystemManager:
    """Create RL system manager."""
    return RLSystemManager(base_dir)


def quick_experiment(agent_type: str,
                    data_path: Optional[str] = None,
                    max_episodes: int = 100) -> Dict[str, Any]:
    """
    Run a quick RL experiment with default settings.

    Args:
        agent_type: Type of RL agent
        data_path: Path to market data CSV file
        max_episodes: Maximum training episodes

    Returns:
        Experiment results
    """
    system = create_rl_system()

    experiment_config = {
        'agent': {
            'type': agent_type,
            'config': {}
        },
        'environment': {},
        'training': {
            'max_episodes': max_episodes,
            'eval_frequency': max(10, max_episodes // 10),
            'save_frequency': max(20, max_episodes // 5)
        },
        'data_path': data_path
    }

    return system.run_experiment(experiment_config)


def compare_algorithms(data_path: Optional[str] = None,
                      max_episodes: int = 100) -> pd.DataFrame:
    """
    Compare different RL algorithms on the same data.

    Args:
        data_path: Path to market data CSV file
        max_episodes: Maximum training episodes per algorithm

    Returns:
        Comparison results
    """
    system = create_rl_system()

    # Load or generate data
    if data_path and os.path.exists(data_path):
        data = pd.read_csv(data_path)
    else:
        data = system._generate_dummy_data()

    environment = system.create_environment(data)
    state_dim = environment.get_observation_space_size()
    action_dim = environment.get_action_space_size()

    # Test algorithms
    algorithms = ['dqn', 'double_dqn', 'q_learning', 'sarsa', 'reinforce', 'a2c']
    agents = {}

    for algo in algorithms:
        try:
            rl_logger.info(f"Training {algo}...")
            agent = system.create_agent(algo, state_dim, action_dim)

            # Quick training
            training_config = {
                'max_episodes': max_episodes,
                'eval_frequency': max(10, max_episodes // 5),
                'save_frequency': max(20, max_episodes // 2),
                'early_stopping': True,
                'patience': max(10, max_episodes // 10)
            }

            system.train_agent(agent, environment, training_config)
            agents[algo] = agent

        except Exception as e:
            rl_logger.error(f"Failed to train {algo}: {e}")

    # Compare results
    if agents:
        comparison = system.compare_agents(agents, environment)
        return comparison
    else:
        rl_logger.error("No agents were successfully trained")
        return pd.DataFrame()
