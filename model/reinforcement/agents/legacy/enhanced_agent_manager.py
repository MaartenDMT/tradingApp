"""
Enhanced Agent Manager with Modern RL Patterns
Integrates with modern_agent.py and provides optimized agent management
"""

import logging
import warnings
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Conditional imports with proper error handling
try:
    from stable_baselines3 import A2C, DQN, PPO, TD3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    STABLE_BASELINES3_AVAILABLE = True
except ImportError as e:
    print(f"Warning: stable_baselines3 import failed: {e}")
    STABLE_BASELINES3_AVAILABLE = False
    # Create dummy classes to prevent import errors
    class A2C:
        pass

    class DQN:
        pass

    class PPO:
        pass

    class TD3:
        pass

    class BaseCallback:
        pass

    class DummyVecEnv:
        pass

# Modern agent implementation
try:
    from model.reinforcement.agents.modern_agent import (BaseAgent, Callback,
                                                         CheckpointCallback,
                                                         EvaluationCallback,
                                                         LoggingCallback,
                                                         ModernDQNAgent)
    MODERN_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Modern agent import failed: {e}")
    MODERN_AGENT_AVAILABLE = False

# Environment imports
from model.reinforcement.environments import TradingEnvironment

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class ModernAgentManager:
    """
    Modern Agent Manager with Enhanced Features

    Features:
    - Unified interface for all agent types
    - Modern training patterns with callbacks
    - Performance monitoring and optimization
    - Automatic checkpointing and evaluation
    - Multi-agent support with efficient parallel execution
    """

    def __init__(self, params: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the agent manager.

        Args:
            params: Configuration parameters for agents and training
            logger: Optional logger instance
        """
        self.params = params
        self.logger = logger or logging.getLogger(__name__)

        # Initialize environment
        self.env = self._create_environment()

        # Agent registry
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_configs: Dict[str, Dict] = {}

        # Training state
        self.training_metrics: Dict[str, Dict] = {}
        self.callbacks: List[Callback] = []

        self.logger.info("ModernAgentManager initialized")

    def _create_environment(self) -> TradingEnvironment:
        """Create trading environment with parameters."""
        env = TradingEnvironment(
            symbol=self.params.get('symbol', 'BTCUSDT'),
            features=self.params.get('features', ['close', 'volume']),
            limit=self.params.get('limit', 300),
            time=self.params.get('time', '30m'),
            actions=self.params.get('env_actions', 3),
            min_acc=self.params.get('min_acc', 0.55)
        )

        self.logger.info(f"Environment created - Observation space: {env.observation_space.shape}")
        self.logger.info(f"Action space: {env.action_space.n}")
        return env

    def create_agent(self, agent_type: str, agent_name: str, **agent_params) -> BaseAgent:
        """
        Create and register a new agent.

        Args:
            agent_type: Type of agent ('modern_dqn', 'stable_baselines', 'legacy')
            agent_name: Unique name for the agent
            **agent_params: Agent-specific parameters

        Returns:
            Created agent instance
        """
        # Merge default params with agent-specific params
        config = {**self.params, **agent_params}

        if agent_type == 'modern_dqn' and MODERN_AGENT_AVAILABLE:
            agent = self._create_modern_dqn_agent(config)
        elif agent_type == 'stable_baselines' and STABLE_BASELINES3_AVAILABLE:
            agent = self._create_stable_baselines_agent(config)
        elif agent_type == 'legacy':
            agent = self._create_legacy_agent(config)
        else:
            raise ValueError(f"Unknown agent type: {agent_type} or dependencies not available")

        self.agents[agent_name] = agent
        self.agent_configs[agent_name] = config

        self.logger.info(f"Agent '{agent_name}' of type '{agent_type}' created successfully")
        return agent

    def _create_modern_dqn_agent(self, config: Dict) -> ModernDQNAgent:
        """Create modern DQN agent with optimized parameters."""
        agent = ModernDQNAgent(
            state_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.n,
            learning_rate=config.get('learning_rate', 0.001),
            gamma=config.get('gamma', 0.95),
            epsilon=config.get('epsilon', 1.0),
            epsilon_min=config.get('epsilon_min', 0.01),
            epsilon_decay=config.get('epsilon_decay', 0.995),
            batch_size=config.get('batch_size', 32),
            memory_size=config.get('memory_size', 10000),
            target_update_freq=config.get('target_update_freq', 100),
            model_type=config.get('model_type', 'standard'),
            hidden_units=config.get('hidden_units', 64),
            dropout=config.get('dropout', 0.1)
        )

        # Add default callbacks
        self._add_default_callbacks(agent, config)
        return agent

    def _create_stable_baselines_agent(self, config: Dict) -> Any:
        """Create Stable Baselines3 agent."""
        algorithm = config.get('algorithm', 'DQN')

        # Wrap environment for Stable Baselines3
        vec_env = DummyVecEnv([lambda: self.env])

        agents = {
            'DQN': lambda: DQN('MlpPolicy', vec_env, verbose=1, **config.get('sb3_params', {})),
            'PPO': lambda: PPO('MlpPolicy', vec_env, verbose=1, **config.get('sb3_params', {})),
            'A2C': lambda: A2C('MlpPolicy', vec_env, verbose=1, **config.get('sb3_params', {})),
            'TD3': lambda: TD3('MlpPolicy', vec_env, verbose=1, **config.get('sb3_params', {}))
        }

        if algorithm not in agents:
            raise ValueError(f"Unknown Stable Baselines3 algorithm: {algorithm}")

        return agents[algorithm]()

    def _create_legacy_agent(self, config: Dict) -> Any:
        """Create legacy agent for backward compatibility."""
        # Legacy agent creation logic would go here
        # This maintains compatibility with existing code
        raise NotImplementedError("Legacy agent creation not yet implemented")

    def _add_default_callbacks(self, agent: BaseAgent, config: Dict) -> None:
        """Add default callbacks to modern agents."""
        if not isinstance(agent, ModernDQNAgent):
            return

        # Logging callback
        log_freq = config.get('log_freq', 10)
        agent.add_callback(LoggingCallback(log_freq=log_freq))

        # Checkpoint callback
        save_path = config.get('save_path', 'data/saved_model/checkpoints')
        save_freq = config.get('checkpoint_freq', 500)
        agent.add_callback(CheckpointCallback(agent, save_path, save_freq))

        # Evaluation callback
        eval_freq = config.get('eval_freq', 100)
        num_eval_episodes = config.get('num_eval_episodes', 5)
        agent.add_callback(EvaluationCallback(self.env, eval_freq, num_eval_episodes))

    def train_agent(self, agent_name: str, episodes: int, **training_params) -> Dict[str, Any]:
        """
        Train a specific agent.

        Args:
            agent_name: Name of the agent to train
            episodes: Number of training episodes
            **training_params: Additional training parameters

        Returns:
            Training metrics and results
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")

        agent = self.agents[agent_name]

        self.logger.info(f"Starting training for agent '{agent_name}' - {episodes} episodes")

        if isinstance(agent, ModernDQNAgent):
            metrics = self._train_modern_agent(agent, episodes, **training_params)
        elif STABLE_BASELINES3_AVAILABLE and hasattr(agent, 'learn'):
            metrics = self._train_stable_baselines_agent(agent, episodes, **training_params)
        else:
            metrics = self._train_legacy_agent(agent, episodes, **training_params)

        self.training_metrics[agent_name] = metrics
        self.logger.info(f"Training completed for agent '{agent_name}'")

        return metrics

    def _train_modern_agent(self, agent: ModernDQNAgent, episodes: int, **params) -> Dict[str, Any]:
        """Train modern DQN agent with advanced features."""
        eval_freq = params.get('eval_freq', 100)

        # Training with modern patterns
        metrics = agent.train(self.env, episodes, eval_freq)

        # Additional performance analysis
        self._analyze_training_performance(agent, metrics)

        return metrics

    def _train_stable_baselines_agent(self, agent: Any, episodes: int, **params) -> Dict[str, Any]:
        """Train Stable Baselines3 agent."""
        total_timesteps = episodes * 200  # Approximate timesteps per episode

        # Add callbacks if specified
        callbacks = params.get('callbacks', [])

        agent.learn(total_timesteps=total_timesteps, callback=callbacks)

        return {
            'algorithm': agent.__class__.__name__,
            'total_timesteps': total_timesteps,
            'episodes': episodes
        }

    def _train_legacy_agent(self, agent: Any, episodes: int, **params) -> Dict[str, Any]:
        """Train legacy agent for backward compatibility."""
        # Legacy training logic would go here
        raise NotImplementedError("Legacy agent training not yet implemented")

    def _analyze_training_performance(self, agent: ModernDQNAgent, metrics: Dict) -> None:
        """Analyze and log training performance."""
        episode_rewards = metrics.get('episode_rewards', [])

        if episode_rewards:
            avg_reward = np.mean(episode_rewards[-100:])  # Last 100 episodes
            max_reward = np.max(episode_rewards)
            min_reward = np.min(episode_rewards)

            self.logger.info("Training Performance Analysis:")
            self.logger.info(f"  Average Reward (last 100): {avg_reward:.2f}")
            self.logger.info(f"  Max Reward: {max_reward:.2f}")
            self.logger.info(f"  Min Reward: {min_reward:.2f}")

            # Plot learning curve
            self._plot_learning_curve(episode_rewards, agent.__class__.__name__)

    def _plot_learning_curve(self, rewards: List[float], agent_type: str) -> None:
        """Plot and save learning curve."""
        try:
            plt.figure(figsize=(12, 6))

            # Moving average for smoother curve
            window_size = min(50, len(rewards) // 10)
            if window_size > 1:
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(range(window_size-1, len(rewards)), moving_avg,
                        label=f'Moving Average (window={window_size})', linewidth=2)

            plt.plot(rewards, alpha=0.3, label='Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title(f'Training Progress - {agent_type}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save plot
            plt.savefig(f'data/plots/learning_curve_{agent_type}.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.warning(f"Failed to plot learning curve: {e}")

    def evaluate_agent(self, agent_name: str, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate agent performance.

        Args:
            agent_name: Name of the agent to evaluate
            num_episodes: Number of evaluation episodes

        Returns:
            Evaluation metrics
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")

        agent = self.agents[agent_name]

        if isinstance(agent, ModernDQNAgent):
            avg_reward = agent.evaluate(self.env, num_episodes)
            return {
                'average_reward': avg_reward,
                'num_episodes': num_episodes,
                'agent_type': agent.__class__.__name__
            }
        else:
            # Implement evaluation for other agent types
            self.logger.warning(f"Evaluation not implemented for agent type: {type(agent)}")
            return {}

    def compare_agents(self, agent_names: List[str], num_episodes: int = 10) -> Dict[str, Dict]:
        """
        Compare performance of multiple agents.

        Args:
            agent_names: List of agent names to compare
            num_episodes: Number of episodes for comparison

        Returns:
            Comparison results
        """
        results = {}

        for name in agent_names:
            if name in self.agents:
                results[name] = self.evaluate_agent(name, num_episodes)
            else:
                self.logger.warning(f"Agent '{name}' not found, skipping comparison")

        # Log comparison results
        self.logger.info("Agent Performance Comparison:")
        for name, metrics in results.items():
            avg_reward = metrics.get('average_reward', 'N/A')
            self.logger.info(f"  {name}: {avg_reward:.2f}")

        return results

    def save_agent(self, agent_name: str, filepath: str) -> None:
        """Save agent to file."""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")

        agent = self.agents[agent_name]

        if hasattr(agent, 'save'):
            agent.save(filepath)
            self.logger.info(f"Agent '{agent_name}' saved to {filepath}")
        else:
            self.logger.error(f"Agent '{agent_name}' does not support saving")

    def load_agent(self, agent_name: str, filepath: str) -> None:
        """Load agent from file."""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")

        agent = self.agents[agent_name]

        if hasattr(agent, 'load'):
            agent.load(filepath)
            self.logger.info(f"Agent '{agent_name}' loaded from {filepath}")
        else:
            self.logger.error(f"Agent '{agent_name}' does not support loading")

    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """Get information about a specific agent."""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")

        agent = self.agents[agent_name]
        config = self.agent_configs.get(agent_name, {})

        return {
            'agent_type': agent.__class__.__name__,
            'config': config,
            'training_metrics': self.training_metrics.get(agent_name, {}),
            'state_size': getattr(agent, 'state_size', 'N/A'),
            'action_size': getattr(agent, 'action_size', 'N/A')
        }

    def list_agents(self) -> List[str]:
        """List all registered agents."""
        return list(self.agents.keys())

    def remove_agent(self, agent_name: str) -> None:
        """Remove an agent from the manager."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            if agent_name in self.agent_configs:
                del self.agent_configs[agent_name]
            if agent_name in self.training_metrics:
                del self.training_metrics[agent_name]
            self.logger.info(f"Agent '{agent_name}' removed")
        else:
            self.logger.warning(f"Agent '{agent_name}' not found")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary for all agents."""
        summary = {
            'total_agents': len(self.agents),
            'agent_types': {},
            'best_performing_agent': None,
            'best_average_reward': float('-inf')
        }

        # Count agent types
        for agent in self.agents.values():
            agent_type = agent.__class__.__name__
            summary['agent_types'][agent_type] = summary['agent_types'].get(agent_type, 0) + 1

        # Find best performing agent
        for name, metrics in self.training_metrics.items():
            episode_rewards = metrics.get('episode_rewards', [])
            if episode_rewards:
                avg_reward = np.mean(episode_rewards[-100:])
                if avg_reward > summary['best_average_reward']:
                    summary['best_average_reward'] = avg_reward
                    summary['best_performing_agent'] = name

        return summary


# Legacy compatibility - keeping original classes with deprecation warnings
class TensorflowModel:
    """Legacy TensorFlow model wrapper - DEPRECATED"""

    def __init__(self, params, rl_logger, num_agents=1):
        warnings.warn("TensorflowModel is deprecated. Use ModernAgentManager instead.",
                     DeprecationWarning, stacklevel=2)
        self.manager = ModernAgentManager(params, rl_logger)
        self.num_agents = num_agents
        self.env = self.manager.env
        self.rl_logger = rl_logger
        self.params = params


class TorchModel:
    """Legacy PyTorch model wrapper - DEPRECATED"""

    def __init__(self, params, rl_logger, num_agents=1):
        warnings.warn("TorchModel is deprecated. Use ModernAgentManager instead.",
                     DeprecationWarning, stacklevel=2)
        self.manager = ModernAgentManager(params, rl_logger)
        self.num_agents = num_agents
        self.env = self.manager.env
        self.rl_logger = rl_logger
        self.params = params


class StablebaselineModel:
    """Legacy Stable Baselines model wrapper - DEPRECATED"""

    def __init__(self, params, rl_logger, num_agents=1):
        warnings.warn("StablebaselineModel is deprecated. Use ModernAgentManager instead.",
                     DeprecationWarning, stacklevel=2)
        self.manager = ModernAgentManager(params, rl_logger)
        self.num_agents = num_agents
        self.env = self.manager.env
        self.rl_logger = rl_logger
        self.params = params
        self.params = params
