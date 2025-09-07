"""
Comprehensive Integration Manager.

This module provides a unified interface for managing the entire RL system
with enhanced integration capabilities, Context7 optimizations, and seamless
coordination between all components.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..utils.visualization.curiosity_analysis import CuriosityVisualizer
from ..utils.visualization.training_visualization import TrainingVisualizer
from .algorithm_factory import (
    AgentWrapper,
    AlgorithmSpec,
    AlgorithmType,
    EnsembleAgent,
    algorithm_factory,
)
from .algorithm_switching import AlgorithmSwitcher

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training sessions."""
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    save_frequency: int = 100
    evaluation_frequency: int = 50
    early_stopping_patience: int = 200
    performance_threshold: float = 0.95
    enable_switching: bool = True
    enable_visualization: bool = True
    enable_curiosity_tracking: bool = False


@dataclass
class EnvironmentSpec:
    """Specification for environment setup."""
    env_type: str  # "discrete", "continuous", "mixed"
    state_dim: int
    action_dim: int
    complexity: str = "medium"  # "simple", "medium", "complex"
    has_sparse_rewards: bool = False
    is_continuous_control: bool = False


class IntegrationManager:
    """
    Comprehensive manager for the entire RL system integration.

    This class coordinates all components of the RL system including:
    - Algorithm creation and management
    - Training coordination
    - Performance monitoring
    - Algorithm switching
    - Visualization
    - State management
    """

    def __init__(self, save_dir: str = "rl_training_runs"):
        """
        Initialize the integration manager.

        Args:
            save_dir: Directory for saving training data and models
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Core components
        self.factory = algorithm_factory
        self.switcher = None
        self.training_visualizer = None
        self.curiosity_visualizer = None

        # State tracking
        self.current_session = None
        self.training_history = []
        self.active_agents = {}

        # Configuration
        self.training_config = TrainingConfig()
        self.environment_spec = None

        logger.info(f"IntegrationManager initialized with save directory: {save_dir}")

    def setup_environment(self, env_spec: EnvironmentSpec):
        """
        Setup environment specification for the training session.

        Args:
            env_spec: Environment specification
        """
        self.environment_spec = env_spec

        # Initialize switcher with environment-specific settings
        self.switcher = AlgorithmSwitcher(save_dir=str(self.save_dir / "switches"))

        # Initialize visualizers
        self.training_visualizer = TrainingVisualizer()

        if env_spec.has_sparse_rewards:
            self.curiosity_visualizer = CuriosityVisualizer()
            self.training_config.enable_curiosity_tracking = True

        logger.info(f"Environment setup completed: {env_spec.env_type}, "
                   f"state_dim={env_spec.state_dim}, action_dim={env_spec.action_dim}")

    def create_recommended_agent(self, agent_id: str = "main_agent") -> AgentWrapper:
        """
        Create an agent with recommended algorithm for the current environment.

        Args:
            agent_id: Identifier for the agent

        Returns:
            Created agent wrapper
        """
        if self.environment_spec is None:
            raise ValueError("Environment specification not set. Call setup_environment() first.")

        # Get recommendations
        recommendations = self.factory.get_recommendations(
            environment_type=self.environment_spec.env_type,
            complexity=self.environment_spec.complexity
        )

        if not recommendations:
            raise ValueError("No algorithm recommendations available for this environment")

        # Use the first recommendation
        recommended_algorithm = recommendations[0]

        # Check if we should use exploration enhancement
        exploration_enhancement = self.environment_spec.has_sparse_rewards

        if exploration_enhancement and recommended_algorithm in [AlgorithmType.DQN, AlgorithmType.PPO, AlgorithmType.SAC]:
            # Use ICM-enhanced version
            algorithm_map = {
                AlgorithmType.DQN: AlgorithmType.DQN_ICM,
                AlgorithmType.PPO: AlgorithmType.PPO_ICM,
                AlgorithmType.SAC: AlgorithmType.SAC_ICM
            }
            recommended_algorithm = algorithm_map.get(recommended_algorithm, recommended_algorithm)

        # Create agent specification
        agent_spec = AlgorithmSpec(
            algorithm_type=recommended_algorithm,
            state_dim=self.environment_spec.state_dim,
            action_dim=self.environment_spec.action_dim,
            exploration_enhancement=exploration_enhancement
        )

        # Create agent
        agent = self.factory.create_agent(agent_spec, agent_id)
        self.active_agents[agent_id] = agent

        # Set as initial agent for switcher
        self.switcher.set_initial_agent(agent)

        logger.info(f"Created recommended agent: {recommended_algorithm.value} for {agent_id}")
        return agent

    def create_custom_agent(self, algorithm_type: Union[str, AlgorithmType],
                          agent_id: str,
                          config: Optional[Dict[str, Any]] = None) -> AgentWrapper:
        """
        Create a custom agent with specified algorithm and configuration.

        Args:
            algorithm_type: Algorithm type
            agent_id: Identifier for the agent
            config: Optional custom configuration

        Returns:
            Created agent wrapper
        """
        if self.environment_spec is None:
            raise ValueError("Environment specification not set. Call setup_environment() first.")

        if isinstance(algorithm_type, str):
            algorithm_type = AlgorithmType(algorithm_type)

        agent_spec = AlgorithmSpec(
            algorithm_type=algorithm_type,
            state_dim=self.environment_spec.state_dim,
            action_dim=self.environment_spec.action_dim,
            config=config
        )

        agent = self.factory.create_agent(agent_spec, agent_id)
        self.active_agents[agent_id] = agent

        logger.info(f"Created custom agent: {algorithm_type.value} for {agent_id}")
        return agent

    def create_ensemble_agent(self, algorithm_types: List[Union[str, AlgorithmType]],
                            ensemble_id: str = "ensemble") -> EnsembleAgent:
        """
        Create an ensemble agent with multiple algorithms.

        Args:
            algorithm_types: List of algorithm types
            ensemble_id: Identifier for the ensemble

        Returns:
            Created ensemble agent
        """
        if self.environment_spec is None:
            raise ValueError("Environment specification not set. Call setup_environment() first.")

        specs = []
        for alg_type in algorithm_types:
            if isinstance(alg_type, str):
                alg_type = AlgorithmType(alg_type)

            spec = AlgorithmSpec(
                algorithm_type=alg_type,
                state_dim=self.environment_spec.state_dim,
                action_dim=self.environment_spec.action_dim
            )
            specs.append(spec)

        ensemble = self.factory.create_ensemble(specs, ensemble_id)
        self.active_agents[ensemble_id] = ensemble

        logger.info(f"Created ensemble agent with {len(algorithm_types)} algorithms")
        return ensemble

    def setup_adaptive_switching(self, algorithms: List[Union[str, AlgorithmType]],
                               performance_threshold: float = 0.1):
        """
        Setup adaptive algorithm switching strategy.

        Args:
            algorithms: List of algorithms to cycle through
            performance_threshold: Performance threshold for switching
        """
        if not self.switcher:
            raise ValueError("Switcher not initialized. Call setup_environment() first.")

        algorithm_enums = []
        for alg in algorithms:
            if isinstance(alg, str):
                alg = AlgorithmType(alg)
            algorithm_enums.append(alg)

        conditions = self.switcher.create_adaptive_switching_strategy(
            algorithm_enums, performance_threshold
        )

        # Add conditions to switcher
        for i, condition in enumerate(conditions):
            if i < len(algorithm_enums) - 1:
                target_alg = algorithm_enums[i + 1]
                self.switcher.add_switch_condition(condition, target_alg)

        logger.info(f"Setup adaptive switching with {len(algorithms)} algorithms")

    def train_agent(self, environment, agent: AgentWrapper,
                   config: Optional[TrainingConfig] = None) -> Dict[str, Any]:
        """
        Train an agent in the specified environment.

        Args:
            environment: Training environment
            agent: Agent to train
            config: Optional training configuration

        Returns:
            Training results and metrics
        """
        if config is None:
            config = self.training_config

        logger.info(f"Starting training for {agent.algorithm_type.value}")

        # Training metrics
        episode_rewards = []
        episode_lengths = []
        training_losses = []

        # Training loop
        for episode in range(config.max_episodes):
            state = environment.reset()
            episode_reward = 0
            episode_length = 0
            episode_losses = []

            for step in range(config.max_steps_per_episode):
                # Get action
                action = agent.get_action(state)

                # Environment step
                next_state, reward, done, info = environment.step(action)

                # Store transition
                agent.store_transition(state, action, reward, next_state, done)

                # Update agent
                if hasattr(agent.agent, 'memory') and len(agent.agent.memory) > agent.agent.batch_size:
                    losses = agent.update()
                    episode_losses.append(losses)

                # Update state and metrics
                state = next_state
                episode_reward += reward
                episode_length += 1

                if done:
                    break

            # Record episode performance
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if episode_losses:
                avg_losses = {}
                for key in episode_losses[0].keys():
                    avg_losses[key] = np.mean([loss[key] for loss in episode_losses])
                training_losses.append(avg_losses)
            else:
                training_losses.append({})

            # Record performance for switching
            if self.switcher:
                self.switcher.record_episode_performance(
                    episode_reward, episode_length, training_losses[-1]
                )

                # Check for switching
                if config.enable_switching:
                    new_agent = self.switcher.check_switch_conditions(episode, step)
                    if new_agent:
                        agent = new_agent
                        logger.info(f"Switched to {agent.algorithm_type.value} at episode {episode}")

            # Progress logging
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

            # Save checkpoint
            if episode % config.save_frequency == 0:
                self._save_checkpoint(agent, episode, episode_rewards, training_losses)

            # Visualization updates
            if config.enable_visualization and episode % 50 == 0:
                self._update_visualizations(episode_rewards, training_losses, agent)

            # Early stopping
            if self._check_early_stopping(episode_rewards, config):
                logger.info(f"Early stopping triggered at episode {episode}")
                break

        # Final results
        results = {
            'total_episodes': len(episode_rewards),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'training_losses': training_losses,
            'final_performance': {
                'mean_reward': np.mean(episode_rewards[-100:]) if episode_rewards else 0,
                'std_reward': np.std(episode_rewards[-100:]) if episode_rewards else 0,
                'mean_length': np.mean(episode_lengths[-100:]) if episode_lengths else 0
            },
            'agent_info': agent.get_info(),
            'switch_history': self.switcher.get_switch_history() if self.switcher else []
        }

        # Save final results
        self._save_training_results(results, agent)

        logger.info(f"Training completed. Final mean reward: {results['final_performance']['mean_reward']:.2f}")
        return results

    def evaluate_agent(self, environment, agent: AgentWrapper,
                      num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate an agent's performance.

        Args:
            environment: Evaluation environment
            agent: Agent to evaluate
            num_episodes: Number of evaluation episodes

        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating {agent.algorithm_type.value} for {num_episodes} episodes")

        eval_rewards = []
        eval_lengths = []

        for episode in range(num_episodes):
            state = environment.reset()
            episode_reward = 0
            episode_length = 0

            while True:
                action = agent.get_action(state)
                next_state, reward, done, info = environment.step(action)

                state = next_state
                episode_reward += reward
                episode_length += 1

                if done:
                    break

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)

        metrics = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'std_length': np.std(eval_lengths)
        }

        logger.info(f"Evaluation completed. Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        return metrics

    def compare_algorithms(self, environment, algorithm_types: List[Union[str, AlgorithmType]],
                         num_episodes: int = 100) -> Dict[str, Any]:
        """
        Compare multiple algorithms on the same environment.

        Args:
            environment: Environment for comparison
            algorithm_types: List of algorithms to compare
            num_episodes: Number of episodes for comparison

        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(algorithm_types)} algorithms over {num_episodes} episodes")

        results = {}

        for alg_type in algorithm_types:
            if isinstance(alg_type, str):
                alg_type = AlgorithmType(alg_type)

            # Create agent
            agent = self.create_custom_agent(alg_type, f"compare_{alg_type.value}")

            # Train agent
            config = TrainingConfig(
                max_episodes=num_episodes,
                enable_switching=False,
                enable_visualization=False
            )

            training_results = self.train_agent(environment, agent, config)

            # Evaluate agent
            eval_results = self.evaluate_agent(environment, agent, 10)

            results[alg_type.value] = {
                'training': training_results,
                'evaluation': eval_results,
                'final_performance': training_results['final_performance']
            }

        # Generate comparison summary
        comparison_summary = self._generate_comparison_summary(results)

        logger.info("Algorithm comparison completed")
        return {
            'individual_results': results,
            'summary': comparison_summary
        }

    def _save_checkpoint(self, agent: AgentWrapper, episode: int,
                        rewards: List[float], losses: List[Dict[str, float]]):
        """Save training checkpoint."""
        checkpoint_dir = self.save_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{agent.algorithm_type.value}_episode_{episode}.pth"
        agent.save_state(str(checkpoint_path))

        # Save training data
        data_path = checkpoint_dir / f"training_data_episode_{episode}.json"
        training_data = {
            'episode': episode,
            'rewards': rewards[-1000:],  # Keep last 1000 episodes
            'losses': losses[-1000:] if losses else []
        }

        with open(data_path, 'w') as f:
            json.dump(training_data, f, indent=2)

    def _update_visualizations(self, rewards: List[float], losses: List[Dict[str, float]],
                             agent: AgentWrapper):
        """Update training visualizations."""
        if not self.training_visualizer:
            return

        try:
            # Update training plots
            if rewards:
                self.training_visualizer.plot_training_progress(
                    rewards, title=f"{agent.algorithm_type.value} Training Progress"
                )

            # Update curiosity plots if available
            if (self.curiosity_visualizer and
                hasattr(agent.agent, 'icm_losses') and
                agent.agent.icm_losses):

                self.curiosity_visualizer.plot_icm_losses(agent.agent.icm_losses)

        except Exception as e:
            logger.warning(f"Visualization update failed: {e}")

    def _check_early_stopping(self, rewards: List[float], config: TrainingConfig) -> bool:
        """Check if early stopping criteria are met."""
        if len(rewards) < config.early_stopping_patience:
            return False

        recent_rewards = rewards[-config.early_stopping_patience:]
        mean_reward = np.mean(recent_rewards)

        return mean_reward >= config.performance_threshold

    def _save_training_results(self, results: Dict[str, Any], agent: AgentWrapper):
        """Save final training results."""
        results_dir = self.save_dir / "results"
        results_dir.mkdir(exist_ok=True)

        timestamp = "latest"
        results_file = results_dir / f"{agent.algorithm_type.value}_results_{timestamp}.json"

        # Convert numpy arrays to lists for JSON serialization
        json_results = self._prepare_results_for_json(results)

        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"Training results saved: {results_file}")

    def _prepare_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results for JSON serialization."""
        json_results = {}

        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # Handle list of dictionaries (like training_losses)
                json_results[key] = [
                    {k: v.tolist() if isinstance(v, np.ndarray) else v
                     for k, v in item.items()}
                    for item in value
                ]
            else:
                json_results[key] = value

        return json_results

    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of algorithm comparison."""
        summary = {
            'best_algorithm': None,
            'performance_ranking': [],
            'convergence_analysis': {},
            'efficiency_analysis': {}
        }

        # Rank by final performance
        performance_scores = {}
        for alg_name, result in results.items():
            score = result['final_performance']['mean_reward']
            performance_scores[alg_name] = score

        # Sort by performance
        ranking = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
        summary['performance_ranking'] = ranking
        summary['best_algorithm'] = ranking[0][0] if ranking else None

        return summary

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'environment_spec': asdict(self.environment_spec) if self.environment_spec else None,
            'training_config': asdict(self.training_config),
            'active_agents': {
                agent_id: agent.get_info()
                for agent_id, agent in self.active_agents.items()
            },
            'switching_enabled': self.switcher is not None,
            'visualization_enabled': self.training_visualizer is not None,
            'curiosity_tracking_enabled': self.curiosity_visualizer is not None
        }

        if self.switcher:
            status['switching_status'] = self.switcher.get_performance_summary()

        return status

    def save_system_state(self, filepath: Optional[str] = None):
        """Save the complete system state."""
        if filepath is None:
            filepath = str(self.save_dir / "system_state.json")

        system_state = self.get_system_status()

        with open(filepath, 'w') as f:
            json.dump(system_state, f, indent=2)

        logger.info(f"System state saved: {filepath}")

    def load_system_state(self, filepath: str):
        """Load system state from file."""
        with open(filepath, 'r') as f:
            system_state = json.load(f)

        # Restore configuration
        if 'training_config' in system_state:
            config_dict = system_state['training_config']
            self.training_config = TrainingConfig(**config_dict)

        if 'environment_spec' in system_state and system_state['environment_spec']:
            env_dict = system_state['environment_spec']
            self.environment_spec = EnvironmentSpec(**env_dict)

        logger.info(f"System state loaded: {filepath}")


# Global integration manager instance
integration_manager = IntegrationManager()


# Convenience functions
def setup_training_environment(env_type: str, state_dim: int, action_dim: int,
                             complexity: str = "medium",
                             has_sparse_rewards: bool = False) -> IntegrationManager:
    """
    Setup training environment with the integration manager.

    Args:
        env_type: Environment type
        state_dim: State dimension
        action_dim: Action dimension
        complexity: Problem complexity
        has_sparse_rewards: Whether environment has sparse rewards

    Returns:
        Configured integration manager
    """
    env_spec = EnvironmentSpec(
        env_type=env_type,
        state_dim=state_dim,
        action_dim=action_dim,
        complexity=complexity,
        has_sparse_rewards=has_sparse_rewards,
        is_continuous_control=(env_type == "continuous")
    )

    integration_manager.setup_environment(env_spec)
    return integration_manager


def quick_train(environment, algorithm: str = "auto",
               episodes: int = 1000) -> Dict[str, Any]:
    """
    Quick training function for simple use cases.

    Args:
        environment: Training environment
        algorithm: Algorithm to use ("auto" for recommended)
        episodes: Number of training episodes

    Returns:
        Training results
    """
    # Determine environment specs (simple heuristic)
    sample_state = environment.reset()
    state_dim = len(sample_state) if hasattr(sample_state, '__len__') else 1

    # Assume discrete action space for simplicity
    action_dim = getattr(environment.action_space, 'n', environment.action_space.shape[0])
    env_type = "discrete" if hasattr(environment.action_space, 'n') else "continuous"

    # Setup environment
    manager = setup_training_environment(env_type, state_dim, action_dim)

    # Create agent
    if algorithm == "auto":
        agent = manager.create_recommended_agent()
    else:
        agent = manager.create_custom_agent(algorithm, "quick_train_agent")

    # Configure training
    config = TrainingConfig(max_episodes=episodes)

    # Train
    results = manager.train_agent(environment, agent, config)
    return results
