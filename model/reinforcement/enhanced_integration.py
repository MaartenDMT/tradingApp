"""
Enhanced RL Trading System Integration.

Complete integration example showing how to use all enhanced components
together for professional reinforcement learning trading applications.
"""

import os
import sys
from datetime import datetime
from typing import Dict, Optional

import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import util.loggers as loggers
from model.reinforcement.ddqn.ddqn_tf import EnhancedDDQNAgent

# Import enhanced components
from model.reinforcement.environment.trading_environment import (
    OptimizedTradingEnvironment,
)
from model.reinforcement.td3.td3_tf import EnhancedTD3Agent
from model.reinforcement.utils.enhanced_models import ModelManager
from model.reinforcement.utils.enhanced_training import (
    OPTIMAL_DDQN_CONFIG,
    OPTIMAL_TD3_CONFIG,
    OPTIMAL_TRAINING_CONFIG,
    PerformanceTracker,
    ResultsAnalyzer,
    TrainingLoop,
)

# Setup logging
logger = loggers.setup_loggers()
app_logger = logger['app']

class IntegratedTradingSystem:
    """
    Complete integrated RL trading system with enhanced components.

    Provides a professional, production-ready reinforcement learning
    trading system with optimized algorithms and comprehensive monitoring.
    """

    def __init__(self,
                 data_file: str,
                 algorithm: str = 'td3',
                 config: Optional[Dict] = None):
        """
        Initialize the integrated trading system.

        Args:
            data_file: Path to trading data CSV file
            algorithm: RL algorithm to use ('td3', 'ddqn')
            config: Optional configuration overrides
        """
        self.data_file = data_file
        self.algorithm = algorithm.lower()
        self.config = config or {}

        # Initialize components
        self.environment = None
        self.agent = None
        self.trainer = None
        self.model_manager = ModelManager()
        self.performance_tracker = PerformanceTracker()

        # Training state
        self.is_trained = False
        self.best_episode_reward = float('-inf')
        self.training_history = []

        app_logger.info(f"Initialized trading system with {algorithm.upper()} algorithm")

    def setup_environment(self, **env_kwargs) -> None:
        """
        Setup the enhanced trading environment.

        Args:
            **env_kwargs: Additional environment configuration
        """
        try:
            # Default environment configuration
            default_config = {
                'initial_balance': 10000,
                'lookback_window': 20,
                'transaction_cost_pct': 0.001,
                'time_cost_pct': 0.0001,
                'normalize_observations': True
            }
            default_config.update(env_kwargs)

            self.environment = OptimizedTradingEnvironment(
                data_file=self.data_file,
                **default_config
            )

            app_logger.info("Enhanced trading environment setup complete")

        except Exception as e:
            app_logger.error(f"Failed to setup environment: {e}")
            raise

    def setup_agent(self, **agent_kwargs) -> None:
        """
        Setup the enhanced RL agent.

        Args:
            **agent_kwargs: Additional agent configuration
        """
        try:
            if not self.environment:
                raise ValueError("Environment must be setup before agent")

            state_dim = self.environment.observation_space.shape[0]

            if self.algorithm == 'td3':
                # Setup TD3 agent
                config = OPTIMAL_TD3_CONFIG.copy()
                config.update(self.config)
                config.update(agent_kwargs)

                self.agent = EnhancedTD3Agent(
                    state_dim=state_dim,
                    action_dim=1,  # Single continuous action for position sizing
                    **config
                )

            elif self.algorithm == 'ddqn':
                # Setup DDQN agent
                config = OPTIMAL_DDQN_CONFIG.copy()
                config.update(self.config)
                config.update(agent_kwargs)

                num_actions = 3  # Hold, Buy, Sell
                self.agent = EnhancedDDQNAgent(
                    state_size=state_dim,
                    action_size=num_actions,
                    **config
                )

            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")

            app_logger.info(f"Enhanced {self.algorithm.upper()} agent setup complete")

        except Exception as e:
            app_logger.error(f"Failed to setup agent: {e}")
            raise

    def train(self,
              episodes: int = 1000,
              save_interval: int = 100,
              evaluation_interval: int = 50,
              **training_kwargs) -> Dict:
        """
        Train the RL agent with enhanced monitoring.

        Args:
            episodes: Number of training episodes
            save_interval: Interval for saving models
            evaluation_interval: Interval for evaluation
            **training_kwargs: Additional training configuration

        Returns:
            Training results dictionary
        """
        try:
            if not self.agent or not self.environment:
                raise ValueError("Agent and environment must be setup before training")

            # Setup training configuration
            config = OPTIMAL_TRAINING_CONFIG.copy()
            config.update(training_kwargs)

            # Initialize training loop
            self.trainer = TrainingLoop(
                agent=self.agent,
                environment=self.environment,
                performance_tracker=self.performance_tracker,
                **config
            )

            # Training loop with enhanced monitoring
            app_logger.info(f"Starting training for {episodes} episodes")
            start_time = datetime.now()

            for episode in range(episodes):
                # Training step
                episode_results = self.trainer.train_episode()
                self.training_history.append(episode_results)

                # Track performance
                episode_reward = episode_results['total_reward']
                self.performance_tracker.update_metrics({
                    'episode': episode,
                    'reward': episode_reward,
                    'profit': episode_results.get('total_profit', 0),
                    'trades': episode_results.get('num_trades', 0),
                    'exploration_rate': getattr(self.agent, 'epsilon', 0)
                })

                # Check for best performance
                if episode_reward > self.best_episode_reward:
                    self.best_episode_reward = episode_reward
                    self.save_best_model(episode)

                # Periodic saving
                if (episode + 1) % save_interval == 0:
                    self.save_model(episode)

                # Periodic evaluation
                if (episode + 1) % evaluation_interval == 0:
                    eval_results = self.evaluate(num_episodes=5)
                    app_logger.info(f"Episode {episode+1}: Eval reward = {eval_results['mean_reward']:.2f}")

                # Progress logging
                if (episode + 1) % 50 == 0:
                    elapsed = datetime.now() - start_time
                    app_logger.info(f"Episode {episode+1}/{episodes} - "
                                  f"Reward: {episode_reward:.2f} - "
                                  f"Best: {self.best_episode_reward:.2f} - "
                                  f"Time: {elapsed}")

            # Final results
            training_time = datetime.now() - start_time
            results = self.performance_tracker.get_summary()
            results['training_time'] = training_time
            results['episodes_completed'] = episodes

            self.is_trained = True
            app_logger.info(f"Training completed in {training_time}")

            return results

        except Exception as e:
            app_logger.error(f"Training failed: {e}")
            raise

    def evaluate(self,
                 num_episodes: int = 10,
                 render: bool = False) -> Dict:
        """
        Evaluate the trained agent.

        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render environment

        Returns:
            Evaluation results
        """
        try:
            if not self.agent or not self.environment:
                raise ValueError("Agent and environment must be setup before evaluation")

            app_logger.info(f"Starting evaluation for {num_episodes} episodes")

            evaluation_rewards = []
            evaluation_profits = []

            for episode in range(num_episodes):
                state = self.environment.reset()
                episode_reward = 0
                done = False

                while not done:
                    # Agent action (no exploration during evaluation)
                    if self.algorithm == 'td3':
                        action = self.agent.act(state, add_noise=False)
                    else:  # DDQN
                        action = self.agent.act(state, training=False)

                    # Environment step
                    next_state, reward, done, info = self.environment.step(action)
                    episode_reward += reward
                    state = next_state

                    if render and episode == 0:  # Only render first episode
                        self.environment.render()

                evaluation_rewards.append(episode_reward)
                evaluation_profits.append(info.get('total_profit', 0))

            # Calculate evaluation metrics
            results = {
                'mean_reward': np.mean(evaluation_rewards),
                'std_reward': np.std(evaluation_rewards),
                'max_reward': np.max(evaluation_rewards),
                'min_reward': np.min(evaluation_rewards),
                'mean_profit': np.mean(evaluation_profits),
                'std_profit': np.std(evaluation_profits),
                'success_rate': len([r for r in evaluation_rewards if r > 0]) / len(evaluation_rewards),
                'episodes': num_episodes
            }

            app_logger.info(f"Evaluation complete - Mean reward: {results['mean_reward']:.2f}")
            return results

        except Exception as e:
            app_logger.error(f"Evaluation failed: {e}")
            raise

    def save_model(self, episode: Optional[int] = None) -> None:
        """Save the current model."""
        try:
            model_name = f"{self.algorithm}_trading_agent"

            if hasattr(self.agent, 'save'):
                # Agent has custom save method
                self.agent.save(os.path.join(self.model_manager.model_dir, model_name))
            else:
                # Use model manager for standard Keras models
                if hasattr(self.agent, 'model'):
                    self.model_manager.save_model(self.agent.model, model_name, episode)

            app_logger.info(f"Model saved for episode {episode}")

        except Exception as e:
            app_logger.error(f"Failed to save model: {e}")

    def save_best_model(self, episode: int) -> None:
        """Save the best performing model."""
        try:
            model_name = f"{self.algorithm}_trading_agent_best"

            if hasattr(self.agent, 'save'):
                self.agent.save(os.path.join(self.model_manager.model_dir, model_name))
            else:
                if hasattr(self.agent, 'model'):
                    self.model_manager.save_model(self.agent.model, model_name, episode)

            app_logger.info(f"Best model saved at episode {episode} with reward {self.best_episode_reward:.2f}")

        except Exception as e:
            app_logger.error(f"Failed to save best model: {e}")

    def load_model(self, model_path: str) -> None:
        """Load a trained model."""
        try:
            if hasattr(self.agent, 'load'):
                self.agent.load(model_path)
            else:
                # Load using model manager
                model = self.model_manager.load_model(model_path)
                if hasattr(self.agent, 'model'):
                    self.agent.model = model

            self.is_trained = True
            app_logger.info(f"Model loaded from {model_path}")

        except Exception as e:
            app_logger.error(f"Failed to load model: {e}")
            raise

    def get_trading_performance_report(self) -> Dict:
        """
        Generate comprehensive trading performance report.

        Returns:
            Performance report dictionary
        """
        try:
            if not self.training_history:
                return {"error": "No training history available"}

            # Analyze training history
            analyzer = ResultsAnalyzer()
            results = analyzer.analyze_training_session(self.training_history)

            # Add system-specific metrics
            results['system_info'] = {
                'algorithm': self.algorithm,
                'data_file': self.data_file,
                'total_episodes': len(self.training_history),
                'best_episode_reward': self.best_episode_reward,
                'is_trained': self.is_trained
            }

            # Performance summary
            if self.training_history:
                rewards = [ep['total_reward'] for ep in self.training_history]
                profits = [ep.get('total_profit', 0) for ep in self.training_history]

                results['performance_summary'] = {
                    'final_reward': rewards[-1] if rewards else 0,
                    'average_reward': np.mean(rewards) if rewards else 0,
                    'best_reward': max(rewards) if rewards else 0,
                    'final_profit': profits[-1] if profits else 0,
                    'total_profit': sum(profits) if profits else 0,
                    'profitable_episodes': len([p for p in profits if p > 0])
                }

            return results

        except Exception as e:
            app_logger.error(f"Failed to generate performance report: {e}")
            return {"error": str(e)}


def create_enhanced_trading_system(data_file: str,
                                  algorithm: str = 'td3',
                                  custom_config: Optional[Dict] = None) -> IntegratedTradingSystem:
    """
    Factory function to create a fully configured enhanced trading system.

    Args:
        data_file: Path to trading data
        algorithm: RL algorithm ('td3' or 'ddqn')
        custom_config: Optional configuration overrides

    Returns:
        Configured IntegratedTradingSystem
    """
    # Create system
    system = IntegratedTradingSystem(
        data_file=data_file,
        algorithm=algorithm,
        config=custom_config
    )

    # Setup components
    system.setup_environment()
    system.setup_agent()

    app_logger.info(f"Enhanced {algorithm.upper()} trading system created successfully")
    return system


# Example usage and testing
if __name__ == "__main__":
    # Example: Create and train a TD3 trading system
    try:
        # Sample data file path
        data_file = "data/csv/BTC_1h.csv"

        if not os.path.exists(data_file):
            print(f"Data file not found: {data_file}")
            print("Please ensure you have trading data available.")
            sys.exit(1)

        # Create enhanced trading system
        print("Creating enhanced TD3 trading system...")
        system = create_enhanced_trading_system(
            data_file=data_file,
            algorithm='td3'
        )

        # Train the system
        print("Starting training...")
        training_results = system.train(
            episodes=100,  # Short training for example
            save_interval=25,
            evaluation_interval=20
        )

        # Evaluate performance
        print("Evaluating performance...")
        eval_results = system.evaluate(num_episodes=10)

        # Generate performance report
        print("Generating performance report...")
        report = system.get_trading_performance_report()

        # Display results
        print("\n=== Training Results ===")
        print(f"Total episodes: {training_results.get('episodes_completed', 0)}")
        print(f"Training time: {training_results.get('training_time', 'N/A')}")
        print(f"Best episode reward: {system.best_episode_reward:.2f}")

        print("\n=== Evaluation Results ===")
        print(f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"Success rate: {eval_results['success_rate']:.2%}")
        print(f"Mean profit: {eval_results['mean_profit']:.2f}")

        print("\n=== System Ready ===")
        print("Enhanced trading system setup and tested successfully!")

    except Exception as e:
        print(f"Error in example: {e}")
        import traceback
        traceback.print_exc()
