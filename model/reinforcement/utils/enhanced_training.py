"""
Enhanced Training Workflows for Reinforcement Learning.

This module provides improved training patterns and utilities based on
latest research in reinforcement learning for trading applications.
"""

import time
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import util.loggers as loggers

logger = loggers.setup_loggers()
training_logger = logger['rl']

# Enhanced Training Constants (from latest ML research)
DEFAULT_MAX_EPISODES = 1000
DEFAULT_EARLY_STOP_THRESHOLD = 25  # Episodes of consistent positive performance
DEFAULT_PERFORMANCE_WINDOW = 100   # Episodes to track for performance metrics
DEFAULT_LOG_INTERVAL = 10          # Log every N episodes
DEFAULT_SAVE_INTERVAL = 50         # Save model every N episodes

class PerformanceTracker:
    """
    Enhanced performance tracking for RL training.

    Tracks key metrics like NAV, returns, win rates, and provides
    early stopping capabilities based on performance criteria.
    """

    def __init__(self, window_size: int = DEFAULT_PERFORMANCE_WINDOW):
        self.window_size = window_size
        self.reset()

    def reset(self):
        """Reset all tracking metrics."""
        self.episode_navs = []
        self.market_navs = []
        self.episode_returns = []
        self.episode_rewards = []
        self.win_streak = 0
        self.total_episodes = 0
        self.start_time = time.time()

    def track_episode(self, episode_result: Dict):
        """Track results from a single episode."""
        self.total_episodes += 1

        # Extract key metrics
        nav = episode_result.get('nav', 1.0)
        market_nav = episode_result.get('market_nav', 1.0)
        episode_return = episode_result.get('strategy_return', 0.0)
        total_reward = episode_result.get('total_reward', 0.0)

        # Store metrics
        self.episode_navs.append(nav)
        self.market_navs.append(market_nav)
        self.episode_returns.append(episode_return)
        self.episode_rewards.append(total_reward)

        # Track win streak
        if episode_return > 0:
            self.win_streak += 1
        else:
            self.win_streak = 0

    def get_recent_performance(self, window: Optional[int] = None) -> Dict:
        """Get performance metrics for recent episodes."""
        if not self.episode_navs:
            return {}

        window = window or min(self.window_size, len(self.episode_navs))
        recent_navs = self.episode_navs[-window:]
        recent_market_navs = self.market_navs[-window:]
        recent_returns = self.episode_returns[-window:]

        return {
            'avg_nav': np.mean(recent_navs),
            'avg_market_nav': np.mean(recent_market_navs),
            'avg_return': np.mean(recent_returns),
            'win_rate': np.sum([r > 0 for r in recent_returns]) / len(recent_returns),
            'total_episodes': self.total_episodes,
            'win_streak': self.win_streak,
            'elapsed_time': time.time() - self.start_time
        }

    def should_early_stop(self, threshold: int = DEFAULT_EARLY_STOP_THRESHOLD) -> bool:
        """Check if training should stop early based on consistent performance."""
        if len(self.episode_returns) < threshold:
            return False

        # Check if last N episodes are all positive
        recent_returns = self.episode_returns[-threshold:]
        return all(r > 0 for r in recent_returns)


class TrainingLoop:
    """
    Enhanced training loop for RL agents with professional patterns.

    Features:
    - Performance tracking and logging
    - Early stopping based on performance
    - Model checkpointing
    - Professional metrics reporting
    - Flexible episode management
    """

    def __init__(self,
                 agent,
                 environment,
                 max_episodes: int = DEFAULT_MAX_EPISODES,
                 early_stop_threshold: int = DEFAULT_EARLY_STOP_THRESHOLD,
                 log_interval: int = DEFAULT_LOG_INTERVAL,
                 save_interval: int = DEFAULT_SAVE_INTERVAL,
                 performance_window: int = DEFAULT_PERFORMANCE_WINDOW):

        self.agent = agent
        self.environment = environment
        self.max_episodes = max_episodes
        self.early_stop_threshold = early_stop_threshold
        self.log_interval = log_interval
        self.save_interval = save_interval

        self.performance_tracker = PerformanceTracker(performance_window)
        self.training_history = []

    def run_episode(self) -> Dict:
        """Run a single training episode."""
        state = self.environment.reset()
        episode_reward = 0
        step_count = 0

        # Get state dimensions for compatibility
        if hasattr(self.environment, 'get_state_dimensions'):
            state_dim, _ = self.environment.get_state_dimensions()
        else:
            state_dim = len(state) if hasattr(state, '__len__') else state.shape[0]

        while not self.environment.done:
            # Agent action selection
            if hasattr(self.agent, 'epsilon_greedy_policy'):
                # For DDQN-style agents
                action = self.agent.epsilon_greedy_policy(state.reshape(-1, state_dim))
            elif hasattr(self.agent, 'choose_action'):
                # For TD3/SAC-style agents
                action = self.agent.choose_action(state)
            else:
                # Generic action selection
                action = self.agent.act(state)

            # Environment step
            next_state, reward, done, info = self.environment.step(action)
            episode_reward += reward
            step_count += 1

            # Agent learning
            if hasattr(self.agent, 'memorize_transition'):
                # For DDQN-style agents
                self.agent.memorize_transition(state, action, reward, next_state,
                                             0.0 if done else 1.0)
                if hasattr(self.agent, 'train') and self.agent.train:
                    self.agent.experience_replay()
            elif hasattr(self.agent, 'remember'):
                # For TD3/SAC-style agents
                self.agent.remember(state, action, reward, next_state, done)
                if hasattr(self.agent, 'learn'):
                    self.agent.learn()

            state = next_state

            if done:
                break

        # Get episode results
        if hasattr(self.environment, 'get_episode_result'):
            episode_result = self.environment.get_episode_result()
        else:
            # Fallback result format
            episode_result = {
                'nav': 1.0 + episode_reward / 100,  # Approximate NAV
                'market_nav': 1.0,
                'strategy_return': episode_reward,
                'total_reward': episode_reward,
                'episode_steps': step_count
            }

        return episode_result

    def run_training(self) -> Dict:
        """Run the complete training loop."""
        training_logger.info("Starting enhanced training loop:")
        training_logger.info(f"  Max episodes: {self.max_episodes}")
        training_logger.info(f"  Early stop threshold: {self.early_stop_threshold}")
        training_logger.info(f"  Performance window: {self.performance_tracker.window_size}")

        for episode in range(1, self.max_episodes + 1):
            # Run episode
            episode_result = self.run_episode()

            # Track performance
            self.performance_tracker.track_episode(episode_result)
            self.training_history.append(episode_result)

            # Logging
            if episode % self.log_interval == 0:
                self._log_progress(episode)

            # Model saving
            if episode % self.save_interval == 0:
                self._save_checkpoint(episode)

            # Early stopping check
            if self.performance_tracker.should_early_stop(self.early_stop_threshold):
                training_logger.info(f"Early stopping at episode {episode} - "
                                   f"consistent performance achieved!")
                break

        # Final results
        final_performance = self.performance_tracker.get_recent_performance()
        training_logger.info("Training completed!")
        training_logger.info(f"Final performance: {final_performance}")

        return {
            'final_performance': final_performance,
            'total_episodes': self.performance_tracker.total_episodes,
            'training_history': self.training_history,
            'early_stopped': episode < self.max_episodes
        }

    def _log_progress(self, episode: int):
        """Log training progress."""
        performance = self.performance_tracker.get_recent_performance(self.log_interval)

        if performance:
            training_logger.info(
                f"Episode {episode:>4d} | "
                f"Avg Return: {performance['avg_return']:>6.2f}% | "
                f"Win Rate: {performance['win_rate']:>5.1%} | "
                f"Win Streak: {performance['win_streak']:>2d} | "
                f"Time: {self._format_time(performance['elapsed_time'])}"
            )

    def _save_checkpoint(self, episode: int):
        """Save model checkpoint."""
        if hasattr(self.agent, 'save_models'):
            try:
                self.agent.save_models()
                training_logger.info(f"Model checkpoint saved at episode {episode}")
            except Exception as e:
                training_logger.warning(f"Failed to save checkpoint: {e}")

    def _format_time(self, seconds: float) -> str:
        """Format elapsed time in HH:MM:SS format."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ResultsAnalyzer:
    """
    Enhanced results analysis and visualization.

    Provides professional analysis of training results with
    comprehensive metrics and visualizations.
    """

    def __init__(self, training_history: List[Dict]):
        self.training_history = training_history
        self.results_df = pd.DataFrame(training_history)

    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        if self.results_df.empty:
            return {}

        # Calculate key metrics
        total_episodes = len(self.results_df)
        final_nav = self.results_df['nav'].iloc[-1] if 'nav' in self.results_df else 1.0
        total_return = (final_nav - 1.0) * 100

        returns = self.results_df.get('strategy_return', pd.Series([0] * total_episodes))
        win_rate = (returns > 0).mean()

        # Performance statistics
        report = {
            'total_episodes': total_episodes,
            'final_nav': final_nav,
            'total_return_pct': total_return,
            'win_rate': win_rate,
            'avg_episode_return': returns.mean(),
            'max_episode_return': returns.max(),
            'min_episode_return': returns.min(),
            'return_volatility': returns.std(),
            'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'consecutive_wins': self._calculate_max_consecutive_wins()
        }

        return report

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if 'nav' not in self.results_df:
            return 0.0

        nav_series = self.results_df['nav']
        peak = nav_series.expanding().max()
        drawdown = (nav_series - peak) / peak
        return drawdown.min()

    def _calculate_max_consecutive_wins(self) -> int:
        """Calculate maximum consecutive winning episodes."""
        if 'strategy_return' not in self.results_df:
            return 0

        returns = self.results_df['strategy_return']
        wins = returns > 0

        max_consecutive = 0
        current_consecutive = 0

        for win in wins:
            if win:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def plot_performance(self, save_path: Optional[str] = None):
        """Create comprehensive performance plots."""
        if self.results_df.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: NAV over time
        if 'nav' in self.results_df:
            axes[0, 0].plot(self.results_df.index, self.results_df['nav'],
                          label='Strategy NAV', color='blue', linewidth=2)
            if 'market_nav' in self.results_df:
                axes[0, 0].plot(self.results_df.index, self.results_df['market_nav'],
                              label='Market NAV', color='red', linewidth=1)
            axes[0, 0].set_title('Net Asset Value Over Time')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('NAV')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Episode returns distribution
        if 'strategy_return' in self.results_df:
            axes[0, 1].hist(self.results_df['strategy_return'], bins=30,
                          alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title('Episode Returns Distribution')
            axes[0, 1].set_xlabel('Return (%)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
            axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Rolling win rate
        if 'strategy_return' in self.results_df:
            window = min(50, len(self.results_df) // 4)
            if window > 0:
                rolling_wins = (self.results_df['strategy_return'] > 0).rolling(window).mean()
                axes[1, 0].plot(self.results_df.index, rolling_wins,
                              color='purple', linewidth=2)
                axes[1, 0].set_title(f'Rolling Win Rate (Window: {window})')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Win Rate')
                axes[1, 0].axhline(0.5, color='red', linestyle='--', alpha=0.7)
                axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Cumulative returns
        if 'strategy_return' in self.results_df:
            cumulative_returns = self.results_df['strategy_return'].cumsum()
            axes[1, 1].plot(self.results_df.index, cumulative_returns,
                          color='orange', linewidth=2)
            axes[1, 1].set_title('Cumulative Returns')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Cumulative Return (%)')
            axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.7)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            training_logger.info(f"Performance plots saved to: {save_path}")

        plt.show()


# Utility functions for enhanced training
def run_enhanced_training(agent, environment, **kwargs) -> Dict:
    """
    Convenience function to run enhanced training with default settings.

    Args:
        agent: RL agent to train
        environment: Trading environment
        **kwargs: Additional arguments for TrainingLoop

    Returns:
        Training results dictionary
    """
    training_loop = TrainingLoop(agent, environment, **kwargs)
    return training_loop.run_training()


def analyze_training_results(training_history: List[Dict],
                           save_plots: bool = True,
                           plots_path: str = 'training_performance.png') -> Dict:
    """
    Convenience function to analyze training results.

    Args:
        training_history: List of episode results
        save_plots: Whether to save performance plots
        plots_path: Path to save plots

    Returns:
        Performance report dictionary
    """
    analyzer = ResultsAnalyzer(training_history)
    report = analyzer.generate_performance_report()

    if save_plots:
        analyzer.plot_performance(plots_path)

    return report


# Example usage patterns
def create_training_workflow(agent_class, environment_class, **config):
    """
    Create a complete training workflow with enhanced patterns.

    This is a template function showing how to set up professional
    RL training with the enhanced utilities.
    """
    # Initialize components
    environment = environment_class(**config.get('env_config', {}))
    agent = agent_class(**config.get('agent_config', {}))

    # Run training
    training_results = run_enhanced_training(
        agent=agent,
        environment=environment,
        max_episodes=config.get('max_episodes', DEFAULT_MAX_EPISODES),
        early_stop_threshold=config.get('early_stop_threshold', DEFAULT_EARLY_STOP_THRESHOLD)
    )

    # Analyze results
    performance_report = analyze_training_results(
        training_results['training_history'],
        save_plots=config.get('save_plots', True),
        plots_path=config.get('plots_path', 'enhanced_training_results.png')
    )

    return {
        'training_results': training_results,
        'performance_report': performance_report,
        'agent': agent,
        'environment': environment
    }
