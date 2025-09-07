"""
Context7 Enhanced Training Workflows for Trading RL Agents.

This module implements professional training patterns following Context7 best practices:
- Enhanced training loops with performance tracking
- Early stopping based on consistent performance
- Professional metrics and logging
- Model selection and hyperparameter optimization
- Portfolio-level performance evaluation
"""

import time
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

import util.loggers as loggers

logger = loggers.setup_loggers()
rl_logger = logger['rl']

# Context7 Training Constants
CONTEXT7_MAX_EPISODES = 1000
CONTEXT7_EARLY_STOP_EPISODES = 25  # Consistent performance episodes
CONTEXT7_PERFORMANCE_WINDOW = 100  # Performance averaging window
CONTEXT7_LOG_INTERVAL = 10        # Logging frequency
CONTEXT7_SAVE_INTERVAL = 50       # Model saving frequency


class Context7PerformanceTracker:
    """
    Professional performance tracking following Context7 patterns.
    """

    def __init__(self):
        self.episode_results = []
        self.nav_history = []
        self.market_nav_history = []
        self.differences = []
        self.episode_times = []
        self.epsilon_history = []

        # Context7 metrics
        self.strategy_wins = []
        self.rolling_sharpe = []
        self.rolling_volatility = []
        self.max_drawdown_history = []

    def track_episode(self, episode: int, result: Dict, epsilon: float = None):
        """Track episode results with Context7 metrics."""
        self.episode_results.append(result)
        self.nav_history.append(result['nav'])
        self.market_nav_history.append(result['market_nav'])
        self.differences.append(result['difference'])

        if epsilon is not None:
            self.epsilon_history.append(epsilon)

        # Calculate Context7 strategy wins
        win_rate = sum([1 for d in self.differences[-100:] if d > 0]) / min(len(self.differences), 100)
        self.strategy_wins.append(win_rate)

        rl_logger.debug(f"Episode {episode}: NAV={result['nav']:.4f}, "
                       f"Market={result['market_nav']:.4f}, "
                       f"Difference={result['difference']:.2f}%, "
                       f"Win Rate={win_rate:.2%}")

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        if not self.nav_history:
            return {}

        nav_array = np.array(self.nav_history)
        market_array = np.array(self.market_nav_history)
        diff_array = np.array(self.differences)

        return {
            'episodes': len(self.nav_history),
            'final_nav': nav_array[-1] if len(nav_array) > 0 else 1.0,
            'final_market_nav': market_array[-1] if len(market_array) > 0 else 1.0,
            'avg_nav_100': np.mean(nav_array[-100:]) if len(nav_array) >= 100 else np.mean(nav_array),
            'avg_nav_10': np.mean(nav_array[-10:]) if len(nav_array) >= 10 else np.mean(nav_array),
            'avg_market_100': np.mean(market_array[-100:]) if len(market_array) >= 100 else np.mean(market_array),
            'avg_market_10': np.mean(market_array[-10:]) if len(market_array) >= 10 else np.mean(market_array),
            'win_rate': np.mean([1 for d in diff_array if d > 0]) if len(diff_array) > 0 else 0,
            'avg_outperformance': np.mean(diff_array) if len(diff_array) > 0 else 0,
            'volatility': np.std(diff_array) if len(diff_array) > 1 else 0,
            'max_drawdown': self._calculate_max_drawdown(nav_array),
            'sharpe_ratio': self._calculate_sharpe_ratio(diff_array)
        }

    def _calculate_max_drawdown(self, nav_array: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(nav_array) < 2:
            return 0.0

        peak = nav_array[0]
        max_dd = 0.0

        for nav in nav_array:
            if nav > peak:
                peak = nav
            dd = (peak - nav) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0.0


class Context7TrainingLoop:
    """
    Enhanced training loop following Context7 best practices.
    """

    def __init__(self,
                 env,
                 agent,
                 max_episodes: int = CONTEXT7_MAX_EPISODES,
                 early_stop_episodes: int = CONTEXT7_EARLY_STOP_EPISODES,
                 performance_window: int = CONTEXT7_PERFORMANCE_WINDOW,
                 log_interval: int = CONTEXT7_LOG_INTERVAL,
                 save_interval: int = CONTEXT7_SAVE_INTERVAL,
                 results_path: Optional[Path] = None):

        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.early_stop_episodes = early_stop_episodes
        self.performance_window = performance_window
        self.log_interval = log_interval
        self.save_interval = save_interval

        # Results tracking
        self.results_path = results_path or Path("results/context7_training")
        self.results_path.mkdir(parents=True, exist_ok=True)

        # Performance tracking
        self.tracker = Context7PerformanceTracker()
        self.start_time = None

        rl_logger.info("Context7 Training Loop initialized:")
        rl_logger.info(f"  Max episodes: {max_episodes}")
        rl_logger.info(f"  Early stop threshold: {early_stop_episodes}")
        rl_logger.info(f"  Performance window: {performance_window}")
        rl_logger.info(f"  Results path: {self.results_path}")

    def train(self) -> Dict:
        """
        Execute Context7 enhanced training loop.

        Returns:
            Training results dictionary
        """
        rl_logger.info("Starting Context7 training...")
        self.start_time = time.time()

        try:
            for episode in range(1, self.max_episodes + 1):
                episode_start_time = time.time()

                # Run episode
                episode_result = self._run_episode(episode)

                # Track performance
                epsilon = getattr(self.agent, 'epsilon', None)
                self.tracker.track_episode(episode, episode_result, epsilon)

                # Track episode time
                episode_time = time.time() - episode_start_time
                self.tracker.episode_times.append(episode_time)

                # Logging
                if episode % self.log_interval == 0:
                    self._log_progress(episode)

                # Model saving
                if episode % self.save_interval == 0:
                    self._save_checkpoint(episode)

                # Early stopping check
                if self._check_early_stopping():
                    rl_logger.info(f"Early stopping at episode {episode} - "
                                  f"Consistent performance achieved!")
                    break

            # Final results
            training_results = self._finalize_training()
            rl_logger.info("Context7 training completed successfully!")

            return training_results

        except Exception as e:
            rl_logger.error(f"Training failed: {e}")
            raise
        finally:
            self.env.close()

    def _run_episode(self, episode: int) -> Dict:
        """Run a single training episode."""
        state = self.env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # Agent action selection
            action = self.agent.choose_action(state)

            # Environment step
            next_state, reward, done, info = self.env.step(action)

            # Agent learning
            if hasattr(self.agent, 'remember'):
                self.agent.remember(state, action, reward, next_state, done)

            if hasattr(self.agent, 'learn'):
                self.agent.learn()
            elif hasattr(self.agent, 'experience_replay'):
                self.agent.experience_replay()

            state = next_state
            episode_reward += reward
            steps += 1

            # Safety check for infinite episodes
            if steps > self.env.max_episode_steps:
                rl_logger.warning(f"Episode {episode} exceeded max steps ({steps})")
                break

        # Get episode result from environment
        result = self.env.get_episode_result()
        result['episode_reward'] = episode_reward
        result['episode_steps'] = steps

        return result

    def _log_progress(self, episode: int):
        """Log training progress with Context7 metrics."""
        summary = self.tracker.get_performance_summary()

        elapsed_time = time.time() - self.start_time

        rl_logger.info(f"Episode {episode:4d} | {self._format_time(elapsed_time)} | "
                      f"Agent: {summary['avg_nav_100']-1:6.1%} ({summary['avg_nav_10']-1:6.1%}) | "
                      f"Market: {summary['avg_market_100']-1:6.1%} ({summary['avg_market_10']-1:6.1%}) | "
                      f"Wins: {summary['win_rate']:5.1%} | "
                      f"Sharpe: {summary['sharpe_ratio']:6.3f}")

        # Agent-specific metrics
        if hasattr(self.agent, 'get_performance_metrics'):
            agent_metrics = self.agent.get_performance_metrics()
            if 'epsilon' in agent_metrics:
                rl_logger.info(f"  Agent - ε: {agent_metrics['epsilon']:.3f}, "
                              f"Buffer: {agent_metrics.get('replay_buffer_size', 'N/A')}")

    def _save_checkpoint(self, episode: int):
        """Save model checkpoint."""
        try:
            checkpoint_dir = self.results_path / f"checkpoints/episode_{episode}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            if hasattr(self.agent, 'save_models'):
                self.agent.save_models(str(checkpoint_dir))
            elif hasattr(self.agent, 'save_model'):
                self.agent.save_model(str(checkpoint_dir / "model"))

            rl_logger.info(f"Checkpoint saved at episode {episode}")

        except Exception as e:
            rl_logger.error(f"Failed to save checkpoint: {e}")

    def _check_early_stopping(self) -> bool:
        """Check if early stopping criteria are met."""
        if len(self.tracker.differences) < self.early_stop_episodes:
            return False

        # Check if last N episodes all have positive performance
        recent_performance = self.tracker.differences[-self.early_stop_episodes:]
        return all(diff > 0 for diff in recent_performance)

    def _finalize_training(self) -> Dict:
        """Finalize training and generate results."""
        # Save final model
        final_model_path = self.results_path / "final_model"
        final_model_path.mkdir(parents=True, exist_ok=True)

        if hasattr(self.agent, 'save_models'):
            self.agent.save_models(str(final_model_path))

        # Generate results DataFrame
        results_df = pd.DataFrame({
            'Episode': list(range(1, len(self.tracker.nav_history) + 1)),
            'Agent_NAV': self.tracker.nav_history,
            'Market_NAV': self.tracker.market_nav_history,
            'Difference': self.tracker.differences,
            'Strategy_Wins_Pct': self.tracker.strategy_wins
        })

        # Calculate rolling metrics
        results_df['Strategy_Wins_100'] = results_df['Difference'].rolling(100).apply(
            lambda x: (x > 0).sum()
        )

        # Save results
        results_df.to_csv(self.results_path / 'training_results.csv', index=False)

        # Generate visualizations
        self._generate_visualizations(results_df)

        # Final performance summary
        final_summary = self.tracker.get_performance_summary()
        final_summary['total_time'] = time.time() - self.start_time
        final_summary['results_path'] = str(self.results_path)

        return final_summary

    def _generate_visualizations(self, results_df: pd.DataFrame):
        """Generate Context7 training visualizations."""
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")

            # Performance plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Context7 Training Results', fontsize=16, fontweight='bold')

            # NAV comparison
            axes[0, 0].plot(results_df['Episode'], results_df['Agent_NAV'],
                           label='Agent NAV', linewidth=2)
            axes[0, 0].plot(results_df['Episode'], results_df['Market_NAV'],
                           label='Market NAV', linewidth=2)
            axes[0, 0].set_title('Net Asset Value Comparison')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('NAV')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Rolling returns
            agent_returns = (results_df['Agent_NAV'] - 1) * 100
            market_returns = (results_df['Market_NAV'] - 1) * 100

            axes[0, 1].plot(results_df['Episode'], agent_returns.rolling(50).mean(),
                           label='Agent (50-ep avg)', linewidth=2)
            axes[0, 1].plot(results_df['Episode'], market_returns.rolling(50).mean(),
                           label='Market (50-ep avg)', linewidth=2)
            axes[0, 1].set_title('Rolling Average Returns (%)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Return (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Outperformance
            axes[1, 0].plot(results_df['Episode'], results_df['Difference'],
                           alpha=0.6, linewidth=1)
            axes[1, 0].plot(results_df['Episode'],
                           pd.Series(results_df['Difference']).rolling(50).mean(),
                           label='50-episode average', linewidth=2, color='red')
            axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('Agent Outperformance (%)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Outperformance (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Strategy wins
            if 'Strategy_Wins_100' in results_df.columns:
                win_rate = results_df['Strategy_Wins_100'] / 100
                axes[1, 1].plot(results_df['Episode'], win_rate, linewidth=2)
                axes[1, 1].axhline(0.5, color='black', linestyle='--', alpha=0.5)
                axes[1, 1].set_title('Strategy Win Rate (100-episode window)')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Win Rate')
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.results_path / 'training_performance.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

            rl_logger.info(f"Visualizations saved to {self.results_path}")

        except Exception as e:
            rl_logger.error(f"Failed to generate visualizations: {e}")

    def _format_time(self, seconds: float) -> str:
        """Format time in HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def context7_train_agent(env, agent, **kwargs) -> Dict:
    """
    Convenience function for Context7 training.

    Args:
        env: Trading environment
        agent: RL agent
        **kwargs: Additional training parameters

    Returns:
        Training results dictionary
    """
    trainer = Context7TrainingLoop(env, agent, **kwargs)
    return trainer.train()


# Export main classes and functions
__all__ = [
    'Context7PerformanceTracker',
    'Context7TrainingLoop',
    'context7_train_agent'
]
