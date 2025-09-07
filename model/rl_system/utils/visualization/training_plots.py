"""
Training Progress Visualization for RL Trading System.

This module provides comprehensive visualization tools for monitoring
and analyzing RL training progress, performance metrics, and algorithm comparison.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TrainingVisualizer:
    """
    Comprehensive visualization system for RL training progress.

    Provides methods to plot training metrics, performance analysis,
    and algorithm comparisons with professional styling.
    """

    def __init__(self, save_dir: str = "training_plots"):
        """
        Initialize the training visualizer.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Color scheme for consistency
        self.colors = {
            'reward': '#2E86C1',
            'loss': '#E74C3C',
            'epsilon': '#F39C12',
            'profit': '#27AE60',
            'drawdown': '#8E44AD',
            'sharpe': '#17A2B8',
            'trades': '#FD7E14'
        }

        logger.info(f"TrainingVisualizer initialized - Save dir: {save_dir}")

    def plot_training_progress(self,
                             training_stats: Dict[str, List],
                             title: str = "Training Progress",
                             figsize: Tuple[int, int] = (15, 10),
                             save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive training progress with multiple metrics.

        Args:
            training_stats: Dictionary containing training statistics
            title: Plot title
            figsize: Figure size
            save_name: Name to save the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Episode rewards
        if 'episode_rewards' in training_stats:
            episodes = range(len(training_stats['episode_rewards']))
            rewards = training_stats['episode_rewards']

            axes[0, 0].plot(episodes, rewards, color=self.colors['reward'], alpha=0.7)
            axes[0, 0].plot(episodes, self._smooth_curve(rewards),
                           color=self.colors['reward'], linewidth=2, label='Smoothed')
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Training losses
        if 'losses' in training_stats:
            episodes = range(len(training_stats['losses']))
            losses = training_stats['losses']

            axes[0, 1].plot(episodes, losses, color=self.colors['loss'], alpha=0.7)
            axes[0, 1].plot(episodes, self._smooth_curve(losses),
                           color=self.colors['loss'], linewidth=2, label='Smoothed')
            axes[0, 1].set_title('Training Loss')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_yscale('log')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Exploration rate (epsilon for DQN-style algorithms)
        if 'exploration_rates' in training_stats:
            episodes = range(len(training_stats['exploration_rates']))
            exploration = training_stats['exploration_rates']

            axes[0, 2].plot(episodes, exploration, color=self.colors['epsilon'], linewidth=2)
            axes[0, 2].set_title('Exploration Rate')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Epsilon')
            axes[0, 2].grid(True, alpha=0.3)

        # Cumulative rewards
        if 'episode_rewards' in training_stats:
            rewards = training_stats['episode_rewards']
            cumulative_rewards = np.cumsum(rewards)
            episodes = range(len(cumulative_rewards))

            axes[1, 0].plot(episodes, cumulative_rewards, color=self.colors['profit'], linewidth=2)
            axes[1, 0].set_title('Cumulative Rewards')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Cumulative Reward')
            axes[1, 0].grid(True, alpha=0.3)

        # Rolling average performance
        if 'episode_rewards' in training_stats:
            rewards = training_stats['episode_rewards']
            rolling_avg = self._rolling_average(rewards, window=50)
            episodes = range(len(rolling_avg))

            axes[1, 1].plot(episodes, rolling_avg, color=self.colors['reward'], linewidth=2)
            axes[1, 1].axhline(y=np.mean(rewards), color=self.colors['loss'],
                              linestyle='--', alpha=0.7, label=f'Overall Avg: {np.mean(rewards):.2f}')
            axes[1, 1].set_title('Rolling Average (50 episodes)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Average Reward')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # Training statistics summary
        if 'episode_rewards' in training_stats:
            rewards = training_stats['episode_rewards']
            stats_text = (
                f"Episodes: {len(rewards)}\n"
                f"Mean Reward: {np.mean(rewards):.2f}\n"
                f"Std Reward: {np.std(rewards):.2f}\n"
                f"Max Reward: {np.max(rewards):.2f}\n"
                f"Min Reward: {np.min(rewards):.2f}\n"
                f"Final 100 Avg: {np.mean(rewards[-100:]):.2f}"
            )

            axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                           fontsize=12, verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            axes[1, 2].set_title('Training Statistics')
            axes[1, 2].axis('off')

        plt.tight_layout()

        if save_name:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training progress plot saved to {save_path}")

        return fig

    def plot_algorithm_comparison(self,
                                algorithm_results: Dict[str, Dict],
                                metrics: List[str] = ['mean_reward', 'std_reward', 'max_reward'],
                                title: str = "Algorithm Comparison",
                                figsize: Tuple[int, int] = (15, 8),
                                save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison between different algorithms.

        Args:
            algorithm_results: Dictionary with algorithm names as keys and results as values
            metrics: List of metrics to compare
            title: Plot title
            figsize: Figure size
            save_name: Name to save the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=16, fontweight='bold')

        algorithms = list(algorithm_results.keys())

        for idx, metric in enumerate(metrics):
            values = [algorithm_results[algo].get(metric, 0) for algo in algorithms]
            colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))

            bars = axes[idx].bar(algorithms, values, color=colors, alpha=0.8)
            axes[idx].set_title(f'{metric.replace("_", " ").title()}')
            axes[idx].set_ylabel('Value')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                              f'{value:.2f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_name:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Algorithm comparison plot saved to {save_path}")

        return fig

    def plot_learning_curves(self,
                           training_histories: Dict[str, List],
                           title: str = "Learning Curves",
                           figsize: Tuple[int, int] = (12, 8),
                           save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot learning curves for multiple algorithms.

        Args:
            training_histories: Dictionary with algorithm names and their reward histories
            title: Plot title
            figsize: Figure size
            save_name: Name to save the plot

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        colors = plt.cm.tab10(np.linspace(0, 1, len(training_histories)))

        # Raw learning curves
        for (algo_name, rewards), color in zip(training_histories.items(), colors):
            episodes = range(len(rewards))
            ax1.plot(episodes, rewards, color=color, alpha=0.3, linewidth=0.5)
            ax1.plot(episodes, self._smooth_curve(rewards, window=20),
                    color=color, linewidth=2, label=algo_name)

        ax1.set_title('Learning Curves')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Rolling average comparison
        for (algo_name, rewards), color in zip(training_histories.items(), colors):
            rolling_avg = self._rolling_average(rewards, window=100)
            episodes = range(len(rolling_avg))
            ax2.plot(episodes, rolling_avg, color=color, linewidth=2, label=algo_name)

        ax2.set_title('Rolling Average (100 episodes)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curves plot saved to {save_path}")

        return fig

    def plot_loss_components(self,
                           loss_data: Dict[str, List],
                           title: str = "Loss Components",
                           figsize: Tuple[int, int] = (15, 6),
                           save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot different loss components for algorithms like PPO, SAC, etc.

        Args:
            loss_data: Dictionary with loss component names and values
            title: Plot title
            figsize: Figure size
            save_name: Name to save the plot

        Returns:
            Matplotlib figure
        """
        n_components = len(loss_data)
        fig, axes = plt.subplots(1, n_components, figsize=figsize)
        if n_components == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=16, fontweight='bold')

        colors = ['#E74C3C', '#3498DB', '#F39C12', '#27AE60', '#8E44AD']

        for idx, (component_name, values) in enumerate(loss_data.items()):
            episodes = range(len(values))
            color = colors[idx % len(colors)]

            axes[idx].plot(episodes, values, color=color, alpha=0.7, linewidth=1)
            axes[idx].plot(episodes, self._smooth_curve(values),
                          color=color, linewidth=2, label='Smoothed')
            axes[idx].set_title(f'{component_name.replace("_", " ").title()}')
            axes[idx].set_xlabel('Update Step')
            axes[idx].set_ylabel('Loss')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Loss components plot saved to {save_path}")

        return fig

    def plot_hyperparameter_sensitivity(self,
                                      sensitivity_results: Dict[str, Dict],
                                      parameter_name: str,
                                      metric: str = 'mean_reward',
                                      title: Optional[str] = None,
                                      figsize: Tuple[int, int] = (10, 6),
                                      save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot hyperparameter sensitivity analysis.

        Args:
            sensitivity_results: Results from hyperparameter sensitivity analysis
            parameter_name: Name of the parameter being analyzed
            metric: Metric to plot
            title: Plot title
            figsize: Figure size
            save_name: Name to save the plot

        Returns:
            Matplotlib figure
        """
        if title is None:
            title = f"Hyperparameter Sensitivity: {parameter_name}"

        fig, ax = plt.subplots(figsize=figsize)

        param_values = []
        metric_values = []
        error_bars = []

        for param_val, results in sensitivity_results.items():
            param_values.append(float(param_val))
            metric_values.append(results.get(metric, 0))
            error_bars.append(results.get(f'{metric}_std', 0))

        # Sort by parameter value
        sorted_data = sorted(zip(param_values, metric_values, error_bars))
        param_values, metric_values, error_bars = zip(*sorted_data)

        ax.errorbar(param_values, metric_values, yerr=error_bars,
                   marker='o', linewidth=2, markersize=8, capsize=5,
                   color=self.colors['reward'])

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(parameter_name.replace('_', ' ').title())
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)

        # Highlight best value
        best_idx = np.argmax(metric_values)
        ax.scatter(param_values[best_idx], metric_values[best_idx],
                  color='red', s=100, zorder=5, marker='*',
                  label=f'Best: {param_values[best_idx]}')
        ax.legend()

        plt.tight_layout()

        if save_name:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Hyperparameter sensitivity plot saved to {save_path}")

        return fig

    def create_training_dashboard(self,
                                training_data: Dict[str, Any],
                                title: str = "RL Training Dashboard",
                                figsize: Tuple[int, int] = (20, 12),
                                save_name: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive training dashboard with multiple visualizations.

        Args:
            training_data: Complete training data dictionary
            title: Dashboard title
            figsize: Figure size
            save_name: Name to save the dashboard

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=20, fontweight='bold')

        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. Training progress (top left, 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        if 'episode_rewards' in training_data:
            rewards = training_data['episode_rewards']
            episodes = range(len(rewards))
            ax1.plot(episodes, rewards, alpha=0.5, color=self.colors['reward'])
            ax1.plot(episodes, self._smooth_curve(rewards),
                    linewidth=2, color=self.colors['reward'])
            ax1.set_title('Training Progress', fontweight='bold')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True, alpha=0.3)

        # 2. Loss progression (top right, 1x2)
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'losses' in training_data:
            losses = training_data['losses']
            steps = range(len(losses))
            ax2.plot(steps, losses, color=self.colors['loss'])
            ax2.set_title('Training Loss', fontweight='bold')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Loss')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)

        # 3. Performance distribution (middle right)
        ax3 = fig.add_subplot(gs[1, 2:])
        if 'episode_rewards' in training_data:
            rewards = training_data['episode_rewards']
            ax3.hist(rewards, bins=30, alpha=0.7, color=self.colors['reward'], edgecolor='black')
            ax3.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
            ax3.set_title('Reward Distribution', fontweight='bold')
            ax3.set_xlabel('Reward')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Training statistics (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        if 'episode_rewards' in training_data:
            rewards = training_data['episode_rewards']
            stats_text = (
                f"Total Episodes: {len(rewards)}\n"
                f"Mean Reward: {np.mean(rewards):.2f}\n"
                f"Std Reward: {np.std(rewards):.2f}\n"
                f"Best Reward: {np.max(rewards):.2f}\n"
                f"Worst Reward: {np.min(rewards):.2f}\n"
                f"Success Rate: {len([r for r in rewards if r > 0])/len(rewards)*100:.1f}%"
            )
            ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
                    fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            ax4.set_title('Key Statistics', fontweight='bold')
            ax4.axis('off')

        # 5. Rolling performance (bottom center)
        ax5 = fig.add_subplot(gs[2, 1])
        if 'episode_rewards' in training_data:
            rewards = training_data['episode_rewards']
            rolling_avg = self._rolling_average(rewards, window=50)
            episodes = range(len(rolling_avg))
            ax5.plot(episodes, rolling_avg, color=self.colors['profit'], linewidth=2)
            ax5.set_title('Rolling Average (50)', fontweight='bold')
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Avg Reward')
            ax5.grid(True, alpha=0.3)

        # 6. Recent performance trend (bottom right)
        ax6 = fig.add_subplot(gs[2, 2:])
        if 'episode_rewards' in training_data:
            rewards = training_data['episode_rewards']
            recent_rewards = rewards[-100:] if len(rewards) > 100 else rewards
            recent_episodes = range(len(recent_rewards))
            ax6.plot(recent_episodes, recent_rewards, color=self.colors['sharpe'], linewidth=2)
            ax6.axhline(y=np.mean(recent_rewards), color='red', linestyle='--', alpha=0.7,
                       label=f'Recent Avg: {np.mean(recent_rewards):.2f}')
            ax6.set_title('Recent Performance (Last 100)', fontweight='bold')
            ax6.set_xlabel('Recent Episode')
            ax6.set_ylabel('Reward')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        if save_name:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training dashboard saved to {save_path}")

        return fig

    def _smooth_curve(self, values: List[float], window: int = 20) -> np.ndarray:
        """Apply smoothing to a curve using moving average."""
        if len(values) < window:
            return np.array(values)

        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            smoothed.append(np.mean(values[start_idx:i+1]))

        return np.array(smoothed)

    def _rolling_average(self, values: List[float], window: int = 50) -> np.ndarray:
        """Compute rolling average with specified window."""
        if len(values) < window:
            return np.array(values)

        rolling_avg = []
        for i in range(len(values)):
            if i < window - 1:
                rolling_avg.append(np.mean(values[:i+1]))
            else:
                rolling_avg.append(np.mean(values[i-window+1:i+1]))

        return np.array(rolling_avg)


# Utility functions for quick plotting
def quick_plot_training(rewards: List[float],
                       losses: Optional[List[float]] = None,
                       title: str = "Training Progress",
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Quick function to plot training progress.

    Args:
        rewards: List of episode rewards
        losses: Optional list of training losses
        title: Plot title
        save_path: Path to save the plot

    Returns:
        Matplotlib figure
    """
    visualizer = TrainingVisualizer()

    training_stats = {'episode_rewards': rewards}
    if losses:
        training_stats['losses'] = losses

    fig = visualizer.plot_training_progress(
        training_stats, title=title,
        save_name=save_path.split('/')[-1].split('.')[0] if save_path else None
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def quick_compare_algorithms(algorithm_results: Dict[str, Dict],
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Quick function to compare algorithm performance.

    Args:
        algorithm_results: Dictionary with algorithm results
        save_path: Path to save the plot

    Returns:
        Matplotlib figure
    """
    visualizer = TrainingVisualizer()

    fig = visualizer.plot_algorithm_comparison(
        algorithm_results,
        save_name=save_path.split('/')[-1].split('.')[0] if save_path else None
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
