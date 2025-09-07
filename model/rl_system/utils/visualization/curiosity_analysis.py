"""
Curiosity and Exploration Visualization for RL Trading System.

This module provides visualization tools specifically for analyzing
curiosity-driven exploration, intrinsic rewards, and ICM performance.
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CuriosityVisualizer:
    """
    Visualization system for curiosity-driven exploration analysis.

    Provides methods to analyze ICM performance, intrinsic rewards,
    and exploration effectiveness.
    """

    def __init__(self, save_dir: str = "curiosity_plots"):
        """
        Initialize the curiosity visualizer.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Color scheme for curiosity analysis
        self.colors = {
            'intrinsic': '#9B59B6',
            'extrinsic': '#3498DB',
            'forward_loss': '#E74C3C',
            'inverse_loss': '#F39C12',
            'curiosity': '#2ECC71',
            'exploration': '#E67E22'
        }

        logger.info(f"CuriosityVisualizer initialized - Save dir: {save_dir}")

    def plot_icm_losses(self,
                       forward_losses: List[float],
                       inverse_losses: List[float],
                       total_losses: List[float],
                       title: str = "ICM Training Losses",
                       figsize: Tuple[int, int] = (15, 6),
                       save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot ICM training losses over time.

        Args:
            forward_losses: Forward model losses
            inverse_losses: Inverse model losses
            total_losses: Total ICM losses
            title: Plot title
            figsize: Figure size
            save_name: Name to save the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        steps = range(len(forward_losses))

        # Forward model loss
        axes[0].plot(steps, forward_losses, color=self.colors['forward_loss'], linewidth=2)
        axes[0].plot(steps, self._smooth_curve(forward_losses),
                    color='darkred', linewidth=2, alpha=0.8, label='Smoothed')
        axes[0].set_title('Forward Model Loss')
        axes[0].set_xlabel('Update Step')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Inverse model loss
        axes[1].plot(steps, inverse_losses, color=self.colors['inverse_loss'], linewidth=2)
        axes[1].plot(steps, self._smooth_curve(inverse_losses),
                    color='darkorange', linewidth=2, alpha=0.8, label='Smoothed')
        axes[1].set_title('Inverse Model Loss')
        axes[1].set_xlabel('Update Step')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Total ICM loss
        axes[2].plot(steps, total_losses, color=self.colors['curiosity'], linewidth=2)
        axes[2].plot(steps, self._smooth_curve(total_losses),
                    color='darkgreen', linewidth=2, alpha=0.8, label='Smoothed')
        axes[2].set_title('Total ICM Loss')
        axes[2].set_xlabel('Update Step')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            import os
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ICM losses plot saved to {save_path}")

        return fig

    def plot_reward_comparison(self,
                             intrinsic_rewards: List[float],
                             extrinsic_rewards: List[float],
                             combined_rewards: Optional[List[float]] = None,
                             title: str = "Intrinsic vs Extrinsic Rewards",
                             figsize: Tuple[int, int] = (15, 10),
                             save_name: Optional[str] = None) -> plt.Figure:
        """
        Compare intrinsic and extrinsic rewards over time.

        Args:
            intrinsic_rewards: Intrinsic reward history
            extrinsic_rewards: Extrinsic reward history
            combined_rewards: Optional combined reward history
            title: Plot title
            figsize: Figure size
            save_name: Name to save the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        steps = range(len(intrinsic_rewards))

        # Reward time series
        axes[0, 0].plot(steps, intrinsic_rewards, color=self.colors['intrinsic'],
                       alpha=0.7, linewidth=1, label='Intrinsic')
        axes[0, 0].plot(steps, self._smooth_curve(intrinsic_rewards),
                       color=self.colors['intrinsic'], linewidth=2, label='Intrinsic (Smoothed)')

        if len(extrinsic_rewards) == len(intrinsic_rewards):
            axes[0, 0].plot(steps, extrinsic_rewards, color=self.colors['extrinsic'],
                           alpha=0.7, linewidth=1, label='Extrinsic')
            axes[0, 0].plot(steps, self._smooth_curve(extrinsic_rewards),
                           color=self.colors['extrinsic'], linewidth=2, label='Extrinsic (Smoothed)')

        axes[0, 0].set_title('Reward Time Series')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Reward distributions
        axes[0, 1].hist(intrinsic_rewards, bins=30, alpha=0.7,
                       color=self.colors['intrinsic'], density=True,
                       label=f'Intrinsic (μ={np.mean(intrinsic_rewards):.4f})')

        if len(extrinsic_rewards) == len(intrinsic_rewards):
            axes[0, 1].hist(extrinsic_rewards, bins=30, alpha=0.7,
                           color=self.colors['extrinsic'], density=True,
                           label=f'Extrinsic (μ={np.mean(extrinsic_rewards):.4f})')

        axes[0, 1].set_title('Reward Distributions')
        axes[0, 1].set_xlabel('Reward Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Rolling correlation
        if len(extrinsic_rewards) == len(intrinsic_rewards):
            correlation = self._rolling_correlation(intrinsic_rewards, extrinsic_rewards)
            axes[1, 0].plot(range(len(correlation)), correlation,
                           color=self.colors['exploration'], linewidth=2)
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[1, 0].set_title('Rolling Correlation (Intrinsic vs Extrinsic)')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Correlation')
            axes[1, 0].grid(True, alpha=0.3)

        # Reward statistics
        stats_text = (
            f"Intrinsic Rewards:\n"
            f"  Mean: {np.mean(intrinsic_rewards):.4f}\n"
            f"  Std: {np.std(intrinsic_rewards):.4f}\n"
            f"  Max: {np.max(intrinsic_rewards):.4f}\n"
            f"  Min: {np.min(intrinsic_rewards):.4f}\n\n"
        )

        if len(extrinsic_rewards) == len(intrinsic_rewards):
            stats_text += (
                f"Extrinsic Rewards:\n"
                f"  Mean: {np.mean(extrinsic_rewards):.4f}\n"
                f"  Std: {np.std(extrinsic_rewards):.4f}\n"
                f"  Max: {np.max(extrinsic_rewards):.4f}\n"
                f"  Min: {np.min(extrinsic_rewards):.4f}\n\n"
                f"Exploration Metrics:\n"
                f"  Intrinsic/Extrinsic Ratio: {np.mean(intrinsic_rewards)/(np.mean(extrinsic_rewards)+1e-8):.3f}\n"
                f"  Correlation: {np.corrcoef(intrinsic_rewards, extrinsic_rewards)[0,1]:.3f}"
            )

        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 1].set_title('Reward Statistics')
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_name:
            import os
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Reward comparison plot saved to {save_path}")

        return fig

    def plot_exploration_effectiveness(self,
                                     states: List,
                                     intrinsic_rewards: List[float],
                                     actions: List[int],
                                     title: str = "Exploration Effectiveness",
                                     figsize: Tuple[int, int] = (15, 8),
                                     save_name: Optional[str] = None) -> plt.Figure:
        """
        Analyze exploration effectiveness using state visitation and curiosity.

        Args:
            states: State history
            intrinsic_rewards: Intrinsic reward history
            actions: Action history
            title: Plot title
            figsize: Figure size
            save_name: Name to save the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # State space exploration (simplified 2D projection)
        if len(states) > 0 and hasattr(states[0], '__len__'):
            # Use first two dimensions or PCA projection
            if len(states[0]) >= 2:
                state_x = [s[0] for s in states]
                state_y = [s[1] for s in states]
            else:
                state_x = [s[0] if hasattr(s, '__len__') else s for s in states]
                state_y = intrinsic_rewards

            # Color by intrinsic reward
            scatter = axes[0, 0].scatter(state_x, state_y, c=intrinsic_rewards,
                                       cmap='viridis', alpha=0.6, s=20)
            axes[0, 0].set_title('State Space Exploration')
            axes[0, 0].set_xlabel('State Dimension 1')
            axes[0, 0].set_ylabel('State Dimension 2')
            plt.colorbar(scatter, ax=axes[0, 0], label='Intrinsic Reward')

        # Curiosity-driven action frequency
        if len(actions) > 0:
            # Bin intrinsic rewards
            high_curiosity_indices = [i for i, r in enumerate(intrinsic_rewards)
                                    if r > np.percentile(intrinsic_rewards, 75)]
            low_curiosity_indices = [i for i, r in enumerate(intrinsic_rewards)
                                   if r < np.percentile(intrinsic_rewards, 25)]

            high_curiosity_actions = [actions[i] for i in high_curiosity_indices if i < len(actions)]
            low_curiosity_actions = [actions[i] for i in low_curiosity_indices if i < len(actions)]

            action_names = ['Hold', 'Buy', 'Sell']

            if high_curiosity_actions and low_curiosity_actions:
                high_counts = np.bincount(high_curiosity_actions, minlength=3)
                low_counts = np.bincount(low_curiosity_actions, minlength=3)

                x = np.arange(len(action_names))
                width = 0.35

                axes[0, 1].bar(x - width/2, high_counts, width,
                              label='High Curiosity', color=self.colors['curiosity'])
                axes[0, 1].bar(x + width/2, low_counts, width,
                              label='Low Curiosity', color=self.colors['exploration'])

                axes[0, 1].set_title('Action Frequency by Curiosity Level')
                axes[0, 1].set_xlabel('Action')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(action_names)
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

        # Intrinsic reward over time with exploration phases
        window_size = max(50, len(intrinsic_rewards) // 20)
        rolling_intrinsic = self._rolling_average(intrinsic_rewards, window_size)

        axes[1, 0].plot(range(len(intrinsic_rewards)), intrinsic_rewards,
                       alpha=0.3, color=self.colors['intrinsic'], linewidth=0.5)
        axes[1, 0].plot(range(len(rolling_intrinsic)), rolling_intrinsic,
                       color=self.colors['intrinsic'], linewidth=2, label='Rolling Average')

        # Mark exploration phases
        exploration_threshold = np.percentile(rolling_intrinsic, 75)
        exploration_phases = np.where(np.array(rolling_intrinsic) > exploration_threshold)[0]

        if len(exploration_phases) > 0:
            axes[1, 0].scatter(exploration_phases,
                             [rolling_intrinsic[i] for i in exploration_phases],
                             color='red', s=10, alpha=0.7, label='High Exploration')

        axes[1, 0].set_title('Exploration Phases')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Intrinsic Reward')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Exploration statistics
        exploration_stats = self._calculate_exploration_stats(
            intrinsic_rewards, actions if len(actions) == len(intrinsic_rewards) else []
        )

        stats_text = "\n".join([f"{k}: {v}" for k, v in exploration_stats.items()])

        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        axes[1, 1].set_title('Exploration Statistics')
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_name:
            import os
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Exploration effectiveness plot saved to {save_path}")

        return fig

    def create_curiosity_dashboard(self,
                                 curiosity_data: Dict[str, List],
                                 title: str = "Curiosity-Driven Exploration Dashboard",
                                 figsize: Tuple[int, int] = (20, 12),
                                 save_name: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive curiosity analysis dashboard.

        Args:
            curiosity_data: Complete curiosity data dictionary
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

        # 1. ICM Losses (top left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, 0:2])
        if 'forward_losses' in curiosity_data and 'inverse_losses' in curiosity_data:
            forward_losses = curiosity_data['forward_losses']
            inverse_losses = curiosity_data['inverse_losses']

            steps = range(len(forward_losses))
            ax1.plot(steps, forward_losses, color=self.colors['forward_loss'],
                    linewidth=2, label='Forward')
            ax1.plot(steps, inverse_losses, color=self.colors['inverse_loss'],
                    linewidth=2, label='Inverse')
            ax1.set_title('ICM Training Losses')
            ax1.set_xlabel('Update Step')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. Reward comparison (top right, spans 2 columns)
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'intrinsic_rewards' in curiosity_data and 'extrinsic_rewards' in curiosity_data:
            intrinsic = curiosity_data['intrinsic_rewards']
            extrinsic = curiosity_data['extrinsic_rewards']

            steps = range(len(intrinsic))
            ax2.plot(steps, self._smooth_curve(intrinsic),
                    color=self.colors['intrinsic'], linewidth=2, label='Intrinsic')

            if len(extrinsic) == len(intrinsic):
                ax2.plot(steps, self._smooth_curve(extrinsic),
                        color=self.colors['extrinsic'], linewidth=2, label='Extrinsic')

            ax2.set_title('Reward Evolution')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Reward')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. Intrinsic reward distribution (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        if 'intrinsic_rewards' in curiosity_data:
            intrinsic = curiosity_data['intrinsic_rewards']
            ax3.hist(intrinsic, bins=30, alpha=0.7, color=self.colors['intrinsic'],
                    edgecolor='black')
            ax3.axvline(np.mean(intrinsic), color='red', linestyle='--',
                       label=f'Mean: {np.mean(intrinsic):.4f}')
            ax3.set_title('Intrinsic Reward Distribution')
            ax3.set_xlabel('Reward')
            ax3.set_ylabel('Frequency')
            ax3.legend()

        # 4. Exploration effectiveness (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        if 'intrinsic_rewards' in curiosity_data:
            intrinsic = curiosity_data['intrinsic_rewards']
            window_size = max(50, len(intrinsic) // 10)
            rolling_avg = self._rolling_average(intrinsic, window_size)

            ax4.plot(range(len(rolling_avg)), rolling_avg,
                    color=self.colors['exploration'], linewidth=2)
            ax4.set_title('Exploration Trend')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Avg Intrinsic Reward')
            ax4.grid(True, alpha=0.3)

        # 5. Action distribution by curiosity (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        if 'actions' in curiosity_data and 'intrinsic_rewards' in curiosity_data:
            actions = curiosity_data['actions']
            intrinsic = curiosity_data['intrinsic_rewards']

            if len(actions) == len(intrinsic):
                high_curiosity_actions = [actions[i] for i, r in enumerate(intrinsic)
                                        if r > np.percentile(intrinsic, 75)]

                if high_curiosity_actions:
                    action_counts = np.bincount(high_curiosity_actions, minlength=3)
                    action_names = ['Hold', 'Buy', 'Sell']

                    colors = [self.colors['exploration'], self.colors['curiosity'],
                             self.colors['intrinsic']][:len(action_counts)]

                    ax5.pie(action_counts, labels=action_names, colors=colors,
                           autopct='%1.1f%%')
                    ax5.set_title('High Curiosity Actions')

        # 6. Forward model accuracy trend (middle far right)
        ax6 = fig.add_subplot(gs[1, 3])
        if 'forward_losses' in curiosity_data:
            forward_losses = curiosity_data['forward_losses']
            # Convert loss to accuracy approximation
            accuracy_approx = [1 / (1 + loss) for loss in forward_losses]
            smoothed_accuracy = self._smooth_curve(accuracy_approx)

            ax6.plot(range(len(smoothed_accuracy)), smoothed_accuracy,
                    color=self.colors['forward_loss'], linewidth=2)
            ax6.set_title('Forward Model Performance')
            ax6.set_xlabel('Update Step')
            ax6.set_ylabel('Accuracy Approx.')
            ax6.grid(True, alpha=0.3)

        # 7. Curiosity statistics (bottom left)
        ax7 = fig.add_subplot(gs[2, 0])
        if 'intrinsic_rewards' in curiosity_data:
            intrinsic = curiosity_data['intrinsic_rewards']

            stats_text = (
                f"Curiosity Stats:\n"
                f"Mean Intrinsic: {np.mean(intrinsic):.4f}\n"
                f"Std Intrinsic: {np.std(intrinsic):.4f}\n"
                f"Max Curiosity: {np.max(intrinsic):.4f}\n"
                f"Exploration Rate: {len([r for r in intrinsic if r > np.mean(intrinsic)])/len(intrinsic)*100:.1f}%"
            )

            ax7.text(0.1, 0.5, stats_text, transform=ax7.transAxes,
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.7))
            ax7.set_title('Curiosity Statistics')
            ax7.axis('off')

        # 8. ICM component comparison (bottom center)
        ax8 = fig.add_subplot(gs[2, 1])
        if 'forward_losses' in curiosity_data and 'inverse_losses' in curiosity_data:
            forward_losses = curiosity_data['forward_losses']
            inverse_losses = curiosity_data['inverse_losses']

            recent_forward = np.mean(forward_losses[-50:]) if len(forward_losses) > 50 else np.mean(forward_losses)
            recent_inverse = np.mean(inverse_losses[-50:]) if len(inverse_losses) > 50 else np.mean(inverse_losses)

            components = ['Forward', 'Inverse']
            values = [recent_forward, recent_inverse]
            colors = [self.colors['forward_loss'], self.colors['inverse_loss']]

            bars = ax8.bar(components, values, color=colors)
            ax8.set_title('Recent ICM Performance')
            ax8.set_ylabel('Average Loss')

            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}', ha='center', va='bottom')

        # 9. Exploration phases (bottom right, spans 2 columns)
        ax9 = fig.add_subplot(gs[2, 2:])
        if 'intrinsic_rewards' in curiosity_data:
            intrinsic = curiosity_data['intrinsic_rewards']

            # Identify exploration phases
            window_size = max(50, len(intrinsic) // 20)
            rolling_avg = self._rolling_average(intrinsic, window_size)
            threshold = np.percentile(rolling_avg, 75)

            ax9.plot(range(len(intrinsic)), intrinsic, alpha=0.3,
                    color=self.colors['intrinsic'], linewidth=0.5)
            ax9.plot(range(len(rolling_avg)), rolling_avg,
                    color=self.colors['intrinsic'], linewidth=2, label='Rolling Avg')
            ax9.axhline(y=threshold, color='red', linestyle='--',
                       alpha=0.7, label=f'Exploration Threshold: {threshold:.4f}')

            # Highlight high exploration periods
            high_exploration = np.where(np.array(rolling_avg) > threshold)[0]
            if len(high_exploration) > 0:
                ax9.fill_between(high_exploration,
                               [rolling_avg[i] for i in high_exploration],
                               threshold, alpha=0.3, color='red', label='High Exploration')

            ax9.set_title('Exploration Timeline')
            ax9.set_xlabel('Time Step')
            ax9.set_ylabel('Intrinsic Reward')
            ax9.legend()
            ax9.grid(True, alpha=0.3)

        if save_name:
            import os
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Curiosity dashboard saved to {save_path}")

        return fig

    # Helper methods
    def _smooth_curve(self, values: List[float], window: int = 20) -> List[float]:
        """Apply smoothing to a curve using moving average."""
        if len(values) < window:
            return values

        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            smoothed.append(np.mean(values[start_idx:i+1]))

        return smoothed

    def _rolling_average(self, values: List[float], window: int = 50) -> List[float]:
        """Compute rolling average with specified window."""
        if len(values) < window:
            return values

        rolling_avg = []
        for i in range(len(values)):
            if i < window - 1:
                rolling_avg.append(np.mean(values[:i+1]))
            else:
                rolling_avg.append(np.mean(values[i-window+1:i+1]))

        return rolling_avg

    def _rolling_correlation(self, x: List[float], y: List[float], window: int = 50) -> List[float]:
        """Compute rolling correlation between two series."""
        if len(x) != len(y) or len(x) < window:
            return []

        correlations = []
        for i in range(window-1, len(x)):
            start_idx = i - window + 1
            x_window = x[start_idx:i+1]
            y_window = y[start_idx:i+1]
            corr = np.corrcoef(x_window, y_window)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)

        return correlations

    def _calculate_exploration_stats(self, intrinsic_rewards: List[float],
                                   actions: List[int] = None) -> Dict[str, str]:
        """Calculate exploration effectiveness statistics."""
        stats = {
            'Avg Intrinsic Reward': f"{np.mean(intrinsic_rewards):.4f}",
            'Intrinsic Reward Std': f"{np.std(intrinsic_rewards):.4f}",
            'Max Curiosity': f"{np.max(intrinsic_rewards):.4f}",
            'High Curiosity Rate': f"{len([r for r in intrinsic_rewards if r > np.percentile(intrinsic_rewards, 75)])/len(intrinsic_rewards)*100:.1f}%"
        }

        if actions and len(actions) == len(intrinsic_rewards):
            # Calculate exploration-driven action diversity
            high_curiosity_indices = [i for i, r in enumerate(intrinsic_rewards)
                                    if r > np.percentile(intrinsic_rewards, 75)]
            high_curiosity_actions = [actions[i] for i in high_curiosity_indices]

            if high_curiosity_actions:
                action_diversity = len(set(high_curiosity_actions)) / len(set(actions))
                stats['Exploration Diversity'] = f"{action_diversity:.3f}"

        return stats


# Utility function for quick curiosity plotting
def quick_plot_curiosity(intrinsic_rewards: List[float],
                        extrinsic_rewards: List[float],
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Quick function to plot curiosity comparison.

    Args:
        intrinsic_rewards: Intrinsic reward history
        extrinsic_rewards: Extrinsic reward history
        save_path: Path to save the plot

    Returns:
        Matplotlib figure
    """
    visualizer = CuriosityVisualizer()

    fig = visualizer.plot_reward_comparison(
        intrinsic_rewards, extrinsic_rewards,
        save_name=save_path.split('/')[-1].split('.')[0] if save_path else None
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
