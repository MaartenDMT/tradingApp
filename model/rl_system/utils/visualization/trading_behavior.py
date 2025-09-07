"""
Trading Behavior Visualization for RL Trading System.

This module provides visualization tools for analyzing trading behavior,
action patterns, market interaction, and strategy insights.
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


class TradingBehaviorVisualizer:
    """
    Comprehensive trading behavior visualization system.

    Provides methods to analyze and visualize trading patterns,
    action distributions, market timing, and strategic insights.
    """

    def __init__(self, save_dir: str = "behavior_plots"):
        """
        Initialize the trading behavior visualizer.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Color scheme for trading actions
        self.colors = {
            'buy': '#27AE60',
            'sell': '#E74C3C',
            'hold': '#F39C12',
            'price': '#3498DB',
            'volume': '#9B59B6',
            'profit': '#2ECC71',
            'loss': '#E67E22'
        }

        logger.info(f"TradingBehaviorVisualizer initialized - Save dir: {save_dir}")

    def plot_action_distribution(self,
                               actions: List[int],
                               action_names: List[str] = None,
                               title: str = "Action Distribution",
                               figsize: Tuple[int, int] = (12, 8),
                               save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of trading actions taken by the agent.

        Args:
            actions: List of action indices
            action_names: Optional list of action names
            title: Plot title
            figsize: Figure size
            save_name: Name to save the plot

        Returns:
            Matplotlib figure
        """
        if action_names is None:
            action_names = ['Hold', 'Buy', 'Sell']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Action frequency
        unique_actions, counts = np.unique(actions, return_counts=True)
        colors = [self.colors['hold'], self.colors['buy'], self.colors['sell']][:len(unique_actions)]

        bars = ax1.bar([action_names[i] for i in unique_actions], counts, color=colors)
        ax1.set_title('Action Frequency')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)

        # Add percentage labels
        total_actions = len(actions)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = count / total_actions * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')

        # Action sequence over time
        ax2.plot(range(len(actions)), actions, alpha=0.7, linewidth=1, color='darkblue')
        ax2.scatter(range(len(actions)), actions, alpha=0.5, s=10, color='red')
        ax2.set_title('Action Sequence Over Time')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Action')
        ax2.set_yticks(range(len(action_names)))
        ax2.set_yticklabels(action_names)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            import os
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Action distribution plot saved to {save_path}")

        return fig

    def plot_trading_patterns(self,
                            prices: List[float],
                            actions: List[int],
                            portfolio_values: List[float],
                            action_names: List[str] = None,
                            title: str = "Trading Patterns",
                            figsize: Tuple[int, int] = (15, 10),
                            save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot trading patterns overlaid on price chart.

        Args:
            prices: Asset prices
            actions: Trading actions
            portfolio_values: Portfolio values over time
            action_names: Optional action names
            title: Plot title
            figsize: Figure size
            save_name: Name to save the plot

        Returns:
            Matplotlib figure
        """
        if action_names is None:
            action_names = ['Hold', 'Buy', 'Sell']

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize,
                                           gridspec_kw={'height_ratios': [2, 1, 1]})
        fig.suptitle(title, fontsize=16, fontweight='bold')

        time_steps = range(len(prices))

        # Price chart with trading actions
        ax1.plot(time_steps, prices, color=self.colors['price'], linewidth=2, label='Price')

        # Mark buy/sell actions
        buy_points = [i for i, action in enumerate(actions) if action == 1]  # Assuming 1 = buy
        sell_points = [i for i, action in enumerate(actions) if action == 2]  # Assuming 2 = sell

        if buy_points:
            ax1.scatter([i for i in buy_points], [prices[i] for i in buy_points],
                       color=self.colors['buy'], marker='^', s=50, label='Buy', zorder=5)

        if sell_points:
            ax1.scatter([i for i in sell_points], [prices[i] for i in sell_points],
                       color=self.colors['sell'], marker='v', s=50, label='Sell', zorder=5)

        ax1.set_title('Price Chart with Trading Actions')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Portfolio value evolution
        ax2.plot(time_steps, portfolio_values, color=self.colors['profit'], linewidth=2)
        ax2.set_title('Portfolio Value Evolution')
        ax2.set_ylabel('Portfolio Value')
        ax2.grid(True, alpha=0.3)

        # Action timeline
        action_colors = [self.colors['hold'] if a == 0 else
                        self.colors['buy'] if a == 1 else
                        self.colors['sell'] for a in actions]

        ax3.scatter(time_steps, actions, c=action_colors, alpha=0.7, s=20)
        ax3.set_title('Action Timeline')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Action')
        ax3.set_yticks(range(len(action_names)))
        ax3.set_yticklabels(action_names)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            import os
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trading patterns plot saved to {save_path}")

        return fig

    def plot_market_timing_analysis(self,
                                  prices: List[float],
                                  actions: List[int],
                                  returns: List[float],
                                  title: str = "Market Timing Analysis",
                                  figsize: Tuple[int, int] = (15, 8),
                                  save_name: Optional[str] = None) -> plt.Figure:
        """
        Analyze market timing effectiveness.

        Args:
            prices: Asset prices
            actions: Trading actions
            returns: Market returns
            title: Plot title
            figsize: Figure size
            save_name: Name to save the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        returns = np.array(returns)

        # 1. Buy timing analysis
        buy_indices = [i for i, action in enumerate(actions) if action == 1]
        buy_returns = [returns[i] if i < len(returns) else 0 for i in buy_indices]

        if buy_returns:
            axes[0, 0].hist(buy_returns, bins=20, alpha=0.7, color=self.colors['buy'],
                           edgecolor='black')
            axes[0, 0].axvline(np.mean(buy_returns), color='red', linestyle='--',
                              linewidth=2, label=f'Avg: {np.mean(buy_returns):.4f}')
            axes[0, 0].set_title('Buy Timing Returns')
            axes[0, 0].set_xlabel('Next Period Return')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Sell timing analysis
        sell_indices = [i for i, action in enumerate(actions) if action == 2]
        sell_returns = [returns[i] if i < len(returns) else 0 for i in sell_indices]

        if sell_returns:
            axes[0, 1].hist(sell_returns, bins=20, alpha=0.7, color=self.colors['sell'],
                           edgecolor='black')
            axes[0, 1].axvline(np.mean(sell_returns), color='red', linestyle='--',
                              linewidth=2, label=f'Avg: {np.mean(sell_returns):.4f}')
            axes[0, 1].set_title('Sell Timing Returns')
            axes[0, 1].set_xlabel('Next Period Return')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Action effectiveness over market conditions
        positive_market_days = returns > 0
        negative_market_days = returns < 0

        action_effectiveness = {
            'Bull Market': {
                'Buy': len([i for i in buy_indices if i < len(returns) and positive_market_days[i]]),
                'Sell': len([i for i in sell_indices if i < len(returns) and positive_market_days[i]]),
                'Hold': len([i for i, a in enumerate(actions) if a == 0 and i < len(returns) and positive_market_days[i]])
            },
            'Bear Market': {
                'Buy': len([i for i in buy_indices if i < len(returns) and negative_market_days[i]]),
                'Sell': len([i for i in sell_indices if i < len(returns) and negative_market_days[i]]),
                'Hold': len([i for i, a in enumerate(actions) if a == 0 and i < len(returns) and negative_market_days[i]])
            }
        }

        market_conditions = list(action_effectiveness.keys())
        action_types = list(action_effectiveness['Bull Market'].keys())

        x = np.arange(len(market_conditions))
        width = 0.25

        for i, action_type in enumerate(action_types):
            values = [action_effectiveness[market][action_type] for market in market_conditions]
            color = self.colors['buy'] if action_type == 'Buy' else \
                   self.colors['sell'] if action_type == 'Sell' else self.colors['hold']
            axes[1, 0].bar(x + i*width, values, width, label=action_type, color=color)

        axes[1, 0].set_title('Actions by Market Condition')
        axes[1, 0].set_xlabel('Market Condition')
        axes[1, 0].set_ylabel('Action Count')
        axes[1, 0].set_xticks(x + width)
        axes[1, 0].set_xticklabels(market_conditions)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Market timing effectiveness metrics
        timing_metrics = self._calculate_timing_metrics(actions, returns)

        metrics_text = "\n".join([f"{k}: {v}" for k, v in timing_metrics.items()])
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 1].set_title('Timing Effectiveness')
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_name:
            import os
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Market timing analysis plot saved to {save_path}")

        return fig

    def plot_position_sizing_analysis(self,
                                    actions: List[int],
                                    position_sizes: List[float],
                                    portfolio_values: List[float],
                                    title: str = "Position Sizing Analysis",
                                    figsize: Tuple[int, int] = (15, 6),
                                    save_name: Optional[str] = None) -> plt.Figure:
        """
        Analyze position sizing behavior.

        Args:
            actions: Trading actions
            position_sizes: Position sizes over time
            portfolio_values: Portfolio values
            title: Plot title
            figsize: Figure size
            save_name: Name to save the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 1. Position size distribution
        non_zero_positions = [p for p in position_sizes if p != 0]
        if non_zero_positions:
            axes[0].hist(non_zero_positions, bins=20, alpha=0.7,
                        color=self.colors['volume'], edgecolor='black')
            axes[0].axvline(np.mean(non_zero_positions), color='red',
                           linestyle='--', linewidth=2,
                           label=f'Mean: {np.mean(non_zero_positions):.2f}')
            axes[0].set_title('Position Size Distribution')
            axes[0].set_xlabel('Position Size')
            axes[0].set_ylabel('Frequency')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # 2. Position size over time
        axes[1].plot(range(len(position_sizes)), position_sizes,
                    color=self.colors['volume'], linewidth=2)
        axes[1].set_title('Position Size Evolution')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Position Size')
        axes[1].grid(True, alpha=0.3)

        # 3. Position utilization
        max_position = max(position_sizes) if position_sizes else 1
        utilization = [p / max_position * 100 for p in position_sizes]

        axes[2].plot(range(len(utilization)), utilization,
                    color=self.colors['profit'], linewidth=2)
        axes[2].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Max Utilization')
        axes[2].set_title('Position Utilization (%)')
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Utilization (%)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            import os
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Position sizing analysis plot saved to {save_path}")

        return fig

    def plot_strategy_heatmap(self,
                            state_action_data: Dict[str, List],
                            title: str = "Strategy Heatmap",
                            figsize: Tuple[int, int] = (12, 8),
                            save_name: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap showing action preferences under different market conditions.

        Args:
            state_action_data: Dictionary with state and action information
            title: Plot title
            figsize: Figure size
            save_name: Name to save the plot

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Create action-state frequency matrix
        if 'states' in state_action_data and 'actions' in state_action_data:
            states = state_action_data['states']
            actions = state_action_data['actions']

            # Discretize states for heatmap (simplified example)
            state_bins = 5  # Number of state bins
            action_types = 3  # Hold, Buy, Sell

            # Create state-action frequency matrix
            frequency_matrix = np.zeros((state_bins, action_types))

            for state, action in zip(states, actions):
                if isinstance(state, (list, np.ndarray)):
                    state_val = state[0] if len(state) > 0 else 0  # Use first state feature
                else:
                    state_val = state

                # Discretize state
                state_bin = min(int(state_val * state_bins), state_bins - 1)
                action_bin = min(max(int(action), 0), action_types - 1)

                frequency_matrix[state_bin, action_bin] += 1

            # Normalize to probabilities
            row_sums = frequency_matrix.sum(axis=1, keepdims=True)
            probability_matrix = np.divide(frequency_matrix, row_sums,
                                         out=np.zeros_like(frequency_matrix),
                                         where=row_sums!=0)

            # Plot heatmap
            sns.heatmap(probability_matrix, annot=True, fmt='.2f', cmap='viridis',
                       xticklabels=['Hold', 'Buy', 'Sell'],
                       yticklabels=[f'State {i}' for i in range(state_bins)],
                       ax=ax1)
            ax1.set_title('Action Probability by State')
            ax1.set_xlabel('Action')
            ax1.set_ylabel('Market State')

        # Action transition matrix
        if 'actions' in state_action_data:
            actions = state_action_data['actions']
            transition_matrix = self._calculate_action_transitions(actions)

            sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='plasma',
                       xticklabels=['Hold', 'Buy', 'Sell'],
                       yticklabels=['Hold', 'Buy', 'Sell'],
                       ax=ax2)
            ax2.set_title('Action Transition Probabilities')
            ax2.set_xlabel('Next Action')
            ax2.set_ylabel('Current Action')

        plt.tight_layout()

        if save_name:
            import os
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Strategy heatmap saved to {save_path}")

        return fig

    def create_behavior_dashboard(self,
                                trading_data: Dict[str, List],
                                title: str = "Trading Behavior Dashboard",
                                figsize: Tuple[int, int] = (20, 12),
                                save_name: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive trading behavior dashboard.

        Args:
            trading_data: Complete trading data dictionary
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

        # 1. Action distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'actions' in trading_data:
            actions = trading_data['actions']
            unique_actions, counts = np.unique(actions, return_counts=True)
            colors = [self.colors['hold'], self.colors['buy'], self.colors['sell']][:len(unique_actions)]
            ax1.pie(counts, labels=['Hold', 'Buy', 'Sell'][:len(unique_actions)],
                   colors=colors, autopct='%1.1f%%')
            ax1.set_title('Action Distribution')

        # 2. Trading patterns (top center-right, spans 2 columns)
        ax2 = fig.add_subplot(gs[0, 1:3])
        if 'prices' in trading_data and 'actions' in trading_data:
            prices = trading_data['prices']
            actions = trading_data['actions']

            ax2.plot(range(len(prices)), prices, color=self.colors['price'], linewidth=1)

            buy_points = [i for i, action in enumerate(actions) if action == 1]
            sell_points = [i for i, action in enumerate(actions) if action == 2]

            if buy_points:
                ax2.scatter([i for i in buy_points], [prices[i] for i in buy_points],
                           color=self.colors['buy'], marker='^', s=30, alpha=0.8)

            if sell_points:
                ax2.scatter([i for i in sell_points], [prices[i] for i in sell_points],
                           color=self.colors['sell'], marker='v', s=30, alpha=0.8)

            ax2.set_title('Trading Patterns')
            ax2.set_ylabel('Price')

        # 3. Portfolio evolution (top right)
        ax3 = fig.add_subplot(gs[0, 3])
        if 'portfolio_values' in trading_data:
            portfolio_values = trading_data['portfolio_values']
            ax3.plot(range(len(portfolio_values)), portfolio_values,
                    color=self.colors['profit'], linewidth=2)
            ax3.set_title('Portfolio Value')
            ax3.set_ylabel('Value')

        # 4. Action sequence (middle left, spans 2 columns)
        ax4 = fig.add_subplot(gs[1, 0:2])
        if 'actions' in trading_data:
            actions = trading_data['actions']
            recent_actions = actions[-100:] if len(actions) > 100 else actions

            action_colors = [self.colors['hold'] if a == 0 else
                           self.colors['buy'] if a == 1 else
                           self.colors['sell'] for a in recent_actions]

            ax4.scatter(range(len(recent_actions)), recent_actions,
                       c=action_colors, alpha=0.7, s=20)
            ax4.set_title('Recent Action Sequence')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Action')
            ax4.set_yticks([0, 1, 2])
            ax4.set_yticklabels(['Hold', 'Buy', 'Sell'])

        # 5. Position sizing (middle right, spans 2 columns)
        ax5 = fig.add_subplot(gs[1, 2:])
        if 'position_sizes' in trading_data:
            position_sizes = trading_data['position_sizes']
            ax5.plot(range(len(position_sizes)), position_sizes,
                    color=self.colors['volume'], linewidth=2)
            ax5.fill_between(range(len(position_sizes)), position_sizes,
                           alpha=0.3, color=self.colors['volume'])
            ax5.set_title('Position Sizing')
            ax5.set_xlabel('Time Step')
            ax5.set_ylabel('Position Size')

        # 6. Trading statistics (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        if 'actions' in trading_data:
            actions = trading_data['actions']
            action_counts = np.bincount(actions, minlength=3)
            total_actions = len(actions)

            stats_text = (
                f"Total Actions: {total_actions}\n"
                f"Hold: {action_counts[0]} ({action_counts[0]/total_actions*100:.1f}%)\n"
                f"Buy: {action_counts[1]} ({action_counts[1]/total_actions*100:.1f}%)\n"
                f"Sell: {action_counts[2]} ({action_counts[2]/total_actions*100:.1f}%)\n"
                f"Activity Rate: {(total_actions-action_counts[0])/total_actions*100:.1f}%"
            )

            ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes,
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            ax6.set_title('Trading Statistics')
            ax6.axis('off')

        # 7. Returns distribution (bottom center)
        ax7 = fig.add_subplot(gs[2, 1])
        if 'returns' in trading_data:
            returns = trading_data['returns']
            ax7.hist(returns, bins=20, alpha=0.7, color=self.colors['profit'],
                    edgecolor='black')
            ax7.axvline(np.mean(returns), color='red', linestyle='--',
                       label=f'Mean: {np.mean(returns):.4f}')
            ax7.set_title('Returns Distribution')
            ax7.set_xlabel('Return')
            ax7.legend()

        # 8. Risk metrics (bottom center-right)
        ax8 = fig.add_subplot(gs[2, 2])
        if 'returns' in trading_data:
            returns = trading_data['returns']
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            volatility = np.std(returns) * np.sqrt(252) * 100

            metrics = ['Sharpe', 'Volatility']
            values = [sharpe, volatility]
            colors = [self.colors['profit'], self.colors['loss']]

            bars = ax8.bar(metrics, values, color=colors)
            ax8.set_title('Risk Metrics')
            ax8.set_ylabel('Value')

            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom')

        # 9. Market timing (bottom right)
        ax9 = fig.add_subplot(gs[2, 3])
        if 'actions' in trading_data and 'returns' in trading_data:
            actions = trading_data['actions']
            returns = trading_data['returns']

            timing_metrics = self._calculate_timing_metrics(actions, returns)

            metrics_text = "\n".join([f"{k}: {v}" for k, v in timing_metrics.items()])
            ax9.text(0.1, 0.5, metrics_text, transform=ax9.transAxes,
                    fontsize=9, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            ax9.set_title('Market Timing')
            ax9.axis('off')

        if save_name:
            import os
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trading behavior dashboard saved to {save_path}")

        return fig

    # Helper methods
    def _calculate_timing_metrics(self, actions: List[int], returns: List[float]) -> Dict[str, str]:
        """Calculate market timing effectiveness metrics."""
        buy_indices = [i for i, action in enumerate(actions) if action == 1]
        sell_indices = [i for i, action in enumerate(actions) if action == 2]

        # Buy timing effectiveness
        buy_returns = [returns[i] if i < len(returns) else 0 for i in buy_indices]
        avg_buy_return = np.mean(buy_returns) if buy_returns else 0

        # Sell timing effectiveness
        sell_returns = [returns[i] if i < len(returns) else 0 for i in sell_indices]
        avg_sell_return = np.mean(sell_returns) if sell_returns else 0

        # Overall market timing score
        market_return = np.mean(returns)
        buy_timing_score = avg_buy_return - market_return
        sell_timing_score = -(avg_sell_return - market_return)  # Negative because we want low returns when selling

        return {
            'Buy Timing': f"{buy_timing_score:.4f}",
            'Sell Timing': f"{sell_timing_score:.4f}",
            'Avg Buy Return': f"{avg_buy_return:.4f}",
            'Avg Sell Return': f"{avg_sell_return:.4f}",
            'Market Return': f"{market_return:.4f}"
        }

    def _calculate_action_transitions(self, actions: List[int]) -> np.ndarray:
        """Calculate action transition probability matrix."""
        n_actions = 3  # Hold, Buy, Sell
        transition_matrix = np.zeros((n_actions, n_actions))

        for i in range(len(actions) - 1):
            current_action = min(max(actions[i], 0), n_actions - 1)
            next_action = min(max(actions[i + 1], 0), n_actions - 1)
            transition_matrix[current_action, next_action] += 1

        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums,
                                    out=np.zeros_like(transition_matrix),
                                    where=row_sums!=0)

        return transition_matrix


# Utility functions for quick plotting
def quick_plot_actions(actions: List[int],
                      action_names: List[str] = None,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Quick function to plot action distribution.

    Args:
        actions: List of actions
        action_names: Optional action names
        save_path: Path to save the plot

    Returns:
        Matplotlib figure
    """
    visualizer = TradingBehaviorVisualizer()

    fig = visualizer.plot_action_distribution(
        actions, action_names=action_names,
        save_name=save_path.split('/')[-1].split('.')[0] if save_path else None
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def quick_plot_trading_patterns(prices: List[float],
                              actions: List[int],
                              portfolio_values: List[float],
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Quick function to plot trading patterns.

    Args:
        prices: Asset prices
        actions: Trading actions
        portfolio_values: Portfolio values
        save_path: Path to save the plot

    Returns:
        Matplotlib figure
    """
    visualizer = TradingBehaviorVisualizer()

    fig = visualizer.plot_trading_patterns(
        prices, actions, portfolio_values,
        save_name=save_path.split('/')[-1].split('.')[0] if save_path else None
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
