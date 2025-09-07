"""
Performance Analysis Visualization for RL Trading System.

This module provides comprehensive performance analysis tools for evaluating
trading strategies, portfolio metrics, and risk analysis.
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


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis system for trading strategies.

    Provides methods to analyze trading performance, risk metrics,
    portfolio evolution, and comparative analysis.
    """

    def __init__(self, save_dir: str = "performance_plots"):
        """
        Initialize the performance analyzer.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Color scheme for consistency
        self.colors = {
            'profit': '#27AE60',
            'loss': '#E74C3C',
            'equity': '#3498DB',
            'drawdown': '#8E44AD',
            'sharpe': '#F39C12',
            'volume': '#17A2B8',
            'benchmark': '#95A5A6'
        }

        logger.info(f"PerformanceAnalyzer initialized - Save dir: {save_dir}")

    def plot_equity_curve(self,
                         portfolio_values: List[float],
                         dates: Optional[List] = None,
                         benchmark: Optional[List[float]] = None,
                         title: str = "Equity Curve",
                         figsize: Tuple[int, int] = (12, 8),
                         save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot equity curve with optional benchmark comparison.

        Args:
            portfolio_values: Portfolio values over time
            dates: Optional dates for x-axis
            benchmark: Optional benchmark values for comparison
            title: Plot title
            figsize: Figure size
            save_name: Name to save the plot

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                      gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Prepare x-axis
        if dates is None:
            x_axis = range(len(portfolio_values))
            x_label = 'Time Period'
        else:
            x_axis = dates
            x_label = 'Date'

        # Main equity curve
        ax1.plot(x_axis, portfolio_values, color=self.colors['equity'],
                linewidth=2, label='Strategy')

        if benchmark is not None:
            ax1.plot(x_axis, benchmark, color=self.colors['benchmark'],
                    linewidth=2, alpha=0.7, linestyle='--', label='Benchmark')

        ax1.set_title('Portfolio Value Evolution')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Calculate and plot drawdown
        drawdown = self._calculate_drawdown(portfolio_values)
        ax2.fill_between(x_axis, drawdown, 0, color=self.colors['drawdown'],
                        alpha=0.7, label='Drawdown')
        ax2.set_title('Drawdown')
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            import os
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Equity curve plot saved to {save_path}")

        return fig

    def plot_returns_analysis(self,
                            returns: List[float],
                            benchmark_returns: Optional[List[float]] = None,
                            title: str = "Returns Analysis",
                            figsize: Tuple[int, int] = (15, 10),
                            save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive returns analysis.

        Args:
            returns: Strategy returns
            benchmark_returns: Optional benchmark returns
            title: Plot title
            figsize: Figure size
            save_name: Name to save the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        returns = np.array(returns)

        # 1. Returns distribution
        axes[0, 0].hist(returns, bins=50, alpha=0.7, color=self.colors['equity'],
                       edgecolor='black', density=True)
        axes[0, 0].axvline(np.mean(returns), color=self.colors['profit'],
                          linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.4f}')
        axes[0, 0].axvline(np.median(returns), color=self.colors['loss'],
                          linestyle='--', linewidth=2, label=f'Median: {np.median(returns):.4f}')
        axes[0, 0].set_title('Returns Distribution')
        axes[0, 0].set_xlabel('Returns')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Rolling Sharpe ratio
        rolling_sharpe = self._rolling_sharpe(returns, window=30)
        axes[0, 1].plot(range(len(rolling_sharpe)), rolling_sharpe,
                       color=self.colors['sharpe'], linewidth=2)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Rolling Sharpe Ratio (30-period)')
        axes[0, 1].set_xlabel('Time Period')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Cumulative returns
        cumulative_returns = np.cumprod(1 + returns) - 1
        axes[0, 2].plot(range(len(cumulative_returns)), cumulative_returns * 100,
                       color=self.colors['equity'], linewidth=2)
        if benchmark_returns is not None:
            benchmark_cumulative = np.cumprod(1 + np.array(benchmark_returns)) - 1
            axes[0, 2].plot(range(len(benchmark_cumulative)), benchmark_cumulative * 100,
                           color=self.colors['benchmark'], linewidth=2, alpha=0.7,
                           linestyle='--', label='Benchmark')
            axes[0, 2].legend()
        axes[0, 2].set_title('Cumulative Returns')
        axes[0, 2].set_xlabel('Time Period')
        axes[0, 2].set_ylabel('Cumulative Return (%)')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Monthly returns heatmap
        monthly_returns = self._calculate_monthly_returns(returns)
        if len(monthly_returns) > 0:
            sns.heatmap(monthly_returns, annot=True, fmt='.2%', cmap='RdYlGn',
                       center=0, ax=axes[1, 0])
            axes[1, 0].set_title('Monthly Returns Heatmap')

        # 5. Risk-Return scatter
        if benchmark_returns is not None:
            strategy_risk = np.std(returns) * np.sqrt(252)  # Annualized
            strategy_return = np.mean(returns) * 252  # Annualized
            benchmark_risk = np.std(benchmark_returns) * np.sqrt(252)
            benchmark_return = np.mean(benchmark_returns) * 252

            axes[1, 1].scatter(strategy_risk, strategy_return, s=100,
                             color=self.colors['equity'], label='Strategy')
            axes[1, 1].scatter(benchmark_risk, benchmark_return, s=100,
                             color=self.colors['benchmark'], label='Benchmark')
            axes[1, 1].set_title('Risk-Return Profile')
            axes[1, 1].set_xlabel('Annualized Risk (Std Dev)')
            axes[1, 1].set_ylabel('Annualized Return')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Performance statistics
        stats = self._calculate_performance_stats(returns, benchmark_returns)
        stats_text = ""
        for key, value in stats.items():
            if isinstance(value, float):
                stats_text += f"{key}: {value:.4f}\n"
            else:
                stats_text += f"{key}: {value}\n"

        axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        axes[1, 2].set_title('Performance Statistics')
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_name:
            import os
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Returns analysis plot saved to {save_path}")

        return fig

    def plot_risk_metrics(self,
                         portfolio_values: List[float],
                         returns: List[float],
                         title: str = "Risk Analysis",
                         figsize: Tuple[int, int] = (15, 8),
                         save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive risk analysis.

        Args:
            portfolio_values: Portfolio values over time
            returns: Strategy returns
            title: Plot title
            figsize: Figure size
            save_name: Name to save the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        returns = np.array(returns)

        # 1. Drawdown analysis
        drawdown = self._calculate_drawdown(portfolio_values)
        axes[0, 0].fill_between(range(len(drawdown)), drawdown, 0,
                               color=self.colors['drawdown'], alpha=0.7)
        axes[0, 0].set_title(f'Drawdown (Max: {np.min(drawdown):.2f}%)')
        axes[0, 0].set_xlabel('Time Period')
        axes[0, 0].set_ylabel('Drawdown (%)')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        axes[0, 1].hist(returns, bins=50, alpha=0.7, color=self.colors['equity'],
                       density=True, edgecolor='black')
        axes[0, 1].axvline(var_95, color=self.colors['loss'], linewidth=2,
                          label=f'VaR 95%: {var_95:.4f}')
        axes[0, 1].axvline(var_99, color='darkred', linewidth=2,
                          label=f'VaR 99%: {var_99:.4f}')
        axes[0, 1].set_title('Value at Risk')
        axes[0, 1].set_xlabel('Returns')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Rolling volatility
        rolling_vol = self._rolling_volatility(returns, window=30)
        axes[0, 2].plot(range(len(rolling_vol)), rolling_vol * 100,
                       color=self.colors['sharpe'], linewidth=2)
        axes[0, 2].set_title('Rolling Volatility (30-period)')
        axes[0, 2].set_xlabel('Time Period')
        axes[0, 2].set_ylabel('Volatility (%)')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Underwater plot
        underwater = self._underwater_plot(portfolio_values)
        axes[1, 0].fill_between(range(len(underwater)), underwater, 0,
                               color=self.colors['drawdown'], alpha=0.7)
        axes[1, 0].set_title('Underwater Plot')
        axes[1, 0].set_xlabel('Time Period')
        axes[1, 0].set_ylabel('Underwater (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Risk-adjusted returns
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns, portfolio_values)

        risk_metrics = ['Sharpe', 'Sortino', 'Calmar']
        risk_values = [sharpe_ratio, sortino_ratio, calmar_ratio]

        bars = axes[1, 1].bar(risk_metrics, risk_values,
                             color=[self.colors['sharpe'], self.colors['profit'], self.colors['equity']])
        axes[1, 1].set_title('Risk-Adjusted Returns')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, risk_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')

        # 6. Risk statistics summary
        risk_stats = {
            'Max Drawdown': f"{np.min(drawdown):.2f}%",
            'VaR (95%)': f"{var_95:.4f}",
            'VaR (99%)': f"{var_99:.4f}",
            'Volatility': f"{np.std(returns) * np.sqrt(252):.2f}%",
            'Sharpe Ratio': f"{sharpe_ratio:.3f}",
            'Sortino Ratio': f"{sortino_ratio:.3f}",
            'Calmar Ratio': f"{calmar_ratio:.3f}"
        }

        stats_text = "\n".join([f"{k}: {v}" for k, v in risk_stats.items()])
        axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.3))
        axes[1, 2].set_title('Risk Statistics')
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_name:
            import os
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Risk analysis plot saved to {save_path}")

        return fig

    def plot_trade_analysis(self,
                          trade_data: Dict[str, List],
                          title: str = "Trade Analysis",
                          figsize: Tuple[int, int] = (15, 10),
                          save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive trade analysis.

        Args:
            trade_data: Dictionary containing trade information
            title: Plot title
            figsize: Figure size
            save_name: Name to save the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 1. P&L distribution
        if 'pnl' in trade_data:
            pnl = trade_data['pnl']
            profit_trades = [p for p in pnl if p > 0]
            loss_trades = [p for p in pnl if p < 0]

            axes[0, 0].hist(pnl, bins=30, alpha=0.7, color=self.colors['equity'],
                           edgecolor='black')
            axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[0, 0].set_title(f'P&L Distribution\nWin Rate: {len(profit_trades)/len(pnl)*100:.1f}%')
            axes[0, 0].set_xlabel('P&L')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Trade duration analysis
        if 'duration' in trade_data:
            duration = trade_data['duration']
            axes[0, 1].hist(duration, bins=20, alpha=0.7, color=self.colors['volume'],
                           edgecolor='black')
            axes[0, 1].set_title(f'Trade Duration\nAvg: {np.mean(duration):.1f} periods')
            axes[0, 1].set_xlabel('Duration (periods)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Cumulative P&L
        if 'pnl' in trade_data:
            cumulative_pnl = np.cumsum(trade_data['pnl'])
            axes[0, 2].plot(range(len(cumulative_pnl)), cumulative_pnl,
                           color=self.colors['profit'], linewidth=2)
            axes[0, 2].set_title('Cumulative P&L')
            axes[0, 2].set_xlabel('Trade Number')
            axes[0, 2].set_ylabel('Cumulative P&L')
            axes[0, 2].grid(True, alpha=0.3)

        # 4. Win/Loss streaks
        if 'pnl' in trade_data:
            streaks = self._calculate_streaks(trade_data['pnl'])
            win_streaks = [s for s in streaks if s > 0]
            loss_streaks = [abs(s) for s in streaks if s < 0]

            streak_data = [np.mean(win_streaks) if win_streaks else 0,
                          np.mean(loss_streaks) if loss_streaks else 0]
            colors = [self.colors['profit'], self.colors['loss']]

            bars = axes[1, 0].bar(['Avg Win Streak', 'Avg Loss Streak'], streak_data, color=colors)
            axes[1, 0].set_title('Average Streaks')
            axes[1, 0].set_ylabel('Streak Length')

            for bar, value in zip(bars, streak_data):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.1f}', ha='center', va='bottom')

        # 5. Monthly P&L
        if 'pnl' in trade_data and 'dates' in trade_data:
            monthly_pnl = self._calculate_monthly_pnl(trade_data['pnl'], trade_data['dates'])
            months = list(monthly_pnl.keys())
            values = list(monthly_pnl.values())

            colors = [self.colors['profit'] if v > 0 else self.colors['loss'] for v in values]
            axes[1, 1].bar(range(len(months)), values, color=colors)
            axes[1, 1].set_title('Monthly P&L')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('P&L')
            axes[1, 1].set_xticks(range(len(months)))
            axes[1, 1].set_xticklabels([m[:7] for m in months], rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Trade statistics
        if 'pnl' in trade_data:
            pnl = trade_data['pnl']
            profit_trades = [p for p in pnl if p > 0]
            loss_trades = [p for p in pnl if p < 0]

            stats = {
                'Total Trades': len(pnl),
                'Win Rate': f"{len(profit_trades)/len(pnl)*100:.1f}%",
                'Avg Profit': f"{np.mean(profit_trades):.4f}" if profit_trades else "0",
                'Avg Loss': f"{np.mean(loss_trades):.4f}" if loss_trades else "0",
                'Profit Factor': f"{sum(profit_trades)/abs(sum(loss_trades)):.2f}" if loss_trades else "∞",
                'Total P&L': f"{sum(pnl):.4f}",
                'Best Trade': f"{max(pnl):.4f}",
                'Worst Trade': f"{min(pnl):.4f}"
            }

            stats_text = "\n".join([f"{k}: {v}" for k, v in stats.items()])
            axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                           fontsize=10, verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3))
            axes[1, 2].set_title('Trade Statistics')
            axes[1, 2].axis('off')

        plt.tight_layout()

        if save_name:
            import os
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trade analysis plot saved to {save_path}")

        return fig

    # Helper methods for calculations
    def _calculate_drawdown(self, portfolio_values: List[float]) -> np.ndarray:
        """Calculate drawdown from portfolio values."""
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak * 100
        return drawdown

    def _underwater_plot(self, portfolio_values: List[float]) -> np.ndarray:
        """Calculate underwater plot (time underwater from peaks)."""
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        underwater = np.where(values == peak, 0, (values - peak) / peak * 100)
        return underwater

    def _rolling_sharpe(self, returns: np.ndarray, window: int = 30) -> np.ndarray:
        """Calculate rolling Sharpe ratio."""
        rolling_mean = np.convolve(returns, np.ones(window)/window, mode='valid')
        rolling_std = np.array([np.std(returns[i:i+window]) for i in range(len(returns)-window+1)])
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
        return rolling_sharpe

    def _rolling_volatility(self, returns: np.ndarray, window: int = 30) -> np.ndarray:
        """Calculate rolling volatility."""
        rolling_vol = np.array([np.std(returns[i:i+window]) for i in range(len(returns)-window+1)])
        return rolling_vol * np.sqrt(252)

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate/252
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)

    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - risk_free_rate/252
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
        return np.mean(excess_returns) / downside_std * np.sqrt(252)

    def _calculate_calmar_ratio(self, returns: np.ndarray, portfolio_values: List[float]) -> float:
        """Calculate Calmar ratio."""
        annual_return = np.mean(returns) * 252
        max_drawdown = abs(np.min(self._calculate_drawdown(portfolio_values))) / 100
        return annual_return / max_drawdown if max_drawdown > 0 else 0

    def _calculate_monthly_returns(self, returns: np.ndarray) -> np.ndarray:
        """Calculate monthly returns matrix for heatmap."""
        # This is a simplified version - in practice you'd need actual dates
        monthly_length = 21  # Approximate trading days per month
        months = len(returns) // monthly_length

        if months < 2:
            return np.array([])

        monthly_returns = []
        for i in range(months):
            start_idx = i * monthly_length
            end_idx = min((i + 1) * monthly_length, len(returns))
            monthly_ret = np.prod(1 + returns[start_idx:end_idx]) - 1
            monthly_returns.append(monthly_ret)

        # Reshape for heatmap (this is simplified)
        years = max(1, len(monthly_returns) // 12)
        months_per_year = len(monthly_returns) // years

        return np.array(monthly_returns[:years * months_per_year]).reshape(years, months_per_year)

    def _calculate_performance_stats(self, returns: np.ndarray,
                                   benchmark_returns: Optional[np.ndarray] = None) -> Dict:
        """Calculate comprehensive performance statistics."""
        stats = {
            'Total Return': f"{(np.prod(1 + returns) - 1) * 100:.2f}%",
            'Annual Return': f"{np.mean(returns) * 252 * 100:.2f}%",
            'Volatility': f"{np.std(returns) * np.sqrt(252) * 100:.2f}%",
            'Sharpe Ratio': f"{self._calculate_sharpe_ratio(returns):.3f}",
            'Sortino Ratio': f"{self._calculate_sortino_ratio(returns):.3f}",
            'Skewness': f"{self._calculate_skewness(returns):.3f}",
            'Kurtosis': f"{self._calculate_kurtosis(returns):.3f}"
        }

        if benchmark_returns is not None:
            beta = self._calculate_beta(returns, benchmark_returns)
            alpha = self._calculate_alpha(returns, benchmark_returns, beta)
            stats['Beta'] = f"{beta:.3f}"
            stats['Alpha'] = f"{alpha * 252 * 100:.2f}%"

        return stats

    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness."""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        return np.mean(((returns - mean_return) / std_return) ** 3)

    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis."""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        return np.mean(((returns - mean_return) / std_return) ** 4) - 3

    def _calculate_beta(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate beta."""
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        return covariance / benchmark_variance if benchmark_variance > 0 else 0

    def _calculate_alpha(self, returns: np.ndarray, benchmark_returns: np.ndarray,
                        beta: float, risk_free_rate: float = 0.02) -> float:
        """Calculate alpha."""
        return np.mean(returns) - (risk_free_rate/252 + beta * (np.mean(benchmark_returns) - risk_free_rate/252))

    def _calculate_streaks(self, pnl: List[float]) -> List[int]:
        """Calculate win/loss streaks."""
        streaks = []
        current_streak = 0

        for trade in pnl:
            if trade > 0:  # Win
                if current_streak > 0:
                    current_streak += 1
                else:
                    if current_streak < 0:
                        streaks.append(current_streak)
                    current_streak = 1
            else:  # Loss
                if current_streak < 0:
                    current_streak -= 1
                else:
                    if current_streak > 0:
                        streaks.append(current_streak)
                    current_streak = -1

        if current_streak != 0:
            streaks.append(current_streak)

        return streaks

    def _calculate_monthly_pnl(self, pnl: List[float], dates: List) -> Dict[str, float]:
        """Calculate monthly P&L (simplified version)."""
        # This is a simplified implementation
        # In practice, you'd use actual date parsing
        monthly_pnl = {}
        current_month = "2023-01"  # Placeholder
        current_sum = 0

        for i, p in enumerate(pnl):
            current_sum += p
            # Simulate month change every 21 trades (approximate)
            if (i + 1) % 21 == 0:
                monthly_pnl[current_month] = current_sum
                current_sum = 0
                # Increment month (simplified)
                month_num = int(current_month.split('-')[1])
                year = int(current_month.split('-')[0])
                month_num += 1
                if month_num > 12:
                    month_num = 1
                    year += 1
                current_month = f"{year}-{month_num:02d}"

        if current_sum != 0:
            monthly_pnl[current_month] = current_sum

        return monthly_pnl
