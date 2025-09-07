"""
Multi-Asset Portfolio Management Example.

This example demonstrates professional portfolio management capabilities
including:
- Multiple asset classes (stocks, bonds, commodities)
- Dynamic rebalancing strategies
- Risk constraints and portfolio optimization
- Transaction costs and market impact
"""

import logging
import os
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import RL system components
from model.rl_system.integration import create_agent, setup_training_environment

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAssetPortfolioEnvironment:
    """
    Professional multi-asset portfolio management environment.

    This environment simulates realistic portfolio management with:
    - Multiple asset classes with different risk/return profiles
    - Transaction costs and market impact
    - Portfolio constraints (position limits, sector allocation)
    - Risk management (VaR limits, drawdown constraints)
    """

    def __init__(self, n_assets: int = 5, n_steps: int = 252, initial_value: float = 1000000):
        """
        Initialize the multi-asset portfolio environment.

        Args:
            n_assets: Number of assets in the universe
            n_steps: Number of trading days (252 = 1 year)
            initial_value: Initial portfolio value
        """
        self.n_assets = n_assets
        self.n_steps = n_steps
        self.initial_value = initial_value

        # Asset characteristics
        self.asset_names = ['Large Cap Stocks', 'Small Cap Stocks', 'Government Bonds',
                           'Corporate Bonds', 'Commodities'][:n_assets]

        # Expected returns (annualized)
        self.expected_returns = np.array([0.08, 0.12, 0.03, 0.05, 0.06])[:n_assets]

        # Volatilities (annualized)
        self.volatilities = np.array([0.15, 0.25, 0.05, 0.08, 0.20])[:n_assets]

        # Asset correlations
        self.correlation_matrix = self._create_correlation_matrix()

        self.reset()

        # Action space: portfolio weights for each asset
        self.action_dim = n_assets

        # State space: [prices, returns, volatilities, weights, portfolio_metrics]
        self.state_dim = n_assets * 4 + 5  # 4 features per asset + 5 portfolio metrics

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = 0
        self.portfolio_value = self.initial_value
        self.cash = self.initial_value * 0.1  # 10% cash buffer

        # Initialize asset prices (normalized to 100)
        self.prices = np.ones(self.n_assets) * 100.0
        self.initial_prices = self.prices.copy()

        # Initialize equal-weight portfolio
        self.portfolio_weights = np.ones(self.n_assets) / self.n_assets
        self.position_values = self.portfolio_value * self.portfolio_weights

        # History tracking
        self.price_history = [self.prices.copy()]
        self.return_history = []
        self.portfolio_history = [self.portfolio_value]
        self.weight_history = [self.portfolio_weights.copy()]
        self.transaction_costs = []

        # Risk tracking
        self.returns_window = []
        self.max_portfolio_value = self.portfolio_value
        self.current_drawdown = 0.0

        return self._get_state()

    def step(self, action: np.ndarray) -> tuple:
        """Execute one step in the environment."""
        if self.current_step >= self.n_steps:
            return self._get_state(), 0.0, True, {}

        # Normalize action to valid portfolio weights
        target_weights = self._normalize_weights(action)

        # Generate new asset prices
        returns = self._generate_asset_returns()
        self.prices *= (1 + returns)

        # Calculate portfolio value before rebalancing
        self.position_values *= (1 + returns)
        portfolio_value_before = np.sum(self.position_values) + self.cash

        # Rebalance portfolio to target weights
        transaction_cost = self._rebalance_portfolio(target_weights, portfolio_value_before)

        # Update portfolio value after rebalancing
        self.portfolio_value = np.sum(self.position_values) + self.cash

        # Calculate reward
        reward = self._calculate_reward(portfolio_value_before, returns, transaction_cost)

        # Update risk metrics
        self._update_risk_metrics(returns)

        # Store history
        self.price_history.append(self.prices.copy())
        self.return_history.append(returns)
        self.portfolio_history.append(self.portfolio_value)
        self.weight_history.append(self.portfolio_weights.copy())
        self.transaction_costs.append(transaction_cost)

        self.current_step += 1
        done = self.current_step >= self.n_steps

        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_weights': self.portfolio_weights.copy(),
            'asset_returns': returns,
            'transaction_cost': transaction_cost,
            'drawdown': self.current_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'volatility': self._calculate_portfolio_volatility()
        }

        return self._get_state(), reward, done, info

    def _create_correlation_matrix(self) -> np.ndarray:
        """Create realistic correlation matrix for assets."""
        correlations = np.eye(self.n_assets)

        if self.n_assets >= 5:
            # Realistic correlations between asset classes
            correlations[0, 1] = 0.8   # Large cap - Small cap
            correlations[1, 0] = 0.8
            correlations[2, 3] = 0.6   # Gov bonds - Corp bonds
            correlations[3, 2] = 0.6
            correlations[0, 4] = -0.2  # Stocks - Commodities
            correlations[4, 0] = -0.2
            correlations[1, 4] = -0.1
            correlations[4, 1] = -0.1
            correlations[0, 2] = -0.3  # Stocks - Bonds
            correlations[2, 0] = -0.3
            correlations[1, 2] = -0.2
            correlations[2, 1] = -0.2

        return correlations

    def _generate_asset_returns(self) -> np.ndarray:
        """Generate correlated asset returns."""
        # Convert annual to daily
        daily_returns = self.expected_returns / 252
        daily_vols = self.volatilities / np.sqrt(252)

        # Generate independent random returns
        independent_returns = np.random.normal(daily_returns, daily_vols)

        # Apply correlation structure
        L = np.linalg.cholesky(self.correlation_matrix)
        correlated_returns = L @ independent_returns

        return correlated_returns

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights with constraints."""
        # Apply absolute value and normalize
        weights = np.abs(weights)
        weights = weights / (np.sum(weights) + 1e-8)

        # Apply position limits (max 40% in any single asset)
        weights = np.minimum(weights, 0.4)

        # Renormalize after constraints
        weights = weights / (np.sum(weights) + 1e-8)

        return weights

    def _rebalance_portfolio(self, target_weights: np.ndarray,
                           current_portfolio_value: float) -> float:
        """Rebalance portfolio to target weights and calculate transaction costs."""

        # Calculate target position values
        target_position_values = current_portfolio_value * target_weights

        # Calculate required trades
        trades = target_position_values - self.position_values

        # Calculate transaction costs
        transaction_cost = np.sum(np.abs(trades)) * 0.001  # 0.1% transaction cost

        # Apply market impact (higher for larger trades)
        market_impact_factor = np.sum(np.abs(trades)) / current_portfolio_value
        market_impact = current_portfolio_value * market_impact_factor * 0.0005

        total_transaction_cost = transaction_cost + market_impact

        # Execute trades
        self.position_values = target_position_values
        self.portfolio_weights = target_weights
        self.cash -= total_transaction_cost

        return total_transaction_cost

    def _calculate_reward(self, prev_portfolio_value: float,
                         returns: np.ndarray, transaction_cost: float) -> float:
        """Calculate reward based on risk-adjusted performance."""

        # Base reward: portfolio return
        portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        reward = portfolio_return * 1000  # Scale for learning

        # Penalty for transaction costs
        cost_penalty = transaction_cost / prev_portfolio_value * 100
        reward -= cost_penalty

        # Risk-adjusted reward using Sharpe ratio
        if len(self.return_history) > 20:
            sharpe_ratio = self._calculate_sharpe_ratio()
            reward += sharpe_ratio * 5  # Bonus for good risk-adjusted performance

        # Drawdown penalty
        if self.current_drawdown > 0.1:  # More than 10% drawdown
            reward -= (self.current_drawdown - 0.1) * 50

        # Diversification bonus
        concentration = np.sum(self.portfolio_weights ** 2)
        diversification_bonus = (1/self.n_assets - concentration) * 10
        reward += diversification_bonus

        return reward

    def _update_risk_metrics(self, returns: np.ndarray):
        """Update risk tracking metrics."""
        # Update returns window (keep last 60 days)
        portfolio_return = np.sum(self.portfolio_weights * returns)
        self.returns_window.append(portfolio_return)
        if len(self.returns_window) > 60:
            self.returns_window.pop(0)

        # Update maximum portfolio value
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)

        # Calculate current drawdown
        self.current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(self.returns_window) < 30:
            return 0.0

        returns_array = np.array(self.returns_window)
        mean_return = np.mean(returns_array) * 252  # Annualize
        std_return = np.std(returns_array) * np.sqrt(252)  # Annualize

        risk_free_rate = 0.02  # 2% risk-free rate

        if std_return == 0:
            return 0.0

        return (mean_return - risk_free_rate) / std_return

    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility."""
        if len(self.returns_window) < 20:
            return 0.0

        returns_array = np.array(self.returns_window)
        return np.std(returns_array) * np.sqrt(252)  # Annualized

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        state_components = []

        # Normalized asset prices
        normalized_prices = self.prices / self.initial_prices
        state_components.extend(normalized_prices)

        # Recent returns
        if len(self.return_history) > 0:
            recent_returns = self.return_history[-1]
        else:
            recent_returns = np.zeros(self.n_assets)
        state_components.extend(recent_returns)

        # Historical volatilities
        if len(self.return_history) > 10:
            recent_returns_matrix = np.array(self.return_history[-10:])
            volatilities = np.std(recent_returns_matrix, axis=0)
        else:
            volatilities = self.volatilities / np.sqrt(252)  # Daily volatilities
        state_components.extend(volatilities)

        # Current portfolio weights
        state_components.extend(self.portfolio_weights)

        # Portfolio-level metrics
        portfolio_return_ytd = (self.portfolio_value - self.initial_value) / self.initial_value
        sharpe_ratio = self._calculate_sharpe_ratio()
        portfolio_vol = self._calculate_portfolio_volatility()
        cash_ratio = self.cash / self.portfolio_value
        time_progress = self.current_step / self.n_steps

        state_components.extend([portfolio_return_ytd, sharpe_ratio, portfolio_vol,
                               cash_ratio, time_progress])

        return np.array(state_components, dtype=np.float32)


def create_portfolio_management_example():
    """Create and train a portfolio management agent."""

    print("=== Multi-Asset Portfolio Management Example ===")
    print("This example demonstrates professional portfolio management with RL.")
    print()

    # Create environment
    env = MultiAssetPortfolioEnvironment(n_assets=5, n_steps=252, initial_value=1000000)
    print("Portfolio Environment:")
    print(f"  Assets: {len(env.asset_names)}")
    for i, name in enumerate(env.asset_names):
        print(f"    {i+1}. {name}: E[R]={env.expected_returns[i]:.1%}, Vol={env.volatilities[i]:.1%}")
    print(f"  Initial Value: ${env.initial_value:,.0f}")
    print(f"  Trading Period: {env.n_steps} days (1 year)")
    print()

    # Setup training environment
    setup_training_environment(
        env_type="continuous",
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        complexity="complex"
    )

    # Create PPO agent (good for continuous portfolio allocation)
    agent = create_agent(
        algorithm_type="ppo",
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        config={
            'learning_rate': 0.0003,
            'batch_size': 64,
            'n_epochs': 10,
            'clip_ratio': 0.2,
            'gae_lambda': 0.95,
            'entropy_coef': 0.01
        }
    )

    print("Created PPO agent for portfolio management")
    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    print()

    # Training
    max_episodes = 100
    print(f"Training for {max_episodes} episodes...")
    print()

    # Training metrics
    episode_returns = []
    sharpe_ratios = []
    max_drawdowns = []
    transaction_costs = []
    portfolio_volatilities = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_transactions = 0

        for step in range(env.n_steps):
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            episode_transactions += info['transaction_cost']

            state = next_state
            if done:
                break

        # Update agent
        if hasattr(agent.agent, 'memory') and len(agent.agent.memory) > 0:
            agent.update()

        # Record episode metrics
        final_return = (env.portfolio_value - env.initial_value) / env.initial_value
        episode_returns.append(final_return)
        sharpe_ratios.append(info['sharpe_ratio'])
        max_drawdowns.append(max(env.current_drawdown, 0))
        transaction_costs.append(episode_transactions / env.initial_value)
        portfolio_volatilities.append(info['volatility'])

        # Progress reporting
        if episode % 20 == 0:
            avg_return = np.mean(episode_returns[-20:]) * 100
            avg_sharpe = np.mean(sharpe_ratios[-20:])
            avg_drawdown = np.mean(max_drawdowns[-20:]) * 100
            print(f"Episode {episode:3d} | Return: {avg_return:6.2f}% | "
                  f"Sharpe: {avg_sharpe:5.2f} | Max DD: {avg_drawdown:5.2f}%")

    print()
    print("Training completed!")

    # Final evaluation
    print("=== Portfolio Performance Summary ===")
    final_avg_return = np.mean(episode_returns[-20:]) * 100
    final_avg_sharpe = np.mean(sharpe_ratios[-20:])
    final_avg_drawdown = np.mean(max_drawdowns[-20:]) * 100
    final_avg_vol = np.mean(portfolio_volatilities[-20:]) * 100
    final_avg_costs = np.mean(transaction_costs[-20:]) * 100

    print(f"Average Annual Return:     {final_avg_return:6.2f}%")
    print(f"Average Sharpe Ratio:      {final_avg_sharpe:6.2f}")
    print(f"Average Max Drawdown:      {final_avg_drawdown:6.2f}%")
    print(f"Average Volatility:        {final_avg_vol:6.2f}%")
    print(f"Average Transaction Costs: {final_avg_costs:6.2f}%")
    print()

    # Create comprehensive visualization
    create_portfolio_plots(episode_returns, sharpe_ratios, max_drawdowns,
                          env.weight_history, env.portfolio_history, env.asset_names)

    return agent, {
        'returns': episode_returns,
        'sharpe_ratios': sharpe_ratios,
        'drawdowns': max_drawdowns,
        'weights': env.weight_history,
        'portfolio_values': env.portfolio_history
    }


def create_portfolio_plots(returns: List[float], sharpe_ratios: List[float],
                          drawdowns: List[float], weight_history: List[np.ndarray],
                          portfolio_history: List[float], asset_names: List[str]):
    """Create comprehensive portfolio management visualization."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multi-Asset Portfolio Management Results', fontsize=16)

    episodes = range(len(returns))

    # Plot 1: Portfolio Returns
    returns_pct = np.array(returns) * 100
    axes[0, 0].plot(episodes, returns_pct)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_title('Portfolio Returns by Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return (%)')
    axes[0, 0].grid(True, alpha=0.3)

    # Add moving average
    if len(returns) > 20:
        ma_returns = np.convolve(returns_pct, np.ones(20)/20, mode='valid')
        axes[0, 0].plot(range(19, len(episodes)), ma_returns, color='red', label='MA(20)')
        axes[0, 0].legend()

    # Plot 2: Sharpe Ratios
    axes[0, 1].plot(episodes, sharpe_ratios)
    axes[0, 1].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Good (>1.0)')
    axes[0, 1].axhline(y=2.0, color='blue', linestyle='--', alpha=0.7, label='Excellent (>2.0)')
    axes[0, 1].set_title('Sharpe Ratio Evolution')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Maximum Drawdowns
    drawdowns_pct = np.array(drawdowns) * 100
    axes[0, 2].plot(episodes, drawdowns_pct, color='red')
    axes[0, 2].axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Warning (10%)')
    axes[0, 2].axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Danger (20%)')
    axes[0, 2].set_title('Maximum Drawdown')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Drawdown (%)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Portfolio Weight Evolution (last episode)
    if weight_history:
        weights_array = np.array(weight_history[-252:])  # Last year
        time_steps = range(len(weights_array))

        for i, asset_name in enumerate(asset_names):
            axes[1, 0].plot(time_steps, weights_array[:, i], label=asset_name)

        axes[1, 0].set_title('Portfolio Allocation (Last Episode)')
        axes[1, 0].set_xlabel('Trading Day')
        axes[1, 0].set_ylabel('Weight')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Portfolio Value Evolution (last episode)
    if portfolio_history:
        portfolio_values = portfolio_history[-252:]  # Last year
        time_steps = range(len(portfolio_values))
        initial_value = portfolio_values[0]

        axes[1, 1].plot(time_steps, portfolio_values)
        axes[1, 1].axhline(y=initial_value, color='red', linestyle='--', alpha=0.7, label='Initial')
        axes[1, 1].set_title('Portfolio Value (Last Episode)')
        axes[1, 1].set_xlabel('Trading Day')
        axes[1, 1].set_ylabel('Portfolio Value ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Performance Distribution
    axes[1, 2].hist(returns_pct, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 2].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
    axes[1, 2].axvline(x=np.mean(returns_pct), color='green', linestyle='-', label='Mean')
    axes[1, 2].set_title('Return Distribution')
    axes[1, 2].set_xlabel('Return (%)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('portfolio_management_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Portfolio management plots saved as 'portfolio_management_results.png'")


def benchmark_against_strategies():
    """Compare RL agent against traditional portfolio strategies."""

    print("=== Benchmark Comparison ===")
    print("Comparing RL agent against traditional strategies...")

    env = MultiAssetPortfolioEnvironment(n_assets=5, n_steps=252)

    strategies = {
        'Equal Weight': equal_weight_strategy,
        '60/40 Portfolio': sixty_forty_strategy,
        'Risk Parity': risk_parity_strategy
    }

    results = {}

    for strategy_name, strategy_func in strategies.items():
        print(f"Testing {strategy_name}...")
        strategy_return = test_strategy(env, strategy_func)
        results[strategy_name] = strategy_return
        print(f"  {strategy_name}: {strategy_return*100:.2f}% return")

    return results


def equal_weight_strategy(state: np.ndarray, n_assets: int) -> np.ndarray:
    """Equal weight portfolio strategy."""
    return np.ones(n_assets) / n_assets


def sixty_forty_strategy(state: np.ndarray, n_assets: int) -> np.ndarray:
    """60% stocks, 40% bonds strategy."""
    weights = np.zeros(n_assets)
    if n_assets >= 3:
        weights[0] = 0.3  # Large cap stocks
        weights[1] = 0.3  # Small cap stocks
        weights[2] = 0.4  # Government bonds
    return weights


def risk_parity_strategy(state: np.ndarray, n_assets: int) -> np.ndarray:
    """Risk parity (inverse volatility) strategy."""
    # Extract volatilities from state (simplified)
    vols = state[n_assets*2:n_assets*3]  # Volatility features
    inv_vols = 1.0 / (vols + 1e-8)
    weights = inv_vols / np.sum(inv_vols)
    return weights


def test_strategy(env, strategy_func) -> float:
    """Test a strategy and return final return."""
    state = env.reset()

    for step in range(env.n_steps):
        action = strategy_func(state, env.n_assets)
        next_state, reward, done, info = env.step(action)
        state = next_state

        if done:
            break

    final_return = (env.portfolio_value - env.initial_value) / env.initial_value
    return final_return


if __name__ == "__main__":
    """Run the complete portfolio management example."""

    print("Multi-Asset Portfolio Management Example")
    print("=" * 50)
    print()

    try:
        # Run main portfolio management example
        agent, results = create_portfolio_management_example()

        print()
        print("=" * 50)

        # Run benchmark comparison
        benchmark_results = benchmark_against_strategies()

        print()
        print("Example completed successfully!")
        print("Professional portfolio management with RL demonstrates sophisticated")
        print("risk-adjusted decision making and dynamic allocation strategies.")

    except Exception as e:
        logger.error(f"Example failed with error: {e}")
        raise
