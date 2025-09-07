"""
PPO Continuous Control Example.

This example demonstrates how to use the PPO algorithm for continuous control
in trading scenarios. Shows:
- Continuous action spaces
- Policy gradient methods
- Advantage estimation
- Continuous portfolio allocation
"""

import logging
import os
import sys
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import RL system components
from model.rl_system.integration import (
    TrainingConfig,
    create_agent,
    setup_training_environment,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContinuousPortfolioEnvironment:
    """
    Continuous portfolio allocation environment.

    Agent decides how much of the portfolio to allocate to each asset.
    Actions are continuous values representing portfolio weights.
    """

    def __init__(self, n_assets: int = 3, n_steps: int = 252):
        """
        Initialize the portfolio environment.

        Args:
            n_assets: Number of assets in the portfolio
            n_steps: Number of trading days (252 = 1 year)
        """
        self.n_assets = n_assets
        self.n_steps = n_steps
        self.reset()

        # Action space: portfolio weights for each asset (sum to 1)
        self.action_dim = n_assets

        # State space: [prices, returns, volatilities, portfolio_weights, cash]
        self.state_dim = n_assets * 4 + 1  # 4 features per asset + cash

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = 0
        self.initial_portfolio_value = 100000.0
        self.portfolio_value = self.initial_portfolio_value
        self.cash = self.initial_portfolio_value

        # Initialize asset prices
        self.prices = np.array([100.0, 50.0, 200.0])[:self.n_assets]
        self.initial_prices = self.prices.copy()

        # Initialize portfolio weights (equal weight initially)
        self.portfolio_weights = np.ones(self.n_assets) / self.n_assets

        # Price and return history
        self.price_history = [self.prices.copy()]
        self.return_history = []
        self.portfolio_history = [self.portfolio_value]

        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Portfolio weights for each asset (should sum to ~1)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.current_step >= self.n_steps:
            return self._get_state(), 0.0, True, {}

        # Normalize action to ensure weights sum to 1
        action = np.abs(action)  # Ensure positive weights
        action = action / (np.sum(action) + 1e-8)  # Normalize to sum to 1

        # Store previous values
        prev_portfolio_value = self.portfolio_value
        prev_prices = self.prices.copy()

        # Generate new asset prices (correlated random walk)
        returns = self._generate_asset_returns()
        self.prices = prev_prices * (1 + returns)
        self.prices = np.maximum(self.prices, 0.01)  # Prevent negative prices

        # Update portfolio weights based on action
        self.portfolio_weights = action

        # Calculate new portfolio value
        price_changes = self.prices / prev_prices - 1
        weighted_returns = np.sum(self.portfolio_weights * price_changes)
        self.portfolio_value = prev_portfolio_value * (1 + weighted_returns)

        # Calculate reward
        reward = self._calculate_reward(prev_portfolio_value, weighted_returns)

        # Store history
        self.price_history.append(self.prices.copy())
        self.return_history.append(returns)
        self.portfolio_history.append(self.portfolio_value)

        self.current_step += 1
        done = self.current_step >= self.n_steps

        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_weights': self.portfolio_weights.copy(),
            'asset_returns': returns,
            'portfolio_return': weighted_returns,
            'step': self.current_step
        }

        return self._get_state(), reward, done, info

    def _generate_asset_returns(self) -> np.ndarray:
        """Generate correlated asset returns."""
        # Base volatilities for different asset types
        base_vols = np.array([0.015, 0.020, 0.012])[:self.n_assets]  # Daily volatility

        # Generate correlated returns
        correlation_matrix = np.array([
            [1.0, 0.6, 0.3],
            [0.6, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])[:self.n_assets, :self.n_assets]

        # Generate independent random returns
        independent_returns = np.random.normal(0, base_vols)

        # Apply correlation
        L = np.linalg.cholesky(correlation_matrix)
        correlated_returns = L @ independent_returns

        return correlated_returns

    def _calculate_reward(self, prev_portfolio_value: float, portfolio_return: float) -> float:
        """Calculate reward based on portfolio performance."""
        # Base reward: portfolio return
        reward = portfolio_return * 100  # Scale for better learning

        # Penalty for extreme allocations (encourage diversification)
        concentration_penalty = np.sum(self.portfolio_weights ** 2) - 1/self.n_assets
        reward -= concentration_penalty * 2

        # Small penalty for volatility (risk-adjusted returns)
        if len(self.return_history) > 10:
            recent_returns = [self.return_history[i] for i in range(-10, 0)]
            volatility = np.std(recent_returns)
            reward -= volatility * 5

        return reward

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        state_components = []

        # Normalized asset prices
        normalized_prices = self.prices / self.initial_prices
        state_components.extend(normalized_prices)

        # Recent returns (or zeros if not available)
        if len(self.return_history) > 0:
            recent_returns = self.return_history[-1]
        else:
            recent_returns = np.zeros(self.n_assets)
        state_components.extend(recent_returns)

        # Historical volatility
        if len(self.return_history) > 5:
            recent_returns_history = np.array(self.return_history[-5:])
            volatilities = np.std(recent_returns_history, axis=0)
        else:
            volatilities = np.ones(self.n_assets) * 0.01
        state_components.extend(volatilities)

        # Current portfolio weights
        state_components.extend(self.portfolio_weights)

        # Normalized portfolio value
        normalized_portfolio_value = self.portfolio_value / self.initial_portfolio_value
        state_components.append(normalized_portfolio_value)

        return np.array(state_components, dtype=np.float32)


def create_ppo_example():
    """Create and train a PPO agent for continuous portfolio allocation."""

    print("=== PPO Continuous Portfolio Example ===")
    print("This example shows how to train a PPO agent for continuous portfolio allocation.")
    print()

    # Create environment
    env = ContinuousPortfolioEnvironment(n_assets=3, n_steps=252)
    print(f"Environment created with state dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim} (portfolio weights)")
    print("Assets: 3 (representing different asset classes)")
    print()

    # Setup training environment in the RL system
    setup_training_environment(
        env_type="continuous",
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        complexity="medium"
    )

    # Create PPO agent
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
            'value_function_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5
        }
    )

    print(f"Created PPO agent: {agent.algorithm_type.value}")
    print(f"Agent info: {agent.get_info()}")
    print()

    # Training configuration
    training_config = TrainingConfig(
        max_episodes=200,
        max_steps_per_episode=252,
        save_frequency=25,
        evaluation_frequency=20,
        enable_switching=False,
        enable_visualization=True
    )

    print("Starting training...")
    print(f"Training for {training_config.max_episodes} episodes")
    print()

    # Training loop
    episode_rewards = []
    portfolio_returns = []
    final_portfolio_values = []
    sharpe_ratios = []

    for episode in range(training_config.max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_returns = []

        for step in range(training_config.max_steps_per_episode):
            # Get action from agent
            action = agent.get_action(state)

            # Take step in environment
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Track episode metrics
            episode_reward += reward
            episode_returns.append(info['portfolio_return'])

            # Update for next step
            state = next_state

            if done:
                break

        # Update agent at end of episode
        if hasattr(agent.agent, 'memory') and len(agent.agent.memory) > 0:
            agent.update()

        # Calculate episode metrics
        final_portfolio_value = env.portfolio_value
        total_return = (final_portfolio_value - env.initial_portfolio_value) / env.initial_portfolio_value

        # Calculate Sharpe ratio
        if len(episode_returns) > 1:
            mean_return = np.mean(episode_returns)
            std_return = np.std(episode_returns)
            sharpe_ratio = mean_return / (std_return + 1e-8) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0

        # Record episode metrics
        episode_rewards.append(episode_reward)
        portfolio_returns.append(total_return)
        final_portfolio_values.append(final_portfolio_value)
        sharpe_ratios.append(sharpe_ratio)

        # Progress reporting
        if episode % 25 == 0:
            avg_reward = np.mean(episode_rewards[-25:])
            avg_return = np.mean(portfolio_returns[-25:]) * 100
            avg_sharpe = np.mean(sharpe_ratios[-25:])
            print(f"Episode {episode:3d} | Avg Reward: {avg_reward:8.2f} | "
                  f"Avg Return: {avg_return:6.2f}% | "
                  f"Avg Sharpe: {avg_sharpe:6.3f}")

    print()
    print("Training completed!")

    # Final evaluation
    print("=== Final Evaluation ===")
    final_avg_reward = np.mean(episode_rewards[-25:])
    final_avg_return = np.mean(portfolio_returns[-25:]) * 100
    final_avg_sharpe = np.mean(sharpe_ratios[-25:])

    print(f"Final average reward: {final_avg_reward:.2f}")
    print(f"Final average return: {final_avg_return:.2f}%")
    print(f"Final average Sharpe ratio: {final_avg_sharpe:.3f}")
    print()

    # Create visualization
    create_ppo_training_plots(episode_rewards, portfolio_returns, sharpe_ratios,
                             env.portfolio_history, env.price_history)

    return agent, episode_rewards, portfolio_returns


def create_ppo_training_plots(rewards: list, returns: list, sharpe_ratios: list,
                             portfolio_history: list, price_history: list):
    """Create visualization plots for PPO training results."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('PPO Continuous Portfolio Training Results', fontsize=16)

    # Plot 1: Episode Rewards
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)

    # Add moving average
    if len(rewards) > 25:
        ma_rewards = np.convolve(rewards, np.ones(25)/25, mode='valid')
        axes[0, 0].plot(range(24, len(rewards)), ma_rewards, color='red', label='MA(25)')
        axes[0, 0].legend()

    # Plot 2: Portfolio Returns
    returns_pct = np.array(returns) * 100
    axes[0, 1].plot(returns_pct)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', label='Break-even')
    axes[0, 1].set_title('Portfolio Returns')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Return (%)')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # Plot 3: Sharpe Ratios
    axes[0, 2].plot(sharpe_ratios)
    axes[0, 2].axhline(y=1.0, color='red', linestyle='--', label='Good (>1.0)')
    axes[0, 2].axhline(y=2.0, color='green', linestyle='--', label='Excellent (>2.0)')
    axes[0, 2].set_title('Sharpe Ratios')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Sharpe Ratio')
    axes[0, 2].grid(True)
    axes[0, 2].legend()

    # Plot 4: Portfolio Value Evolution (last episode)
    axes[1, 0].plot(portfolio_history)
    axes[1, 0].axhline(y=100000, color='red', linestyle='--', label='Initial Value')
    axes[1, 0].set_title('Portfolio Value Evolution (Last Episode)')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Portfolio Value ($)')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # Plot 5: Asset Price Evolution (last episode)
    price_array = np.array(price_history)
    asset_names = ['Asset 1', 'Asset 2', 'Asset 3']
    for i in range(min(3, price_array.shape[1])):
        axes[1, 1].plot(price_array[:, i], label=asset_names[i])
    axes[1, 1].set_title('Asset Prices (Last Episode)')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Price')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    # Plot 6: Performance Distribution
    axes[1, 2].hist(returns_pct, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 2].axvline(x=0, color='red', linestyle='--', label='Break-even')
    axes[1, 2].axvline(x=np.mean(returns_pct), color='green', linestyle='-', label='Mean')
    axes[1, 2].set_title('Return Distribution')
    axes[1, 2].set_xlabel('Return (%)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True)
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig('ppo_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Training plots saved as 'ppo_training_results.png'")


def demonstrate_portfolio_allocation():
    """Demonstrate how the trained PPO agent allocates portfolio."""

    print("=== Portfolio Allocation Demonstration ===")

    # Create environment and agent
    env = ContinuousPortfolioEnvironment(n_assets=3, n_steps=50)
    agent = create_agent("ppo", env.state_dim, env.action_dim)

    print("Portfolio allocation over time:")
    print("Step | Asset 1 | Asset 2 | Asset 3 | Portfolio Value")
    print("-" * 55)

    state = env.reset()
    for step in range(10):
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        weights = info['portfolio_weights']
        portfolio_value = info['portfolio_value']

        print(f"{step+1:4d} | {weights[0]:7.1%} | {weights[1]:7.1%} | {weights[2]:7.1%} | ${portfolio_value:12,.2f}")

        state = next_state
        if done:
            break

    print()
    print("Note: The agent learns to dynamically adjust portfolio weights")
    print("based on market conditions and risk-return trade-offs.")


if __name__ == "__main__":
    """Run the complete PPO example."""

    print("PPO Continuous Portfolio Example")
    print("=" * 50)
    print()

    try:
        # Run main training example
        agent, rewards, returns = create_ppo_example()

        print()
        print("=" * 50)

        # Demonstrate portfolio allocation
        demonstrate_portfolio_allocation()

        print()
        print("Example completed successfully!")
        print("Check the generated plots showing portfolio performance.")

    except Exception as e:
        logger.error(f"Example failed with error: {e}")
        raise
