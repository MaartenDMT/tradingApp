"""
Algorithm Switching Example.

This example demonstrates the advanced algorithm switching capabilities,
showing how to:
- Set up automatic switching conditions
- Transfer state between algorithms
- Monitor performance across switches
- Use adaptive switching strategies
"""

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import RL system components
from model.rl_system.integration import create_agent, setup_training_environment
from model.rl_system.integration.algorithm_factory import AlgorithmType
from model.rl_system.integration.algorithm_switching import (
    AlgorithmSwitcher,
    SwitchCondition,
    SwitchTrigger,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VolatileTradingEnvironment:
    """
    Trading environment with changing market conditions.

    This environment simulates different market regimes (bull, bear, sideways)
    to demonstrate the benefits of algorithm switching.
    """

    def __init__(self, n_steps: int = 1000):
        """Initialize the volatile trading environment."""
        self.n_steps = n_steps
        self.reset()

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = type('ActionSpace', (), {'n': 3})()

        # State space: [price, position, cash, portfolio_value, volatility, trend]
        self.state_dim = 6

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = 0
        self.price = 100.0
        self.position = 0.0
        self.cash = 10000.0
        self.portfolio_value = self.cash

        # Market regime: 0=bear, 1=sideways, 2=bull
        self.market_regime = 1
        self.regime_duration = 0
        self.regime_change_probability = 0.01

        self.price_history = [self.price]
        self.regime_history = [self.market_regime]
        self.action_history = []
        self.reward_history = []

        return self._get_state()

    def step(self, action: int) -> tuple:
        """Execute one step in the environment."""
        if self.current_step >= self.n_steps:
            return self._get_state(), 0.0, True, {}

        # Store previous values
        prev_portfolio_value = self.portfolio_value

        # Update market regime
        self._update_market_regime()

        # Generate price movement based on regime
        price_change = self._generate_price_change()
        self.price *= (1 + price_change)
        self.price = max(self.price, 0.01)

        # Execute trading action
        reward = self._execute_action(action)

        # Update portfolio value
        self.portfolio_value = self.cash + self.position * self.price

        # Add performance-based reward
        portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        reward += portfolio_return * 100

        # Store history
        self.price_history.append(self.price)
        self.regime_history.append(self.market_regime)
        self.action_history.append(action)
        self.reward_history.append(reward)

        self.current_step += 1
        done = self.current_step >= self.n_steps

        info = {
            'price': self.price,
            'position': self.position,
            'cash': self.cash,
            'portfolio_value': self.portfolio_value,
            'market_regime': self.market_regime,
            'volatility': self._calculate_volatility(),
            'trend': self._calculate_trend()
        }

        return self._get_state(), reward, done, info

    def _update_market_regime(self):
        """Update the market regime based on probability."""
        self.regime_duration += 1

        # Increase probability of regime change over time
        change_prob = self.regime_change_probability * (1 + self.regime_duration / 100)

        if np.random.random() < change_prob:
            self.market_regime = np.random.choice([0, 1, 2])
            self.regime_duration = 0

    def _generate_price_change(self) -> float:
        """Generate price change based on current market regime."""
        if self.market_regime == 0:  # Bear market
            drift = -0.0005
            volatility = 0.025
        elif self.market_regime == 1:  # Sideways market
            drift = 0.0
            volatility = 0.015
        else:  # Bull market
            drift = 0.0005
            volatility = 0.020

        return np.random.normal(drift, volatility)

    def _execute_action(self, action: int) -> float:
        """Execute trading action and return immediate reward."""
        reward = 0.0
        trade_size = 10
        transaction_cost = 0.001

        if action == 1:  # Buy
            cost = trade_size * self.price * (1 + transaction_cost)
            if self.cash >= cost:
                self.cash -= cost
                self.position += trade_size
                reward = -transaction_cost * trade_size * self.price

        elif action == 2:  # Sell
            if self.position >= trade_size:
                proceeds = trade_size * self.price * (1 - transaction_cost)
                self.cash += proceeds
                self.position -= trade_size
                reward = -transaction_cost * trade_size * self.price

        # Penalty for holding in volatile conditions
        volatility = self._calculate_volatility()
        if action == 0 and volatility > 0.02:
            reward -= 0.01

        return reward

    def _calculate_volatility(self) -> float:
        """Calculate recent price volatility."""
        if len(self.price_history) < 10:
            return 0.015

        recent_prices = self.price_history[-10:]
        returns = [recent_prices[i]/recent_prices[i-1] - 1 for i in range(1, len(recent_prices))]
        return np.std(returns)

    def _calculate_trend(self) -> float:
        """Calculate price trend (-1 to 1)."""
        if len(self.price_history) < 20:
            return 0.0

        recent_prices = self.price_history[-20:]
        x = np.arange(len(recent_prices))
        correlation = np.corrcoef(x, recent_prices)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        volatility = self._calculate_volatility()
        trend = self._calculate_trend()

        normalized_price = self.price / 100.0
        normalized_position = self.position / 100.0
        normalized_cash = self.cash / 10000.0
        normalized_portfolio = self.portfolio_value / 10000.0

        state = np.array([
            normalized_price,
            normalized_position,
            normalized_cash,
            normalized_portfolio,
            volatility * 50,  # Scale volatility
            trend
        ], dtype=np.float32)

        return state


def create_switching_example():
    """Demonstrate algorithm switching in volatile market conditions."""

    print("=== Algorithm Switching Example ===")
    print("This example shows how algorithms can be switched based on performance.")
    print("We'll train in a volatile environment with changing market regimes.")
    print()

    # Create environment
    env = VolatileTradingEnvironment(n_steps=300)
    print("Environment: Volatile trading with regime changes")
    print(f"State dimension: {env.state_dim}")
    print("Market regimes: Bear (0), Sideways (1), Bull (2)")
    print()

    # Setup RL system
    setup_training_environment(
        env_type="discrete",
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        complexity="complex"
    )

    # Create initial agent (DQN)
    initial_agent = create_agent(
        algorithm_type="dqn",
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        config={
            'learning_rate': 0.001,
            'epsilon_decay': 0.99
        }
    )

    # Setup algorithm switcher
    switcher = AlgorithmSwitcher(save_dir="switching_example")
    switcher.set_initial_agent(initial_agent)

    # Add switching conditions
    # Switch to Double DQN if performance is poor
    performance_condition = SwitchCondition(
        trigger=SwitchTrigger.PERFORMANCE_THRESHOLD,
        threshold=-5.0,  # Switch if avg reward < -5
        metric='mean_reward',
        comparison='less_than'
    )

    episode_condition = SwitchCondition(
        trigger=SwitchTrigger.EPISODE_COUNT,
        episodes=50
    )

    # Add switch rules
    from model.rl_system.integration.algorithm_factory import AlgorithmSpec

    double_dqn_spec = AlgorithmSpec(
        algorithm_type=AlgorithmType.DOUBLE_DQN,
        state_dim=env.state_dim,
        action_dim=env.action_space.n
    )

    ppo_spec = AlgorithmSpec(
        algorithm_type=AlgorithmType.PPO,
        state_dim=env.state_dim,
        action_dim=env.action_space.n
    )

    switcher.add_switch_condition(performance_condition, AlgorithmType.DOUBLE_DQN, double_dqn_spec)
    switcher.add_switch_condition(episode_condition, AlgorithmType.PPO, ppo_spec)

    print("Algorithm switching setup:")
    print("- Start with DQN")
    print("- Switch to Double DQN if avg reward < -5")
    print("- Switch to PPO after 50 episodes")
    print()

    # Training loop with switching
    current_agent = initial_agent
    episode_rewards = []
    switch_episodes = []
    algorithm_history = [current_agent.algorithm_type.value]

    max_episodes = 150

    print("Starting training with algorithm switching...")

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(300):
            action = current_agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            current_agent.store_transition(state, action, reward, next_state, done)

            # Update agent
            if hasattr(current_agent.agent, 'memory') and len(current_agent.agent.memory) > 32:
                current_agent.update()

            state = next_state
            episode_reward += reward

            if done:
                break

        # Record episode performance
        episode_rewards.append(episode_reward)
        switcher.record_episode_performance(episode_reward, step + 1, {})

        # Check for algorithm switching
        new_agent = switcher.check_switch_conditions(episode, step)
        if new_agent:
            current_agent = new_agent
            switch_episodes.append(episode)
            algorithm_history.append(current_agent.algorithm_type.value)
            print(f"  -> Switched to {current_agent.algorithm_type.value} at episode {episode}")

        # Progress reporting
        if episode % 25 == 0:
            avg_reward = np.mean(episode_rewards[-25:])
            current_alg = current_agent.algorithm_type.value
            print(f"Episode {episode:3d} | Algorithm: {current_alg:10s} | Avg Reward: {avg_reward:8.2f}")

    print()
    print("Training with switching completed!")

    # Analysis of switching performance
    print("=== Switching Analysis ===")

    switch_history = switcher.get_switch_history()
    print(f"Total switches: {len(switch_history)}")

    for i, switch_event in enumerate(switch_history):
        print(f"Switch {i+1}: {switch_event.from_algorithm} -> {switch_event.to_algorithm} "
              f"at episode {switch_event.episode} (trigger: {switch_event.trigger.value})")

    # Create visualization
    create_switching_plots(episode_rewards, switch_episodes, algorithm_history,
                          env.regime_history, env.price_history)

    return current_agent, episode_rewards, switch_history


def create_switching_plots(rewards: list, switch_episodes: list, algorithm_history: list,
                          regime_history: list, price_history: list):
    """Create visualization plots for algorithm switching results."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Algorithm Switching Training Results', fontsize=16)

    # Plot 1: Episode Rewards with switches marked
    axes[0, 0].plot(rewards, label='Episode Rewards')
    for switch_ep in switch_episodes:
        axes[0, 0].axvline(x=switch_ep, color='red', linestyle='--', alpha=0.7)

    # Add moving average
    if len(rewards) > 25:
        ma_rewards = np.convolve(rewards, np.ones(25)/25, mode='valid')
        axes[0, 0].plot(range(24, len(rewards)), ma_rewards, color='orange', label='MA(25)')

    axes[0, 0].set_title('Episode Rewards (Red lines = Algorithm Switches)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # Plot 2: Algorithm Timeline
    algorithm_nums = {'dqn': 0, 'double_dqn': 1, 'ppo': 2}

    # Create step function for algorithm changes
    episode_ranges = []
    current_ep = 0

    for i, alg in enumerate(algorithm_history[:-1]):
        next_switch = switch_episodes[i] if i < len(switch_episodes) else len(rewards)
        episode_ranges.extend([algorithm_nums[alg]] * (next_switch - current_ep))
        current_ep = next_switch

    if current_ep < len(rewards):
        episode_ranges.extend([algorithm_nums[algorithm_history[-1]]] * (len(rewards) - current_ep))

    axes[0, 1].plot(episode_ranges, linewidth=3)
    axes[0, 1].set_title('Algorithm Timeline')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Algorithm')
    axes[0, 1].set_yticks([0, 1, 2])
    axes[0, 1].set_yticklabels(['DQN', 'Double DQN', 'PPO'])
    axes[0, 1].grid(True)

    # Plot 3: Market Regime vs Performance
    # Group rewards by regime
    if len(regime_history) > len(rewards):
        regime_history = regime_history[:len(rewards)]

    regime_rewards = {0: [], 1: [], 2: []}
    for i, reward in enumerate(rewards):
        if i < len(regime_history):
            regime = regime_history[i]
            regime_rewards[regime].append(reward)

    regime_names = ['Bear', 'Sideways', 'Bull']
    regime_means = [np.mean(regime_rewards[i]) if regime_rewards[i] else 0 for i in range(3)]

    bars = axes[1, 0].bar(regime_names, regime_means, color=['red', 'gray', 'green'], alpha=0.7)
    axes[1, 0].set_title('Average Reward by Market Regime')
    axes[1, 0].set_xlabel('Market Regime')
    axes[1, 0].set_ylabel('Average Reward')
    axes[1, 0].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean_val in zip(bars, regime_means):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean_val:.1f}', ha='center', va='bottom')

    # Plot 4: Price Evolution (sample)
    if len(price_history) > 300:
        sample_prices = price_history[-300:]  # Last 300 steps
    else:
        sample_prices = price_history

    axes[1, 1].plot(sample_prices)
    axes[1, 1].set_title('Asset Price Evolution (Sample)')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Price')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('algorithm_switching_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Switching analysis plots saved as 'algorithm_switching_results.png'")


def compare_switching_vs_static():
    """Compare performance with and without algorithm switching."""

    print("=== Switching vs Static Comparison ===")
    print("Comparing performance with algorithm switching vs static algorithm.")
    print()

    # Create environment
    env = VolatileTradingEnvironment(n_steps=200)

    # Test 1: Static DQN
    print("Training static DQN...")
    static_agent = create_agent("dqn", env.state_dim, env.action_space.n)
    static_rewards = train_agent_simple(env, static_agent, 100)

    # Test 2: Algorithm switching
    print("Training with algorithm switching...")
    switching_agent, switching_rewards, _ = create_switching_example()

    # Compare results
    static_mean = np.mean(static_rewards[-25:])
    switching_mean = np.mean(switching_rewards[-25:])

    print()
    print("=== Comparison Results ===")
    print(f"Static DQN final performance:     {static_mean:.2f}")
    print(f"Algorithm switching performance:  {switching_mean:.2f}")
    print(f"Improvement:                      {switching_mean - static_mean:.2f}")
    print(f"Percentage improvement:           {(switching_mean - static_mean) / abs(static_mean) * 100:.1f}%")

    return static_rewards, switching_rewards


def train_agent_simple(env, agent, episodes):
    """Simple training loop for comparison."""
    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(200):
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)

            if hasattr(agent.agent, 'memory') and len(agent.agent.memory) > 32:
                agent.update()

            state = next_state
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)

    return episode_rewards


if __name__ == "__main__":
    """Run the complete algorithm switching example."""

    print("Algorithm Switching Example")
    print("=" * 50)
    print()

    try:
        # Run main switching example
        agent, rewards, switch_history = create_switching_example()

        print()
        print("=" * 50)

        # Run comparison
        static_rewards, switching_rewards = compare_switching_vs_static()

        print()
        print("Example completed successfully!")
        print("Algorithm switching demonstrates adaptive learning capabilities.")

    except Exception as e:
        logger.error(f"Example failed with error: {e}")
        raise
