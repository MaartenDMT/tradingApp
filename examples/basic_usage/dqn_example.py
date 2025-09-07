"""
Basic DQN Training Example.

This example demonstrates how to use the DQN algorithm for basic trading tasks.
Shows fundamental concepts like:
- Environment setup
- Agent creation
- Training loop
- Performance evaluation
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


class SimpleTradingEnvironment:
    """
    Simple trading environment for demonstration purposes.

    This environment simulates basic price movements and trading decisions.
    """

    def __init__(self, n_steps: int = 1000, initial_price: float = 100.0):
        """
        Initialize the trading environment.

        Args:
            n_steps: Number of time steps in an episode
            initial_price: Starting price for the asset
        """
        self.n_steps = n_steps
        self.initial_price = initial_price
        self.reset()

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = type('ActionSpace', (), {'n': 3})()

        # State space: [price, position, cash, portfolio_value, price_change]
        self.state_dim = 5

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = 0
        self.price = self.initial_price
        self.position = 0.0  # Number of shares held
        self.cash = 10000.0  # Starting cash
        self.portfolio_value = self.cash
        self.price_history = [self.price]
        self.action_history = []
        self.reward_history = []

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0=Hold, 1=Buy, 2=Sell)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.current_step >= self.n_steps:
            return self._get_state(), 0.0, True, {}

        # Store previous portfolio value
        prev_portfolio_value = self.portfolio_value

        # Generate new price (simple random walk)
        price_change = np.random.normal(0, 0.02)  # 2% volatility
        self.price *= (1 + price_change)
        self.price = max(self.price, 0.01)  # Prevent negative prices

        # Execute action
        reward = self._execute_action(action)

        # Update portfolio value
        self.portfolio_value = self.cash + self.position * self.price

        # Calculate reward based on portfolio performance
        portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        reward += portfolio_return * 100  # Scale reward

        # Store history
        self.price_history.append(self.price)
        self.action_history.append(action)
        self.reward_history.append(reward)

        self.current_step += 1
        done = self.current_step >= self.n_steps

        info = {
            'price': self.price,
            'position': self.position,
            'cash': self.cash,
            'portfolio_value': self.portfolio_value,
            'step': self.current_step
        }

        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        price_change = 0.0
        if len(self.price_history) > 1:
            price_change = (self.price - self.price_history[-1]) / self.price_history[-1]

        # Normalize state values
        normalized_price = self.price / self.initial_price
        normalized_position = self.position / 100.0  # Assume max 100 shares
        normalized_cash = self.cash / 10000.0  # Normalize by initial cash
        normalized_portfolio = self.portfolio_value / 10000.0

        state = np.array([
            normalized_price,
            normalized_position,
            normalized_cash,
            normalized_portfolio,
            price_change
        ], dtype=np.float32)

        return state

    def _execute_action(self, action: int) -> float:
        """
        Execute trading action and return immediate reward.

        Args:
            action: Trading action (0=Hold, 1=Buy, 2=Sell)

        Returns:
            Immediate reward for the action
        """
        reward = 0.0
        trade_size = 10  # Number of shares per trade
        transaction_cost = 0.001  # 0.1% transaction cost

        if action == 1:  # Buy
            cost = trade_size * self.price * (1 + transaction_cost)
            if self.cash >= cost:
                self.cash -= cost
                self.position += trade_size
                reward = -transaction_cost * trade_size * self.price  # Negative reward for transaction cost

        elif action == 2:  # Sell
            if self.position >= trade_size:
                proceeds = trade_size * self.price * (1 - transaction_cost)
                self.cash += proceeds
                self.position -= trade_size
                reward = -transaction_cost * trade_size * self.price  # Negative reward for transaction cost

        # Small penalty for holding to encourage action
        if action == 0:
            reward = -0.001

        return reward


def create_dqn_example():
    """Create and train a DQN agent on the simple trading environment."""

    print("=== DQN Trading Example ===")
    print("This example shows how to train a DQN agent for basic trading decisions.")
    print()

    # Create environment
    env = SimpleTradingEnvironment(n_steps=500)
    print(f"Environment created with state dimension: {env.state_dim}")
    print(f"Action space: {env.action_space.n} actions (Hold, Buy, Sell)")
    print()

    # Setup training environment in the RL system
    setup_training_environment(
        env_type="discrete",
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        complexity="simple"
    )

    # Create DQN agent
    agent = create_agent(
        algorithm_type="dqn",
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        config={
            'learning_rate': 0.001,
            'batch_size': 32,
            'memory_size': 10000,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'target_update_frequency': 100
        }
    )

    print(f"Created DQN agent: {agent.algorithm_type.value}")
    print(f"Agent info: {agent.get_info()}")
    print()

    # Training configuration
    training_config = TrainingConfig(
        max_episodes=300,
        max_steps_per_episode=500,
        save_frequency=50,
        evaluation_frequency=25,
        enable_switching=False,
        enable_visualization=True
    )

    print("Starting training...")
    print(f"Training for {training_config.max_episodes} episodes")
    print()

    # Training loop
    episode_rewards = []
    episode_lengths = []
    portfolio_values = []

    for episode in range(training_config.max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(training_config.max_steps_per_episode):
            # Get action from agent
            action = agent.get_action(state)

            # Take step in environment
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Update agent
            if hasattr(agent.agent, 'memory') and len(agent.agent.memory) > agent.agent.batch_size:
                agent.update()

            # Update for next step
            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                break

        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        portfolio_values.append(info['portfolio_value'])

        # Progress reporting
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_portfolio = np.mean(portfolio_values[-50:])
            print(f"Episode {episode:3d} | Avg Reward: {avg_reward:8.2f} | "
                  f"Portfolio Value: ${avg_portfolio:8.2f} | "
                  f"Epsilon: {agent.agent.epsilon:.3f}")

    print()
    print("Training completed!")

    # Final evaluation
    print("=== Final Evaluation ===")
    final_avg_reward = np.mean(episode_rewards[-50:])
    final_avg_portfolio = np.mean(portfolio_values[-50:])
    total_return = (final_avg_portfolio - 10000) / 10000 * 100

    print(f"Final average reward: {final_avg_reward:.2f}")
    print(f"Final portfolio value: ${final_avg_portfolio:.2f}")
    print(f"Total return: {total_return:.2f}%")
    print()

    # Create visualization
    create_training_plots(episode_rewards, portfolio_values, env.price_history, env.action_history)

    return agent, episode_rewards, portfolio_values


def create_training_plots(rewards: list, portfolio_values: list,
                         price_history: list, action_history: list):
    """Create visualization plots for training results."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Trading Training Results', fontsize=16)

    # Plot 1: Episode Rewards
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)

    # Add moving average
    if len(rewards) > 50:
        ma_rewards = np.convolve(rewards, np.ones(50)/50, mode='valid')
        axes[0, 0].plot(range(49, len(rewards)), ma_rewards, color='red', label='MA(50)')
        axes[0, 0].legend()

    # Plot 2: Portfolio Value
    axes[0, 1].plot(portfolio_values)
    axes[0, 1].axhline(y=10000, color='red', linestyle='--', label='Initial Value')
    axes[0, 1].set_title('Portfolio Value Over Episodes')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Portfolio Value ($)')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # Plot 3: Price History (last episode)
    axes[1, 0].plot(price_history)
    axes[1, 0].set_title('Asset Price History (Last Episode)')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Price')
    axes[1, 0].grid(True)

    # Plot 4: Action Distribution
    if action_history:
        action_counts = np.bincount(action_history)
        action_labels = ['Hold', 'Buy', 'Sell']
        axes[1, 1].bar(range(len(action_counts)), action_counts)
        axes[1, 1].set_title('Action Distribution (Last Episode)')
        axes[1, 1].set_xlabel('Action')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_xticks(range(len(action_labels)))
        axes[1, 1].set_xticklabels(action_labels)
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('dqn_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Training plots saved as 'dqn_training_results.png'")


def demonstrate_agent_usage():
    """Demonstrate different ways to use the trained agent."""

    print("=== Agent Usage Demonstration ===")

    # Create environment and agent
    env = SimpleTradingEnvironment(n_steps=100)
    agent = create_agent("dqn", env.state_dim, env.action_space.n)

    print("1. Getting actions from agent:")
    state = env.reset()
    for i in range(5):
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        action_names = ['Hold', 'Buy', 'Sell']
        print(f"   Step {i+1}: Action = {action_names[action]}, Reward = {reward:.3f}")
        state = next_state

    print()
    print("2. Agent information:")
    info = agent.get_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    print()
    print("3. Saving and loading agent:")
    save_path = "example_dqn_agent.pth"
    agent.save_state(save_path)
    print(f"   Agent saved to {save_path}")

    # Create new agent and load state
    new_agent = create_agent("dqn", env.state_dim, env.action_space.n)
    new_agent.load_state(save_path)
    print(f"   Agent loaded from {save_path}")


if __name__ == "__main__":
    """Run the complete DQN example."""

    print("DQN Trading Example")
    print("=" * 50)
    print()

    try:
        # Run main training example
        agent, rewards, portfolio_values = create_dqn_example()

        print()
        print("=" * 50)

        # Demonstrate agent usage
        demonstrate_agent_usage()

        print()
        print("Example completed successfully!")
        print("Check the generated plots and saved agent files.")

    except Exception as e:
        logger.error(f"Example failed with error: {e}")
        raise
