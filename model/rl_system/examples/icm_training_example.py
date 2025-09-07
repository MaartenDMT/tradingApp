"""
Example usage of the Intrinsic Curiosity Module (ICM) with RL agents.

This script demonstrates how to integrate ICM with different RL algorithms
to enhance exploration in trading environments.
"""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

# Import RL system components
from model.rl_system.algorithms.exploration import create_curiosity_driven_agent
from model.rl_system.algorithms.value_based import DQNAgent
from model.rl_system.utils.visualization import CuriosityVisualizer

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SimpleEnvironment:
    """
    Simple trading environment for demonstration.
    """

    def __init__(self, state_dim: int = 10, price_volatility: float = 0.02):
        self.state_dim = state_dim
        self.price_volatility = price_volatility
        self.current_step = 0
        self.max_steps = 1000
        self.price = 100.0
        self.position = 0.0
        self.cash = 10000.0
        self.initial_cash = 10000.0

    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.price = 100.0
        self.position = 0.0
        self.cash = self.initial_cash
        return self._get_state()

    def step(self, action: int):
        """
        Execute action and return next state, reward, done.

        Actions: 0=Hold, 1=Buy, 2=Sell
        """
        # Update price (random walk)
        price_change = np.random.normal(0, self.price_volatility)
        self.price *= (1 + price_change)

        # Execute action
        reward = 0.0
        if action == 1 and self.cash >= self.price:  # Buy
            shares_to_buy = self.cash // self.price
            self.position += shares_to_buy
            self.cash -= shares_to_buy * self.price
            reward = -0.01  # Small transaction cost

        elif action == 2 and self.position > 0:  # Sell
            self.cash += self.position * self.price
            reward = 0.01 * self.position  # Small profit bonus
            self.position = 0

        # Calculate portfolio value change as reward
        current_value = self.cash + self.position * self.price
        reward += (current_value - self.initial_cash) / self.initial_cash * 0.1

        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._get_state(), reward, done

    def _get_state(self):
        """Get current state representation."""
        # Simple state: price history, position, cash ratio, technical indicators
        price_history = [self.price] * 3  # Simplified
        technical_indicators = [
            self.price / 100.0 - 1,  # Normalized price change
            self.position / 100.0,   # Normalized position
            self.cash / self.initial_cash,  # Cash ratio
        ]

        # Add some noise to make exploration meaningful
        noise = np.random.normal(0, 0.01, self.state_dim - 6)
        state = price_history + technical_indicators + noise.tolist()

        return np.array(state[:self.state_dim])


def train_with_icm_example():
    """
    Example of training a DQN agent with ICM for enhanced exploration.
    """
    print("🧠 Training DQN with Intrinsic Curiosity Module (ICM)")
    print("=" * 60)

    # Environment and agent setup
    env = SimpleEnvironment(state_dim=10)
    state_dim = 10
    action_dim = 3  # Hold, Buy, Sell
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Create base DQN agent
    base_agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-3,
        device=device
    )

    # Create ICM-enhanced agent
    curiosity_agent = create_curiosity_driven_agent(
        base_agent=base_agent,
        state_dim=state_dim,
        action_dim=action_dim,
        intrinsic_reward_weight=0.5,
        device=device
    )

    print("✅ Created curiosity-driven agent with intrinsic reward weight: 0.5")

    # Training parameters
    num_episodes = 200
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1

    # Training metrics
    episode_rewards = []
    intrinsic_rewards = []
    extrinsic_rewards = []
    icm_losses = {'forward': [], 'inverse': [], 'total': []}

    print(f"\n🚀 Starting training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        episode_extrinsic = []
        steps = 0

        while True:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(action_dim)
            else:
                action = curiosity_agent.get_action(state)

            # Environment step
            next_state, reward, done = env.step(action)

            # Store transition (this includes ICM intrinsic reward calculation)
            curiosity_agent.store_transition(state, action, reward, next_state, done)

            # Track rewards separately for analysis
            episode_extrinsic.append(reward)

            total_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        # Update agent (includes ICM update)
        if len(curiosity_agent.base_agent.memory) > 32:
            losses = curiosity_agent.update()

            # Track ICM losses
            if 'forward_loss' in losses:
                icm_losses['forward'].append(losses['forward_loss'])
            if 'inverse_loss' in losses:
                icm_losses['inverse'].append(losses['inverse_loss'])
            if 'total_icm_loss' in losses:
                icm_losses['total'].append(losses['total_icm_loss'])

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Track episode metrics
        episode_rewards.append(total_reward)
        intrinsic_rewards.extend(curiosity_agent.intrinsic_rewards_history[-steps:])
        extrinsic_rewards.extend(episode_extrinsic)

        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            exploration_stats = curiosity_agent.get_exploration_stats()

            print(f"Episode {episode + 1:3d}: Avg Reward: {avg_reward:8.3f}, "
                  f"Epsilon: {epsilon:.3f}, "
                  f"Avg Intrinsic: {exploration_stats.get('avg_intrinsic_reward', 0):.4f}")

    print("\n✅ Training completed!")

    # Analyze results
    print("\n📊 Training Analysis:")
    print(f"Total episodes: {num_episodes}")
    print(f"Final average reward (last 50): {np.mean(episode_rewards[-50:]):.3f}")
    print(f"Best episode reward: {np.max(episode_rewards):.3f}")
    print(f"Average intrinsic reward: {np.mean(intrinsic_rewards):.4f}")
    print(f"Average extrinsic reward: {np.mean(extrinsic_rewards):.4f}")

    if icm_losses['forward']:
        print(f"Final forward loss: {icm_losses['forward'][-1]:.4f}")
        print(f"Final inverse loss: {icm_losses['inverse'][-1]:.4f}")

    # Create visualizations
    create_training_visualizations(
        episode_rewards, intrinsic_rewards, extrinsic_rewards, icm_losses
    )

    return curiosity_agent, episode_rewards, intrinsic_rewards, extrinsic_rewards, icm_losses


def create_training_visualizations(episode_rewards: List[float],
                                 intrinsic_rewards: List[float],
                                 extrinsic_rewards: List[float],
                                 icm_losses: Dict[str, List[float]]):
    """
    Create comprehensive visualizations of ICM training.
    """
    print("\n🎨 Creating training visualizations...")

    # Initialize curiosity visualizer
    visualizer = CuriosityVisualizer(save_dir="icm_training_plots")

    # 1. ICM Losses
    if icm_losses['forward'] and icm_losses['inverse'] and icm_losses['total']:
        visualizer.plot_icm_losses(
            forward_losses=icm_losses['forward'],
            inverse_losses=icm_losses['inverse'],
            total_losses=icm_losses['total'],
            title="ICM Training Losses",
            save_name="icm_losses"
        )
        plt.show()

    # 2. Reward Comparison
    if intrinsic_rewards and extrinsic_rewards:
        visualizer.plot_reward_comparison(
            intrinsic_rewards=intrinsic_rewards,
            extrinsic_rewards=extrinsic_rewards,
            title="Intrinsic vs Extrinsic Rewards",
            save_name="reward_comparison"
        )
        plt.show()

    # 3. Episode Rewards
    plt.figure(figsize=(12, 6))
    episodes = range(len(episode_rewards))
    plt.plot(episodes, episode_rewards, alpha=0.7, linewidth=1, label='Episode Rewards')

    # Smooth curve
    if len(episode_rewards) > 20:
        smoothed = []
        window = 20
        for i in range(len(episode_rewards)):
            start_idx = max(0, i - window + 1)
            smoothed.append(np.mean(episode_rewards[start_idx:i+1]))
        plt.plot(episodes, smoothed, linewidth=2, label='Smoothed', color='red')

    plt.title('Episode Rewards During ICM Training')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('icm_training_plots/episode_rewards.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Curiosity Dashboard
    curiosity_data = {
        'forward_losses': icm_losses['forward'],
        'inverse_losses': icm_losses['inverse'],
        'intrinsic_rewards': intrinsic_rewards,
        'extrinsic_rewards': extrinsic_rewards
    }

    visualizer.create_curiosity_dashboard(
        curiosity_data=curiosity_data,
        title="ICM Training Dashboard",
        save_name="curiosity_dashboard"
    )
    plt.show()

    print("✅ Visualizations created and saved to 'icm_training_plots/' directory")


def compare_with_without_icm():
    """
    Compare DQN performance with and without ICM.
    """
    print("\n🔍 Comparing DQN with and without ICM...")
    print("=" * 50)

    env = SimpleEnvironment(state_dim=10)
    state_dim = 10
    action_dim = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_episodes = 100

    # Train without ICM
    print("Training vanilla DQN...")
    vanilla_agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    vanilla_rewards = train_agent(vanilla_agent, env, num_episodes, "Vanilla DQN")

    # Train with ICM
    print("Training DQN with ICM...")
    base_agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    icm_agent = create_curiosity_driven_agent(base_agent, state_dim, action_dim, device=device)
    icm_rewards = train_agent(icm_agent, env, num_episodes, "DQN + ICM")

    # Compare results
    print("\n📊 Comparison Results:")
    print(f"Vanilla DQN - Mean: {np.mean(vanilla_rewards):.3f}, Std: {np.std(vanilla_rewards):.3f}")
    print(f"DQN + ICM   - Mean: {np.mean(icm_rewards):.3f}, Std: {np.std(icm_rewards):.3f}")

    # Plot comparison
    plt.figure(figsize=(12, 6))
    episodes = range(len(vanilla_rewards))

    plt.plot(episodes, vanilla_rewards, alpha=0.7, label='Vanilla DQN', color='blue')
    plt.plot(episodes, icm_rewards, alpha=0.7, label='DQN + ICM', color='red')

    # Smoothed curves
    window = 10
    if len(vanilla_rewards) > window:
        vanilla_smooth = [np.mean(vanilla_rewards[max(0, i-window):i+1]) for i in range(len(vanilla_rewards))]
        icm_smooth = [np.mean(icm_rewards[max(0, i-window):i+1]) for i in range(len(icm_rewards))]

        plt.plot(episodes, vanilla_smooth, linewidth=2, label='Vanilla DQN (Smoothed)', color='darkblue')
        plt.plot(episodes, icm_smooth, linewidth=2, label='DQN + ICM (Smoothed)', color='darkred')

    plt.title('DQN vs DQN + ICM Performance Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('icm_training_plots/dqn_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def train_agent(agent, env, num_episodes: int, agent_name: str) -> List[float]:
    """
    Train an agent and return episode rewards.
    """
    rewards = []
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            # Action selection
            if np.random.random() < epsilon:
                action = np.random.randint(3)
            else:
                action = agent.get_action(state)

            next_state, reward, done = env.step(action)

            # Store transition
            if hasattr(agent, 'store_transition'):
                agent.store_transition(state, action, reward, next_state, done)
            else:
                agent.base_agent.store_transition(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if done:
                break

        # Update
        if hasattr(agent, 'update'):
            if hasattr(agent, 'base_agent'):
                if len(agent.base_agent.memory) > 32:
                    agent.update()
            else:
                if len(agent.memory) > 32:
                    agent.update()

        rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % 25 == 0:
            avg_reward = np.mean(rewards[-25:])
            print(f"{agent_name} - Episode {episode + 1:3d}: Avg Reward: {avg_reward:8.3f}")

    return rewards


if __name__ == "__main__":
    print("🎯 ICM (Intrinsic Curiosity Module) Training Example")
    print("🎯 " + "=" * 55)

    # Run main ICM training example
    agent, rewards, intrinsic, extrinsic, losses = train_with_icm_example()

    # Optional: Run comparison
    print("\n" + "🔄" * 30)
    compare_with_without_icm()

    print("\n🎉 ICM Example completed! Check 'icm_training_plots/' for visualizations.")
