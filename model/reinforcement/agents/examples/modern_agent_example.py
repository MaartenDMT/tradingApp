"""
Modern Agent Implementation Examples.

Demonstrates how to use the enhanced agent base classes to create
professional reinforcement learning agents for trading applications.
"""

import os
import sys
from typing import Dict, Tuple

import numpy as np

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import util.loggers as loggers
from model.reinforcement.agents.base.base_agent import (
    ContinuousAgent,
    DiscreteAgent,
    ReplayBuffer,
)
from model.reinforcement.utils.enhanced_models import EnhancedNetworkBuilder

logger = loggers.setup_loggers()
agent_logger = logger['agent']


class ModernDQNAgent(DiscreteAgent):
    """
    Modern DQN agent implementation using enhanced base classes.

    Demonstrates professional patterns for discrete action RL agents
    with improved exploration, learning, and model management.
    """

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 learning_rate: float = 0.0001,
                 memory_size: int = 100000,
                 batch_size: int = 32,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 target_update_freq: int = 1000,
                 **kwargs):
        """
        Initialize Modern DQN Agent.

        Args:
            state_dim: State space dimension
            num_actions: Number of discrete actions
            learning_rate: Learning rate for optimizer
            memory_size: Replay buffer capacity
            batch_size: Training batch size
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            target_update_freq: Target network update frequency
        """
        config = {
            'learning_rate': learning_rate,
            'memory_size': memory_size,
            'batch_size': batch_size,
            'gamma': gamma,
            'epsilon_start': epsilon_start,
            'epsilon_min': epsilon_min,
            'epsilon_decay': epsilon_decay,
            'target_update_freq': target_update_freq
        }
        config.update(kwargs)

        super().__init__(state_dim, num_actions, "ModernDQN", config)

        # Agent parameters
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        # Initialize replay buffer
        self.memory = ReplayBuffer(memory_size, state_dim, 1)  # 1D action for discrete

        # Build networks
        self.q_network = EnhancedNetworkBuilder.build_dqn_network(
            state_dim=state_dim,
            num_actions=num_actions,
            learning_rate=learning_rate
        )

        self.target_network = EnhancedNetworkBuilder.build_dqn_network(
            state_dim=state_dim,
            num_actions=num_actions,
            learning_rate=learning_rate
        )

        # Initialize target network
        self.update_target_network()

        agent_logger.info(f"Initialized Modern DQN Agent with {num_actions} actions")

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current environment state
            training: Whether in training mode (affects exploration)

        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            # Random exploration
            action = np.random.randint(self.num_actions)
        else:
            # Greedy action
            state_batch = np.expand_dims(state, axis=0)
            q_values = self.q_network.predict(state_batch, verbose=0)[0]
            action = np.argmax(q_values)

        return action

    def learn(self, experiences: Tuple = None, **kwargs) -> Dict[str, float]:
        """
        Learn from batch of experiences.

        Args:
            experiences: Optional pre-sampled experiences
            **kwargs: Additional learning parameters

        Returns:
            Dictionary of learning metrics
        """
        if not self.memory.can_sample(self.batch_size):
            return {'loss': 0.0, 'q_mean': 0.0}

        # Sample experiences
        if experiences is None:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = experiences

        # Compute target Q-values
        next_q_values = self.target_network.predict(next_states, verbose=0)
        max_next_q_values = np.max(next_q_values, axis=1)

        targets = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Current Q-values
        current_q_values = self.q_network.predict(states, verbose=0)

        # Update Q-values for taken actions
        actions_int = actions.astype(int).flatten()
        for i in range(len(actions_int)):
            current_q_values[i][actions_int[i]] = targets[i]

        # Train network
        history = self.q_network.fit(
            states, current_q_values,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0
        )

        # Update exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()

        self.update_training_step()

        return {
            'loss': history.history['loss'][0],
            'q_mean': np.mean(current_q_values),
            'epsilon': self.epsilon
        }

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        self.memory.add(state, action, reward, next_state, done)
        self.add_reward(reward)

    def update_target_network(self) -> None:
        """Update target network weights."""
        self.target_network.set_weights(self.q_network.get_weights())
        agent_logger.debug("Target network updated")

    def save(self, filepath: str) -> None:
        """
        Save agent to file.

        Args:
            filepath: Base filepath for saving
        """
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save networks
            self.q_network.save(f"{filepath}_q_network.h5")
            self.target_network.save(f"{filepath}_target_network.h5")

            # Save replay buffer
            self.memory.save(f"{filepath}_memory.pkl")

            # Save agent stats
            stats = self.get_stats()
            stats.update(self.get_epsilon_stats())

            import json
            with open(f"{filepath}_stats.json", 'w') as f:
                json.dump(stats, f, indent=2, default=str)

            agent_logger.info(f"Modern DQN Agent saved to {filepath}")

        except Exception as e:
            agent_logger.error(f"Failed to save agent: {e}")
            raise

    def load(self, filepath: str) -> None:
        """
        Load agent from file.

        Args:
            filepath: Base filepath for loading
        """
        try:
            import tensorflow as tf

            # Load networks
            self.q_network = tf.keras.models.load_model(f"{filepath}_q_network.h5")
            self.target_network = tf.keras.models.load_model(f"{filepath}_target_network.h5")

            # Load replay buffer
            self.memory.load(f"{filepath}_memory.pkl")

            # Load stats
            import json
            with open(f"{filepath}_stats.json", 'r') as f:
                stats = json.load(f)

            # Restore agent state
            self.training_step = stats.get('training_step', 0)
            self.episode_count = stats.get('episode_count', 0)
            self.epsilon = stats.get('epsilon', self.epsilon)

            agent_logger.info(f"Modern DQN Agent loaded from {filepath}")

        except Exception as e:
            agent_logger.error(f"Failed to load agent: {e}")
            raise


class ModernTD3Agent(ContinuousAgent):
    """
    Modern TD3 agent implementation using enhanced base classes.

    Demonstrates professional patterns for continuous action RL agents
    with twin critics, delayed policy updates, and target policy smoothing.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 action_bounds: Tuple[float, float] = (-1.0, 1.0),
                 actor_lr: float = 0.0001,
                 critic_lr: float = 0.0001,
                 memory_size: int = 1000000,
                 batch_size: int = 32,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 policy_delay: int = 2,
                 noise_std: float = 0.1,
                 noise_clip: float = 0.5,
                 **kwargs):
        """
        Initialize Modern TD3 Agent.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            action_bounds: Action space bounds (min, max)
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            memory_size: Replay buffer capacity
            batch_size: Training batch size
            gamma: Discount factor
            tau: Soft update rate
            policy_delay: Policy update delay
            noise_std: Target policy smoothing noise
            noise_clip: Noise clipping range
        """
        config = {
            'actor_lr': actor_lr,
            'critic_lr': critic_lr,
            'memory_size': memory_size,
            'batch_size': batch_size,
            'gamma': gamma,
            'tau': tau,
            'policy_delay': policy_delay,
            'noise_std': noise_std,
            'noise_clip': noise_clip
        }
        config.update(kwargs)

        super().__init__(state_dim, action_dim, action_bounds, "ModernTD3", config)

        # Agent parameters
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay

        # Initialize replay buffer
        self.memory = ReplayBuffer(memory_size, state_dim, action_dim)

        # Build networks
        self.actor = EnhancedNetworkBuilder.build_actor_network(
            state_dim=state_dim,
            num_actions=action_dim,
            learning_rate=actor_lr,
            action_bound=max(abs(action_bounds[0]), abs(action_bounds[1]))
        )

        self.critic1 = EnhancedNetworkBuilder.build_critic_network(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=critic_lr
        )

        self.critic2 = EnhancedNetworkBuilder.build_critic_network(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=critic_lr
        )

        # Target networks
        self.target_actor = EnhancedNetworkBuilder.build_actor_network(
            state_dim=state_dim,
            num_actions=action_dim,
            learning_rate=actor_lr,
            action_bound=max(abs(action_bounds[0]), abs(action_bounds[1]))
        )

        self.target_critic1 = EnhancedNetworkBuilder.build_critic_network(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=critic_lr
        )

        self.target_critic2 = EnhancedNetworkBuilder.build_critic_network(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=critic_lr
        )

        # Initialize target networks
        self.update_target_networks()

        # Policy update counter
        self.policy_update_counter = 0

        agent_logger.info(f"Initialized Modern TD3 Agent with action dim {action_dim}")

    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select action using current policy.

        Args:
            state: Current environment state
            add_noise: Whether to add exploration noise

        Returns:
            Selected action
        """
        state_batch = np.expand_dims(state, axis=0)
        action = self.actor.predict(state_batch, verbose=0)[0]

        if add_noise:
            action = self.add_noise(action)

        return action

    def learn(self, experiences: Tuple = None, **kwargs) -> Dict[str, float]:
        """
        Learn from batch of experiences using TD3 algorithm.

        Args:
            experiences: Optional pre-sampled experiences
            **kwargs: Additional learning parameters

        Returns:
            Dictionary of learning metrics
        """
        if not self.memory.can_sample(self.batch_size):
            return {'critic_loss': 0.0, 'actor_loss': 0.0}

        # Sample experiences
        if experiences is None:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = experiences

        # Train critics
        import tensorflow as tf

        with tf.GradientTape(persistent=True) as tape:
            # Target actions with noise
            target_actions = self.target_actor(next_states)
            noise = tf.random.normal(tf.shape(target_actions), 0, self.noise_std)
            noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
            target_actions = tf.clip_by_value(
                target_actions + noise,
                self.action_bounds[0],
                self.action_bounds[1]
            )

            # Target Q-values (take minimum of twin critics)
            target_q1 = self.target_critic1([next_states, target_actions])
            target_q2 = self.target_critic2([next_states, target_actions])
            target_q = tf.minimum(target_q1, target_q2)

            # Compute target
            targets = rewards + self.gamma * target_q * (1 - dones)
            targets = tf.stop_gradient(targets)

            # Current Q-values
            current_q1 = self.critic1([states, actions])
            current_q2 = self.critic2([states, actions])

            # Critic losses
            critic1_loss = tf.reduce_mean(tf.square(targets - current_q1))
            critic2_loss = tf.reduce_mean(tf.square(targets - current_q2))

        # Update critics
        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)

        self.critic1.optimizer.apply_gradients(
            zip(critic1_grads, self.critic1.trainable_variables)
        )
        self.critic2.optimizer.apply_gradients(
            zip(critic2_grads, self.critic2.trainable_variables)
        )

        critic_loss = (critic1_loss + critic2_loss) / 2
        actor_loss = 0.0

        # Delayed policy updates
        if self.policy_update_counter % self.policy_delay == 0:
            with tf.GradientTape() as tape:
                # Actor loss
                predicted_actions = self.actor(states)
                actor_loss = -tf.reduce_mean(self.critic1([states, predicted_actions]))

            # Update actor
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(
                zip(actor_grads, self.actor.trainable_variables)
            )

            # Soft update target networks
            self.soft_update_target_networks()

            actor_loss = float(actor_loss)

        self.policy_update_counter += 1
        self.update_training_step()

        del tape

        return {
            'critic_loss': float(critic_loss),
            'actor_loss': actor_loss,
            'policy_updates': self.policy_update_counter
        }

    def remember(self, state: np.ndarray, action: np.ndarray, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        self.memory.add(state, action, reward, next_state, done)
        self.add_reward(reward)

    def update_target_networks(self) -> None:
        """Hard update target networks."""
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())

    def soft_update_target_networks(self) -> None:
        """Soft update target networks."""

        # Update target actor
        for target_param, param in zip(self.target_actor.trainable_variables,
                                      self.actor.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)

        # Update target critics
        for target_param, param in zip(self.target_critic1.trainable_variables,
                                      self.critic1.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)

        for target_param, param in zip(self.target_critic2.trainable_variables,
                                      self.critic2.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)

    def save(self, filepath: str) -> None:
        """
        Save agent to file.

        Args:
            filepath: Base filepath for saving
        """
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save networks
            self.actor.save(f"{filepath}_actor.h5")
            self.critic1.save(f"{filepath}_critic1.h5")
            self.critic2.save(f"{filepath}_critic2.h5")
            self.target_actor.save(f"{filepath}_target_actor.h5")
            self.target_critic1.save(f"{filepath}_target_critic1.h5")
            self.target_critic2.save(f"{filepath}_target_critic2.h5")

            # Save replay buffer
            self.memory.save(f"{filepath}_memory.pkl")

            # Save agent stats
            stats = self.get_stats()
            stats.update(self.get_noise_stats())
            stats['policy_update_counter'] = self.policy_update_counter

            import json
            with open(f"{filepath}_stats.json", 'w') as f:
                json.dump(stats, f, indent=2, default=str)

            agent_logger.info(f"Modern TD3 Agent saved to {filepath}")

        except Exception as e:
            agent_logger.error(f"Failed to save agent: {e}")
            raise

    def load(self, filepath: str) -> None:
        """
        Load agent from file.

        Args:
            filepath: Base filepath for loading
        """
        try:
            import tensorflow as tf

            # Load networks
            self.actor = tf.keras.models.load_model(f"{filepath}_actor.h5")
            self.critic1 = tf.keras.models.load_model(f"{filepath}_critic1.h5")
            self.critic2 = tf.keras.models.load_model(f"{filepath}_critic2.h5")
            self.target_actor = tf.keras.models.load_model(f"{filepath}_target_actor.h5")
            self.target_critic1 = tf.keras.models.load_model(f"{filepath}_target_critic1.h5")
            self.target_critic2 = tf.keras.models.load_model(f"{filepath}_target_critic2.h5")

            # Load replay buffer
            self.memory.load(f"{filepath}_memory.pkl")

            # Load stats
            import json
            with open(f"{filepath}_stats.json", 'r') as f:
                stats = json.load(f)

            # Restore agent state
            self.training_step = stats.get('training_step', 0)
            self.episode_count = stats.get('episode_count', 0)
            self.policy_update_counter = stats.get('policy_update_counter', 0)

            agent_logger.info(f"Modern TD3 Agent loaded from {filepath}")

        except Exception as e:
            agent_logger.error(f"Failed to load agent: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Example: Create and test modern agents
    print("=== Modern Agent Examples ===")

    # Test DQN Agent
    print("\n1. Testing Modern DQN Agent:")
    dqn_agent = ModernDQNAgent(
        state_dim=10,
        num_actions=3,
        learning_rate=0.001,
        memory_size=10000
    )

    # Test basic functionality
    test_state = np.random.random(10)
    action = dqn_agent.act(test_state)
    print(f"DQN Action: {action}")
    print(f"DQN Stats: {dqn_agent.get_stats()}")

    # Test TD3 Agent
    print("\n2. Testing Modern TD3 Agent:")
    td3_agent = ModernTD3Agent(
        state_dim=10,
        action_dim=1,
        action_bounds=(-1.0, 1.0),
        actor_lr=0.001,
        critic_lr=0.001
    )

    # Test basic functionality
    test_state = np.random.random(10)
    action = td3_agent.act(test_state)
    print(f"TD3 Action: {action}")
    print(f"TD3 Stats: {td3_agent.get_stats()}")

    print("\n=== Modern Agents Ready ===")
    print("Enhanced agent implementations created successfully!")
    print("Use these as templates for your trading applications.")
