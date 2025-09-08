"""
Proximal Policy Optimization (PPO) Implementation.

This module implements the PPO algorithm with actor-critic networks,
generalized advantage estimation (GAE), and policy clipping for stable training.
PPO is particularly well-suited for trading applications due to its conservative
policy updates and stability.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

from ...core.base_agents import PolicyBasedAgent

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    state_dim: int
    action_dim: int
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    clip_value: float = 0.5
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    batch_size: int = 64
    hidden_dim: int = 64
    action_std: float = 0.1
    device: str = "cpu"


class PPOMemory:
    """
    Memory buffer for storing trajectory data used in PPO training.

    Stores states, actions, rewards, values, log probabilities, and done flags
    for batch processing during policy updates.
    """

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.clear()

    def clear(self):
        """Clear all stored trajectory data."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def store_transition(self, state, action, log_prob, value, reward, done):
        """Store a single transition."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_advantages_and_returns(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter for bias-variance tradeoff
        """
        advantages = []
        gae = 0

        # Convert to numpy arrays for easier computation
        values = np.array(self.values)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        # Compute advantages using GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        self.advantages = advantages
        self.returns = [adv + val for adv, val in zip(advantages, self.values)]

    def get_batches(self):
        """
        Generate random batches for training.

        Returns:
            Generator yielding batches of training data
        """
        n_samples = len(self.states)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, n_samples, self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]

            yield {
                'states': torch.tensor([self.states[i] for i in batch_indices], dtype=torch.float32),
                'actions': torch.tensor([self.actions[i] for i in batch_indices], dtype=torch.float32),
                'old_log_probs': torch.tensor([self.log_probs[i] for i in batch_indices], dtype=torch.float32),
                'values': torch.tensor([self.values[i] for i in batch_indices], dtype=torch.float32),
                'advantages': torch.tensor([self.advantages[i] for i in batch_indices], dtype=torch.float32),
                'returns': torch.tensor([self.returns[i] for i in batch_indices], dtype=torch.float32)
            }


class ActorNetwork(nn.Module):
    """
    Actor network for PPO that outputs action probabilities or action means/stds.

    Supports both discrete and continuous action spaces.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [128, 128],
                 continuous: bool = False,
                 action_std: float = 0.5):
        super(ActorNetwork, self).__init__()

        self.continuous = continuous
        self.action_dim = action_dim

        # Build network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)  # Layer normalization for stability
            ])
            prev_dim = hidden_dim

        self.base_network = nn.Sequential(*layers)

        if continuous:
            # For continuous actions, output mean and optionally std
            self.action_mean = nn.Linear(prev_dim, action_dim)
            self.action_std = nn.Parameter(torch.ones(action_dim) * action_std)
        else:
            # For discrete actions, output action probabilities
            self.action_probs = nn.Linear(prev_dim, action_dim)

    def forward(self, state):
        """Forward pass through the network."""
        x = self.base_network(state)

        if self.continuous:
            action_mean = torch.tanh(self.action_mean(x))  # Bound actions to [-1, 1]
            action_std = torch.clamp(self.action_std, min=0.01, max=2.0)
            return action_mean, action_std
        else:
            action_logits = self.action_probs(x)
            return torch.softmax(action_logits, dim=-1)

    def get_action_and_log_prob(self, state):
        """Get action and its log probability for the given state."""
        if self.continuous:
            action_mean, action_std = self.forward(state)
            distribution = Normal(action_mean, action_std)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(dim=-1)
            return action, log_prob
        else:
            action_probs = self.forward(state)
            distribution = Categorical(action_probs)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            return action, log_prob

    def get_log_prob(self, state, action):
        """Get log probability of given actions for given states."""
        if self.continuous:
            action_mean, action_std = self.forward(state)
            distribution = Normal(action_mean, action_std)
            log_prob = distribution.log_prob(action).sum(dim=-1)
            return log_prob
        else:
            action_probs = self.forward(state)
            distribution = Categorical(action_probs)
            log_prob = distribution.log_prob(action)
            return log_prob


class CriticNetwork(nn.Module):
    """
    Critic network for PPO that estimates state values.
    """

    def __init__(self,
                 state_dim: int,
                 hidden_dims: List[int] = [128, 128]):
        super(CriticNetwork, self).__init__()

        # Build network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)  # Layer normalization for stability
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))  # Output single value

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """Forward pass through the network."""
        return self.network(state).squeeze(-1)


class PPOAgent(PolicyBasedAgent):
    """
    Proximal Policy Optimization (PPO) Agent.

    Implements the PPO algorithm with actor-critic networks, GAE, and policy clipping.
    Suitable for both discrete and continuous action spaces.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 continuous: bool = False,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coeff: float = 0.5,
                 entropy_coeff: float = 0.01,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 10,
                 batch_size: int = 64,
                 hidden_dims: List[int] = [128, 128],
                 action_std: float = 0.5,
                 device: Optional[str] = None):
        """
        Initialize PPO agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            continuous: Whether action space is continuous
            learning_rate: Learning rate for optimizers
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coeff: Coefficient for value loss
            entropy_coeff: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of epochs per PPO update
            batch_size: Batch size for training
            hidden_dims: Hidden layer dimensions
            action_std: Initial action standard deviation (continuous only)
            device: Device to run on (cpu/cuda)
        """
        super().__init__(state_dim, action_dim)

        self.continuous = continuous
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs

        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize networks
        self.actor = ActorNetwork(
            state_dim, action_dim, hidden_dims, continuous, action_std
        ).to(self.device)

        self.critic = CriticNetwork(
            state_dim, hidden_dims
        ).to(self.device)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Initialize memory
        self.memory = PPOMemory(batch_size)

        # Training statistics
        self.training_step = 0
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []

        logger.info(f"Initialized PPO agent - State dim: {state_dim}, "
                   f"Action dim: {action_dim}, Continuous: {continuous}")

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action for given state.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if training:
                action, log_prob = self.actor.get_action_and_log_prob(state_tensor)
                value = self.critic(state_tensor)

                # Store for training
                self.last_state = state
                self.last_action = action.cpu().numpy().squeeze()
                self.last_log_prob = log_prob.cpu().numpy().squeeze()
                self.last_value = value.cpu().numpy().squeeze()
            else:
                # For evaluation, use deterministic policy
                if self.continuous:
                    action_mean, _ = self.actor(state_tensor)
                    action = action_mean
                else:
                    action_probs = self.actor(state_tensor)
                    action = torch.argmax(action_probs, dim=-1)

                self.last_action = action.cpu().numpy().squeeze()

        return self.last_action

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in memory."""
        if hasattr(self, 'last_log_prob') and hasattr(self, 'last_value'):
            self.memory.store_transition(
                self.last_state,
                self.last_action,
                self.last_log_prob,
                self.last_value,
                reward,
                done
            )

    def update(self):
        """
        Update the agent using collected trajectory data.

        Returns:
            Dictionary containing training statistics
        """
        # Compute advantages and returns
        self.memory.compute_advantages_and_returns(self.gamma, self.gae_lambda)

        # Training statistics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        num_updates = 0

        # Multiple epochs of training
        for epoch in range(self.ppo_epochs):
            for batch in self.memory.get_batches():
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                old_log_probs = batch['old_log_probs'].to(self.device)
                advantages = batch['advantages'].to(self.device)
                returns = batch['returns'].to(self.device)

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Get current policy probabilities and values
                current_log_probs = self.actor.get_log_prob(states, actions)
                current_values = self.critic(states)

                # Compute ratio for policy loss
                ratio = torch.exp(current_log_probs - old_log_probs)

                # Compute policy loss with clipping
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                value_loss = nn.MSELoss()(current_values, returns)

                # Compute entropy bonus
                if self.continuous:
                    action_mean, action_std = self.actor(states)
                    entropy = Normal(action_mean, action_std).entropy().sum(dim=-1).mean()
                else:
                    action_probs = self.actor(states)
                    entropy = Categorical(action_probs).entropy().mean()

                entropy_loss = -entropy

                # Update actor
                self.actor_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # Accumulate statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                num_updates += 1

        # Clear memory
        self.memory.clear()

        # Update training statistics
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates

        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropy_losses.append(avg_entropy_loss)

        self.training_step += 1

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_policy_loss + self.value_coeff * avg_value_loss + self.entropy_coeff * avg_entropy_loss
        }

    def save_model(self, filepath: str):
        """Save model weights and configuration."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'continuous': self.continuous,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'value_coeff': self.value_coeff,
                'entropy_coeff': self.entropy_coeff
            }
        }, filepath)
        logger.info(f"PPO model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model weights and configuration."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_step = checkpoint['training_step']

        logger.info(f"PPO model loaded from {filepath}")

    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'training_step': self.training_step,
            'avg_policy_loss': np.mean(self.policy_losses[-100:]) if self.policy_losses else 0,
            'avg_value_loss': np.mean(self.value_losses[-100:]) if self.value_losses else 0,
            'avg_entropy_loss': np.mean(self.entropy_losses[-100:]) if self.entropy_losses else 0,
            'total_updates': len(self.policy_losses)
        }


# Factory function for easy creation
def create_ppo_agent(state_dim: int,
                     action_dim: int,
                     continuous: bool = False,
                     **kwargs) -> PPOAgent:
    """
    Factory function to create a PPO agent with sensible defaults.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        continuous: Whether action space is continuous
        **kwargs: Additional configuration parameters

    Returns:
        Configured PPO agent
    """
    return PPOAgent(state_dim, action_dim, continuous, **kwargs)
