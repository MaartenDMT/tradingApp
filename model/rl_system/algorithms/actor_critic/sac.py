"""
Soft Actor-Critic (SAC) Implementation.

This module implements the SAC algorithm with automatic entropy tuning,
twin critics, and stochastic policies. SAC is particularly well-suited for
continuous control tasks like trading position sizing.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from ..core.base_agents import ActorCriticAgent, ReplayBuffer

logger = logging.getLogger(__name__)


class GaussianActor(nn.Module):
    """
    Gaussian actor network for SAC that outputs action means and log stds.

    The actor outputs a mean and log standard deviation for each action dimension,
    which are used to construct a Gaussian distribution for action sampling.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 max_action: float = 1.0,
                 log_std_min: float = -20,
                 log_std_max: float = 2):
        super(GaussianActor, self).__init__()

        self.action_dim = action_dim
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Build shared trunk
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)  # Layer normalization for stability
            ])
            prev_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

    def forward(self, state):
        """Forward pass through the network."""
        x = self.trunk(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)

        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, reparameterize: bool = True):
        """
        Sample actions from the policy.

        Args:
            state: Current state
            reparameterize: Whether to use reparameterization trick

        Returns:
            Tuple of (action, log_prob, mean)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Create distribution
        distribution = Normal(mean, std)

        if reparameterize:
            # Reparameterization trick
            raw_action = distribution.rsample()
        else:
            # No gradients through sampling
            raw_action = distribution.sample()

        # Apply tanh to bound actions and compute log probability
        action = torch.tanh(raw_action)

        # Compute log probability with tanh correction
        log_prob = distribution.log_prob(raw_action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Scale actions to desired range
        action = action * self.max_action

        return action, log_prob, mean


class SACCritic(nn.Module):
    """
    Twin critic networks for SAC.

    Implements two Q-networks to reduce overestimation bias.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256]):
        super(SACCritic, self).__init__()

        # First Q-network
        q1_layers = []
        prev_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            q1_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        q1_layers.append(nn.Linear(prev_dim, 1))
        self.q1_network = nn.Sequential(*q1_layers)

        # Second Q-network
        q2_layers = []
        prev_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            q2_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        q2_layers.append(nn.Linear(prev_dim, 1))
        self.q2_network = nn.Sequential(*q2_layers)

    def forward(self, state, action):
        """Forward pass through both Q-networks."""
        state_action = torch.cat([state, action], dim=-1)

        q1 = self.q1_network(state_action)
        q2 = self.q2_network(state_action)

        return q1, q2

    def q1(self, state, action):
        """Get Q-value from first network only."""
        state_action = torch.cat([state, action], dim=-1)
        return self.q1_network(state_action)


class SACAgent(ActorCriticAgent):
    """
    Soft Actor-Critic (SAC) Agent.

    Implements SAC with automatic entropy tuning, twin critics, and stochastic policy.
    Particularly well-suited for continuous control tasks like trading.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 alpha_lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 automatic_entropy_tuning: bool = True,
                 target_entropy: Optional[float] = None,
                 max_action: float = 1.0,
                 replay_buffer_size: int = 1000000,
                 batch_size: int = 256,
                 hidden_dims: List[int] = [256, 256],
                 device: Optional[str] = None):
        """
        Initialize SAC agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            alpha_lr: Entropy coefficient learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Initial entropy coefficient
            automatic_entropy_tuning: Whether to automatically tune entropy
            target_entropy: Target entropy for automatic tuning
            max_action: Maximum action value
            replay_buffer_size: Size of replay buffer
            batch_size: Batch size for training
            hidden_dims: Hidden layer dimensions
            device: Device to run on (cpu/cuda)
        """
        super().__init__(state_dim, action_dim)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize networks
        self.actor = GaussianActor(
            state_dim, action_dim, hidden_dims, max_action
        ).to(self.device)

        self.critic = SACCritic(
            state_dim, action_dim, hidden_dims
        ).to(self.device)

        self.target_critic = SACCritic(
            state_dim, action_dim, hidden_dims
        ).to(self.device)

        # Initialize target network
        self.hard_update(self.target_critic, self.critic)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Entropy coefficient
        if automatic_entropy_tuning:
            if target_entropy is None:
                # Heuristic target entropy
                self.target_entropy = -action_dim
            else:
                self.target_entropy = target_entropy

            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = alpha

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # Training statistics
        self.training_step = 0
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []

        logger.info(f"Initialized SAC agent - State dim: {state_dim}, "
                   f"Action dim: {action_dim}, Auto entropy: {automatic_entropy_tuning}")

    @property
    def alpha(self):
        """Get current entropy coefficient."""
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp()
        else:
            return self._alpha

    @alpha.setter
    def alpha(self, value):
        """Set entropy coefficient."""
        if self.automatic_entropy_tuning:
            self.log_alpha.data.fill_(np.log(value))
        else:
            self._alpha = value

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
                action, _, _ = self.actor.sample(state_tensor)
            else:
                # For evaluation, use mean action
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean) * self.max_action

        return action.cpu().numpy().squeeze()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def hard_update(self, target, source):
        """Hard update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update(self) -> Dict:
        """
        Update the agent using a batch of transitions.

        Returns:
            Dictionary containing training statistics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(batch['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(batch['next_states'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(batch['dones'], dtype=torch.float32).to(self.device)

        # Update critic
        critic_loss = self.update_critic(states, actions, rewards, next_states, dones)

        # Update actor
        actor_loss = self.update_actor(states)

        # Update entropy coefficient
        alpha_loss = None
        if self.automatic_entropy_tuning:
            alpha_loss = self.update_alpha(states)

        # Soft update target networks
        self.soft_update(self.target_critic, self.critic, self.tau)

        # Update statistics
        self.training_step += 1
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        if alpha_loss is not None:
            self.alpha_losses.append(alpha_loss)

        result = {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'alpha': self.alpha.item() if torch.is_tensor(self.alpha) else self.alpha
        }

        if alpha_loss is not None:
            result['alpha_loss'] = alpha_loss

        return result

    def update_critic(self, states, actions, rewards, next_states, dones):
        """Update critic networks."""
        with torch.no_grad():
            # Sample next actions
            next_actions, next_log_probs, _ = self.actor.sample(next_states)

            # Compute target Q-values
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = target_q - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * self.gamma * target_q

        # Compute current Q-values
        current_q1, current_q2 = self.critic(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_actor(self, states):
        """Update actor network."""
        # Sample actions
        actions, log_probs, _ = self.actor.sample(states)

        # Compute Q-values
        q1, q2 = self.critic(states, actions)
        q = torch.min(q1, q2)

        # Compute actor loss
        actor_loss = (self.alpha * log_probs - q).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def update_alpha(self, states):
        """Update entropy coefficient."""
        # Sample actions
        actions, log_probs, _ = self.actor.sample(states)

        # Compute alpha loss
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss.item()

    def save_model(self, filepath: str):
        """Save model weights and configuration."""
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'tau': self.tau,
                'max_action': self.max_action,
                'automatic_entropy_tuning': self.automatic_entropy_tuning
            }
        }

        if self.automatic_entropy_tuning:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
            save_dict['target_entropy'] = self.target_entropy
        else:
            save_dict['alpha'] = self._alpha

        torch.save(save_dict, filepath)
        logger.info(f"SAC model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model weights and configuration."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_step = checkpoint['training_step']

        if self.automatic_entropy_tuning:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.target_entropy = checkpoint['target_entropy']
        else:
            self._alpha = checkpoint['alpha']

        logger.info(f"SAC model loaded from {filepath}")

    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'training_step': self.training_step,
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0,
            'avg_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0,
            'avg_alpha_loss': np.mean(self.alpha_losses[-100:]) if self.alpha_losses else 0,
            'current_alpha': self.alpha.item() if torch.is_tensor(self.alpha) else self.alpha,
            'total_updates': len(self.actor_losses)
        }


# Factory function for easy creation
def create_sac_agent(state_dim: int,
                     action_dim: int,
                     **kwargs) -> SACAgent:
    """
    Factory function to create a SAC agent with sensible defaults.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        **kwargs: Additional configuration parameters

    Returns:
        Configured SAC agent
    """
    return SACAgent(state_dim, action_dim, **kwargs)
