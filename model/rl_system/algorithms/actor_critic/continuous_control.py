"""
Advanced Actor-Critic Algorithms Implementation.

This module contains implementations of advanced actor-critic methods including
Twin Delayed Deep Deterministic Policy Gradient (TD3), Soft Actor-Critic (SAC),
and Deep Deterministic Policy Gradient (DDPG) for continuous control.
"""

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import util.loggers as loggers

from ...core.base_agents import ActorCriticAgent, AgentConfig, ReplayBuffer

logger = loggers.setup_loggers()
rl_logger = logger['rl']


class Actor(nn.Module):
    """Actor network for continuous control."""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float = 1.0,
                 hidden_dims: list = [256, 256],
                 activation: str = 'relu'):
        super(Actor, self).__init__()

        self.max_action = max_action

        # Build network
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())

            prev_dim = hidden_dim

        # Output layer with tanh activation
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to get actions."""
        return self.max_action * self.network(state)


class Critic(nn.Module):
    """Critic network for Q-value estimation."""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: list = [256, 256],
                 activation: str = 'relu'):
        super(Critic, self).__init__()

        # Build network
        layers = []
        prev_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())

            prev_dim = hidden_dim

        # Output single Q-value
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass to get Q-value."""
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class GaussianActor(nn.Module):
    """Gaussian actor for SAC with stochastic policy."""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float = 1.0,
                 hidden_dims: list = [256, 256],
                 log_std_min: float = -20,
                 log_std_max: float = 2):
        super(GaussianActor, self).__init__()

        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.shared_network = nn.Sequential(*layers)

        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in [self.shared_network, self.mean_layer, self.log_std_layer]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.constant_(layer.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor, deterministic: bool = False, with_logprob: bool = True):
        """Forward pass to get action and log probability."""
        features = self.shared_network(state)

        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        if deterministic:
            action = torch.tanh(mean) * self.max_action
            return action

        std = torch.exp(log_std)
        normal = Normal(mean, std)

        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.max_action

        if with_logprob:
            # Compute log probability with tanh correction
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.max_action * (1 - torch.tanh(x_t).pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            return action, log_prob

        return action


@dataclass
class ActorCriticConfig(AgentConfig):
    """Configuration for actor-critic agents."""
    learning_rate: float = 0.001
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    gamma: float = 0.99
    tau: float = 0.005  # Soft update rate
    batch_size: int = 256
    memory_size: int = 1000000
    hidden_dims: list = None
    activation: str = 'relu'
    max_action: float = 1.0
    noise_std: float = 0.1
    noise_clip: float = 0.5
    policy_delay: int = 2  # For TD3
    target_noise: float = 0.2  # For TD3
    alpha: float = 0.2  # For SAC
    automatic_entropy_tuning: bool = True  # For SAC

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]


class DDPGAgent(ActorCriticAgent):
    """Deep Deterministic Policy Gradient agent."""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: ActorCriticConfig = None):

        self.config = config or ActorCriticConfig()
        super().__init__(state_dim, action_dim, self.config, "DDPGAgent")

    def _initialize_agent(self) -> None:
        """Initialize DDPG agent components."""
        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')

        # Actor networks
        self.actor = Actor(
            self.state_dim,
            self.action_dim,
            self.config.max_action,
            self.config.hidden_dims,
            self.config.activation
        ).to(self.device)

        self.actor_target = copy.deepcopy(self.actor).to(self.device)

        # Critic networks
        self.critic = Critic(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            self.config.memory_size,
            self.state_dim,
            self.action_dim,
            self.config.random_seed
        )

        # Training metrics
        self.actor_loss_history = deque(maxlen=1000)
        self.critic_loss_history = deque(maxlen=1000)

        rl_logger.info(f"Initialized DDPG agent with device: {self.device}")

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using actor network with exploration noise."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()

        if training:
            # Add exploration noise
            noise = np.random.normal(0, self.config.noise_std, size=self.action_dim)
            action = np.clip(action + noise, -self.config.max_action, self.config.max_action)

        return action

    def update(self,
               state: np.ndarray,
               action: np.ndarray,
               reward: float,
               next_state: np.ndarray,
               done: bool) -> Dict[str, float]:
        """Update DDPG networks."""
        # Store experience
        self.replay_buffer.store(state, action, reward, next_state, done)

        metrics = {}

        # Update if we have enough experiences
        if self.replay_buffer.can_sample(self.config.batch_size):
            actor_loss, critic_loss = self._update_networks()
            metrics.update({
                'actor_loss': actor_loss,
                'critic_loss': critic_loss
            })

        self.training_step += 1
        return metrics

    def _update_networks(self) -> Tuple[float, float]:
        """Update actor and critic networks."""
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards.unsqueeze(1) + (self.config.gamma * target_q * ~dones.unsqueeze(1))

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update_targets()

        # Record losses
        actor_loss_val = actor_loss.item()
        critic_loss_val = critic_loss.item()
        self.actor_loss_history.append(actor_loss_val)
        self.critic_loss_history.append(critic_loss_val)

        return actor_loss_val, critic_loss_val

    def _soft_update_targets(self) -> None:
        """Soft update target networks."""
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

    def _get_agent_state(self) -> Dict[str, Any]:
        """Get agent state for checkpointing."""
        return {
            'actor_state': self.actor.state_dict(),
            'actor_target_state': self.actor_target.state_dict(),
            'critic_state': self.critic.state_dict(),
            'critic_target_state': self.critic_target.state_dict(),
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'critic_optimizer_state': self.critic_optimizer.state_dict(),
            'actor_loss_history': list(self.actor_loss_history),
            'critic_loss_history': list(self.critic_loss_history)
        }

    def _restore_agent_state(self, state: Dict[str, Any]) -> None:
        """Restore agent state from checkpoint."""
        self.actor.load_state_dict(state['actor_state'])
        self.actor_target.load_state_dict(state['actor_target_state'])
        self.critic.load_state_dict(state['critic_state'])
        self.critic_target.load_state_dict(state['critic_target_state'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer_state'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer_state'])
        self.actor_loss_history = deque(state.get('actor_loss_history', []), maxlen=1000)
        self.critic_loss_history = deque(state.get('critic_loss_history', []), maxlen=1000)

    def _save_model_weights(self, filepath: str) -> None:
        """Save model weights."""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, f"{filepath}.pth")

    def _load_model_weights(self, filepath: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(f"{filepath}.pth", map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


class TD3Agent(DDPGAgent):
    """Twin Delayed Deep Deterministic Policy Gradient agent."""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: ActorCriticConfig = None):

        if config is None:
            config = ActorCriticConfig()

        # Store action_dim before calling super
        self.action_dim = action_dim
        super().__init__(state_dim, action_dim, config)
        self.name = "TD3Agent"

    def _initialize_agent(self) -> None:
        """Initialize TD3 agent with twin critics."""
        super()._initialize_agent()

        # Add second critic network (twin critics)
        self.critic2 = Critic(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation
        ).to(self.device)

        self.critic2_target = copy.deepcopy(self.critic2).to(self.device)

        # Add optimizer for second critic
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.config.critic_lr)

        # Policy update counter
        self.policy_update_counter = 0

        rl_logger.info("Initialized TD3 agent with twin critics")

    def _update_networks(self) -> Tuple[float, float]:
        """Update networks with TD3 improvements."""
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Update critics
        with torch.no_grad():
            # Target policy smoothing
            next_actions = self.actor_target(next_states)
            noise = torch.randn_like(next_actions) * self.config.target_noise
            noise = torch.clamp(noise, -self.config.noise_clip, self.config.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -self.config.max_action, self.config.max_action)

            # Twin critics - take minimum
            target_q1 = self.critic_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(1) + (self.config.gamma * target_q * ~dones.unsqueeze(1))

        # Update both critics
        current_q1 = self.critic(states, actions)
        current_q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        critic_loss = (critic1_loss + critic2_loss) / 2

        # Delayed policy update
        actor_loss = 0.0
        if self.policy_update_counter % self.config.policy_delay == 0:
            # Update actor
            actor_loss = -self.critic(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            self._soft_update_targets()

            actor_loss = actor_loss.item()

        self.policy_update_counter += 1

        # Record losses
        critic_loss_val = critic_loss.item()
        self.actor_loss_history.append(actor_loss)
        self.critic_loss_history.append(critic_loss_val)

        return actor_loss, critic_loss_val

    def _soft_update_targets(self) -> None:
        """Soft update target networks including second critic."""
        super()._soft_update_targets()

        # Update second critic target
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

    def _get_agent_state(self) -> Dict[str, Any]:
        """Get agent state including second critic."""
        state = super()._get_agent_state()
        state.update({
            'critic2_state': self.critic2.state_dict(),
            'critic2_target_state': self.critic2_target.state_dict(),
            'critic2_optimizer_state': self.critic2_optimizer.state_dict(),
            'policy_update_counter': self.policy_update_counter
        })
        return state

    def _restore_agent_state(self, state: Dict[str, Any]) -> None:
        """Restore agent state including second critic."""
        super()._restore_agent_state(state)
        self.critic2.load_state_dict(state['critic2_state'])
        self.critic2_target.load_state_dict(state['critic2_target_state'])
        self.critic2_optimizer.load_state_dict(state['critic2_optimizer_state'])
        self.policy_update_counter = state.get('policy_update_counter', 0)

    def _save_model_weights(self, filepath: str) -> None:
        """Save model weights including second critic."""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict()
        }, f"{filepath}.pth")

    def _load_model_weights(self, filepath: str) -> None:
        """Load model weights including second critic."""
        checkpoint = torch.load(f"{filepath}.pth", map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])


def create_actor_critic_agent(agent_type: str,
                             state_dim: int,
                             action_dim: int,
                             config: ActorCriticConfig = None) -> ActorCriticAgent:
    """
    Factory function to create actor-critic agents.

    Args:
        agent_type: Type of agent ('ddpg', 'td3', 'sac')
        state_dim: State space dimension
        action_dim: Action space dimension
        config: Agent configuration

    Returns:
        Actor-critic agent instance
    """
    agent_type = agent_type.lower()

    if agent_type == 'ddpg':
        return DDPGAgent(state_dim, action_dim, config)
    elif agent_type == 'td3':
        return TD3Agent(state_dim, action_dim, config)
    else:
        raise ValueError(f"Unknown actor-critic agent type: {agent_type}")
