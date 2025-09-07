"""
Asynchronous Actor-Critic (A3C) Implementation.

This module implements the A3C algorithm with multiple parallel workers
for asynchronous training across different environments. A3C is particularly
useful for training on multiple market scenarios simultaneously.
"""

import logging
import time
from queue import Queue
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

from ..core.base_agents import ActorCriticAgent

logger = logging.getLogger(__name__)


class SharedAdam(optim.Adam):
    """
    Shared Adam optimizer for A3C that allows parameter sharing across processes.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        # Initialize shared state
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_step'] = torch.zeros(1).share_memory_()
                state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()

    def step(self, closure=None):
        """Override step to handle shared state."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['shared_step'] += 1

                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['shared_step'].item()
                bias_correction2 = 1 - beta2 ** state['shared_step'].item()
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)


class A3CNetwork(nn.Module):
    """
    Actor-Critic network for A3C with shared features.

    The network has a shared trunk followed by separate actor and critic heads.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [128, 128],
                 continuous: bool = False):
        super(A3CNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous

        # Shared feature extraction
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(prev_dim, action_dim)
            self.actor_std = nn.Linear(prev_dim, action_dim)
        else:
            self.actor = nn.Linear(prev_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(prev_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, state):
        """Forward pass through the network."""
        shared_features = self.shared_layers(state)

        if self.continuous:
            action_mean = torch.tanh(self.actor_mean(shared_features))
            action_std = F.softplus(self.actor_std(shared_features)) + 1e-5
            value = self.critic(shared_features)
            return action_mean, action_std, value
        else:
            action_logits = self.actor(shared_features)
            action_probs = F.softmax(action_logits, dim=-1)
            value = self.critic(shared_features)
            return action_probs, value

    def get_action_and_value(self, state):
        """Get action and value for given state."""
        if self.continuous:
            action_mean, action_std, value = self.forward(state)
            distribution = Normal(action_mean, action_std)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(dim=-1)
            return action, log_prob, value
        else:
            action_probs, value = self.forward(state)
            distribution = Categorical(action_probs)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            return action, log_prob, value

    def evaluate_actions(self, state, action):
        """Evaluate actions for given states."""
        if self.continuous:
            action_mean, action_std, value = self.forward(state)
            distribution = Normal(action_mean, action_std)
            log_prob = distribution.log_prob(action).sum(dim=-1)
            entropy = distribution.entropy().sum(dim=-1)
            return log_prob, entropy, value
        else:
            action_probs, value = self.forward(state)
            distribution = Categorical(action_probs)
            log_prob = distribution.log_prob(action)
            entropy = distribution.entropy()
            return log_prob, entropy, value


class A3CWorker(mp.Process):
    """
    A3C worker process that runs asynchronous training.

    Each worker has its own environment and local network, but shares
    the global network parameters.
    """

    def __init__(self,
                 worker_id: int,
                 global_network: A3CNetwork,
                 optimizer: SharedAdam,
                 environment_fn,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 value_coeff: float = 0.5,
                 entropy_coeff: float = 0.01,
                 max_grad_norm: float = 0.5,
                 update_frequency: int = 20,
                 max_episodes: int = 1000,
                 results_queue: Optional[Queue] = None):
        super(A3CWorker, self).__init__()

        self.worker_id = worker_id
        self.global_network = global_network
        self.optimizer = optimizer
        self.environment_fn = environment_fn
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.update_frequency = update_frequency
        self.max_episodes = max_episodes
        self.results_queue = results_queue

        # Create local network
        self.local_network = A3CNetwork(
            global_network.state_dim,
            global_network.action_dim,
            continuous=global_network.continuous
        )

        # Episode tracking
        self.episode_count = 0
        self.episode_rewards = []
        self.running = True

    def sync_with_global(self):
        """Synchronize local network with global network."""
        self.local_network.load_state_dict(self.global_network.state_dict())

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[i + 1]

            delta = rewards[i] + self.gamma * next_val * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def run(self):
        """Main worker loop."""
        torch.manual_seed(self.worker_id + int(time.time()))
        np.random.seed(self.worker_id + int(time.time()))

        # Create environment
        env = self.environment_fn()

        logger.info(f"A3C Worker {self.worker_id} started")

        while self.running and self.episode_count < self.max_episodes:
            # Sync with global network
            self.sync_with_global()

            # Collect trajectory
            states, actions, rewards, values, log_probs, dones = self.collect_trajectory(env)

            if len(states) > 0:
                # Compute advantages and returns
                next_value = 0  # Assuming episode ended or using 0 for terminal
                if not dones[-1]:  # If episode didn't end, get next value
                    with torch.no_grad():
                        state_tensor = torch.tensor(states[-1], dtype=torch.float32).unsqueeze(0)
                        if self.local_network.continuous:
                            _, _, next_value = self.local_network(state_tensor)
                        else:
                            _, next_value = self.local_network(state_tensor)
                        next_value = next_value.item()

                advantages, returns = self.compute_gae(rewards, values, dones, next_value)

                # Update global network
                self.update_global_network(states, actions, log_probs, advantages, returns)

                # Track episode reward
                episode_reward = sum(rewards)
                self.episode_rewards.append(episode_reward)
                self.episode_count += 1

                # Send results to main process
                if self.results_queue:
                    self.results_queue.put({
                        'worker_id': self.worker_id,
                        'episode': self.episode_count,
                        'reward': episode_reward,
                        'length': len(rewards)
                    })

                if self.episode_count % 100 == 0:
                    avg_reward = np.mean(self.episode_rewards[-100:])
                    logger.info(f"Worker {self.worker_id} - Episode {self.episode_count}, "
                               f"Avg Reward: {avg_reward:.2f}")

        logger.info(f"A3C Worker {self.worker_id} finished")

    def collect_trajectory(self, env):
        """Collect a trajectory for training."""
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []

        state = env.reset()
        done = False
        step_count = 0

        while not done and step_count < self.update_frequency:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, value = self.local_network.get_action_and_value(state_tensor)
                action_np = action.cpu().numpy().squeeze()
                log_prob_np = log_prob.cpu().numpy().squeeze()
                value_np = value.cpu().numpy().squeeze()

            next_state, reward, done, _ = env.step(action_np)

            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            values.append(value_np)
            log_probs.append(log_prob_np)
            dones.append(done)

            state = next_state
            step_count += 1

        return states, actions, rewards, values, log_probs, dones

    def update_global_network(self, states, actions, log_probs, advantages, returns):
        """Update the global network using collected trajectory."""
        # Convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        # Evaluate actions with local network
        current_log_probs, entropy, values = self.local_network.evaluate_actions(
            states_tensor, actions_tensor
        )

        # Compute losses
        policy_loss = -(current_log_probs * advantages_tensor).mean()
        value_loss = F.mse_loss(values.squeeze(), returns_tensor)
        entropy_loss = -entropy.mean()

        total_loss = (policy_loss +
                     self.value_coeff * value_loss +
                     self.entropy_coeff * entropy_loss)

        # Update global network
        self.optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), self.max_grad_norm)

        # Copy gradients to global network
        for local_param, global_param in zip(self.local_network.parameters(),
                                           self.global_network.parameters()):
            if global_param.grad is None:
                global_param._grad = local_param.grad.clone()
            else:
                global_param._grad += local_param.grad

        self.optimizer.step()

    def stop(self):
        """Stop the worker."""
        self.running = False


class A3CAgent(ActorCriticAgent):
    """
    Asynchronous Actor-Critic (A3C) Agent.

    Implements A3C with multiple parallel workers for asynchronous training.
    Each worker runs independently and updates a shared global network.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 continuous: bool = False,
                 num_workers: int = 4,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 value_coeff: float = 0.5,
                 entropy_coeff: float = 0.01,
                 max_grad_norm: float = 0.5,
                 update_frequency: int = 20,
                 hidden_dims: List[int] = [128, 128]):
        """
        Initialize A3C agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            continuous: Whether action space is continuous
            num_workers: Number of parallel workers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            value_coeff: Coefficient for value loss
            entropy_coeff: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            update_frequency: Frequency of network updates
            hidden_dims: Hidden layer dimensions
        """
        super().__init__(state_dim, action_dim)

        self.continuous = continuous
        self.num_workers = num_workers
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.update_frequency = update_frequency

        # Create global network
        self.global_network = A3CNetwork(
            state_dim, action_dim, hidden_dims, continuous
        )
        self.global_network.share_memory()  # Enable parameter sharing

        # Create shared optimizer
        self.optimizer = SharedAdam(self.global_network.parameters(), lr=learning_rate)

        # Worker management
        self.workers = []
        self.results_queue = Queue()
        self.training = False

        # Training statistics
        self.episode_rewards = []
        self.worker_stats = {}

        logger.info(f"Initialized A3C agent - State dim: {state_dim}, "
                   f"Action dim: {action_dim}, Workers: {num_workers}")

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action for given state using the global network.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            if training:
                action, _, _ = self.global_network.get_action_and_value(state_tensor)
            else:
                # For evaluation, use deterministic policy
                if self.continuous:
                    action_mean, _, _ = self.global_network(state_tensor)
                    action = action_mean
                else:
                    action_probs, _ = self.global_network(state_tensor)
                    action = torch.argmax(action_probs, dim=-1)

        return action.cpu().numpy().squeeze()

    def train(self,
              environment_fn,
              max_episodes: int = 1000,
              log_frequency: int = 100) -> Dict[str, Any]:
        """
        Train the A3C agent using multiple workers.

        Args:
            environment_fn: Function that creates environment instances
            max_episodes: Maximum episodes per worker
            log_frequency: Frequency of logging

        Returns:
            Training statistics
        """
        logger.info(f"Starting A3C training with {self.num_workers} workers")
        self.training = True

        # Create and start workers
        for worker_id in range(self.num_workers):
            worker = A3CWorker(
                worker_id=worker_id,
                global_network=self.global_network,
                optimizer=self.optimizer,
                environment_fn=environment_fn,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                value_coeff=self.value_coeff,
                entropy_coeff=self.entropy_coeff,
                max_grad_norm=self.max_grad_norm,
                update_frequency=self.update_frequency,
                max_episodes=max_episodes,
                results_queue=self.results_queue
            )
            worker.start()
            self.workers.append(worker)

        # Monitor training
        total_episodes = 0
        start_time = time.time()

        try:
            while self.training and any(w.is_alive() for w in self.workers):
                try:
                    # Get results from workers
                    result = self.results_queue.get(timeout=1.0)

                    worker_id = result['worker_id']
                    reward = result['reward']

                    # Update statistics
                    if worker_id not in self.worker_stats:
                        self.worker_stats[worker_id] = []
                    self.worker_stats[worker_id].append(reward)
                    self.episode_rewards.append(reward)

                    total_episodes += 1

                    # Log progress
                    if total_episodes % log_frequency == 0:
                        avg_reward = np.mean(self.episode_rewards[-100:])
                        elapsed_time = time.time() - start_time
                        logger.info(f"A3C Training - Episodes: {total_episodes}, "
                                   f"Avg Reward: {avg_reward:.2f}, "
                                   f"Time: {elapsed_time:.1f}s")

                except Exception:
                    continue

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        finally:
            # Stop all workers
            for worker in self.workers:
                worker.stop()
                worker.join(timeout=1.0)
                if worker.is_alive():
                    worker.terminate()

        self.training = False
        training_time = time.time() - start_time

        # Compile results
        results = {
            'total_episodes': total_episodes,
            'training_time': training_time,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'max_reward': np.max(self.episode_rewards) if self.episode_rewards else 0,
            'worker_stats': {k: {'avg': np.mean(v), 'episodes': len(v)}
                           for k, v in self.worker_stats.items()}
        }

        logger.info(f"A3C training completed - Total episodes: {total_episodes}, "
                   f"Avg reward: {results['avg_reward']:.2f}")

        return results

    def save_model(self, filepath: str):
        """Save the global network."""
        torch.save({
            'global_network_state_dict': self.global_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'continuous': self.continuous,
                'num_workers': self.num_workers,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda
            }
        }, filepath)
        logger.info(f"A3C model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load the global network."""
        checkpoint = torch.load(filepath)
        self.global_network.load_state_dict(checkpoint['global_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"A3C model loaded from {filepath}")

    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'total_episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'num_workers': self.num_workers,
            'worker_stats': self.worker_stats
        }


# Factory function for easy creation
def create_a3c_agent(state_dim: int,
                     action_dim: int,
                     continuous: bool = False,
                     **kwargs) -> A3CAgent:
    """
    Factory function to create an A3C agent with sensible defaults.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        continuous: Whether action space is continuous
        **kwargs: Additional configuration parameters

    Returns:
        Configured A3C agent
    """
    return A3CAgent(state_dim, action_dim, continuous, **kwargs)
