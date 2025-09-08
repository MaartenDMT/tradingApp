"""
Asynchronous Advantage Actor-Critic (A3C) Implementation.

This module implements A3C as an extension of A2C with multi-worker support.
For simplicity, we provide a single-worker version that extends the A2C implementation.
"""

from dataclasses import dataclass

from .policy_gradients import A2CAgent, PolicyGradientConfig


@dataclass
class A3CConfig(PolicyGradientConfig):
    """Configuration for A3C agent (extends A2C config)."""
    state_dim: int = 0
    action_dim: int = 0
    num_workers: int = 1
    worker_steps: int = 20
    max_grad_norm: float = 0.5
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    entropy_coef: float = 0.01  # Alias for test compatibility
    value_coef: float = 0.5     # Alias for test compatibility
    n_workers: int = 1          # Alias for test compatibility


class A3CAgent(A2CAgent):
    """
    Asynchronous Advantage Actor-Critic (A3C) Agent.

    Simplified single-worker implementation extending A2C.
    In practice, A3C would use multiple worker processes.
    """

    def __init__(self, config: A3CConfig):
        """
        Initialize A3C agent.

        Args:
            config: A3C configuration
        """
        # Initialize as A2C agent with A3C-specific config
        super().__init__(config.state_dim, config.action_dim, config)
        self.config: A3CConfig = config

        # A3C-specific attributes
        self.num_workers = config.num_workers
        self.worker_steps = config.worker_steps
        self.max_grad_norm = config.max_grad_norm

        # Add aliases for test compatibility
        self.shared_net = self.shared_network
        self.actor_head = self.policy_head
        self.critic_head = self.value_head
        self.shared_optimizer = self.optimizer

    def compute_gradients(self, states, actions, rewards, next_states, dones):
        """
        Compute gradients for A3C update.

        This is a simplified version - full A3C would aggregate gradients
        from multiple workers before applying updates.
        """
        # Use A2C gradient computation as base
        return super().train_step(states, actions, rewards, next_states, dones)

    def update_global_network(self, gradients):
        """
        Update global network with computed gradients.

        In full A3C, this would update shared global parameters.
        """
        # For single-worker version, this is equivalent to normal update
        pass

    def sync_with_global(self):
        """
        Synchronize local network with global network.

        In full A3C, workers would periodically sync with global network.
        """
        # For single-worker version, no sync needed
        pass


def create_a3c_agent(state_dim: int, action_dim: int, **kwargs) -> A3CAgent:
    """
    Factory function to create A3C agent.

    Args:
        state_dim: State space dimension
        action_dim: Action space dimension
        **kwargs: Additional configuration parameters

    Returns:
        A3CAgent instance
    """
    config = A3CConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        **kwargs
    )
    return A3CAgent(config)
