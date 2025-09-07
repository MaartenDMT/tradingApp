"""
Actor-Critic RL Algorithms.

This package contains actor-critic methods for reinforcement learning,
including policy gradient with value function baselines.
"""

from .a3c import A3CAgent, A3CNetwork, A3CWorker, SharedAdam, create_a3c_agent
from .continuous_control import (
                                 Actor,
                                 Critic,
                                 DDPGAgent,
                                 GaussianActor,
                                 TD3Agent,
                                 create_ddpg_agent,
                                 create_td3_agent,
)
from .sac import GaussianActor as SACGaussianActor
from .sac import SACAgent, SACCritic, create_sac_agent

__all__ = [
    # Continuous Control
    'Actor',
    'Critic',
    'GaussianActor',
    'DDPGAgent',
    'TD3Agent',
    'create_ddpg_agent',
    'create_td3_agent',

    # SAC
    'SACAgent',
    'SACGaussianActor',
    'SACCritic',
    'create_sac_agent',

    # A3C
    'A3CAgent',
    'A3CNetwork',
    'A3CWorker',
    'SharedAdam',
    'create_a3c_agent'
]
