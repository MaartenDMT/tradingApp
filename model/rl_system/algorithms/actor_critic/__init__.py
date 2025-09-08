"""
Actor-Critic RL Algorithms.

This package contains actor-critic methods for reinforcement learning,
including policy gradient with value function baselines.
"""

from .a3c import A3CAgent, A3CNetwork, A3CWorker, SharedAdam, create_a3c_agent
from .continuous_control import (Actor, Critic, DDPGAgent, GaussianActor,
                                 TD3Agent, create_actor_critic_agent)
from .ddg import (DDGActor, DDGAgent, DDGConfig, DDGCritic,
                  OrnsteinUhlenbeckNoise)
from .mapddg import (MAPDDGActor, MAPDDGAgent, MAPDDGCentralizedCritic,
                     MAPDDGConfig, MultiAgentReplayBuffer)
from .sac import GaussianActor as SACGaussianActor
from .sac import SACAgent, SACCritic, create_sac_agent
from .simple_ac import (SimpleACConfig, SimpleActorCriticAgent,
                        SimpleActorCriticNetwork)
from .td3 import TD3Actor
from .td3 import TD3Agent as EnhancedTD3Agent
from .td3 import TD3Config, TD3Critic

__all__ = [
    # Continuous Control (original)
    'Actor',
    'Critic',
    'GaussianActor',
    'DDPGAgent',
    'TD3Agent',
    'create_actor_critic_agent',

    # Enhanced DDG implementation
    'DDGAgent',
    'DDGActor',
    'DDGCritic',
    'DDGConfig',
    'OrnsteinUhlenbeckNoise',

    # Enhanced TD3 implementation
    'EnhancedTD3Agent',
    'TD3Actor',
    'TD3Critic',
    'TD3Config',

    # Simple Actor-Critic
    'SimpleActorCriticAgent',
    'SimpleActorCriticNetwork',
    'SimpleACConfig',

    # Multi-Agent DDPG
    'MAPDDGAgent',
    'MAPDDGActor',
    'MAPDDGCentralizedCritic',
    'MAPDDGConfig',
    'MultiAgentReplayBuffer',

    # SAC
    'SACAgent',
    'SACCritic',
    'SACGaussianActor',
    'create_sac_agent',

    # A3C
    'A3CAgent',
    'A3CNetwork',
    'A3CWorker',
    'SharedAdam',
    'create_a3c_agent',
]
