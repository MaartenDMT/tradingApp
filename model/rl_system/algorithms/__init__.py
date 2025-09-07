"""Algorithm implementations for the RL system."""

# Value-based algorithms
# Actor-critic algorithms
from .actor_critic.continuous_control import (
                                              ActorCriticConfig,
                                              DDPGAgent,
                                              TD3Agent,
                                              create_actor_critic_agent,
)

# Exploration algorithms
from .exploration import CuriosityDrivenAgent, ICMModule, create_curiosity_driven_agent

# Policy-based algorithms
from .policy_based.policy_gradients import (
                                              A2CAgent,
                                              PolicyGradientConfig,
                                              REINFORCEAgent,
                                              create_policy_gradient_agent,
)
from .value_based.dqn_family import (
                                              DoubleDQNAgent,
                                              DQNAgent,
                                              DQNConfig,
                                              DuelingDQNAgent,
                                              RainbowDQNAgent,
                                              create_dqn_agent,
)
from .value_based.tabular_methods import (
                                              ExpectedSARSAAgent,
                                              MonteCarloAgent,
                                              QLearningAgent,
                                              SARSAAgent,
                                              TabularConfig,
                                              create_tabular_agent,
)

__all__ = [
    # DQN family
    'DQNAgent',
    'DoubleDQNAgent',
    'DuelingDQNAgent',
    'RainbowDQNAgent',
    'DQNConfig',
    'create_dqn_agent',

    # Tabular methods
    'QLearningAgent',
    'SARSAAgent',
    'MonteCarloAgent',
    'ExpectedSARSAAgent',
    'TabularConfig',
    'create_tabular_agent',

    # Policy gradients
    'REINFORCEAgent',
    'A2CAgent',
    'PolicyGradientConfig',
    'create_policy_gradient_agent',

    # Actor-critic
    'DDPGAgent',
    'TD3Agent',
    'ActorCriticConfig',
    'create_actor_critic_agent',

    # Exploration
    'ICMModule',
    'CuriosityDrivenAgent',
    'create_curiosity_driven_agent',
]
