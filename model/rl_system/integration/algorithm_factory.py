"""
Advanced Algorithm Factory and Integration System.

This module provides a comprehensive factory pattern for creating and managing
RL algorithms with enhanced integration capabilities, Context7 optimizations,
and seamless algorithm switching.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

from ..actor_critic.a3c import A3CAgent
from ..actor_critic.continuous_control import DDPGAgent, TD3Agent
from ..actor_critic.ppo import PPOAgent
from ..actor_critic.sac import SACAgent
from ..config.optimal_hyperparameters import AlgorithmConfig, get_optimal_config
from ..exploration.icm import create_curiosity_driven_agent
from ..policy_based.policy_gradients import A2CAgent, REINFORCEAgent

# Import all algorithm types
from ..value_based.dqn_family import (
    DoubleDQNAgent,
    DQNAgent,
    DuelingDQNAgent,
    RainbowDQNAgent,
)

logger = logging.getLogger(__name__)


class AlgorithmType(Enum):
    """Enumeration of supported algorithm types."""

    # Value-based methods
    DQN = "dqn"
    DOUBLE_DQN = "double_dqn"
    DUELING_DQN = "dueling_dqn"
    RAINBOW_DQN = "rainbow_dqn"

    # Policy-based methods
    REINFORCE = "reinforce"
    A2C = "a2c"

    # Actor-critic methods
    DDPG = "ddpg"
    TD3 = "td3"
    PPO = "ppo"
    SAC = "sac"
    A3C = "a3c"

    # Exploration-enhanced variants
    DQN_ICM = "dqn_icm"
    PPO_ICM = "ppo_icm"
    SAC_ICM = "sac_icm"


@dataclass
class AlgorithmSpec:
    """Specification for algorithm creation."""
    algorithm_type: AlgorithmType
    state_dim: int
    action_dim: int
    config: Optional[Dict[str, Any]] = None
    device: str = 'cpu'
    exploration_enhancement: bool = False
    custom_config: Optional[AlgorithmConfig] = None


class BaseAgentInterface(ABC):
    """Abstract base interface for all RL agents."""

    @abstractmethod
    def get_action(self, state: np.ndarray) -> Union[int, np.ndarray]:
        """Get action from the agent."""
        pass

    @abstractmethod
    def store_transition(self, state: np.ndarray, action: Union[int, np.ndarray],
                        reward: float, next_state: np.ndarray, done: bool):
        """Store a transition in the agent's memory."""
        pass

    @abstractmethod
    def update(self) -> Dict[str, float]:
        """Update the agent's parameters."""
        pass

    @abstractmethod
    def save_state(self, filepath: str):
        """Save agent state."""
        pass

    @abstractmethod
    def load_state(self, filepath: str):
        """Load agent state."""
        pass


class AgentWrapper:
    """
    Universal wrapper for all RL agents to provide consistent interface.
    """

    def __init__(self, agent: Any, algorithm_type: AlgorithmType, spec: AlgorithmSpec):
        """
        Initialize agent wrapper.

        Args:
            agent: The wrapped RL agent
            algorithm_type: Type of algorithm
            spec: Algorithm specification used for creation
        """
        self.agent = agent
        self.algorithm_type = algorithm_type
        self.spec = spec
        self.training_history = []
        self.performance_metrics = {}

        logger.info(f"AgentWrapper created for {algorithm_type.value}")

    def get_action(self, state: np.ndarray) -> Union[int, np.ndarray]:
        """Get action from wrapped agent."""
        return self.agent.get_action(state)

    def store_transition(self, state: np.ndarray, action: Union[int, np.ndarray],
                        reward: float, next_state: np.ndarray, done: bool):
        """Store transition in wrapped agent."""
        self.agent.store_transition(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """Update wrapped agent and track metrics."""
        losses = self.agent.update()

        # Track training history
        if isinstance(losses, dict):
            self.training_history.append(losses)

        return losses

    def save_state(self, filepath: str):
        """Save wrapped agent state with metadata."""
        # Save agent state
        if hasattr(self.agent, 'save_state'):
            self.agent.save_state(filepath)

        # Save wrapper metadata
        metadata = {
            'algorithm_type': self.algorithm_type.value,
            'spec': {
                'state_dim': self.spec.state_dim,
                'action_dim': self.spec.action_dim,
                'device': self.spec.device,
                'exploration_enhancement': self.spec.exploration_enhancement
            },
            'training_history': self.training_history,
            'performance_metrics': self.performance_metrics
        }

        metadata_filepath = filepath.replace('.pth', '_metadata.json')
        import json
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Agent state and metadata saved to {filepath}")

    def load_state(self, filepath: str):
        """Load wrapped agent state with metadata."""
        # Load agent state
        if hasattr(self.agent, 'load_state'):
            self.agent.load_state(filepath)

        # Load wrapper metadata
        metadata_filepath = filepath.replace('.pth', '_metadata.json')
        try:
            import json
            with open(metadata_filepath, 'r') as f:
                metadata = json.load(f)

            self.training_history = metadata.get('training_history', [])
            self.performance_metrics = metadata.get('performance_metrics', {})

            logger.info(f"Agent state and metadata loaded from {filepath}")
        except FileNotFoundError:
            logger.warning(f"Metadata file not found: {metadata_filepath}")

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive agent information."""
        return {
            'algorithm_type': self.algorithm_type.value,
            'state_dim': self.spec.state_dim,
            'action_dim': self.spec.action_dim,
            'device': self.spec.device,
            'exploration_enhancement': self.spec.exploration_enhancement,
            'training_steps': len(self.training_history),
            'performance_metrics': self.performance_metrics,
            'has_memory': hasattr(self.agent, 'memory'),
            'memory_size': len(self.agent.memory) if hasattr(self.agent, 'memory') else 0
        }


class AlgorithmRegistry:
    """Registry for available algorithms and their creation functions."""

    _algorithms: Dict[AlgorithmType, Type] = {}
    _configurations: Dict[AlgorithmType, Dict[str, Any]] = {}

    @classmethod
    def register(cls, algorithm_type: AlgorithmType, agent_class: Type,
                default_config: Optional[Dict[str, Any]] = None):
        """Register an algorithm type with its class and default configuration."""
        cls._algorithms[algorithm_type] = agent_class
        cls._configurations[algorithm_type] = default_config or {}
        logger.info(f"Registered algorithm: {algorithm_type.value}")

    @classmethod
    def get_agent_class(cls, algorithm_type: AlgorithmType) -> Type:
        """Get agent class for algorithm type."""
        if algorithm_type not in cls._algorithms:
            raise ValueError(f"Algorithm type {algorithm_type.value} not registered")
        return cls._algorithms[algorithm_type]

    @classmethod
    def get_default_config(cls, algorithm_type: AlgorithmType) -> Dict[str, Any]:
        """Get default configuration for algorithm type."""
        return cls._configurations.get(algorithm_type, {}).copy()

    @classmethod
    def list_algorithms(cls) -> List[AlgorithmType]:
        """List all registered algorithms."""
        return list(cls._algorithms.keys())


class AdvancedAlgorithmFactory:
    """
    Advanced factory for creating and managing RL algorithms with enhanced features.
    """

    def __init__(self):
        """Initialize the advanced algorithm factory."""
        self._initialize_registry()
        self.created_agents = {}
        self.performance_tracker = {}

        logger.info("AdvancedAlgorithmFactory initialized")

    def _initialize_registry(self):
        """Initialize the algorithm registry with all available algorithms."""

        # Value-based algorithms
        AlgorithmRegistry.register(AlgorithmType.DQN, DQNAgent)
        AlgorithmRegistry.register(AlgorithmType.DOUBLE_DQN, DoubleDQNAgent)
        AlgorithmRegistry.register(AlgorithmType.DUELING_DQN, DuelingDQNAgent)
        AlgorithmRegistry.register(AlgorithmType.RAINBOW_DQN, RainbowDQNAgent)

        # Policy-based algorithms
        AlgorithmRegistry.register(AlgorithmType.REINFORCE, REINFORCEAgent)
        AlgorithmRegistry.register(AlgorithmType.A2C, A2CAgent)

        # Actor-critic algorithms
        AlgorithmRegistry.register(AlgorithmType.DDPG, DDPGAgent)
        AlgorithmRegistry.register(AlgorithmType.TD3, TD3Agent)
        AlgorithmRegistry.register(AlgorithmType.PPO, PPOAgent)
        AlgorithmRegistry.register(AlgorithmType.SAC, SACAgent)
        AlgorithmRegistry.register(AlgorithmType.A3C, A3CAgent)

        logger.info("Algorithm registry initialized with all available algorithms")

    def create_agent(self, spec: AlgorithmSpec, agent_id: Optional[str] = None) -> AgentWrapper:
        """
        Create an agent according to the specification.

        Args:
            spec: Algorithm specification
            agent_id: Optional identifier for the agent

        Returns:
            Wrapped agent instance
        """
        logger.info(f"Creating agent: {spec.algorithm_type.value}")

        # Handle exploration-enhanced variants
        if spec.algorithm_type in [AlgorithmType.DQN_ICM, AlgorithmType.PPO_ICM, AlgorithmType.SAC_ICM]:
            return self._create_exploration_enhanced_agent(spec, agent_id)

        # Get agent class
        agent_class = AlgorithmRegistry.get_agent_class(spec.algorithm_type)

        # Prepare configuration
        config = self._prepare_config(spec)

        # Create agent instance
        agent = agent_class(
            state_dim=spec.state_dim,
            action_dim=spec.action_dim,
            device=spec.device,
            **config
        )

        # Wrap agent
        wrapped_agent = AgentWrapper(agent, spec.algorithm_type, spec)

        # Store agent reference
        if agent_id:
            self.created_agents[agent_id] = wrapped_agent

        logger.info(f"Successfully created {spec.algorithm_type.value} agent")
        return wrapped_agent

    def _create_exploration_enhanced_agent(self, spec: AlgorithmSpec,
                                         agent_id: Optional[str] = None) -> AgentWrapper:
        """Create an exploration-enhanced agent with ICM."""

        # Map ICM variants to base algorithms
        base_algorithm_map = {
            AlgorithmType.DQN_ICM: AlgorithmType.DQN,
            AlgorithmType.PPO_ICM: AlgorithmType.PPO,
            AlgorithmType.SAC_ICM: AlgorithmType.SAC
        }

        base_algorithm = base_algorithm_map[spec.algorithm_type]

        # Create base agent
        base_spec = AlgorithmSpec(
            algorithm_type=base_algorithm,
            state_dim=spec.state_dim,
            action_dim=spec.action_dim,
            config=spec.config,
            device=spec.device,
            custom_config=spec.custom_config
        )

        base_agent_class = AlgorithmRegistry.get_agent_class(base_algorithm)
        config = self._prepare_config(base_spec)

        base_agent = base_agent_class(
            state_dim=spec.state_dim,
            action_dim=spec.action_dim,
            device=spec.device,
            **config
        )

        # Enhance with ICM
        enhanced_agent = create_curiosity_driven_agent(
            base_agent=base_agent,
            state_dim=spec.state_dim,
            action_dim=spec.action_dim,
            device=spec.device
        )

        # Wrap enhanced agent
        wrapped_agent = AgentWrapper(enhanced_agent, spec.algorithm_type, spec)

        # Store agent reference
        if agent_id:
            self.created_agents[agent_id] = wrapped_agent

        logger.info(f"Successfully created exploration-enhanced {spec.algorithm_type.value} agent")
        return wrapped_agent

    def _prepare_config(self, spec: AlgorithmSpec) -> Dict[str, Any]:
        """Prepare configuration for agent creation."""
        config = {}

        # Start with optimal configuration if available
        try:
            optimal_config = get_optimal_config(spec.algorithm_type.value)
            if optimal_config:
                config.update(optimal_config.to_dict())
        except Exception as e:
            logger.warning(f"Could not load optimal config for {spec.algorithm_type.value}: {e}")

        # Override with custom configuration
        if spec.custom_config:
            config.update(spec.custom_config.to_dict())

        # Override with spec configuration
        if spec.config:
            config.update(spec.config)

        return config

    def create_multiple_agents(self, specs: List[AlgorithmSpec],
                             agent_ids: Optional[List[str]] = None) -> List[AgentWrapper]:
        """
        Create multiple agents from specifications.

        Args:
            specs: List of algorithm specifications
            agent_ids: Optional list of agent identifiers

        Returns:
            List of wrapped agent instances
        """
        agents = []

        for i, spec in enumerate(specs):
            agent_id = agent_ids[i] if agent_ids and i < len(agent_ids) else None
            agent = self.create_agent(spec, agent_id)
            agents.append(agent)

        logger.info(f"Created {len(agents)} agents")
        return agents

    def get_agent(self, agent_id: str) -> Optional[AgentWrapper]:
        """Get agent by ID."""
        return self.created_agents.get(agent_id)

    def list_agents(self) -> Dict[str, str]:
        """List all created agents with their types."""
        return {agent_id: agent.algorithm_type.value
                for agent_id, agent in self.created_agents.items()}

    def compare_agents(self, agent_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Compare multiple agents' information."""
        comparison = {}

        for agent_id in agent_ids:
            if agent_id in self.created_agents:
                comparison[agent_id] = self.created_agents[agent_id].get_info()

        return comparison

    def create_ensemble(self, specs: List[AlgorithmSpec],
                       ensemble_id: str) -> 'EnsembleAgent':
        """Create an ensemble of agents."""
        agents = self.create_multiple_agents(specs)
        ensemble = EnsembleAgent(agents, ensemble_id)

        self.created_agents[ensemble_id] = ensemble
        logger.info(f"Created ensemble with {len(agents)} agents: {ensemble_id}")

        return ensemble

    def get_recommendations(self, environment_type: str,
                          complexity: str = "medium") -> List[AlgorithmType]:
        """
        Get algorithm recommendations based on environment characteristics.

        Args:
            environment_type: Type of environment ("discrete", "continuous", "mixed")
            complexity: Problem complexity ("simple", "medium", "complex")

        Returns:
            List of recommended algorithm types
        """
        recommendations = []

        if environment_type == "discrete":
            if complexity == "simple":
                recommendations = [AlgorithmType.DQN, AlgorithmType.REINFORCE]
            elif complexity == "medium":
                recommendations = [AlgorithmType.DOUBLE_DQN, AlgorithmType.A2C, AlgorithmType.PPO]
            else:  # complex
                recommendations = [AlgorithmType.RAINBOW_DQN, AlgorithmType.PPO_ICM, AlgorithmType.A3C]

        elif environment_type == "continuous":
            if complexity == "simple":
                recommendations = [AlgorithmType.DDPG, AlgorithmType.A2C]
            elif complexity == "medium":
                recommendations = [AlgorithmType.TD3, AlgorithmType.PPO, AlgorithmType.SAC]
            else:  # complex
                recommendations = [AlgorithmType.SAC_ICM, AlgorithmType.PPO_ICM, AlgorithmType.TD3]

        else:  # mixed
            recommendations = [AlgorithmType.PPO, AlgorithmType.SAC, AlgorithmType.A3C]

        logger.info(f"Recommendations for {environment_type}/{complexity}: {[r.value for r in recommendations]}")
        return recommendations


class EnsembleAgent:
    """
    Ensemble agent that combines multiple RL agents.
    """

    def __init__(self, agents: List[AgentWrapper], ensemble_id: str):
        """
        Initialize ensemble agent.

        Args:
            agents: List of wrapped agents
            ensemble_id: Identifier for the ensemble
        """
        self.agents = agents
        self.ensemble_id = ensemble_id
        self.weights = np.ones(len(agents)) / len(agents)  # Equal weights initially
        self.performance_history = []

        logger.info(f"EnsembleAgent created with {len(agents)} agents")

    def get_action(self, state: np.ndarray) -> Union[int, np.ndarray]:
        """Get action from ensemble using weighted voting."""
        actions = []

        for agent in self.agents:
            action = agent.get_action(state)
            actions.append(action)

        # For discrete actions, use weighted voting
        if isinstance(actions[0], (int, np.integer)):
            # Weighted voting for discrete actions
            action_counts = {}
            for i, action in enumerate(actions):
                action_counts[action] = action_counts.get(action, 0) + self.weights[i]

            return max(action_counts, key=action_counts.get)

        else:
            # Weighted average for continuous actions
            weighted_action = np.zeros_like(actions[0])
            for i, action in enumerate(actions):
                weighted_action += self.weights[i] * action

            return weighted_action

    def store_transition(self, state: np.ndarray, action: Union[int, np.ndarray],
                        reward: float, next_state: np.ndarray, done: bool):
        """Store transition in all agents."""
        for agent in self.agents:
            agent.store_transition(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """Update all agents and adjust weights based on performance."""
        all_losses = {}

        for i, agent in enumerate(self.agents):
            losses = agent.update()

            # Store losses with agent prefix
            for key, value in losses.items():
                all_losses[f"agent_{i}_{key}"] = value

        # Update ensemble weights based on recent performance
        self._update_weights()

        return all_losses

    def _update_weights(self):
        """Update agent weights based on recent performance."""
        # Simple performance-based weighting
        # In practice, this could be more sophisticated

        recent_performance = []
        for agent in self.agents:
            if agent.training_history:
                # Use recent loss as performance indicator (lower is better)
                recent_losses = agent.training_history[-10:]
                avg_loss = np.mean([loss.get('total_loss', 0) for loss in recent_losses])
                performance = 1.0 / (1.0 + avg_loss)  # Convert loss to performance score
                recent_performance.append(performance)
            else:
                recent_performance.append(1.0)  # Default weight

        # Normalize weights
        total_performance = sum(recent_performance)
        if total_performance > 0:
            self.weights = np.array(recent_performance) / total_performance

        logger.debug(f"Ensemble weights updated: {self.weights}")

    def get_info(self) -> Dict[str, Any]:
        """Get ensemble information."""
        return {
            'ensemble_id': self.ensemble_id,
            'num_agents': len(self.agents),
            'agent_types': [agent.algorithm_type.value for agent in self.agents],
            'weights': self.weights.tolist(),
            'performance_history_length': len(self.performance_history)
        }


# Factory instance for global use
algorithm_factory = AdvancedAlgorithmFactory()


# Convenience functions
def create_agent(algorithm_type: Union[str, AlgorithmType],
                state_dim: int,
                action_dim: int,
                device: str = 'cpu',
                config: Optional[Dict[str, Any]] = None,
                exploration_enhancement: bool = False) -> AgentWrapper:
    """
    Convenience function to create a single agent.

    Args:
        algorithm_type: Algorithm type (string or enum)
        state_dim: State space dimension
        action_dim: Action space dimension
        device: Device to run on
        config: Optional configuration
        exploration_enhancement: Whether to enhance with ICM

    Returns:
        Wrapped agent instance
    """
    if isinstance(algorithm_type, str):
        algorithm_type = AlgorithmType(algorithm_type)

    spec = AlgorithmSpec(
        algorithm_type=algorithm_type,
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        device=device,
        exploration_enhancement=exploration_enhancement
    )

    return algorithm_factory.create_agent(spec)


def get_algorithm_recommendations(environment_type: str = "discrete",
                                complexity: str = "medium") -> List[str]:
    """
    Get algorithm recommendations for an environment.

    Args:
        environment_type: Environment type
        complexity: Problem complexity

    Returns:
        List of recommended algorithm names
    """
    recommendations = algorithm_factory.get_recommendations(environment_type, complexity)
    return [rec.value for rec in recommendations]


def list_available_algorithms() -> List[str]:
    """List all available algorithm types."""
    return [alg.value for alg in AlgorithmRegistry.list_algorithms()]
