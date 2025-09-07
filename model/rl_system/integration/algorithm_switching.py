"""
Seamless Algorithm Switching System.

This module provides capabilities for seamless switching between different RL algorithms
during training, with state transfer, performance monitoring, and automatic optimization.
"""

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .algorithm_factory import (
    AgentWrapper,
    AlgorithmSpec,
    AlgorithmType,
    algorithm_factory,
)

logger = logging.getLogger(__name__)


class SwitchTrigger(Enum):
    """Triggers for algorithm switching."""
    PERFORMANCE_THRESHOLD = "performance_threshold"
    EPISODE_COUNT = "episode_count"
    TRAINING_STEPS = "training_steps"
    MANUAL = "manual"
    ADAPTIVE = "adaptive"


@dataclass
class SwitchCondition:
    """Condition for triggering algorithm switch."""
    trigger: SwitchTrigger
    threshold: Optional[float] = None
    episodes: Optional[int] = None
    steps: Optional[int] = None
    metric: Optional[str] = None
    comparison: str = "less_than"  # "less_than", "greater_than", "equal_to"


@dataclass
class SwitchEvent:
    """Record of an algorithm switch event."""
    from_algorithm: str
    to_algorithm: str
    trigger: SwitchTrigger
    episode: int
    step: int
    performance_before: float
    performance_after: Optional[float] = None
    timestamp: str = ""


class StateTransferManager:
    """
    Manages state transfer between different algorithms.
    """

    def __init__(self):
        """Initialize state transfer manager."""
        self.transfer_strategies = {
            'value_to_value': self._transfer_value_networks,
            'policy_to_policy': self._transfer_policy_networks,
            'critic_to_critic': self._transfer_critic_networks,
            'experience_replay': self._transfer_experience_replay,
            'general': self._transfer_general_state
        }

        logger.info("StateTransferManager initialized")

    def transfer_state(self, from_agent: AgentWrapper, to_agent: AgentWrapper) -> bool:
        """
        Transfer compatible state from one agent to another.

        Args:
            from_agent: Source agent
            to_agent: Target agent

        Returns:
            Success status of transfer
        """
        logger.info(f"Transferring state from {from_agent.algorithm_type.value} to {to_agent.algorithm_type.value}")

        success = False

        try:
            # Determine transfer strategy
            strategy = self._determine_transfer_strategy(from_agent, to_agent)

            # Execute transfer
            if strategy in self.transfer_strategies:
                success = self.transfer_strategies[strategy](from_agent, to_agent)
            else:
                # Fallback to general transfer
                success = self._transfer_general_state(from_agent, to_agent)

            if success:
                logger.info("State transfer completed successfully")
            else:
                logger.warning("State transfer failed or partially completed")

        except Exception as e:
            logger.error(f"Error during state transfer: {e}")
            success = False

        return success

    def _determine_transfer_strategy(self, from_agent: AgentWrapper,
                                   to_agent: AgentWrapper) -> str:
        """Determine the best transfer strategy for the agent pair."""

        from_type = from_agent.algorithm_type
        to_type = to_agent.algorithm_type

        # Value-based to value-based
        value_based = [AlgorithmType.DQN, AlgorithmType.DOUBLE_DQN,
                      AlgorithmType.DUELING_DQN, AlgorithmType.RAINBOW_DQN]

        if from_type in value_based and to_type in value_based:
            return 'value_to_value'

        # Policy-based to policy-based
        policy_based = [AlgorithmType.REINFORCE, AlgorithmType.A2C,
                       AlgorithmType.PPO]

        if from_type in policy_based and to_type in policy_based:
            return 'policy_to_policy'

        # Actor-critic transfers
        actor_critic = [AlgorithmType.DDPG, AlgorithmType.TD3,
                       AlgorithmType.SAC, AlgorithmType.A3C]

        if from_type in actor_critic and to_type in actor_critic:
            return 'critic_to_critic'

        # Experience replay transfer
        if (hasattr(from_agent.agent, 'memory') and
            hasattr(to_agent.agent, 'memory')):
            return 'experience_replay'

        return 'general'

    def _transfer_value_networks(self, from_agent: AgentWrapper,
                               to_agent: AgentWrapper) -> bool:
        """Transfer value networks between value-based agents."""
        try:
            # Transfer Q-network weights if architectures are compatible
            if (hasattr(from_agent.agent, 'q_network') and
                hasattr(to_agent.agent, 'q_network')):

                from_state = from_agent.agent.q_network.state_dict()
                to_state = to_agent.agent.q_network.state_dict()

                # Transfer compatible layers
                transferred_layers = 0
                for name, param in from_state.items():
                    if name in to_state and param.shape == to_state[name].shape:
                        to_state[name] = param.clone()
                        transferred_layers += 1

                to_agent.agent.q_network.load_state_dict(to_state)
                logger.info(f"Transferred {transferred_layers} layers in value network")

                return transferred_layers > 0

        except Exception as e:
            logger.error(f"Value network transfer failed: {e}")

        return False

    def _transfer_policy_networks(self, from_agent: AgentWrapper,
                                to_agent: AgentWrapper) -> bool:
        """Transfer policy networks between policy-based agents."""
        try:
            # Transfer policy network weights
            if (hasattr(from_agent.agent, 'policy_network') and
                hasattr(to_agent.agent, 'policy_network')):

                from_state = from_agent.agent.policy_network.state_dict()
                to_state = to_agent.agent.policy_network.state_dict()

                transferred_layers = 0
                for name, param in from_state.items():
                    if name in to_state and param.shape == to_state[name].shape:
                        to_state[name] = param.clone()
                        transferred_layers += 1

                to_agent.agent.policy_network.load_state_dict(to_state)
                logger.info(f"Transferred {transferred_layers} layers in policy network")

                return transferred_layers > 0

        except Exception as e:
            logger.error(f"Policy network transfer failed: {e}")

        return False

    def _transfer_critic_networks(self, from_agent: AgentWrapper,
                                to_agent: AgentWrapper) -> bool:
        """Transfer critic networks between actor-critic agents."""
        try:
            success = False

            # Transfer critic network
            if (hasattr(from_agent.agent, 'critic') and
                hasattr(to_agent.agent, 'critic')):

                from_state = from_agent.agent.critic.state_dict()
                to_state = to_agent.agent.critic.state_dict()

                transferred_layers = 0
                for name, param in from_state.items():
                    if name in to_state and param.shape == to_state[name].shape:
                        to_state[name] = param.clone()
                        transferred_layers += 1

                to_agent.agent.critic.load_state_dict(to_state)
                logger.info(f"Transferred {transferred_layers} layers in critic network")
                success = transferred_layers > 0

            # Transfer actor network if available
            if (hasattr(from_agent.agent, 'actor') and
                hasattr(to_agent.agent, 'actor')):

                from_state = from_agent.agent.actor.state_dict()
                to_state = to_agent.agent.actor.state_dict()

                transferred_layers = 0
                for name, param in from_state.items():
                    if name in to_state and param.shape == to_state[name].shape:
                        to_state[name] = param.clone()
                        transferred_layers += 1

                to_agent.agent.actor.load_state_dict(to_state)
                logger.info(f"Transferred {transferred_layers} layers in actor network")
                success = success or (transferred_layers > 0)

            return success

        except Exception as e:
            logger.error(f"Critic network transfer failed: {e}")

        return False

    def _transfer_experience_replay(self, from_agent: AgentWrapper,
                                  to_agent: AgentWrapper) -> bool:
        """Transfer experience replay buffer."""
        try:
            if (hasattr(from_agent.agent, 'memory') and
                hasattr(to_agent.agent, 'memory')):

                # Copy experience buffer
                from_memory = from_agent.agent.memory
                to_memory = to_agent.agent.memory

                # Transfer experiences if buffer structure is compatible
                if hasattr(from_memory, 'buffer') and hasattr(to_memory, 'buffer'):
                    # Simple copy of experiences
                    transferred_experiences = min(len(from_memory.buffer),
                                                to_memory.capacity)

                    for i in range(transferred_experiences):
                        to_memory.buffer[i] = from_memory.buffer[i]

                    to_memory.position = transferred_experiences % to_memory.capacity
                    to_memory.size = min(to_memory.capacity, transferred_experiences)

                    logger.info(f"Transferred {transferred_experiences} experiences")
                    return True

        except Exception as e:
            logger.error(f"Experience replay transfer failed: {e}")

        return False

    def _transfer_general_state(self, from_agent: AgentWrapper,
                              to_agent: AgentWrapper) -> bool:
        """General state transfer for any compatible components."""
        try:
            # Transfer training history
            to_agent.training_history.extend(from_agent.training_history)

            # Transfer performance metrics
            to_agent.performance_metrics.update(from_agent.performance_metrics)

            logger.info("Transferred general state information")
            return True

        except Exception as e:
            logger.error(f"General state transfer failed: {e}")

        return False


class PerformanceMonitor:
    """
    Monitors agent performance for switching decisions.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor.

        Args:
            window_size: Size of the sliding window for performance calculation
        """
        self.window_size = window_size
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.performance_history = []

        logger.info(f"PerformanceMonitor initialized with window size {window_size}")

    def record_episode(self, reward: float, length: int, losses: Dict[str, float]):
        """Record episode performance."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.training_losses.append(losses)

        # Calculate moving averages
        performance = self._calculate_performance()
        self.performance_history.append(performance)

    def _calculate_performance(self) -> Dict[str, float]:
        """Calculate current performance metrics."""
        recent_rewards = self.episode_rewards[-self.window_size:]
        recent_lengths = self.episode_lengths[-self.window_size:]

        performance = {
            'mean_reward': np.mean(recent_rewards) if recent_rewards else 0.0,
            'std_reward': np.std(recent_rewards) if recent_rewards else 0.0,
            'mean_length': np.mean(recent_lengths) if recent_lengths else 0.0,
            'reward_trend': self._calculate_trend(recent_rewards),
            'total_episodes': len(self.episode_rewards)
        }

        return performance

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values (-1 to 1, where 1 is strongly increasing)."""
        if len(values) < 2:
            return 0.0

        # Simple linear regression slope
        x = np.arange(len(values))
        y = np.array(values)

        if np.std(x) == 0:
            return 0.0

        correlation = np.corrcoef(x, y)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    def get_current_performance(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self._calculate_performance()

    def should_switch(self, condition: SwitchCondition) -> bool:
        """Check if switching condition is met."""
        current_perf = self.get_current_performance()

        if condition.trigger == SwitchTrigger.PERFORMANCE_THRESHOLD:
            if condition.metric not in current_perf:
                return False

            value = current_perf[condition.metric]
            threshold = condition.threshold

            if condition.comparison == "less_than":
                return value < threshold
            elif condition.comparison == "greater_than":
                return value > threshold
            elif condition.comparison == "equal_to":
                return abs(value - threshold) < 1e-6

        elif condition.trigger == SwitchTrigger.EPISODE_COUNT:
            return current_perf['total_episodes'] >= condition.episodes

        return False


class AlgorithmSwitcher:
    """
    Main class for seamless algorithm switching during training.
    """

    def __init__(self, save_dir: str = "algorithm_switches"):
        """
        Initialize algorithm switcher.

        Args:
            save_dir: Directory to save switch history and states
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        self.state_transfer_manager = StateTransferManager()
        self.performance_monitor = PerformanceMonitor()

        self.current_agent = None
        self.switch_history = []
        self.switch_conditions = []

        logger.info(f"AlgorithmSwitcher initialized with save directory: {save_dir}")

    def set_initial_agent(self, agent: AgentWrapper):
        """Set the initial agent for training."""
        self.current_agent = agent
        logger.info(f"Initial agent set: {agent.algorithm_type.value}")

    def add_switch_condition(self, condition: SwitchCondition,
                           target_algorithm: AlgorithmType,
                           target_spec: Optional[AlgorithmSpec] = None):
        """
        Add a condition for automatic switching.

        Args:
            condition: Condition that triggers the switch
            target_algorithm: Algorithm to switch to
            target_spec: Specification for the target algorithm
        """
        switch_rule = {
            'condition': condition,
            'target_algorithm': target_algorithm,
            'target_spec': target_spec
        }

        self.switch_conditions.append(switch_rule)
        logger.info(f"Added switch condition: {condition.trigger.value} -> {target_algorithm.value}")

    def check_switch_conditions(self, episode: int, step: int) -> Optional[AgentWrapper]:
        """
        Check if any switch conditions are met and perform switch if needed.

        Args:
            episode: Current episode number
            step: Current training step

        Returns:
            New agent if switch occurred, None otherwise
        """
        for rule in self.switch_conditions:
            condition = rule['condition']

            if self.performance_monitor.should_switch(condition):
                target_algorithm = rule['target_algorithm']
                target_spec = rule['target_spec']

                logger.info(f"Switch condition met: {condition.trigger.value}")

                new_agent = self.switch_algorithm(
                    target_algorithm=target_algorithm,
                    target_spec=target_spec,
                    trigger=condition.trigger,
                    episode=episode,
                    step=step
                )

                return new_agent

        return None

    def switch_algorithm(self, target_algorithm: AlgorithmType,
                        target_spec: Optional[AlgorithmSpec] = None,
                        trigger: SwitchTrigger = SwitchTrigger.MANUAL,
                        episode: int = 0,
                        step: int = 0) -> AgentWrapper:
        """
        Perform algorithm switch.

        Args:
            target_algorithm: Algorithm to switch to
            target_spec: Specification for target algorithm
            trigger: What triggered the switch
            episode: Current episode
            step: Current step

        Returns:
            New agent instance
        """
        if self.current_agent is None:
            raise ValueError("No current agent set")

        logger.info(f"Switching from {self.current_agent.algorithm_type.value} to {target_algorithm.value}")

        # Create target specification if not provided
        if target_spec is None:
            target_spec = AlgorithmSpec(
                algorithm_type=target_algorithm,
                state_dim=self.current_agent.spec.state_dim,
                action_dim=self.current_agent.spec.action_dim,
                device=self.current_agent.spec.device
            )

        # Save current agent state
        self._save_agent_state(self.current_agent, episode, step)

        # Create new agent
        new_agent = algorithm_factory.create_agent(target_spec)

        # Transfer state
        transfer_success = self.state_transfer_manager.transfer_state(
            self.current_agent, new_agent
        )

        # Record switch event
        current_performance = self.performance_monitor.get_current_performance()
        switch_event = SwitchEvent(
            from_algorithm=self.current_agent.algorithm_type.value,
            to_algorithm=target_algorithm.value,
            trigger=trigger,
            episode=episode,
            step=step,
            performance_before=current_performance.get('mean_reward', 0.0),
            timestamp=f"switch_{episode}_{step}"
        )

        self.switch_history.append(switch_event)
        self._save_switch_history()

        # Update current agent
        self.current_agent = new_agent

        logger.info(f"Algorithm switch completed. Transfer success: {transfer_success}")

        return new_agent

    def _save_agent_state(self, agent: AgentWrapper, episode: int, step: int):
        """Save agent state before switching."""
        timestamp = f"ep{episode}_step{step}"
        filename = f"{agent.algorithm_type.value}_ep{episode}_step{step}_{timestamp}.pth"
        filepath = self.save_dir / filename

        try:
            agent.save_state(str(filepath))
            logger.info(f"Saved agent state: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")

    def _save_switch_history(self):
        """Save switch history to JSON file."""
        history_file = self.save_dir / "switch_history.json"

        try:
            # Convert switch events to dictionaries
            history_data = []
            for event in self.switch_history:
                event_dict = asdict(event)
                history_data.append(event_dict)

            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)

            logger.info(f"Saved switch history: {history_file}")
        except Exception as e:
            logger.error(f"Failed to save switch history: {e}")

    def record_episode_performance(self, reward: float, length: int,
                                 losses: Dict[str, float]):
        """Record episode performance for monitoring."""
        self.performance_monitor.record_episode(reward, length, losses)

    def get_switch_history(self) -> List[SwitchEvent]:
        """Get history of algorithm switches."""
        return self.switch_history.copy()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary including switch history."""
        current_perf = self.performance_monitor.get_current_performance()

        summary = {
            'current_performance': current_perf,
            'current_algorithm': self.current_agent.algorithm_type.value if self.current_agent else None,
            'total_switches': len(self.switch_history),
            'switch_history': [asdict(event) for event in self.switch_history[-5:]]  # Last 5 switches
        }

        return summary

    def create_adaptive_switching_strategy(self, algorithms: List[AlgorithmType],
                                         performance_threshold: float = 0.1) -> List[SwitchCondition]:
        """
        Create an adaptive switching strategy that cycles through algorithms
        based on performance stagnation.

        Args:
            algorithms: List of algorithms to cycle through
            performance_threshold: Threshold for performance improvement

        Returns:
            List of switch conditions
        """
        conditions = []

        for i, algorithm in enumerate(algorithms[1:], 1):
            condition = SwitchCondition(
                trigger=SwitchTrigger.PERFORMANCE_THRESHOLD,
                threshold=performance_threshold,
                metric='reward_trend',
                comparison='less_than'
            )

            # Add condition with delay based on position
            episodes_delay = i * 100  # Increase delay for later algorithms

            episode_condition = SwitchCondition(
                trigger=SwitchTrigger.EPISODE_COUNT,
                episodes=episodes_delay
            )

            conditions.extend([condition, episode_condition])

        return conditions


# Convenience functions
def create_switcher(save_dir: str = "algorithm_switches") -> AlgorithmSwitcher:
    """Create an algorithm switcher instance."""
    return AlgorithmSwitcher(save_dir)


def create_performance_based_strategy(algorithms: List[str],
                                     threshold: float = 0.1) -> List[Dict[str, Any]]:
    """
    Create a performance-based switching strategy.

    Args:
        algorithms: List of algorithm names
        threshold: Performance threshold for switching

    Returns:
        List of strategy configurations
    """
    strategy = []

    for algorithm in algorithms:
        strategy.append({
            'algorithm': algorithm,
            'condition': {
                'trigger': 'performance_threshold',
                'threshold': threshold,
                'metric': 'reward_trend',
                'comparison': 'less_than'
            }
        })

    return strategy
