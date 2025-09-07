"""
Tabular Reinforcement Learning Algorithms.

This module implements classic tabular RL algorithms including Q-Learning, SARSA,
Monte Carlo methods, and Expected SARSA. These are the foundational algorithms
that work with discrete state and action spaces.
"""

import pickle
from collections import defaultdict, deque
from typing import Any, Dict, Optional, Tuple

import numpy as np

import util.loggers as loggers

from ..core.base_agents import AgentConfig, BaseRLAgent

logger = loggers.setup_loggers()
rl_logger = logger['rl']


class TabularConfig(AgentConfig):
    """Configuration for tabular RL agents."""
    learning_rate: float = 0.1
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    state_discretization: int = 100  # Number of bins for continuous states
    min_visits_for_update: int = 1   # Minimum visits before updating

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class StateDiscretizer:
    """Discretizes continuous states for tabular methods."""

    def __init__(self,
                 state_dim: int,
                 num_bins: int = 100,
                 state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Initialize state discretizer.

        Args:
            state_dim: Dimension of state space
            num_bins: Number of bins per dimension
            state_bounds: (min_values, max_values) for each dimension
        """
        self.state_dim = state_dim
        self.num_bins = num_bins

        if state_bounds is None:
            # Default bounds - will be updated as we see data
            self.min_values = np.full(state_dim, np.inf)
            self.max_values = np.full(state_dim, -np.inf)
            self.adaptive = True
        else:
            self.min_values, self.max_values = state_bounds
            self.adaptive = False

        self._update_bins()

    def _update_bins(self):
        """Update bin edges based on current bounds."""
        self.bin_edges = []
        for i in range(self.state_dim):
            if self.min_values[i] != self.max_values[i]:
                edges = np.linspace(self.min_values[i], self.max_values[i], self.num_bins + 1)
            else:
                edges = np.array([self.min_values[i] - 0.5, self.max_values[i] + 0.5])
            self.bin_edges.append(edges)

    def discretize(self, state: np.ndarray) -> Tuple[int, ...]:
        """
        Discretize continuous state to discrete state.

        Args:
            state: Continuous state vector

        Returns:
            Discrete state tuple
        """
        if self.adaptive:
            # Update bounds if necessary
            updated = False
            for i in range(self.state_dim):
                if state[i] < self.min_values[i]:
                    self.min_values[i] = state[i]
                    updated = True
                if state[i] > self.max_values[i]:
                    self.max_values[i] = state[i]
                    updated = True

            if updated:
                self._update_bins()

        # Discretize each dimension
        discrete_state = []
        for i in range(self.state_dim):
            # Find which bin this value falls into
            bin_idx = np.digitize(state[i], self.bin_edges[i]) - 1
            bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)
            discrete_state.append(bin_idx)

        return tuple(discrete_state)

    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state bounds."""
        return self.min_values.copy(), self.max_values.copy()


class TabularRLAgent(BaseRLAgent):
    """Base class for tabular RL agents."""

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 config: TabularConfig,
                 name: str = "TabularRLAgent"):

        self.config = config
        self.num_actions = num_actions
        super().__init__(state_dim, num_actions, config, name)

    def _initialize_agent(self) -> None:
        """Initialize tabular agent components."""
        # Q-table: default dict that returns zeros for unseen states
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))

        # State discretizer for continuous states
        self.state_discretizer = StateDiscretizer(
            self.state_dim,
            self.config.state_discretization
        )

        # Exploration parameters
        self.epsilon = self.config.epsilon_start

        # Visit counts for each state-action pair
        self.visit_counts = defaultdict(lambda: np.zeros(self.num_actions))

        # Value tracking
        self.value_history = deque(maxlen=1000)

        rl_logger.info(f"Initialized {self.name} with {self.config.state_discretization} bins per dimension")

    def _discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """Convert continuous state to discrete state."""
        return self.state_discretizer.discretize(state)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        discrete_state = self._discretize_state(state)
        q_values = self.q_table[discrete_state]

        # Record Q-values for monitoring
        self.value_history.append(np.mean(q_values))

        if training and np.random.random() < self.epsilon:
            # Random action
            return np.random.randint(self.num_actions)
        else:
            # Greedy action
            return int(np.argmax(q_values))

    def update_epsilon(self) -> None:
        """Update exploration parameter."""
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def get_q_value(self, state: np.ndarray, action: int) -> float:
        """Get Q-value for state-action pair."""
        discrete_state = self._discretize_state(state)
        return self.q_table[discrete_state][action]

    def get_state_value(self, state: np.ndarray) -> float:
        """Get state value (max Q-value)."""
        discrete_state = self._discretize_state(state)
        return np.max(self.q_table[discrete_state])

    def _get_agent_state(self) -> Dict[str, Any]:
        """Get agent-specific state for checkpointing."""
        return {
            'q_table': dict(self.q_table),
            'visit_counts': dict(self.visit_counts),
            'epsilon': self.epsilon,
            'state_bounds': self.state_discretizer.get_state_bounds(),
            'value_history': list(self.value_history)
        }

    def _restore_agent_state(self, state: Dict[str, Any]) -> None:
        """Restore agent-specific state from checkpoint."""
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions), state['q_table'])
        self.visit_counts = defaultdict(lambda: np.zeros(self.num_actions), state['visit_counts'])
        self.epsilon = state['epsilon']

        if 'state_bounds' in state:
            min_vals, max_vals = state['state_bounds']
            self.state_discretizer = StateDiscretizer(
                self.state_dim,
                self.config.state_discretization,
                (min_vals, max_vals)
            )

        self.value_history = deque(state.get('value_history', []), maxlen=1000)

    def _save_model_weights(self, filepath: str) -> None:
        """Save Q-table and visit counts."""
        data = {
            'q_table': dict(self.q_table),
            'visit_counts': dict(self.visit_counts),
            'state_bounds': self.state_discretizer.get_state_bounds()
        }
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(data, f)

    def _load_model_weights(self, filepath: str) -> None:
        """Load Q-table and visit counts."""
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)

        self.q_table = defaultdict(lambda: np.zeros(self.num_actions), data['q_table'])
        self.visit_counts = defaultdict(lambda: np.zeros(self.num_actions), data['visit_counts'])

        if 'state_bounds' in data:
            min_vals, max_vals = data['state_bounds']
            self.state_discretizer = StateDiscretizer(
                self.state_dim,
                self.config.state_discretization,
                (min_vals, max_vals)
            )

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics."""
        base_metrics = super().get_training_metrics()

        tabular_metrics = {
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'avg_q_value': np.mean(self.value_history) if self.value_history else 0.0,
            'states_visited': len(self.q_table),
            'total_state_action_pairs': len(self.q_table) * self.num_actions
        }

        base_metrics.update(tabular_metrics)
        return base_metrics


class QLearningAgent(TabularRLAgent):
    """Q-Learning agent with off-policy learning."""

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 config: TabularConfig = None):

        if config is None:
            config = TabularConfig()

        super().__init__(state_dim, num_actions, config, "QLearningAgent")

    def update(self,
               state: np.ndarray,
               action: int,
               reward: float,
               next_state: np.ndarray,
               done: bool) -> Dict[str, float]:
        """Update Q-table using Q-learning update rule."""

        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)

        # Update visit count
        self.visit_counts[discrete_state][action] += 1

        # Q-learning update
        current_q = self.q_table[discrete_state][action]

        if done:
            target = reward
        else:
            target = reward + self.config.gamma * np.max(self.q_table[discrete_next_state])

        # Update Q-value
        td_error = target - current_q
        self.q_table[discrete_state][action] += self.config.learning_rate * td_error

        # Update exploration
        self.update_epsilon()

        self.training_step += 1

        return {
            'td_error': abs(td_error),
            'q_value': self.q_table[discrete_state][action],
            'epsilon': self.epsilon,
            'visits': self.visit_counts[discrete_state][action]
        }


class SARSAAgent(TabularRLAgent):
    """SARSA agent with on-policy learning."""

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 config: TabularConfig = None):

        if config is None:
            config = TabularConfig()

        super().__init__(state_dim, num_actions, config, "SARSAAgent")

        # Store last state-action for SARSA update
        self.last_state = None
        self.last_action = None

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action and store for SARSA update."""
        action = super().select_action(state, training)

        if training:
            self.last_state = state.copy()
            self.last_action = action

        return action

    def update(self,
               state: np.ndarray,
               action: int,
               reward: float,
               next_state: np.ndarray,
               done: bool) -> Dict[str, float]:
        """Update Q-table using SARSA update rule."""

        if self.last_state is None or self.last_action is None:
            return {}

        discrete_state = self._discretize_state(self.last_state)
        discrete_next_state = self._discretize_state(next_state)

        # Update visit count
        self.visit_counts[discrete_state][self.last_action] += 1

        # SARSA update
        current_q = self.q_table[discrete_state][self.last_action]

        if done:
            target = reward
        else:
            # Use the action that will be taken in next state (on-policy)
            next_action = self.select_action(next_state, training=True)
            target = reward + self.config.gamma * self.q_table[discrete_next_state][next_action]

        # Update Q-value
        td_error = target - current_q
        self.q_table[discrete_state][self.last_action] += self.config.learning_rate * td_error

        # Update exploration
        self.update_epsilon()

        self.training_step += 1

        return {
            'td_error': abs(td_error),
            'q_value': self.q_table[discrete_state][self.last_action],
            'epsilon': self.epsilon,
            'visits': self.visit_counts[discrete_state][self.last_action]
        }


class ExpectedSARSAAgent(TabularRLAgent):
    """Expected SARSA agent."""

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 config: TabularConfig = None):

        if config is None:
            config = TabularConfig()

        super().__init__(state_dim, num_actions, config, "ExpectedSARSAAgent")

    def update(self,
               state: np.ndarray,
               action: int,
               reward: float,
               next_state: np.ndarray,
               done: bool) -> Dict[str, float]:
        """Update Q-table using Expected SARSA update rule."""

        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)

        # Update visit count
        self.visit_counts[discrete_state][action] += 1

        # Expected SARSA update
        current_q = self.q_table[discrete_state][action]

        if done:
            target = reward
        else:
            # Expected value under epsilon-greedy policy
            next_q_values = self.q_table[discrete_next_state]

            # Probability of taking each action
            best_action = np.argmax(next_q_values)
            action_probs = np.full(self.num_actions, self.epsilon / self.num_actions)
            action_probs[best_action] += 1 - self.epsilon

            expected_value = np.sum(action_probs * next_q_values)
            target = reward + self.config.gamma * expected_value

        # Update Q-value
        td_error = target - current_q
        self.q_table[discrete_state][action] += self.config.learning_rate * td_error

        # Update exploration
        self.update_epsilon()

        self.training_step += 1

        return {
            'td_error': abs(td_error),
            'q_value': self.q_table[discrete_state][action],
            'epsilon': self.epsilon,
            'visits': self.visit_counts[discrete_state][action]
        }


class MonteCarloAgent(TabularRLAgent):
    """Monte Carlo agent with episode-based learning."""

    def __init__(self,
                 state_dim: int,
                 num_actions: int,
                 config: TabularConfig = None):

        if config is None:
            config = TabularConfig()

        super().__init__(state_dim, num_actions, config, "MonteCarloAgent")

        # Episode buffer
        self.episode_buffer = []
        self.returns = defaultdict(list)  # Returns for each state-action pair

    def update(self,
               state: np.ndarray,
               action: int,
               reward: float,
               next_state: np.ndarray,
               done: bool) -> Dict[str, float]:
        """Store experience and update at episode end."""

        # Store experience
        discrete_state = self._discretize_state(state)
        self.episode_buffer.append((discrete_state, action, reward))

        if done:
            # Update Q-table using episode returns
            return self._update_from_episode()

        return {}

    def _update_from_episode(self) -> Dict[str, float]:
        """Update Q-table using Monte Carlo returns."""
        if not self.episode_buffer:
            return {}

        # Calculate returns for each state-action pair
        G = 0  # Return
        visited_pairs = set()

        # Work backwards through episode
        for t in reversed(range(len(self.episode_buffer))):
            state, action, reward = self.episode_buffer[t]
            G = reward + self.config.gamma * G

            # First-visit Monte Carlo
            if (state, action) not in visited_pairs:
                visited_pairs.add((state, action))

                # Update visit count
                self.visit_counts[state][action] += 1

                # Store return
                self.returns[(state, action)].append(G)

                # Update Q-value as average of returns
                self.q_table[state][action] = np.mean(self.returns[(state, action)])

        # Clear episode buffer
        num_updates = len(visited_pairs)
        self.episode_buffer.clear()

        # Update exploration
        self.update_epsilon()

        self.training_step += 1

        return {
            'num_updates': num_updates,
            'episode_return': G,
            'epsilon': self.epsilon
        }


def create_tabular_agent(agent_type: str,
                        state_dim: int,
                        num_actions: int,
                        config: TabularConfig = None) -> TabularRLAgent:
    """
    Factory function to create tabular RL agents.

    Args:
        agent_type: Type of agent ('qlearning', 'sarsa', 'expected_sarsa', 'monte_carlo')
        state_dim: State space dimension
        num_actions: Number of actions
        config: Agent configuration

    Returns:
        Tabular RL agent instance
    """
    agent_type = agent_type.lower()

    if agent_type == 'qlearning' or agent_type == 'q_learning':
        return QLearningAgent(state_dim, num_actions, config)
    elif agent_type == 'sarsa':
        return SARSAAgent(state_dim, num_actions, config)
    elif agent_type == 'expected_sarsa':
        return ExpectedSARSAAgent(state_dim, num_actions, config)
    elif agent_type == 'monte_carlo' or agent_type == 'mc':
        return MonteCarloAgent(state_dim, num_actions, config)
    else:
        raise ValueError(f"Unknown tabular agent type: {agent_type}")
