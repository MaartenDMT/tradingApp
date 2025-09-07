"""
Context7 Enhanced Agent Management for Trading RL.

This module provides professional agent management utilities following Context7 patterns:
- Advanced model selection and hyperparameter optimization
- Professional performance monitoring and comparison
- Context7 ensemble methods and model averaging
- Production deployment and monitoring systems
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import optuna

import util.loggers as loggers

logger = loggers.setup_loggers()
rl_logger = logger['rl']

# Context7 Agent Management Constants
CONTEXT7_OPTIMIZATION_TRIALS = 100
CONTEXT7_ENSEMBLE_SIZE = 5
CONTEXT7_PERFORMANCE_METRICS = ['sharpe_ratio', 'max_drawdown', 'win_rate', 'total_return']
CONTEXT7_VALIDATION_EPISODES = 50


@dataclass
class Context7AgentConfig:
    """Configuration for Context7 agents."""
    agent_type: str
    hyperparameters: Dict
    performance_metrics: Dict = None
    training_episodes: int = 1000
    validation_episodes: int = CONTEXT7_VALIDATION_EPISODES
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.performance_metrics is None:
            self.performance_metrics = {}


@dataclass
class Context7PerformanceReport:
    """Performance report for Context7 agents."""
    agent_id: str
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    volatility: float
    trades_count: int
    avg_trade_return: float
    evaluation_period: str
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class Context7HyperparameterOptimizer:
    """
    Professional hyperparameter optimization following Context7 patterns.
    """

    def __init__(self,
                 agent_factory: Callable,
                 env_factory: Callable,
                 n_trials: int = CONTEXT7_OPTIMIZATION_TRIALS):

        self.agent_factory = agent_factory
        self.env_factory = env_factory
        self.n_trials = n_trials
        self.study = None
        self.best_params = None

        rl_logger.info(f"Context7 Hyperparameter Optimizer initialized with {n_trials} trials")

    def optimize(self,
                 param_space: Dict,
                 objective_metric: str = 'sharpe_ratio',
                 direction: str = 'maximize') -> Dict:
        """
        Optimize hyperparameters using Optuna.

        Args:
            param_space: Parameter search space
            objective_metric: Metric to optimize
            direction: Optimization direction

        Returns:
            Best parameters found
        """
        rl_logger.info(f"Starting hyperparameter optimization for {objective_metric}")

        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )

            # Evaluate agent with these parameters
            try:
                performance = self._evaluate_params(params)
                return performance[objective_metric]
            except Exception as e:
                rl_logger.warning(f"Trial failed: {e}")
                return float('-inf') if direction == 'maximize' else float('inf')

        # Create and run study
        self.study = optuna.create_study(direction=direction)
        self.study.optimize(objective, n_trials=self.n_trials)

        self.best_params = self.study.best_params

        rl_logger.info(f"Optimization completed. Best {objective_metric}: {self.study.best_value:.4f}")
        rl_logger.info(f"Best parameters: {self.best_params}")

        return self.best_params

    def _evaluate_params(self, params: Dict) -> Dict:
        """Evaluate agent performance with given parameters."""
        # Create environment and agent
        env = self.env_factory()
        agent = self.agent_factory(**params)

        # Quick training (fewer episodes for optimization)
        quick_episodes = min(200, self.n_trials // 5)

        episode_results = []
        for episode in range(quick_episodes):
            state = env.reset()
            done = False

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)

                if hasattr(agent, 'remember'):
                    agent.remember(state, action, reward, next_state, done)

                if hasattr(agent, 'learn'):
                    agent.learn()

                state = next_state

            result = env.get_episode_result()
            episode_results.append(result)

        # Calculate performance metrics
        performance = self._calculate_performance_metrics(episode_results)

        env.close()
        return performance

    def _calculate_performance_metrics(self, episode_results: List[Dict]) -> Dict:
        """Calculate performance metrics from episode results."""
        if not episode_results:
            return {metric: 0.0 for metric in CONTEXT7_PERFORMANCE_METRICS}

        navs = [result['nav'] for result in episode_results]
        differences = [result['difference'] for result in episode_results]

        # Calculate metrics
        total_return = navs[-1] - 1.0 if navs else 0.0
        win_rate = sum(1 for d in differences if d > 0) / len(differences) if differences else 0.0

        # Sharpe ratio calculation
        returns = np.diff(navs) if len(navs) > 1 else [0.0]
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0

        # Max drawdown
        peak = navs[0] if navs else 1.0
        max_dd = 0.0
        for nav in navs:
            if nav > peak:
                peak = nav
            dd = (peak - nav) / peak
            if dd > max_dd:
                max_dd = dd

        return {
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'max_drawdown': max_dd,
            'win_rate': win_rate
        }


class Context7AgentRegistry:
    """
    Professional agent registry following Context7 patterns.
    """

    def __init__(self, registry_path: str = "model/reinforcement/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.agents = {}
        self.performance_history = {}

        # Load existing registry
        self._load_registry()

        rl_logger.info(f"Context7 Agent Registry initialized: {len(self.agents)} agents loaded")

    def register_agent(self,
                      agent_id: str,
                      agent,
                      config: Context7AgentConfig,
                      performance: Context7PerformanceReport = None) -> None:
        """Register a new agent."""
        # Save agent model
        agent_dir = self.registry_path / agent_id
        agent_dir.mkdir(exist_ok=True)

        if hasattr(agent, 'save_models'):
            agent.save_models(str(agent_dir / "model"))
        elif hasattr(agent, 'save_model'):
            agent.save_model(str(agent_dir / "model"))

        # Save configuration
        with open(agent_dir / "config.json", 'w') as f:
            json.dump(asdict(config), f, indent=2)

        # Save performance if provided
        if performance:
            with open(agent_dir / "performance.json", 'w') as f:
                json.dump(asdict(performance), f, indent=2)

        # Update registry
        self.agents[agent_id] = {
            'config': config,
            'performance': performance,
            'path': str(agent_dir)
        }

        self._save_registry()

        rl_logger.info(f"Agent {agent_id} registered successfully")

    def get_agent(self, agent_id: str):
        """Load and return an agent."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found in registry")

        agent_info = self.agents[agent_id]
        agent_path = Path(agent_info['path'])

        # Load agent based on type
        config = agent_info['config']
        agent_type = config.agent_type

        # Import and create agent (this would need to be adapted based on your agent classes)
        if agent_type == 'ddqn':
            from model.reinforcement.ddqn_tf import Context7DDQNAgent
            agent = Context7DDQNAgent(**config.hyperparameters)
            if (agent_path / "model").exists():
                agent.load_models(str(agent_path / "model"))
        elif agent_type == 'td3':
            from model.reinforcement.td3_tf import Context7TD3Agent
            agent = Context7TD3Agent(**config.hyperparameters)
            if (agent_path / "model").exists():
                agent.load_models(str(agent_path / "model"))
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        return agent

    def list_agents(self) -> List[Dict]:
        """List all registered agents with their performance."""
        agents_list = []
        for agent_id, info in self.agents.items():
            agent_summary = {
                'id': agent_id,
                'type': info['config'].agent_type,
                'created_at': info['config'].created_at,
                'performance': info['performance']
            }
            agents_list.append(agent_summary)

        return sorted(agents_list, key=lambda x: x['created_at'], reverse=True)

    def get_best_agents(self,
                       metric: str = 'sharpe_ratio',
                       top_k: int = 5) -> List[str]:
        """Get top performing agents by metric."""
        agent_scores = []

        for agent_id, info in self.agents.items():
            if info['performance'] and hasattr(info['performance'], metric):
                score = getattr(info['performance'], metric)
                agent_scores.append((agent_id, score))

        # Sort by score (descending)
        agent_scores.sort(key=lambda x: x[1], reverse=True)

        return [agent_id for agent_id, _ in agent_scores[:top_k]]

    def _load_registry(self):
        """Load registry from disk."""
        registry_file = self.registry_path / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)

                for agent_id, agent_data in data.items():
                    config = Context7AgentConfig(**agent_data['config'])
                    performance = None
                    if agent_data.get('performance'):
                        performance = Context7PerformanceReport(**agent_data['performance'])

                    self.agents[agent_id] = {
                        'config': config,
                        'performance': performance,
                        'path': agent_data['path']
                    }
            except Exception as e:
                rl_logger.warning(f"Failed to load registry: {e}")

    def _save_registry(self):
        """Save registry to disk."""
        registry_file = self.registry_path / "registry.json"

        data = {}
        for agent_id, info in self.agents.items():
            data[agent_id] = {
                'config': asdict(info['config']),
                'performance': asdict(info['performance']) if info['performance'] else None,
                'path': info['path']
            }

        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)


class Context7EnsembleManager:
    """
    Professional ensemble management following Context7 patterns.
    """

    def __init__(self,
                 registry: Context7AgentRegistry,
                 ensemble_size: int = CONTEXT7_ENSEMBLE_SIZE):

        self.registry = registry
        self.ensemble_size = ensemble_size
        self.ensemble_agents = []
        self.weights = None

        rl_logger.info(f"Context7 Ensemble Manager initialized (size: {ensemble_size})")

    def create_ensemble(self,
                       selection_metric: str = 'sharpe_ratio',
                       weighting_method: str = 'performance') -> List[str]:
        """
        Create ensemble of best performing agents.

        Args:
            selection_metric: Metric for agent selection
            weighting_method: Method for weight calculation

        Returns:
            List of selected agent IDs
        """
        # Get best agents
        best_agents = self.registry.get_best_agents(
            metric=selection_metric,
            top_k=self.ensemble_size
        )

        if len(best_agents) < self.ensemble_size:
            rl_logger.warning(f"Only {len(best_agents)} agents available, using all")

        # Load agents
        self.ensemble_agents = []
        performances = []

        for agent_id in best_agents:
            agent = self.registry.get_agent(agent_id)
            self.ensemble_agents.append(agent)

            # Get performance for weighting
            agent_info = self.registry.agents[agent_id]
            if agent_info['performance']:
                performance = getattr(agent_info['performance'], selection_metric)
                performances.append(max(performance, 0.01))  # Avoid zero weights
            else:
                performances.append(1.0)

        # Calculate weights
        if weighting_method == 'performance':
            total_performance = sum(performances)
            self.weights = [p / total_performance for p in performances]
        elif weighting_method == 'equal':
            self.weights = [1.0 / len(self.ensemble_agents)] * len(self.ensemble_agents)
        else:
            raise ValueError(f"Unknown weighting method: {weighting_method}")

        rl_logger.info(f"Ensemble created with {len(self.ensemble_agents)} agents")
        rl_logger.info(f"Weights: {[f'{w:.3f}' for w in self.weights]}")

        return best_agents

    def ensemble_action(self, state) -> int:
        """Get ensemble action using weighted voting."""
        if not self.ensemble_agents:
            raise ValueError("No ensemble created")

        # Get actions from all agents
        actions = []
        for agent in self.ensemble_agents:
            action = agent.choose_action(state)
            actions.append(action)

        # For discrete actions, use weighted voting
        if isinstance(actions[0], int):
            # Count votes for each action
            action_votes = {}
            for action, weight in zip(actions, self.weights):
                action_votes[action] = action_votes.get(action, 0) + weight

            # Return action with highest weighted vote
            return max(action_votes.keys(), key=lambda x: action_votes[x])

        # For continuous actions, use weighted average
        else:
            weighted_action = np.average(actions, weights=self.weights, axis=0)
            return weighted_action

    def evaluate_ensemble(self, env, episodes: int = 50) -> Context7PerformanceReport:
        """Evaluate ensemble performance."""
        rl_logger.info(f"Evaluating ensemble over {episodes} episodes")

        episode_results = []

        for episode in range(episodes):
            state = env.reset()
            done = False

            while not done:
                action = self.ensemble_action(state)
                state, reward, done, info = env.step(action)

            result = env.get_episode_result()
            episode_results.append(result)

        # Calculate performance metrics
        navs = [result['nav'] for result in episode_results]
        differences = [result['difference'] for result in episode_results]

        total_return = navs[-1] - 1.0
        win_rate = sum(1 for d in differences if d > 0) / len(differences)

        returns = np.diff(navs)
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        volatility = np.std(returns)

        # Max drawdown
        peak = navs[0]
        max_dd = 0.0
        for nav in navs:
            if nav > peak:
                peak = nav
            dd = (peak - nav) / peak
            if dd > max_dd:
                max_dd = dd

        performance = Context7PerformanceReport(
            agent_id="ensemble",
            sharpe_ratio=sharpe,
            total_return=total_return,
            max_drawdown=max_dd,
            win_rate=win_rate,
            volatility=volatility,
            trades_count=len(episode_results),
            avg_trade_return=np.mean(differences),
            evaluation_period=f"{episodes}_episodes"
        )

        rl_logger.info("Ensemble evaluation completed:")
        rl_logger.info(f"  Sharpe Ratio: {sharpe:.3f}")
        rl_logger.info(f"  Total Return: {total_return:.2%}")
        rl_logger.info(f"  Win Rate: {win_rate:.2%}")
        rl_logger.info(f"  Max Drawdown: {max_dd:.2%}")

        return performance


def context7_optimize_agent(agent_factory: Callable,
                           env_factory: Callable,
                           param_space: Dict,
                           **kwargs) -> Dict:
    """
    Convenience function for Context7 hyperparameter optimization.

    Args:
        agent_factory: Function to create agent
        env_factory: Function to create environment
        param_space: Parameter search space
        **kwargs: Additional optimization parameters

    Returns:
        Best parameters found
    """
    optimizer = Context7HyperparameterOptimizer(agent_factory, env_factory, **kwargs)
    return optimizer.optimize(param_space)


# Export main classes and functions
__all__ = [
    'Context7AgentConfig',
    'Context7PerformanceReport',
    'Context7HyperparameterOptimizer',
    'Context7AgentRegistry',
    'Context7EnsembleManager',
    'context7_optimize_agent'
]
