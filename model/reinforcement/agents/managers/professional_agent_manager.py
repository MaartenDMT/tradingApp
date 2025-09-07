"""
Professional Agent Manager for Reinforcement Learning Trading.

Provides comprehensive agent management, training coordination,
and performance monitoring for multiple RL agents.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

import util.loggers as loggers

# Import agent base classes
from model.reinforcement.agents.base.base_agent import BaseAgent
from model.reinforcement.agents.examples.modern_agent_example import (
    ModernDQNAgent,
    ModernTD3Agent,
)
from model.reinforcement.algorithms.tensorflow.TF.DDQN.ddqn_tf import EnhancedDDQNAgent

# Import algorithms
from model.reinforcement.algorithms.tensorflow.TF.TD3.td3_tf import EnhancedTD3Agent

logger = loggers.setup_loggers()
manager_logger = logger['agent']


class AgentFactory:
    """
    Professional factory for creating and configuring RL agents.

    Provides a centralized way to create agents with standardized
    configurations and professional defaults.
    """

    # Registry of available agent types
    AGENT_REGISTRY = {
        'modern_dqn': ModernDQNAgent,
        'modern_td3': ModernTD3Agent,
        'enhanced_td3': EnhancedTD3Agent,
        'enhanced_ddqn': EnhancedDDQNAgent,
    }

    # Default configurations for each agent type
    DEFAULT_CONFIGS = {
        'modern_dqn': {
            'learning_rate': 0.0001,
            'memory_size': 100000,
            'batch_size': 32,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'target_update_freq': 1000
        },
        'modern_td3': {
            'actor_lr': 0.0001,
            'critic_lr': 0.0001,
            'memory_size': 1000000,
            'batch_size': 32,
            'gamma': 0.99,
            'tau': 0.005,
            'policy_delay': 2,
            'noise_std': 0.1,
            'noise_clip': 0.5
        },
        'enhanced_td3': {
            'alpha': 0.0001,
            'beta': 0.0001,
            'gamma': 0.99,
            'tau': 0.005,
            'batch_size': 100,
            'max_size': 1000000,
            'noise': 0.1,
            'noise_clip': 0.5
        },
        'enhanced_ddqn': {
            'lr': 0.0001,
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_dec': 0.995,
            'mem_size': 100000,
            'batch_size': 32,
            'replace': 1000
        }
    }

    @classmethod
    def create_agent(cls,
                     agent_type: str,
                     state_dim: int,
                     action_dim: int,
                     custom_config: Optional[Dict] = None,
                     **kwargs) -> BaseAgent:
        """
        Create an agent with professional configuration.

        Args:
            agent_type: Type of agent to create
            state_dim: State space dimension
            action_dim: Action space dimension
            custom_config: Optional custom configuration
            **kwargs: Additional arguments

        Returns:
            Configured agent instance
        """
        if agent_type not in cls.AGENT_REGISTRY:
            raise ValueError(f"Unknown agent type: {agent_type}. "
                           f"Available: {list(cls.AGENT_REGISTRY.keys())}")

        # Get agent class and default config
        agent_class = cls.AGENT_REGISTRY[agent_type]
        config = cls.DEFAULT_CONFIGS[agent_type].copy()

        # Update with custom configuration
        if custom_config:
            config.update(custom_config)

        # Update with keyword arguments
        config.update(kwargs)

        try:
            # Create agent based on type
            if agent_type in ['modern_dqn', 'enhanced_ddqn']:
                # Discrete action agents
                agent = agent_class(
                    state_dim=state_dim,
                    num_actions=action_dim,
                    **config
                )
            elif agent_type in ['modern_td3', 'enhanced_td3']:
                # Continuous action agents
                agent = agent_class(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    **config
                )
            else:
                # Generic creation
                agent = agent_class(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    **config
                )

            manager_logger.info(f"Created {agent_type} agent with config: {config}")
            return agent

        except Exception as e:
            manager_logger.error(f"Failed to create {agent_type} agent: {e}")
            raise

    @classmethod
    def list_available_agents(cls) -> List[str]:
        """List all available agent types."""
        return list(cls.AGENT_REGISTRY.keys())

    @classmethod
    def get_agent_info(cls, agent_type: str) -> Dict:
        """Get information about an agent type."""
        if agent_type not in cls.AGENT_REGISTRY:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent_class = cls.AGENT_REGISTRY[agent_type]
        default_config = cls.DEFAULT_CONFIGS[agent_type]

        return {
            'agent_type': agent_type,
            'agent_class': agent_class.__name__,
            'module': agent_class.__module__,
            'default_config': default_config,
            'description': agent_class.__doc__ or "No description available"
        }


class AgentPerformanceTracker:
    """
    Professional performance tracking for RL agents.

    Tracks training progress, evaluation metrics, and provides
    comprehensive analysis and reporting capabilities.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.training_history = []
        self.evaluation_history = []
        self.created_at = datetime.now()

    def record_training_step(self,
                           episode: int,
                           step: int,
                           reward: float,
                           metrics: Dict[str, Any]):
        """
        Record training step information.

        Args:
            episode: Episode number
            step: Step number within episode
            reward: Step reward
            metrics: Additional training metrics
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'episode': episode,
            'step': step,
            'reward': reward,
            'metrics': metrics
        }
        self.training_history.append(record)

    def record_episode_end(self,
                          episode: int,
                          total_reward: float,
                          total_steps: int,
                          episode_metrics: Dict[str, Any]):
        """
        Record episode completion information.

        Args:
            episode: Episode number
            total_reward: Total episode reward
            total_steps: Total steps in episode
            episode_metrics: Episode-level metrics
        """
        # Find episode records
        episode_records = [r for r in self.training_history if r['episode'] == episode]

        if episode_records:
            # Update the last record with episode summary
            episode_records[-1].update({
                'episode_end': True,
                'total_reward': total_reward,
                'total_steps': total_steps,
                'episode_metrics': episode_metrics
            })

    def record_evaluation(self,
                         evaluation_id: str,
                         num_episodes: int,
                         results: Dict[str, Any]):
        """
        Record evaluation results.

        Args:
            evaluation_id: Unique evaluation identifier
            num_episodes: Number of evaluation episodes
            results: Evaluation results
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_id': evaluation_id,
            'num_episodes': num_episodes,
            'results': results
        }
        self.evaluation_history.append(record)

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.training_history:
            return {'error': 'No training history available'}

        episode_records = [r for r in self.training_history if r.get('episode_end', False)]

        if not episode_records:
            return {'error': 'No complete episodes found'}

        rewards = [r['total_reward'] for r in episode_records]
        steps = [r['total_steps'] for r in episode_records]

        return {
            'total_episodes': len(episode_records),
            'total_training_steps': len(self.training_history),
            'reward_stats': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards),
                'final': rewards[-1] if rewards else 0
            },
            'steps_stats': {
                'mean': np.mean(steps),
                'std': np.std(steps),
                'min': np.min(steps),
                'max': np.max(steps),
                'total': np.sum(steps)
            },
            'training_duration': (datetime.now() - self.created_at).total_seconds()
        }

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get evaluation summary statistics."""
        if not self.evaluation_history:
            return {'error': 'No evaluation history available'}

        latest_eval = self.evaluation_history[-1]
        all_results = [e['results'] for e in self.evaluation_history]

        return {
            'total_evaluations': len(self.evaluation_history),
            'latest_evaluation': latest_eval,
            'evaluation_trend': all_results,
            'last_updated': self.evaluation_history[-1]['timestamp']
        }

    def save(self, filepath: str):
        """Save performance data to file."""
        try:
            data = {
                'agent_id': self.agent_id,
                'created_at': self.created_at.isoformat(),
                'training_history': self.training_history,
                'evaluation_history': self.evaluation_history
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            manager_logger.info(f"Performance data saved to {filepath}")

        except Exception as e:
            manager_logger.error(f"Failed to save performance data: {e}")
            raise

    def load(self, filepath: str):
        """Load performance data from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.agent_id = data['agent_id']
            self.created_at = datetime.fromisoformat(data['created_at'])
            self.training_history = data['training_history']
            self.evaluation_history = data['evaluation_history']

            manager_logger.info(f"Performance data loaded from {filepath}")

        except Exception as e:
            manager_logger.error(f"Failed to load performance data: {e}")
            raise


class ProfessionalAgentManager:
    """
    Professional manager for multiple RL agents.

    Provides centralized management for training, evaluation,
    and monitoring of multiple agents with comprehensive
    experiment tracking and comparison capabilities.
    """

    def __init__(self, workspace_dir: str = "agents_workspace"):
        self.workspace_dir = workspace_dir
        self.agents: Dict[str, BaseAgent] = {}
        self.performance_trackers: Dict[str, AgentPerformanceTracker] = {}
        self.experiment_metadata = {}

        # Create workspace directory
        os.makedirs(workspace_dir, exist_ok=True)

        manager_logger.info(f"Professional Agent Manager initialized in {workspace_dir}")

    def register_agent(self,
                      agent_id: str,
                      agent: BaseAgent,
                      experiment_name: Optional[str] = None) -> None:
        """
        Register an agent for management.

        Args:
            agent_id: Unique agent identifier
            agent: Agent instance
            experiment_name: Optional experiment name
        """
        if agent_id in self.agents:
            manager_logger.warning(f"Agent {agent_id} already registered, overwriting")

        self.agents[agent_id] = agent
        self.performance_trackers[agent_id] = AgentPerformanceTracker(agent_id)

        # Store experiment metadata
        self.experiment_metadata[agent_id] = {
            'experiment_name': experiment_name or f"experiment_{agent_id}",
            'registered_at': datetime.now().isoformat(),
            'agent_type': type(agent).__name__,
            'agent_config': getattr(agent, 'config', {})
        }

        manager_logger.info(f"Registered agent {agent_id} of type {type(agent).__name__}")

    def create_and_register_agent(self,
                                 agent_id: str,
                                 agent_type: str,
                                 state_dim: int,
                                 action_dim: int,
                                 experiment_name: Optional[str] = None,
                                 custom_config: Optional[Dict] = None,
                                 **kwargs) -> BaseAgent:
        """
        Create and register an agent in one step.

        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent to create
            state_dim: State space dimension
            action_dim: Action space dimension
            experiment_name: Optional experiment name
            custom_config: Optional custom configuration
            **kwargs: Additional creation arguments

        Returns:
            Created and registered agent
        """
        agent = AgentFactory.create_agent(
            agent_type=agent_type,
            state_dim=state_dim,
            action_dim=action_dim,
            custom_config=custom_config,
            **kwargs
        )

        self.register_agent(agent_id, agent, experiment_name)
        return agent

    def get_agent(self, agent_id: str) -> BaseAgent:
        """Get registered agent by ID."""
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found")
        return self.agents[agent_id]

    def list_agents(self) -> List[str]:
        """List all registered agent IDs."""
        return list(self.agents.keys())

    def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive information about an agent."""
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found")

        agent = self.agents[agent_id]
        tracker = self.performance_trackers[agent_id]
        metadata = self.experiment_metadata[agent_id]

        info = {
            'agent_id': agent_id,
            'agent_stats': agent.get_stats() if hasattr(agent, 'get_stats') else {},
            'training_summary': tracker.get_training_summary(),
            'evaluation_summary': tracker.get_evaluation_summary(),
            'experiment_metadata': metadata
        }

        return info

    def save_agent(self, agent_id: str, include_performance: bool = True) -> str:
        """
        Save agent and optionally its performance data.

        Args:
            agent_id: Agent to save
            include_performance: Whether to save performance data

        Returns:
            Filepath where agent was saved
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found")

        agent = self.agents[agent_id]

        # Create agent directory
        agent_dir = os.path.join(self.workspace_dir, agent_id)
        os.makedirs(agent_dir, exist_ok=True)

        # Save agent
        agent_filepath = os.path.join(agent_dir, "agent")
        agent.save(agent_filepath)

        # Save performance data
        if include_performance:
            performance_filepath = os.path.join(agent_dir, "performance.json")
            self.performance_trackers[agent_id].save(performance_filepath)

        # Save metadata
        metadata_filepath = os.path.join(agent_dir, "metadata.json")
        with open(metadata_filepath, 'w') as f:
            json.dump(self.experiment_metadata[agent_id], f, indent=2)

        manager_logger.info(f"Agent {agent_id} saved to {agent_dir}")
        return agent_dir

    def load_agent(self, agent_id: str, agent_dir: str) -> None:
        """
        Load agent from directory.

        Args:
            agent_id: Agent identifier
            agent_dir: Directory containing agent files
        """
        try:
            # Load metadata
            metadata_filepath = os.path.join(agent_dir, "metadata.json")
            with open(metadata_filepath, 'r') as f:
                metadata = json.load(f)

            agent_type = metadata.get('agent_type', '').lower()

            # Determine agent type and create instance
            # This is a simplified approach - in practice, you'd need more sophisticated type detection
            if 'dqn' in agent_type:
                agent = ModernDQNAgent(state_dim=1, num_actions=1)  # Placeholder dimensions
            elif 'td3' in agent_type:
                agent = ModernTD3Agent(state_dim=1, action_dim=1)  # Placeholder dimensions
            else:
                raise ValueError(f"Cannot determine agent type from metadata: {agent_type}")

            # Load agent
            agent_filepath = os.path.join(agent_dir, "agent")
            agent.load(agent_filepath)

            # Register agent
            self.agents[agent_id] = agent
            self.experiment_metadata[agent_id] = metadata

            # Load performance data
            performance_filepath = os.path.join(agent_dir, "performance.json")
            if os.path.exists(performance_filepath):
                tracker = AgentPerformanceTracker(agent_id)
                tracker.load(performance_filepath)
                self.performance_trackers[agent_id] = tracker
            else:
                self.performance_trackers[agent_id] = AgentPerformanceTracker(agent_id)

            manager_logger.info(f"Agent {agent_id} loaded from {agent_dir}")

        except Exception as e:
            manager_logger.error(f"Failed to load agent {agent_id}: {e}")
            raise

    def compare_agents(self, agent_ids: List[str]) -> Dict[str, Any]:
        """
        Compare performance of multiple agents.

        Args:
            agent_ids: List of agent IDs to compare

        Returns:
            Comparison results
        """
        if not all(aid in self.agents for aid in agent_ids):
            missing = [aid for aid in agent_ids if aid not in self.agents]
            raise KeyError(f"Agents not found: {missing}")

        comparison = {
            'agents': agent_ids,
            'comparison_timestamp': datetime.now().isoformat(),
            'agent_summaries': {},
            'performance_comparison': {}
        }

        # Get summaries for each agent
        for agent_id in agent_ids:
            comparison['agent_summaries'][agent_id] = self.get_agent_info(agent_id)

        # Compare training performance
        training_rewards = {}
        training_episodes = {}

        for agent_id in agent_ids:
            summary = comparison['agent_summaries'][agent_id]['training_summary']
            if 'reward_stats' in summary:
                training_rewards[agent_id] = summary['reward_stats']
                training_episodes[agent_id] = summary['total_episodes']

        comparison['performance_comparison'] = {
            'training_rewards': training_rewards,
            'training_episodes': training_episodes,
            'best_mean_reward': max(training_rewards.keys(),
                                  key=lambda x: training_rewards[x].get('mean', float('-inf'))) if training_rewards else None,
            'best_final_reward': max(training_rewards.keys(),
                                   key=lambda x: training_rewards[x].get('final', float('-inf'))) if training_rewards else None
        }

        return comparison

    def get_workspace_summary(self) -> Dict[str, Any]:
        """Get summary of entire workspace."""
        summary = {
            'workspace_dir': self.workspace_dir,
            'total_agents': len(self.agents),
            'agent_list': list(self.agents.keys()),
            'experiment_count': len(set(meta.get('experiment_name', '')
                                       for meta in self.experiment_metadata.values())),
            'agent_types': list(set(type(agent).__name__ for agent in self.agents.values()))
        }

        return summary


# Example usage and testing
if __name__ == "__main__":
    # Example: Professional agent management
    print("=== Professional Agent Manager Example ===")

    # Create manager
    manager = ProfessionalAgentManager("example_workspace")

    # List available agent types
    print("\nAvailable agent types:")
    for agent_type in AgentFactory.list_available_agents():
        info = AgentFactory.get_agent_info(agent_type)
        print(f"  - {agent_type}: {info['agent_class']}")

    # Create and register agents
    print("\nCreating agents...")

    # Create DQN agent
    dqn_agent = manager.create_and_register_agent(
        agent_id="dqn_trader_001",
        agent_type="modern_dqn",
        state_dim=20,
        action_dim=3,
        experiment_name="DQN Trading Experiment",
        custom_config={'learning_rate': 0.0005}
    )

    # Create TD3 agent
    td3_agent = manager.create_and_register_agent(
        agent_id="td3_trader_001",
        agent_type="modern_td3",
        state_dim=20,
        action_dim=1,
        experiment_name="TD3 Trading Experiment"
    )

    print(f"Registered agents: {manager.list_agents()}")

    # Get workspace summary
    summary = manager.get_workspace_summary()
    print(f"\nWorkspace summary: {summary}")

    print("\n=== Professional Agent Manager Ready ===")
    print("Use this manager for comprehensive agent management in your trading system!")
