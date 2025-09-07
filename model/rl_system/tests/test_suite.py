"""
Comprehensive Testing Framework for RL System.

This module provides extensive testing capabilities for all components
of the RL system including unit tests, integration tests, and benchmarks.
"""

import os
import time
import unittest
import warnings
from typing import Dict

import numpy as np
import pandas as pd

import util.loggers as loggers

from ..algorithms.actor_critic.continuous_control import ActorCriticConfig, DDPGAgent
from ..algorithms.policy_based.policy_gradients import (
    PolicyGradientConfig,
    REINFORCEAgent,
)
from ..algorithms.value_based.dqn_family import DQNAgent, DQNConfig
from ..algorithms.value_based.tabular_methods import QLearningAgent, TabularConfig

# Import all components to test
from ..core.base_agents import BaseRLAgent
from ..environments.trading_env import TradingConfig, TradingEnvironment
from ..integration.rl_system import RLSystemManager
from ..training.trainer import RLTrainer, TrainingConfig

logger = loggers.setup_loggers()
rl_logger = logger['rl']

warnings.filterwarnings('ignore', category=FutureWarning)


class TestTradingEnvironment(unittest.TestCase):
    """Test cases for TradingEnvironment."""

    def setUp(self):
        """Set up test fixtures."""
        # Create dummy market data
        np.random.seed(42)
        self.data = pd.DataFrame({
            'open': np.random.uniform(95, 105, 100),
            'high': np.random.uniform(100, 110, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 105, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })

        # Ensure high >= low
        self.data['high'] = np.maximum(self.data['high'], self.data['low'])

        self.config = TradingConfig(
            initial_balance=10000,
            lookback_window=10
        )

    def test_environment_initialization(self):
        """Test environment initialization."""
        env = TradingEnvironment(self.data, self.config)

        self.assertEqual(env.initial_balance, 10000)
        self.assertEqual(env.balance, 10000)
        self.assertEqual(env.position, 0.0)
        self.assertEqual(env.current_step, 0)

    def test_environment_reset(self):
        """Test environment reset functionality."""
        env = TradingEnvironment(self.data, self.config)

        # Take some steps
        env.step(1)  # Buy
        env.step(2)  # Sell

        # Reset environment
        obs = env.reset()

        self.assertEqual(env.current_step, 0)
        self.assertEqual(env.balance, 10000)
        self.assertEqual(env.position, 0.0)
        self.assertIsInstance(obs, np.ndarray)

    def test_environment_step(self):
        """Test environment step functionality."""
        env = TradingEnvironment(self.data, self.config)
        initial_obs = env.reset()

        # Test different actions
        for action in range(env.get_action_space_size()):
            obs, reward, done, info = env.step(action)

            self.assertIsInstance(obs, np.ndarray)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(done, bool)
            self.assertIsInstance(info, dict)

            if done:
                break

    def test_observation_space(self):
        """Test observation space consistency."""
        env = TradingEnvironment(self.data, self.config)
        obs = env.reset()

        expected_size = env.get_observation_space_size()
        self.assertEqual(len(obs), expected_size)

        # Test multiple observations
        for _ in range(10):
            obs, _, done, _ = env.step(0)  # Hold action
            self.assertEqual(len(obs), expected_size)
            if done:
                break


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQN agent."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 50
        self.num_actions = 5
        self.config = DQNConfig(
            learning_rate=0.001,
            memory_size=1000,
            batch_size=32,
            hidden_dims=[64, 64]
        )

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = DQNAgent(self.state_dim, self.num_actions, self.config)

        self.assertEqual(agent.state_dim, self.state_dim)
        self.assertEqual(agent.num_actions, self.num_actions)
        self.assertEqual(agent.training_step, 0)
        self.assertEqual(agent.episode_count, 0)

    def test_action_selection(self):
        """Test action selection."""
        agent = DQNAgent(self.state_dim, self.num_actions, self.config)
        state = np.random.random(self.state_dim).astype(np.float32)

        # Test training mode
        action = agent.select_action(state, training=True)
        self.assertIsInstance(action, (int, np.integer))
        self.assertTrue(0 <= action < self.num_actions)

        # Test evaluation mode
        action = agent.select_action(state, training=False)
        self.assertIsInstance(action, (int, np.integer))
        self.assertTrue(0 <= action < self.num_actions)

    def test_agent_update(self):
        """Test agent update mechanism."""
        agent = DQNAgent(self.state_dim, self.num_actions, self.config)

        state = np.random.random(self.state_dim).astype(np.float32)
        action = 1
        reward = 1.0
        next_state = np.random.random(self.state_dim).astype(np.float32)
        done = False

        # Fill replay buffer
        for _ in range(100):
            metrics = agent.update(state, action, reward, next_state, done)
            state = next_state
            next_state = np.random.random(self.state_dim).astype(np.float32)

        # Should start returning metrics once batch size is reached
        self.assertIsInstance(metrics, dict)

    def test_save_load_checkpoint(self):
        """Test agent save/load functionality."""
        agent = DQNAgent(self.state_dim, self.num_actions, self.config)

        # Train for a few steps
        state = np.random.random(self.state_dim).astype(np.float32)
        for _ in range(50):
            action = agent.select_action(state, training=True)
            next_state = np.random.random(self.state_dim).astype(np.float32)
            agent.update(state, action, 1.0, next_state, False)
            state = next_state

        # Save checkpoint
        checkpoint_path = "test_checkpoint"
        agent.save_checkpoint(checkpoint_path)

        # Create new agent and load checkpoint
        new_agent = DQNAgent(self.state_dim, self.num_actions, self.config)
        new_agent.load_checkpoint(checkpoint_path)

        # Check that states match
        self.assertEqual(agent.training_step, new_agent.training_step)
        self.assertEqual(agent.episode_count, new_agent.episode_count)

        # Cleanup
        for ext in ['.json', '_weights.pth']:
            filepath = f"{checkpoint_path}{ext}"
            if os.path.exists(filepath):
                os.remove(filepath)


class TestTabularAgent(unittest.TestCase):
    """Test cases for tabular agents."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 4
        self.num_actions = 3
        self.config = TabularConfig(
            learning_rate=0.1,
            state_discretization=10
        )

    def test_q_learning_agent(self):
        """Test Q-Learning agent."""
        agent = QLearningAgent(self.state_dim, self.num_actions, self.config)

        self.assertEqual(agent.state_dim, self.state_dim)
        self.assertEqual(agent.num_actions, self.num_actions)

        # Test action selection
        state = np.random.random(self.state_dim)
        action = agent.select_action(state)
        self.assertTrue(0 <= action < self.num_actions)

        # Test update
        next_state = np.random.random(self.state_dim)
        metrics = agent.update(state, action, 1.0, next_state, False)
        self.assertIsInstance(metrics, dict)


class TestPolicyGradientAgent(unittest.TestCase):
    """Test cases for policy gradient agents."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 20
        self.num_actions = 4
        self.config = PolicyGradientConfig(
            learning_rate=0.001,
            hidden_dims=[32, 32]
        )

    def test_reinforce_agent(self):
        """Test REINFORCE agent."""
        agent = REINFORCEAgent(self.state_dim, self.num_actions, self.config)

        self.assertEqual(agent.state_dim, self.state_dim)
        self.assertEqual(agent.num_actions, self.num_actions)

        # Test action selection
        state = np.random.random(self.state_dim).astype(np.float32)
        action = agent.select_action(state)
        self.assertTrue(0 <= action < self.num_actions)

        # Test episode update
        for _ in range(10):
            state = np.random.random(self.state_dim).astype(np.float32)
            action = agent.select_action(state)
            reward = np.random.random()
            next_state = np.random.random(self.state_dim).astype(np.float32)
            done = (_ == 9)  # Last step

            metrics = agent.update(state, action, reward, next_state, done)

            if done:
                self.assertIsInstance(metrics, dict)
                break


class TestActorCriticAgent(unittest.TestCase):
    """Test cases for actor-critic agents."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 15
        self.action_dim = 2  # Continuous actions
        self.config = ActorCriticConfig(
            learning_rate=0.001,
            hidden_dims=[32, 32],
            memory_size=1000
        )

    def test_ddpg_agent(self):
        """Test DDPG agent."""
        agent = DDPGAgent(self.state_dim, self.action_dim, self.config)

        self.assertEqual(agent.state_dim, self.state_dim)
        self.assertEqual(agent.action_dim, self.action_dim)

        # Test action selection
        state = np.random.random(self.state_dim).astype(np.float32)
        action = agent.select_action(state)
        self.assertEqual(len(action), self.action_dim)
        self.assertTrue(np.all(np.abs(action) <= self.config.max_action))


class TestTrainingFramework(unittest.TestCase):
    """Test cases for training framework."""

    def setUp(self):
        """Set up test fixtures."""
        # Create simple environment
        data = pd.DataFrame({
            'open': [100] * 50,
            'high': [105] * 50,
            'low': [95] * 50,
            'close': [100] * 50,
            'volume': [1000] * 50
        })

        env_config = TradingConfig(lookback_window=5)
        self.environment = TradingEnvironment(data, env_config)

        # Create simple agent
        state_dim = self.environment.get_observation_space_size()
        action_dim = self.environment.get_action_space_size()
        agent_config = DQNConfig(hidden_dims=[16, 16], memory_size=100)
        self.agent = DQNAgent(state_dim, action_dim, agent_config)

        self.training_config = TrainingConfig(
            max_episodes=10,
            eval_frequency=5,
            save_frequency=10
        )

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = RLTrainer(self.agent, self.environment, self.training_config)

        self.assertEqual(trainer.agent, self.agent)
        self.assertEqual(trainer.environment, self.environment)
        self.assertEqual(trainer.config, self.training_config)

    def test_short_training(self):
        """Test short training run."""
        trainer = RLTrainer(self.agent, self.environment, self.training_config)

        # Run short training
        results = trainer.train()

        self.assertIsInstance(results, dict)
        self.assertIn('total_episodes', results)
        self.assertGreater(results['total_episodes'], 0)


class TestRLSystemIntegration(unittest.TestCase):
    """Test cases for RL system integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.system = RLSystemManager("test_experiments")

    def tearDown(self):
        """Clean up test files."""
        # Clean up test directory if it exists
        import shutil
        if os.path.exists("test_experiments"):
            shutil.rmtree("test_experiments")

    def test_system_initialization(self):
        """Test system initialization."""
        self.assertIsInstance(self.system.agent_registry, dict)
        self.assertGreater(len(self.system.agent_registry), 0)

    def test_agent_creation(self):
        """Test agent creation through system."""
        agent_types = ['dqn', 'q_learning', 'reinforce']

        for agent_type in agent_types:
            try:
                agent = self.system.create_agent(agent_type, 10, 5)
                self.assertIsInstance(agent, BaseRLAgent)
                self.assertEqual(agent.state_dim, 10)
            except Exception as e:
                self.fail(f"Failed to create {agent_type} agent: {e}")

    def test_environment_creation(self):
        """Test environment creation through system."""
        data = pd.DataFrame({
            'open': [100] * 30,
            'high': [105] * 30,
            'low': [95] * 30,
            'close': [100] * 30,
            'volume': [1000] * 30
        })

        env = self.system.create_environment(data)
        self.assertIsInstance(env, TradingEnvironment)

    def test_available_agents(self):
        """Test getting available agent types."""
        available = self.system.get_available_agents()
        self.assertIsInstance(available, dict)
        self.assertIn('dqn', available)
        self.assertIn('q_learning', available)


class BenchmarkSuite:
    """Benchmark suite for performance testing."""

    def __init__(self):
        self.results = {}

    def benchmark_agent_creation(self, agent_type: str, state_dim: int, action_dim: int, iterations: int = 100):
        """Benchmark agent creation time."""
        system = RLSystemManager()

        start_time = time.time()
        for _ in range(iterations):
            agent = system.create_agent(agent_type, state_dim, action_dim)
        end_time = time.time()

        avg_time = (end_time - start_time) / iterations
        self.results[f"{agent_type}_creation"] = avg_time

        return avg_time

    def benchmark_action_selection(self, agent: BaseRLAgent, state_dim: int, iterations: int = 1000):
        """Benchmark action selection speed."""
        state = np.random.random(state_dim).astype(np.float32)

        start_time = time.time()
        for _ in range(iterations):
            _ = agent.select_action(state, training=True)
        end_time = time.time()

        avg_time = (end_time - start_time) / iterations
        self.results[f"{agent.name}_action_selection"] = avg_time

        return avg_time

    def benchmark_training_step(self, agent: BaseRLAgent, state_dim: int, iterations: int = 100):
        """Benchmark training step speed."""
        state = np.random.random(state_dim).astype(np.float32)
        action = 0
        reward = 1.0
        next_state = np.random.random(state_dim).astype(np.float32)
        done = False

        start_time = time.time()
        for _ in range(iterations):
            agent.update(state, action, reward, next_state, done)
        end_time = time.time()

        avg_time = (end_time - start_time) / iterations
        self.results[f"{agent.name}_training_step"] = avg_time

        return avg_time

    def run_comprehensive_benchmark(self) -> Dict[str, float]:
        """Run comprehensive benchmark suite."""
        system = RLSystemManager()

        # Test parameters
        state_dim = 50
        action_dim = 5

        # Benchmark different agent types
        agent_types = ['dqn', 'q_learning', 'reinforce']

        for agent_type in agent_types:
            try:
                # Benchmark creation
                creation_time = self.benchmark_agent_creation(agent_type, state_dim, action_dim)
                print(f"{agent_type} creation: {creation_time:.6f}s")

                # Create agent for further benchmarks
                test_agent = system.create_agent(agent_type, state_dim, action_dim)

                # Benchmark action selection
                action_time = self.benchmark_action_selection(test_agent, state_dim)
                print(f"{agent_type} action selection: {action_time:.6f}s")

                # Benchmark training step
                training_time = self.benchmark_training_step(test_agent, state_dim)
                print(f"{agent_type} training step: {training_time:.6f}s")

            except Exception as e:
                print(f"Failed to benchmark {agent_type}: {e}")

        return self.results

    def save_results(self, filepath: str):
        """Save benchmark results to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)


def run_all_tests():
    """Run all unit tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestTradingEnvironment,
        TestDQNAgent,
        TestTabularAgent,
        TestPolicyGradientAgent,
        TestActorCriticAgent,
        TestTrainingFramework,
        TestRLSystemIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


def run_benchmarks():
    """Run performance benchmarks."""
    print("Running RL System Benchmarks...")
    print("=" * 50)

    benchmark = BenchmarkSuite()
    results = benchmark.comprehensive_benchmark()

    print("\nBenchmark Results:")
    print("=" * 50)
    for test_name, avg_time in results.items():
        print(f"{test_name}: {avg_time:.6f}s")

    # Save results
    benchmark.save_results("benchmark_results.json")
    print("\nBenchmark results saved to benchmark_results.json")

    return results


if __name__ == "__main__":
    print("RL System Test Suite")
    print("=" * 50)

    # Run unit tests
    print("Running unit tests...")
    test_success = run_all_tests()

    if test_success:
        print("\n✅ All tests passed!")

        # Run benchmarks
        print("\nRunning benchmarks...")
        run_benchmarks()

        print("\n✅ Testing and benchmarking completed successfully!")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
