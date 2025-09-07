"""
Comprehensive test suite for RL algorithms (PPO, SAC, A3C, ICM).

This module provides extensive testing for all modern RL algorithms
including unit tests, integration tests, and performance benchmarks.
"""

import os
import shutil
import sys
import tempfile
import unittest
import warnings
from unittest.mock import Mock

import numpy as np
import torch
import torch.nn as nn

# Add parent directories to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_root)

from model.rl_system.algorithms.actor_critic.sac import SACAgent, SACConfig
from model.rl_system.algorithms.exploration.icm import ICMConfig, ICMModule
from model.rl_system.algorithms.policy_based.a3c import A3CAgent, A3CConfig
from model.rl_system.algorithms.policy_based.ppo import PPOAgent, PPOConfig
from model.rl_system.integration import create_agent, setup_training_environment

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class TestPPOAlgorithm(unittest.TestCase):
    """Test cases for PPO algorithm implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 10
        self.action_dim = 4
        self.config = PPOConfig(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            learning_rate=0.001,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            n_epochs=4,
            batch_size=64,
            entropy_coef=0.01,
            value_coef=0.5
        )
        self.agent = PPOAgent(self.config)

        # Create temporary directory for model saving
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_ppo_initialization(self):
        """Test PPO agent initialization."""
        self.assertIsInstance(self.agent.policy_net, nn.Module)
        self.assertIsInstance(self.agent.value_net, nn.Module)
        self.assertIsInstance(self.agent.optimizer, torch.optim.Optimizer)

        # Test network architectures
        self.assertTrue(hasattr(self.agent.policy_net, 'mean'))
        self.assertTrue(hasattr(self.agent.policy_net, 'log_std'))

        # Test configuration
        self.assertEqual(self.agent.config.clip_ratio, 0.2)
        self.assertEqual(self.agent.config.gae_lambda, 0.95)

    def test_ppo_action_selection(self):
        """Test PPO action selection."""
        state = np.random.randn(self.state_dim).astype(np.float32)

        # Test action selection
        action = self.agent.get_action(state)
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertTrue(np.isfinite(action).all())

        # Test action with evaluation mode
        action_eval = self.agent.get_action(state, deterministic=True)
        self.assertEqual(action_eval.shape, (self.action_dim,))

        # Test multiple actions (batch)
        states = np.random.randn(5, self.state_dim).astype(np.float32)
        actions = self.agent.get_actions(states)
        self.assertEqual(actions.shape, (5, self.action_dim))

    def test_ppo_memory_storage(self):
        """Test PPO memory storage functionality."""
        state = np.random.randn(self.state_dim).astype(np.float32)
        action = np.random.randn(self.action_dim).astype(np.float32)
        reward = 1.0
        next_state = np.random.randn(self.state_dim).astype(np.float32)
        done = False

        # Store transition
        self.agent.store_transition(state, action, reward, next_state, done)

        # Check memory
        self.assertEqual(len(self.agent.memory), 1)

        # Store multiple transitions
        for _ in range(10):
            self.agent.store_transition(state, action, reward, next_state, done)

        self.assertEqual(len(self.agent.memory), 11)

    def test_ppo_gae_computation(self):
        """Test Generalized Advantage Estimation."""
        # Store several transitions
        for _ in range(10):
            state = np.random.randn(self.state_dim).astype(np.float32)
            action = np.random.randn(self.action_dim).astype(np.float32)
            reward = np.random.randn()
            next_state = np.random.randn(self.state_dim).astype(np.float32)
            done = False

            self.agent.store_transition(state, action, reward, next_state, done)

        # Test GAE computation
        advantages = self.agent._compute_gae()
        self.assertEqual(len(advantages), len(self.agent.memory))
        self.assertTrue(np.isfinite(advantages).all())

    def test_ppo_update(self):
        """Test PPO policy update."""
        # Fill memory with transitions
        for _ in range(100):
            state = np.random.randn(self.state_dim).astype(np.float32)
            action = np.random.randn(self.action_dim).astype(np.float32)
            reward = np.random.randn()
            next_state = np.random.randn(self.state_dim).astype(np.float32)
            done = np.random.choice([True, False])

            self.agent.store_transition(state, action, reward, next_state, done)

        # Perform update
        update_info = self.agent.update()

        # Check update results
        self.assertIsInstance(update_info, dict)
        self.assertIn('policy_loss', update_info)
        self.assertIn('value_loss', update_info)
        self.assertIn('entropy', update_info)

        # Check that losses are finite
        self.assertTrue(np.isfinite(update_info['policy_loss']))
        self.assertTrue(np.isfinite(update_info['value_loss']))

    def test_ppo_save_load(self):
        """Test PPO model save/load functionality."""
        # Save model
        save_path = os.path.join(self.temp_dir, 'ppo_model.pth')
        self.agent.save(save_path)
        self.assertTrue(os.path.exists(save_path))

        # Create new agent and load
        new_agent = PPOAgent(self.config)
        new_agent.load(save_path)

        # Test that loaded model produces same outputs
        state = np.random.randn(self.state_dim).astype(np.float32)
        action1 = self.agent.get_action(state, deterministic=True)
        action2 = new_agent.get_action(state, deterministic=True)

        np.testing.assert_allclose(action1, action2, rtol=1e-5)


class TestSACAlgorithm(unittest.TestCase):
    """Test cases for SAC algorithm implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 8
        self.action_dim = 3
        self.config = SACConfig(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            learning_rate=0.0003,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            automatic_entropy_tuning=True,
            buffer_size=100000,
            batch_size=256
        )
        self.agent = SACAgent(self.config)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_sac_initialization(self):
        """Test SAC agent initialization."""
        # Test network components
        self.assertIsInstance(self.agent.actor, nn.Module)
        self.assertIsInstance(self.agent.critic1, nn.Module)
        self.assertIsInstance(self.agent.critic2, nn.Module)
        self.assertIsInstance(self.agent.critic1_target, nn.Module)
        self.assertIsInstance(self.agent.critic2_target, nn.Module)

        # Test optimizers
        self.assertIsInstance(self.agent.actor_optimizer, torch.optim.Optimizer)
        self.assertIsInstance(self.agent.critic1_optimizer, torch.optim.Optimizer)
        self.assertIsInstance(self.agent.critic2_optimizer, torch.optim.Optimizer)

        # Test automatic entropy tuning
        if self.config.automatic_entropy_tuning:
            self.assertTrue(hasattr(self.agent, 'log_alpha'))
            self.assertTrue(hasattr(self.agent, 'alpha_optimizer'))

    def test_sac_action_selection(self):
        """Test SAC action selection."""
        state = np.random.randn(self.state_dim).astype(np.float32)

        # Test stochastic action
        action = self.agent.get_action(state)
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertTrue((action >= -1).all() and (action <= 1).all())  # Tanh bounded

        # Test deterministic action
        det_action = self.agent.get_action(state, deterministic=True)
        self.assertEqual(det_action.shape, (self.action_dim,))

    def test_sac_replay_buffer(self):
        """Test SAC replay buffer functionality."""
        # Test buffer initialization
        self.assertEqual(len(self.agent.replay_buffer), 0)

        # Add transitions
        for _ in range(50):
            state = np.random.randn(self.state_dim).astype(np.float32)
            action = np.random.randn(self.action_dim).astype(np.float32)
            reward = np.random.randn()
            next_state = np.random.randn(self.state_dim).astype(np.float32)
            done = np.random.choice([True, False])

            self.agent.store_transition(state, action, reward, next_state, done)

        self.assertEqual(len(self.agent.replay_buffer), 50)

        # Test sampling
        if len(self.agent.replay_buffer) >= self.agent.config.batch_size:
            batch = self.agent.replay_buffer.sample(32)
            self.assertIn('states', batch)
            self.assertIn('actions', batch)
            self.assertIn('rewards', batch)
            self.assertIn('next_states', batch)
            self.assertIn('dones', batch)

    def test_sac_twin_critics(self):
        """Test SAC twin critic architecture."""
        state = torch.randn(5, self.state_dim)
        action = torch.randn(5, self.action_dim)

        # Test both critics
        q1 = self.agent.critic1(state, action)
        q2 = self.agent.critic2(state, action)

        self.assertEqual(q1.shape, (5, 1))
        self.assertEqual(q2.shape, (5, 1))

        # Test target critics
        q1_target = self.agent.critic1_target(state, action)
        q2_target = self.agent.critic2_target(state, action)

        self.assertEqual(q1_target.shape, (5, 1))
        self.assertEqual(q2_target.shape, (5, 1))

    def test_sac_soft_update(self):
        """Test SAC soft target network updates."""
        # Get initial target parameters
        initial_target_params = list(self.agent.critic1_target.parameters())

        # Perform soft update
        self.agent._soft_update_targets()

        # Check that target parameters changed
        updated_target_params = list(self.agent.critic1_target.parameters())

        # Target should have moved toward main network
        for initial, updated in zip(initial_target_params, updated_target_params):
            self.assertFalse(torch.equal(initial, updated))

    def test_sac_update(self):
        """Test SAC update mechanism."""
        # Fill replay buffer
        for _ in range(1000):  # Need more samples for SAC
            state = np.random.randn(self.state_dim).astype(np.float32)
            action = np.random.randn(self.action_dim).astype(np.float32)
            reward = np.random.randn()
            next_state = np.random.randn(self.state_dim).astype(np.float32)
            done = np.random.choice([True, False])

            self.agent.store_transition(state, action, reward, next_state, done)

        # Perform update
        update_info = self.agent.update()

        # Check update results
        self.assertIsInstance(update_info, dict)
        self.assertIn('critic1_loss', update_info)
        self.assertIn('critic2_loss', update_info)
        self.assertIn('actor_loss', update_info)

        if self.config.automatic_entropy_tuning:
            self.assertIn('alpha_loss', update_info)
            self.assertIn('alpha_value', update_info)


class TestA3CAlgorithm(unittest.TestCase):
    """Test cases for A3C algorithm implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 6
        self.action_dim = 2
        self.config = A3CConfig(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            learning_rate=0.001,
            gamma=0.99,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            n_workers=2  # Use fewer workers for testing
        )
        self.agent = A3CAgent(self.config)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_a3c_initialization(self):
        """Test A3C agent initialization."""
        # Test shared network
        self.assertIsInstance(self.agent.shared_net, nn.Module)
        self.assertTrue(hasattr(self.agent.shared_net, 'actor_head'))
        self.assertTrue(hasattr(self.agent.shared_net, 'critic_head'))

        # Test optimizer
        self.assertIsInstance(self.agent.shared_optimizer, torch.optim.Optimizer)

        # Test worker configuration
        self.assertEqual(self.agent.config.n_workers, 2)

    def test_a3c_action_selection(self):
        """Test A3C action selection."""
        state = np.random.randn(self.state_dim).astype(np.float32)

        # Test action selection
        action = self.agent.get_action(state)
        self.assertEqual(action.shape, (self.action_dim,))

        # Test value estimation
        value = self.agent.get_value(state)
        self.assertIsInstance(value, (int, float))

    def test_a3c_gradient_computation(self):
        """Test A3C gradient computation."""
        # Create some sample data
        states = []
        actions = []
        rewards = []
        values = []

        for _ in range(10):
            state = np.random.randn(self.state_dim).astype(np.float32)
            action = np.random.randn(self.action_dim).astype(np.float32)
            reward = np.random.randn()
            value = np.random.randn()

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)

        # Compute gradients
        grads = self.agent._compute_gradients(states, actions, rewards, values)

        # Check gradients exist and are finite
        self.assertIsInstance(grads, list)
        for grad in grads:
            if grad is not None:
                self.assertTrue(torch.isfinite(grad).all())

    def test_a3c_worker_simulation(self):
        """Test A3C worker process simulation."""
        # Mock environment for testing
        mock_env = Mock()
        mock_env.reset.return_value = np.random.randn(self.state_dim)
        mock_env.step.return_value = (
            np.random.randn(self.state_dim),  # next_state
            np.random.randn(),  # reward
            False,  # done
            {}  # info
        )

        # Test worker rollout simulation
        states, actions, rewards, values = [], [], [], []
        state = mock_env.reset()

        for _ in range(5):  # Short rollout
            action = self.agent.get_action(state)
            value = self.agent.get_value(state)
            next_state, reward, done, _ = mock_env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)

            state = next_state
            if done:
                break

        # Verify rollout data
        self.assertEqual(len(states), len(actions))
        self.assertEqual(len(actions), len(rewards))
        self.assertEqual(len(rewards), len(values))


class TestICMModule(unittest.TestCase):
    """Test cases for Intrinsic Curiosity Module."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 12
        self.action_dim = 4
        self.feature_dim = 64

        self.config = ICMConfig(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            feature_dim=self.feature_dim,
            learning_rate=0.001,
            eta=0.01,  # Intrinsic reward scaling
            beta=0.2   # Inverse model loss weight
        )
        self.icm = ICMModule(self.config)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_icm_initialization(self):
        """Test ICM module initialization."""
        # Test network components
        self.assertIsInstance(self.icm.feature_net, nn.Module)
        self.assertIsInstance(self.icm.inverse_net, nn.Module)
        self.assertIsInstance(self.icm.forward_net, nn.Module)

        # Test optimizer
        self.assertIsInstance(self.icm.optimizer, torch.optim.Optimizer)

    def test_icm_feature_extraction(self):
        """Test ICM feature extraction."""
        states = torch.randn(5, self.state_dim)

        # Extract features
        features = self.icm.feature_net(states)

        self.assertEqual(features.shape, (5, self.feature_dim))
        self.assertTrue(torch.isfinite(features).all())

    def test_icm_inverse_model(self):
        """Test ICM inverse model prediction."""
        states = torch.randn(5, self.state_dim)
        next_states = torch.randn(5, self.state_dim)

        # Get features
        features = self.icm.feature_net(states)
        next_features = self.icm.feature_net(next_states)

        # Predict action
        predicted_actions = self.icm.inverse_net(torch.cat([features, next_features], dim=1))

        self.assertEqual(predicted_actions.shape, (5, self.action_dim))

    def test_icm_forward_model(self):
        """Test ICM forward model prediction."""
        states = torch.randn(5, self.state_dim)
        actions = torch.randn(5, self.action_dim)

        # Get features
        features = self.icm.feature_net(states)

        # Predict next features
        predicted_next_features = self.icm.forward_net(torch.cat([features, actions], dim=1))

        self.assertEqual(predicted_next_features.shape, (5, self.feature_dim))

    def test_icm_intrinsic_reward(self):
        """Test ICM intrinsic reward computation."""
        states = np.random.randn(3, self.state_dim).astype(np.float32)
        actions = np.random.randn(3, self.action_dim).astype(np.float32)
        next_states = np.random.randn(3, self.state_dim).astype(np.float32)

        # Compute intrinsic rewards
        intrinsic_rewards = self.icm.compute_intrinsic_reward(states, actions, next_states)

        self.assertEqual(intrinsic_rewards.shape, (3,))
        self.assertTrue(np.isfinite(intrinsic_rewards).all())
        self.assertTrue((intrinsic_rewards >= 0).all())  # Should be non-negative

    def test_icm_update(self):
        """Test ICM module update."""
        # Generate batch data
        batch_size = 32
        states = np.random.randn(batch_size, self.state_dim).astype(np.float32)
        actions = np.random.randn(batch_size, self.action_dim).astype(np.float32)
        next_states = np.random.randn(batch_size, self.state_dim).astype(np.float32)

        # Update ICM
        update_info = self.icm.update(states, actions, next_states)

        # Check update results
        self.assertIsInstance(update_info, dict)
        self.assertIn('forward_loss', update_info)
        self.assertIn('inverse_loss', update_info)
        self.assertIn('total_loss', update_info)

        # Check losses are finite
        self.assertTrue(np.isfinite(update_info['forward_loss']))
        self.assertTrue(np.isfinite(update_info['inverse_loss']))


class TestRLIntegration(unittest.TestCase):
    """Test cases for RL system integration."""

    def test_agent_creation_factory(self):
        """Test agent creation through factory system."""
        state_dim = 8
        action_dim = 3

        # Test PPO creation
        ppo_agent = create_agent('ppo', state_dim, action_dim)
        self.assertIsInstance(ppo_agent, PPOAgent)

        # Test SAC creation
        sac_agent = create_agent('sac', state_dim, action_dim)
        self.assertIsInstance(sac_agent, SACAgent)

        # Test A3C creation
        a3c_agent = create_agent('a3c', state_dim, action_dim)
        self.assertIsInstance(a3c_agent, A3CAgent)

    def test_training_environment_setup(self):
        """Test training environment setup."""
        # Test discrete environment setup
        setup_training_environment('discrete', 10, 4)

        # Test continuous environment setup
        setup_training_environment('continuous', 8, 3)

        # Should not raise exceptions
        self.assertTrue(True)

    def test_algorithm_compatibility(self):
        """Test algorithm compatibility with different environment types."""
        state_dim = 6
        action_dim = 2

        # Test algorithm-environment compatibility
        algorithms = ['ppo', 'sac', 'a3c']
        env_types = ['continuous', 'discrete']

        for algorithm in algorithms:
            for env_type in env_types:
                try:
                    create_agent(algorithm, state_dim, action_dim)
                    setup_training_environment(env_type, state_dim, action_dim)
                    # Should succeed for compatible combinations
                    self.assertTrue(True)
                except Exception as e:
                    # Some combinations might be incompatible
                    if 'not supported' in str(e).lower():
                        continue
                    else:
                        raise


class TestRLPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests for RL algorithms."""

    def test_algorithm_training_speed(self):
        """Benchmark training speed of different algorithms."""
        state_dim = 8
        action_dim = 3
        n_updates = 10

        algorithms = ['ppo', 'sac']  # Test subset for speed
        timing_results = {}

        for algorithm in algorithms:
            agent = create_agent(algorithm, state_dim, action_dim)

            # Generate random training data
            for _ in range(100):
                state = np.random.randn(state_dim).astype(np.float32)
                action = np.random.randn(action_dim).astype(np.float32)
                reward = np.random.randn()
                next_state = np.random.randn(state_dim).astype(np.float32)
                done = False

                agent.store_transition(state, action, reward, next_state, done)

            # Time updates
            import time
            start_time = time.time()

            for _ in range(n_updates):
                if hasattr(agent, 'update'):
                    agent.update()

            elapsed_time = time.time() - start_time
            timing_results[algorithm] = elapsed_time / n_updates

        # Print results for manual inspection
        print("Algorithm timing results (avg per update):")
        for alg, time_per_update in timing_results.items():
            print(f"  {alg}: {time_per_update:.4f} seconds")

        # Basic sanity check - updates should complete in reasonable time
        for time_per_update in timing_results.values():
            self.assertLess(time_per_update, 5.0)  # Should be under 5 seconds per update

    def test_memory_efficiency(self):
        """Test memory efficiency of different algorithms."""
        import gc

        import psutil

        state_dim = 10
        action_dim = 4

        algorithms = ['ppo', 'sac']
        memory_usage = {}

        for algorithm in algorithms:
            # Measure memory before
            gc.collect()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Create agent
            agent = create_agent(algorithm, state_dim, action_dim)

            # Add some data
            for _ in range(1000):
                state = np.random.randn(state_dim).astype(np.float32)
                action = np.random.randn(action_dim).astype(np.float32)
                reward = np.random.randn()
                next_state = np.random.randn(state_dim).astype(np.float32)
                done = False

                agent.store_transition(state, action, reward, next_state, done)

            # Measure memory after
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_usage[algorithm] = memory_after - memory_before

            # Clean up
            del agent
            gc.collect()

        # Print results
        print("Memory usage results:")
        for alg, usage in memory_usage.items():
            print(f"  {alg}: {usage:.2f} MB")

        # Sanity check - should use less than 1GB
        for usage in memory_usage.values():
            self.assertLess(usage, 1000)  # Less than 1GB


def create_test_suite():
    """Create comprehensive test suite for RL algorithms."""
    test_suite = unittest.TestSuite()

    # Algorithm-specific tests
    test_suite.addTest(unittest.makeSuite(TestPPOAlgorithm))
    test_suite.addTest(unittest.makeSuite(TestSACAlgorithm))
    test_suite.addTest(unittest.makeSuite(TestA3CAlgorithm))
    test_suite.addTest(unittest.makeSuite(TestICMModule))

    # Integration tests
    test_suite.addTest(unittest.makeSuite(TestRLIntegration))

    # Performance benchmarks
    test_suite.addTest(unittest.makeSuite(TestRLPerformanceBenchmarks))

    return test_suite


if __name__ == '__main__':
    # Run the test suite
    print("Running RL Algorithm Test Suite")
    print("=" * 50)

    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASSED' if success else 'FAILED'}")
