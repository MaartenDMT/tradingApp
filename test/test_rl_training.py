"""
Test suite for RL training system and utilities.

This module tests the training infrastructure, configuration management,
and training loop functionality.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

# Add parent directories to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_root)

# RL system imports (after path setup)
try:
    from model.rl_system.config.algorithm_configs import (
        A3CConfig,
        ICMConfig,
        PPOConfig,
        SACConfig,
    )
    from model.rl_system.training.trainer import RLTrainer, TrainingConfig
    from model.rl_system.utils.metrics import MetricsTracker
    from model.rl_system.utils.visualization import TrainingVisualizer
except ImportError:
    # Handle case where RL system modules are not available
    RLTrainer = None
    TrainingConfig = None
    PPOConfig = None
    SACConfig = None
    A3CConfig = None
    ICMConfig = None
    MetricsTracker = None
    TrainingVisualizer = None


class TestTrainingConfig(unittest.TestCase):
    """Test cases for training configuration management."""

    def test_config_creation(self):
        """Test training configuration creation."""
        config = TrainingConfig(
            algorithm='ppo',
            max_episodes=1000,
            max_steps_per_episode=500,
            eval_frequency=100,
            save_frequency=200,
            log_frequency=10
        )

        self.assertEqual(config.algorithm, 'ppo')
        self.assertEqual(config.max_episodes, 1000)
        self.assertEqual(config.max_steps_per_episode, 500)
        self.assertEqual(config.eval_frequency, 100)

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid algorithm
        with self.assertRaises(ValueError):
            TrainingConfig(algorithm='invalid_algorithm')

        # Test negative values
        with self.assertRaises(ValueError):
            TrainingConfig(algorithm='ppo', max_episodes=-1)

        with self.assertRaises(ValueError):
            TrainingConfig(algorithm='ppo', max_steps_per_episode=0)

    def test_config_defaults(self):
        """Test configuration default values."""
        config = TrainingConfig(algorithm='ppo')

        # Check defaults are reasonable
        self.assertGreater(config.max_episodes, 0)
        self.assertGreater(config.max_steps_per_episode, 0)
        self.assertGreater(config.eval_frequency, 0)


class TestRLTrainer(unittest.TestCase):
    """Test cases for RL trainer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Mock agent
        self.mock_agent = Mock()
        self.mock_agent.get_action.return_value = np.array([0.5, -0.3, 0.1])
        self.mock_agent.update.return_value = {
            'policy_loss': 0.1,
            'value_loss': 0.05,
            'entropy': 2.5
        }
        self.mock_agent.save = Mock()
        self.mock_agent.load = Mock()

        # Mock environment
        self.mock_env = Mock()
        self.mock_env.reset.return_value = np.random.randn(8)
        self.mock_env.step.return_value = (
            np.random.randn(8),  # next_state
            1.0,  # reward
            False,  # done
            {'episode_reward': 10.0}  # info
        )

        # Training config
        self.config = TrainingConfig(
            algorithm='ppo',
            max_episodes=10,
            max_steps_per_episode=50,
            eval_frequency=5,
            save_frequency=5,
            log_frequency=2
        )

        self.trainer = RLTrainer(
            agent=self.mock_agent,
            env=self.mock_env,
            config=self.config,
            save_dir=self.temp_dir
        )

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertEqual(self.trainer.agent, self.mock_agent)
        self.assertEqual(self.trainer.env, self.mock_env)
        self.assertEqual(self.trainer.config, self.config)
        self.assertEqual(self.trainer.save_dir, self.temp_dir)

        # Check metrics tracker initialization
        self.assertIsNotNone(self.trainer.metrics)
        self.assertIsInstance(self.trainer.metrics, MetricsTracker)

    def test_single_episode_training(self):
        """Test single episode training."""
        # Setup episode ending
        self.mock_env.step.side_effect = [
            (np.random.randn(8), 1.0, False, {}),
            (np.random.randn(8), 1.0, False, {}),
            (np.random.randn(8), 1.0, True, {})  # Episode ends
        ]

        episode_reward = self.trainer._run_episode()

        # Check interactions happened
        self.mock_env.reset.assert_called_once()
        self.assertGreaterEqual(self.mock_env.step.call_count, 1)
        self.assertGreaterEqual(self.mock_agent.get_action.call_count, 1)

        # Check return value
        self.assertIsInstance(episode_reward, (int, float))

    def test_training_loop(self):
        """Test complete training loop."""
        # Setup early episode termination
        self.mock_env.step.return_value = (
            np.random.randn(8), 1.0, True, {'episode_reward': 10.0}
        )

        # Run training
        training_results = self.trainer.train()

        # Check results structure
        self.assertIsInstance(training_results, dict)
        self.assertIn('episode_rewards', training_results)
        self.assertIn('training_losses', training_results)
        self.assertIn('best_reward', training_results)

        # Check episode rewards
        episode_rewards = training_results['episode_rewards']
        self.assertEqual(len(episode_rewards), self.config.max_episodes)

        # Check that all episodes ran
        self.assertEqual(self.mock_env.reset.call_count, self.config.max_episodes)

    def test_evaluation_during_training(self):
        """Test evaluation functionality during training."""
        # Configure short episodes
        self.mock_env.step.return_value = (
            np.random.randn(8), 5.0, True, {'episode_reward': 5.0}
        )

        # Mock evaluation environment
        eval_env = Mock()
        eval_env.reset.return_value = np.random.randn(8)
        eval_env.step.return_value = (
            np.random.randn(8), 3.0, True, {'episode_reward': 3.0}
        )

        self.trainer.eval_env = eval_env

        # Run training with evaluation
        results = self.trainer.train()

        # Check evaluation occurred
        self.assertIn('eval_rewards', results)
        eval_rewards = results['eval_rewards']

        # Should have evaluations based on eval_frequency
        expected_evals = self.config.max_episodes // self.config.eval_frequency
        self.assertGreaterEqual(len(eval_rewards), expected_evals)

    def test_model_saving(self):
        """Test model saving functionality."""
        # Run short training
        self.mock_env.step.return_value = (
            np.random.randn(8), 1.0, True, {}
        )

        self.trainer.train()

        # Check that save was called
        self.assertGreater(self.mock_agent.save.call_count, 0)

        # Check save directory exists
        self.assertTrue(os.path.exists(self.temp_dir))

    def test_metrics_collection(self):
        """Test metrics collection during training."""
        # Run training
        self.mock_env.step.return_value = (
            np.random.randn(8), 2.0, True, {}
        )

        results = self.trainer.train()

        # Check metrics were collected
        self.assertIn('episode_rewards', results)
        self.assertIn('training_losses', results)

        # Check metrics tracker state
        self.assertGreater(len(self.trainer.metrics.episode_rewards), 0)


class TestMetricsTracker(unittest.TestCase):
    """Test cases for metrics tracking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = MetricsTracker()

    def test_metrics_initialization(self):
        """Test metrics tracker initialization."""
        self.assertEqual(len(self.metrics.episode_rewards), 0)
        self.assertEqual(len(self.metrics.training_losses), 0)
        self.assertEqual(len(self.metrics.eval_rewards), 0)

    def test_episode_reward_tracking(self):
        """Test episode reward tracking."""
        rewards = [10.0, 15.0, 8.0, 20.0]

        for reward in rewards:
            self.metrics.add_episode_reward(reward)

        self.assertEqual(len(self.metrics.episode_rewards), len(rewards))
        self.assertEqual(self.metrics.episode_rewards, rewards)

    def test_training_loss_tracking(self):
        """Test training loss tracking."""
        losses = {
            'policy_loss': 0.1,
            'value_loss': 0.05,
            'entropy': 2.3
        }

        self.metrics.add_training_loss(losses)

        self.assertEqual(len(self.metrics.training_losses), 1)
        self.assertEqual(self.metrics.training_losses[0], losses)

    def test_metrics_statistics(self):
        """Test metrics statistics computation."""
        rewards = [10.0, 20.0, 15.0, 25.0, 5.0]

        for reward in rewards:
            self.metrics.add_episode_reward(reward)

        stats = self.metrics.get_statistics()

        self.assertIn('mean_reward', stats)
        self.assertIn('std_reward', stats)
        self.assertIn('max_reward', stats)
        self.assertIn('min_reward', stats)

        self.assertEqual(stats['mean_reward'], np.mean(rewards))
        self.assertEqual(stats['max_reward'], max(rewards))
        self.assertEqual(stats['min_reward'], min(rewards))

    def test_moving_average(self):
        """Test moving average computation."""
        rewards = list(range(1, 21))  # 1 to 20

        for reward in rewards:
            self.metrics.add_episode_reward(reward)

        # Test different window sizes
        ma_5 = self.metrics.get_moving_average(window=5)
        ma_10 = self.metrics.get_moving_average(window=10)

        self.assertEqual(len(ma_5), len(rewards))
        self.assertEqual(len(ma_10), len(rewards))

        # Check last value (should be mean of last window)
        self.assertEqual(ma_5[-1], np.mean(rewards[-5:]))
        self.assertEqual(ma_10[-1], np.mean(rewards[-10:]))

    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        # Add some data
        self.metrics.add_episode_reward(10.0)
        self.metrics.add_training_loss({'loss': 0.1})

        # Check data exists
        self.assertGreater(len(self.metrics.episode_rewards), 0)
        self.assertGreater(len(self.metrics.training_losses), 0)

        # Reset
        self.metrics.reset()

        # Check data cleared
        self.assertEqual(len(self.metrics.episode_rewards), 0)
        self.assertEqual(len(self.metrics.training_losses), 0)


class TestTrainingVisualizer(unittest.TestCase):
    """Test cases for training visualization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = TrainingVisualizer(save_dir=self.temp_dir)

        # Create sample metrics
        self.metrics = MetricsTracker()
        for i in range(50):
            self.metrics.add_episode_reward(10 + np.random.randn() * 2)
            self.metrics.add_training_loss({
                'policy_loss': 0.1 + np.random.randn() * 0.02,
                'value_loss': 0.05 + np.random.randn() * 0.01,
                'entropy': 2.0 + np.random.randn() * 0.1
            })

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        self.assertEqual(self.visualizer.save_dir, self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))

    def test_reward_plot_creation(self):
        """Test reward plot creation."""
        plot_path = self.visualizer.plot_episode_rewards(
            self.metrics.episode_rewards,
            save_name='test_rewards.png'
        )

        # Check plot was saved
        self.assertTrue(os.path.exists(plot_path))
        self.assertTrue(plot_path.endswith('test_rewards.png'))

    def test_loss_plot_creation(self):
        """Test loss plot creation."""
        plot_path = self.visualizer.plot_training_losses(
            self.metrics.training_losses,
            save_name='test_losses.png'
        )

        # Check plot was saved
        self.assertTrue(os.path.exists(plot_path))
        self.assertTrue(plot_path.endswith('test_losses.png'))

    def test_comprehensive_plot_creation(self):
        """Test comprehensive training plot creation."""
        plot_path = self.visualizer.create_training_summary(
            metrics=self.metrics,
            save_name='training_summary.png'
        )

        # Check plot was saved
        self.assertTrue(os.path.exists(plot_path))
        self.assertTrue(plot_path.endswith('training_summary.png'))

    def test_plot_customization(self):
        """Test plot customization options."""
        plot_path = self.visualizer.plot_episode_rewards(
            self.metrics.episode_rewards,
            title='Custom Title',
            xlabel='Custom X Label',
            ylabel='Custom Y Label',
            color='red',
            save_name='custom_plot.png'
        )

        # Check plot was created
        self.assertTrue(os.path.exists(plot_path))


class TestAlgorithmConfigs(unittest.TestCase):
    """Test cases for algorithm configuration classes."""

    def test_ppo_config(self):
        """Test PPO configuration."""
        config = PPOConfig(
            state_dim=10,
            action_dim=4,
            learning_rate=0.0003,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2
        )

        self.assertEqual(config.state_dim, 10)
        self.assertEqual(config.action_dim, 4)
        self.assertEqual(config.learning_rate, 0.0003)
        self.assertEqual(config.clip_ratio, 0.2)

        # Test validation
        with self.assertRaises(ValueError):
            PPOConfig(state_dim=0, action_dim=4)  # Invalid state_dim

        with self.assertRaises(ValueError):
            PPOConfig(state_dim=10, action_dim=4, clip_ratio=2.0)  # Invalid clip_ratio

    def test_sac_config(self):
        """Test SAC configuration."""
        config = SACConfig(
            state_dim=8,
            action_dim=3,
            learning_rate=0.0003,
            gamma=0.99,
            tau=0.005,
            alpha=0.2
        )

        self.assertEqual(config.state_dim, 8)
        self.assertEqual(config.action_dim, 3)
        self.assertEqual(config.tau, 0.005)
        self.assertEqual(config.alpha, 0.2)

        # Test validation
        with self.assertRaises(ValueError):
            SACConfig(state_dim=8, action_dim=3, tau=1.5)  # Invalid tau

        with self.assertRaises(ValueError):
            SACConfig(state_dim=8, action_dim=3, alpha=-0.1)  # Invalid alpha

    def test_a3c_config(self):
        """Test A3C configuration."""
        config = A3CConfig(
            state_dim=6,
            action_dim=2,
            learning_rate=0.001,
            gamma=0.99,
            n_workers=4
        )

        self.assertEqual(config.state_dim, 6)
        self.assertEqual(config.action_dim, 2)
        self.assertEqual(config.n_workers, 4)

        # Test validation
        with self.assertRaises(ValueError):
            A3CConfig(state_dim=6, action_dim=2, n_workers=0)  # Invalid n_workers

    def test_icm_config(self):
        """Test ICM configuration."""
        config = ICMConfig(
            state_dim=12,
            action_dim=4,
            feature_dim=64,
            eta=0.01,
            beta=0.2
        )

        self.assertEqual(config.state_dim, 12)
        self.assertEqual(config.action_dim, 4)
        self.assertEqual(config.feature_dim, 64)
        self.assertEqual(config.eta, 0.01)
        self.assertEqual(config.beta, 0.2)

        # Test validation
        with self.assertRaises(ValueError):
            ICMConfig(state_dim=12, action_dim=4, beta=1.5)  # Invalid beta


def create_training_test_suite():
    """Create test suite for training system."""
    test_suite = unittest.TestSuite()

    # Training system tests
    test_suite.addTest(unittest.makeSuite(TestTrainingConfig))
    test_suite.addTest(unittest.makeSuite(TestRLTrainer))

    # Utilities tests
    test_suite.addTest(unittest.makeSuite(TestMetricsTracker))
    test_suite.addTest(unittest.makeSuite(TestTrainingVisualizer))

    # Configuration tests
    test_suite.addTest(unittest.makeSuite(TestAlgorithmConfigs))

    return test_suite


if __name__ == '__main__':
    # Run the test suite
    print("Running RL Training System Test Suite")
    print("=" * 50)

    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_training_test_suite()
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"Overall result: {'PASSED' if success else 'FAILED'}")
