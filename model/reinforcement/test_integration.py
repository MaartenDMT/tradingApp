"""
Comprehensive Integration Test for Enhanced RL Trading System.

Tests all components together to ensure proper integration and functionality.
This serves as both a test suite and a comprehensive example of system usage.
"""

import os
import sys
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import util.loggers as loggers
from model.reinforcement.agents.base.base_agent import ReplayBuffer
from model.reinforcement.agents.examples.modern_agent_example import (
    ModernDQNAgent,
    ModernTD3Agent,
)
from model.reinforcement.agents.managers.professional_agent_manager import (
    AgentFactory,
    ProfessionalAgentManager,
)
from model.reinforcement.enhanced_integration import create_enhanced_trading_system

# Import all components
from model.reinforcement.environments.trading_environment import (
    OptimizedTradingEnvironment,
)
from model.reinforcement.utils.enhanced_models import ModelManager
from model.reinforcement.utils.enhanced_training import PerformanceTracker

logger = loggers.setup_loggers()
test_logger = logger['app']


class ComprehensiveIntegrationTest:
    """
    Comprehensive integration test suite for the enhanced RL trading system.

    Tests all major components and their interactions to ensure
    the system works correctly as an integrated whole.
    """

    def __init__(self, data_file: str, test_output_dir: str = "integration_test_output"):
        self.data_file = data_file
        self.test_output_dir = test_output_dir
        self.test_results = {}
        self.test_passed = 0
        self.test_failed = 0

        # Create output directory
        os.makedirs(test_output_dir, exist_ok=True)

        test_logger.info(f"Integration test initialized with data: {data_file}")

    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "PASSED" if passed else "FAILED"
        test_logger.info(f"TEST {test_name}: {status} - {details}")

        self.test_results[test_name] = {
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }

        if passed:
            self.test_passed += 1
        else:
            self.test_failed += 1

    def test_environment_creation(self) -> bool:
        """Test environment creation and basic functionality."""
        try:
            # Test environment creation
            env = OptimizedTradingEnvironment(
                data_file=self.data_file,
                initial_balance=10000,
                lookback_window=20,
                transaction_cost_pct=0.001
            )

            # Test reset
            state = env.reset()
            assert state is not None, "Environment reset returned None"
            assert len(state) > 0, "State is empty"

            # Test step
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)

            assert next_state is not None, "Next state is None"
            assert isinstance(reward, (int, float)), "Reward is not numeric"
            assert isinstance(done, bool), "Done is not boolean"
            assert isinstance(info, dict), "Info is not dictionary"

            self.log_test_result("environment_creation", True,
                               f"Environment created successfully with state dim {len(state)}")
            return True

        except Exception as e:
            self.log_test_result("environment_creation", False, f"Error: {e}")
            return False

    def test_base_agent_functionality(self) -> bool:
        """Test base agent classes and replay buffer."""
        try:
            # Test ReplayBuffer
            buffer = ReplayBuffer(capacity=1000, state_dim=10, action_dim=1)

            # Add experiences
            for i in range(100):
                state = np.random.random(10)
                action = np.random.random(1)
                reward = np.random.random()
                next_state = np.random.random(10)
                done = i % 10 == 0

                buffer.add(state, action, reward, next_state, done)

            # Test sampling
            assert buffer.can_sample(32), "Buffer should be able to sample"
            batch = buffer.sample(32)
            assert len(batch) == 5, "Batch should have 5 elements"

            # Test buffer stats
            stats = buffer.get_stats()
            assert stats['size'] == 100, "Buffer size should be 100"

            self.log_test_result("base_agent_functionality", True,
                               f"ReplayBuffer working correctly with {stats['size']} experiences")
            return True

        except Exception as e:
            self.log_test_result("base_agent_functionality", False, f"Error: {e}")
            return False

    def test_modern_agents(self) -> bool:
        """Test modern agent implementations."""
        try:
            # Test ModernDQNAgent
            dqn_agent = ModernDQNAgent(
                state_dim=20,
                num_actions=3,
                learning_rate=0.001,
                memory_size=10000
            )

            # Test action selection
            test_state = np.random.random(20)
            action = dqn_agent.act(test_state, training=True)
            assert isinstance(action, (int, np.integer)), "DQN action should be integer"
            assert 0 <= action < 3, "DQN action out of bounds"

            # Test experience storage
            dqn_agent.remember(test_state, action, 1.0, test_state, False)

            # Test ModernTD3Agent
            td3_agent = ModernTD3Agent(
                state_dim=20,
                action_dim=1,
                action_bounds=(-1.0, 1.0),
                actor_lr=0.001,
                critic_lr=0.001
            )

            # Test action selection
            action = td3_agent.act(test_state, add_noise=True)
            assert isinstance(action, np.ndarray), "TD3 action should be numpy array"
            assert len(action) == 1, "TD3 action should have dimension 1"
            assert -1.0 <= action[0] <= 1.0, "TD3 action out of bounds"

            # Test experience storage
            td3_agent.remember(test_state, action, 1.0, test_state, False)

            self.log_test_result("modern_agents", True,
                               "DQN and TD3 agents created and tested successfully")
            return True

        except Exception as e:
            self.log_test_result("modern_agents", False, f"Error: {e}")
            return False

    def test_agent_manager(self) -> bool:
        """Test professional agent manager."""
        try:
            # Create agent manager
            manager = ProfessionalAgentManager(
                os.path.join(self.test_output_dir, "test_agents")
            )

            # Test agent factory
            available_agents = AgentFactory.list_available_agents()
            assert len(available_agents) > 0, "No agents available in factory"

            # Create and register agents
            dqn_agent = manager.create_and_register_agent(
                agent_id="test_dqn",
                agent_type="modern_dqn",
                state_dim=20,
                action_dim=3,
                experiment_name="Test DQN"
            )

            td3_agent = manager.create_and_register_agent(
                agent_id="test_td3",
                agent_type="modern_td3",
                state_dim=20,
                action_dim=1,
                experiment_name="Test TD3"
            )

            # Test agent retrieval
            retrieved_dqn = manager.get_agent("test_dqn")
            assert retrieved_dqn is dqn_agent, "Retrieved agent doesn't match registered agent"

            # Test agent listing
            agent_list = manager.list_agents()
            assert "test_dqn" in agent_list, "test_dqn not in agent list"
            assert "test_td3" in agent_list, "test_td3 not in agent list"

            # Test agent info
            info = manager.get_agent_info("test_dqn")
            assert 'agent_stats' in info, "Agent info missing stats"

            # Test workspace summary
            summary = manager.get_workspace_summary()
            assert summary['total_agents'] == 2, "Workspace should have 2 agents"

            self.log_test_result("agent_manager", True,
                               f"Agent manager working with {len(agent_list)} agents")
            return True

        except Exception as e:
            self.log_test_result("agent_manager", False, f"Error: {e}")
            return False

    def test_training_utilities(self) -> bool:
        """Test enhanced training utilities."""
        try:
            # Test performance tracker
            tracker = PerformanceTracker()

            # Add some metrics
            for i in range(10):
                tracker.update_metrics({
                    'episode': i,
                    'reward': np.random.random(),
                    'profit': np.random.random() * 100,
                    'trades': np.random.randint(1, 10)
                })

            # Get summary
            summary = tracker.get_summary()
            assert 'reward_stats' in summary, "Summary missing reward stats"
            assert summary['episodes_tracked'] == 10, "Should have tracked 10 episodes"

            # Test model manager
            model_manager = ModelManager(
                os.path.join(self.test_output_dir, "test_models")
            )

            # Test directory creation
            assert os.path.exists(model_manager.model_dir), "Model directory not created"

            self.log_test_result("training_utilities", True,
                               "Performance tracker and model manager working correctly")
            return True

        except Exception as e:
            self.log_test_result("training_utilities", False, f"Error: {e}")
            return False

    def test_integrated_system(self) -> bool:
        """Test the complete integrated trading system."""
        try:
            # Create integrated system
            system = create_enhanced_trading_system(
                data_file=self.data_file,
                algorithm='modern_dqn'
            )

            # Test short training run
            training_results = system.train(
                episodes=10,
                save_interval=5,
                evaluation_interval=5
            )

            assert 'episodes_completed' in training_results, "Training results missing episodes"
            assert training_results['episodes_completed'] == 10, "Should have completed 10 episodes"

            # Test evaluation
            eval_results = system.evaluate(num_episodes=3)
            assert 'mean_reward' in eval_results, "Evaluation results missing mean reward"
            assert eval_results['episodes'] == 3, "Should have evaluated 3 episodes"

            # Test performance report
            report = system.get_trading_performance_report()
            assert 'system_info' in report, "Report missing system info"
            assert report['system_info']['algorithm'] == 'modern_dqn', "Wrong algorithm in report"

            self.log_test_result("integrated_system", True,
                               f"Integrated system completed {training_results['episodes_completed']} episodes")
            return True

        except Exception as e:
            self.log_test_result("integrated_system", False, f"Error: {e}")
            return False

    def test_td3_integrated_system(self) -> bool:
        """Test the integrated system with TD3 algorithm."""
        try:
            # Create TD3 system
            system = create_enhanced_trading_system(
                data_file=self.data_file,
                algorithm='modern_td3'
            )

            # Test short training run
            training_results = system.train(
                episodes=5,
                save_interval=3,
                evaluation_interval=3
            )

            assert 'episodes_completed' in training_results, "TD3 training results missing episodes"
            assert training_results['episodes_completed'] == 5, "TD3 should have completed 5 episodes"

            # Test evaluation
            eval_results = system.evaluate(num_episodes=2)
            assert 'mean_reward' in eval_results, "TD3 evaluation results missing mean reward"

            self.log_test_result("td3_integrated_system", True,
                               f"TD3 system completed {training_results['episodes_completed']} episodes")
            return True

        except Exception as e:
            self.log_test_result("td3_integrated_system", False, f"Error: {e}")
            return False

    def test_save_load_functionality(self) -> bool:
        """Test save and load functionality."""
        try:
            # Create and train a small agent
            agent = ModernDQNAgent(
                state_dim=10,
                num_actions=3,
                learning_rate=0.001,
                memory_size=1000
            )

            # Add some experiences
            for i in range(50):
                state = np.random.random(10)
                action = np.random.randint(0, 3)
                reward = np.random.random()
                next_state = np.random.random(10)
                done = i % 10 == 0

                agent.remember(state, action, reward, next_state, done)

            # Train a few steps
            if agent.memory.can_sample(32):
                for _ in range(5):
                    agent.learn()

            # Get initial stats
            initial_stats = agent.get_stats()

            # Save agent
            save_path = os.path.join(self.test_output_dir, "test_save_load")
            agent.save(save_path)

            # Create new agent and load
            new_agent = ModernDQNAgent(
                state_dim=10,
                num_actions=3,
                learning_rate=0.001,
                memory_size=1000
            )

            new_agent.load(save_path)

            # Compare stats
            loaded_stats = new_agent.get_stats()
            assert loaded_stats['training_step'] == initial_stats['training_step'], "Training step not preserved"

            self.log_test_result("save_load_functionality", True,
                               "Agent save/load working correctly")
            return True

        except Exception as e:
            self.log_test_result("save_load_functionality", False, f"Error: {e}")
            return False

    def run_all_tests(self) -> Dict[str, any]:
        """Run all integration tests."""
        test_logger.info("=== Starting Comprehensive Integration Tests ===")

        # List of all tests
        tests = [
            self.test_environment_creation,
            self.test_base_agent_functionality,
            self.test_modern_agents,
            self.test_agent_manager,
            self.test_training_utilities,
            self.test_integrated_system,
            self.test_td3_integrated_system,
            self.test_save_load_functionality
        ]

        # Run all tests
        for test_func in tests:
            try:
                test_func()
            except Exception as e:
                test_name = test_func.__name__
                self.log_test_result(test_name, False, f"Unexpected error: {e}")

        # Generate summary
        summary = {
            'total_tests': len(tests),
            'passed': self.test_passed,
            'failed': self.test_failed,
            'pass_rate': self.test_passed / len(tests) if tests else 0,
            'test_results': self.test_results,
            'timestamp': datetime.now().isoformat()
        }

        # Log summary
        test_logger.info("=== Integration Test Summary ===")
        test_logger.info(f"Total tests: {summary['total_tests']}")
        test_logger.info(f"Passed: {summary['passed']}")
        test_logger.info(f"Failed: {summary['failed']}")
        test_logger.info(f"Pass rate: {summary['pass_rate']:.1%}")

        # Save summary to file
        import json
        summary_file = os.path.join(self.test_output_dir, "integration_test_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        test_logger.info(f"Test summary saved to: {summary_file}")

        return summary


def create_sample_data(filepath: str, num_samples: int = 1000) -> str:
    """
    Create sample trading data for testing.

    Args:
        filepath: Path to save sample data
        num_samples: Number of data samples

    Returns:
        Path to created data file
    """
    # Generate sample OHLCV data
    np.random.seed(42)

    # Create realistic price movement
    initial_price = 100.0
    prices = [initial_price]

    for _ in range(num_samples - 1):
        change = np.random.normal(0, 0.02)  # 2% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # Prevent negative prices

    # Generate OHLCV data
    data = []
    for i, close in enumerate(prices):
        open_price = close * (1 + np.random.normal(0, 0.005))
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.randint(1000, 10000)

        data.append({
            'timestamp': i,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    # Save to CSV
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

    test_logger.info(f"Sample data created: {filepath} with {num_samples} samples")
    return filepath


# Example usage and main execution
if __name__ == "__main__":
    # Check if data file exists or create sample data
    data_file = "data/csv/BTC_1h.csv"

    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        print("Creating sample data for testing...")
        data_file = create_sample_data("test_data/sample_trading_data.csv")

    # Run comprehensive integration tests
    print("=== Running Comprehensive Integration Tests ===")

    test_suite = ComprehensiveIntegrationTest(
        data_file=data_file,
        test_output_dir="integration_test_results"
    )

    summary = test_suite.run_all_tests()

    # Display results
    print("\n=== Test Results ===")
    print(f"Total tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")

    if summary['failed'] == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("Enhanced RL Trading System is fully functional!")
    else:
        print(f"\n⚠️ {summary['failed']} tests failed. Check logs for details.")

    print("\nDetailed results saved to: integration_test_results/")
    print("=== Integration Testing Complete ===")
