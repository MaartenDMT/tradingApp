"""
Comprehensive Example and Usage Guide for RL Trading System.

This script demonstrates how to use the complete RL system for trading
applications, showcasing all major features and capabilities.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import util.loggers as loggers

# Import the RL system
from model.rl_system import (
    RLSystemManager,
    compare_algorithms,
    quick_experiment,
    run_all_tests,
    run_benchmarks,
)

logger = loggers.setup_loggers()
rl_logger = logger['rl']


def generate_sample_data(days: int = 365) -> pd.DataFrame:
    """Generate realistic sample trading data."""
    np.random.seed(42)

    # Generate date range
    start_date = datetime.now() - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days, freq='D')

    # Generate realistic price movements
    returns = np.random.normal(0.0005, 0.02, days)  # Small positive drift with volatility

    # Add some market cycles
    cycle = np.sin(np.arange(days) * 2 * np.pi / 252) * 0.01  # Annual cycle
    returns += cycle

    # Calculate cumulative prices
    initial_price = 100.0
    prices = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLCV data
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, days))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, days)
    })

    # Ensure high >= low
    data['high'] = np.maximum(data['high'], data['low'])
    data['high'] = np.maximum(data['high'], data['open'])
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['open'])
    data['low'] = np.minimum(data['low'], data['close'])

    return data


def example_1_basic_usage():
    """Example 1: Basic system usage."""
    print("=" * 60)
    print("EXAMPLE 1: Basic RL System Usage")
    print("=" * 60)

    # Create RL system manager
    system = RLSystemManager("example_experiments")

    # Generate sample data
    data = generate_sample_data(100)  # 100 days of data
    print(f"Generated {len(data)} days of sample trading data")

    # Create trading environment
    env_config = {
        'initial_balance': 10000,
        'transaction_cost': 0.001,
        'lookback_window': 20
    }
    environment = system.create_environment(data, env_config)
    print(f"Created trading environment with balance: ${env_config['initial_balance']}")

    # Create DQN agent
    state_dim = environment.get_observation_space_size()
    action_dim = environment.get_action_space_size()

    agent_config = {
        'learning_rate': 0.001,
        'memory_size': 10000,
        'batch_size': 32,
        'hidden_dims': [128, 128]
    }
    agent = system.create_agent('dqn', state_dim, action_dim, agent_config)
    print(f"Created DQN agent with state_dim={state_dim}, action_dim={action_dim}")

    # Train the agent
    training_config = {
        'max_episodes': 50,
        'eval_frequency': 10,
        'save_frequency': 25,
        'early_stopping': True,
        'patience': 20
    }

    print("Starting training...")
    results = system.train_agent(agent, environment, training_config)

    print("Training completed:")
    print(f"- Total episodes: {results['total_episodes']}")
    print(f"- Best reward: {results['best_reward']:.2f}")
    print(f"- Final performance: {results['final_performance']}")

    # Evaluate the trained agent
    evaluation = system.evaluate_agent(agent, environment, num_episodes=5)
    print("Evaluation results:")
    print(f"- Mean reward: {evaluation['mean_reward']:.2f}")
    print(f"- Std reward: {evaluation['std_reward']:.2f}")
    if 'sharpe_ratio' in evaluation:
        print(f"- Sharpe ratio: {evaluation['sharpe_ratio']:.2f}")


def example_2_algorithm_comparison():
    """Example 2: Compare different algorithms."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Algorithm Comparison")
    print("=" * 60)

    # Use the quick comparison function
    print("Comparing different RL algorithms...")
    comparison_results = compare_algorithms(max_episodes=30)

    if not comparison_results.empty:
        print("\nComparison Results:")
        print(comparison_results[['agent_name', 'mean_reward', 'std_reward']].to_string(index=False))

        # Find best performing algorithm
        best_agent = comparison_results.loc[comparison_results['mean_reward'].idxmax()]
        print(f"\nBest performing algorithm: {best_agent['agent_name']}")
        print(f"Mean reward: {best_agent['mean_reward']:.2f}")
    else:
        print("No comparison results available")


def example_3_custom_configuration():
    """Example 3: Custom agent and environment configuration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Configuration")
    print("=" * 60)

    system = RLSystemManager("custom_experiments")

    # Generate data
    data = generate_sample_data(150)

    # Custom environment configuration
    custom_env_config = {
        'initial_balance': 50000,
        'transaction_cost': 0.0005,  # Lower transaction cost
        'max_position_size': 0.8,    # Allow larger positions
        'lookback_window': 30,       # Longer lookback
        'reward_scaling': 2.0,       # Higher reward scaling
        'include_technical_indicators': True,
        'normalize_features': True
    }

    environment = system.create_environment(data, custom_env_config)

    # Create multiple agents with different configurations
    state_dim = environment.get_observation_space_size()
    action_dim = environment.get_action_space_size()

    agents = {}

    # DQN with large network
    dqn_config = {
        'learning_rate': 0.0005,
        'memory_size': 50000,
        'batch_size': 64,
        'hidden_dims': [256, 256, 128],
        'target_update_frequency': 500
    }
    agents['DQN_Large'] = system.create_agent('dqn', state_dim, action_dim, dqn_config)

    # Double DQN
    double_dqn_config = {
        'learning_rate': 0.001,
        'memory_size': 30000,
        'batch_size': 32,
        'hidden_dims': [128, 128],
        'double_dqn': True
    }
    agents['DoubleDQN'] = system.create_agent('double_dqn', state_dim, action_dim, double_dqn_config)

    # REINFORCE
    reinforce_config = {
        'learning_rate': 0.002,
        'gamma': 0.95,
        'hidden_dims': [128, 64],
        'entropy_coefficient': 0.05
    }
    agents['REINFORCE'] = system.create_agent('reinforce', state_dim, action_dim, reinforce_config)

    print(f"Created {len(agents)} agents with custom configurations")

    # Train each agent
    training_config = {
        'max_episodes': 40,
        'eval_frequency': 10,
        'early_stopping': True,
        'patience': 15
    }

    trained_agents = {}
    for name, agent in agents.items():
        print(f"\nTraining {name}...")
        try:
            results = system.train_agent(agent, environment, training_config)
            trained_agents[name] = agent
            print(f"{name} training completed - Best reward: {results['best_reward']:.2f}")
        except Exception as e:
            print(f"Training failed for {name}: {e}")

    # Compare trained agents
    if trained_agents:
        print(f"\nComparing {len(trained_agents)} trained agents...")
        comparison = system.compare_agents(trained_agents, environment, num_episodes=5)
        print("\nFinal Comparison:")
        print(comparison[['agent_name', 'mean_reward', 'std_reward', 'mean_total_return']].to_string(index=False))


def example_4_save_load_agents():
    """Example 4: Save and load trained agents."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Save and Load Agents")
    print("=" * 60)

    system = RLSystemManager("save_load_experiments")

    # Create and train an agent
    data = generate_sample_data(60)
    environment = system.create_environment(data)

    state_dim = environment.get_observation_space_size()
    action_dim = environment.get_action_space_size()

    agent = system.create_agent('dqn', state_dim, action_dim)
    print("Created DQN agent")

    # Quick training
    training_config = {'max_episodes': 20, 'eval_frequency': 10}
    results = system.train_agent(agent, environment, training_config)
    print(f"Training completed - Best reward: {results['best_reward']:.2f}")

    # Save the agent
    save_path = "saved_agents/trained_dqn_agent"
    system.save_agent(agent, save_path)
    print(f"Agent saved to {save_path}")

    # Load the agent
    loaded_agent = system.load_agent(save_path, 'dqn', state_dim, action_dim)
    print("Agent loaded successfully")

    # Verify the loaded agent works
    evaluation = system.evaluate_agent(loaded_agent, environment, num_episodes=3)
    print(f"Loaded agent evaluation - Mean reward: {evaluation['mean_reward']:.2f}")

    # Clean up
    import shutil
    if os.path.exists("saved_agents"):
        shutil.rmtree("saved_agents")


def example_5_quick_experiment():
    """Example 5: Using the quick experiment function."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Quick Experiment")
    print("=" * 60)

    # Generate and save sample data
    data = generate_sample_data(100)
    data_path = "sample_data.csv"
    data.to_csv(data_path, index=False)
    print(f"Sample data saved to {data_path}")

    # Run quick experiments with different agents
    algorithms = ['dqn', 'q_learning', 'reinforce']

    for algo in algorithms:
        print(f"\nRunning quick experiment with {algo.upper()}...")
        try:
            results = quick_experiment(algo, data_path, max_episodes=25)

            training_results = results['training_results']
            eval_results = results['evaluation_results']

            print(f"Results for {algo.upper()}:")
            print(f"- Training episodes: {training_results['total_episodes']}")
            print(f"- Best training reward: {training_results['best_reward']:.2f}")
            print(f"- Final evaluation: {eval_results['mean_reward']:.2f}")

        except Exception as e:
            print(f"Experiment failed for {algo}: {e}")

    # Clean up
    if os.path.exists(data_path):
        os.remove(data_path)


def example_6_comprehensive_experiment():
    """Example 6: Comprehensive experiment configuration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Comprehensive Experiment")
    print("=" * 60)

    system = RLSystemManager("comprehensive_experiments")

    # Generate data and save it
    data = generate_sample_data(200)
    data_path = "comprehensive_data.csv"
    data.to_csv(data_path, index=False)

    # Comprehensive experiment configuration
    experiment_config = {
        'data_path': data_path,
        'agent': {
            'type': 'rainbow_dqn',
            'config': {
                'learning_rate': 0.0005,
                'memory_size': 50000,
                'batch_size': 64,
                'hidden_dims': [256, 256, 128],
                'target_update_frequency': 1000,
                'epsilon_decay': 0.998,
                'double_dqn': True,
                'dueling_dqn': True
            }
        },
        'environment': {
            'initial_balance': 100000,
            'transaction_cost': 0.0005,
            'lookback_window': 40,
            'reward_scaling': 1.5,
            'include_technical_indicators': True,
            'normalize_features': True
        },
        'training': {
            'max_episodes': 100,
            'eval_frequency': 20,
            'save_frequency': 50,
            'early_stopping': True,
            'patience': 30,
            'use_lr_scheduler': True,
            'lr_decay_factor': 0.98,
            'target_reward': 500
        }
    }

    print("Running comprehensive experiment...")
    results = system.run_experiment(experiment_config)

    print("Comprehensive Experiment Results:")
    print(f"- Agent type: {results['agent_info']['type']}")
    print(f"- Training episodes: {results['training_results']['total_episodes']}")
    print(f"- Best training reward: {results['training_results']['best_reward']:.2f}")
    print(f"- Final evaluation reward: {results['evaluation_results']['mean_reward']:.2f}")

    if 'sharpe_ratio' in results['evaluation_results']:
        print(f"- Sharpe ratio: {results['evaluation_results']['sharpe_ratio']:.2f}")

    # Clean up
    if os.path.exists(data_path):
        os.remove(data_path)


def example_7_testing_and_benchmarking():
    """Example 7: Testing and benchmarking the system."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Testing and Benchmarking")
    print("=" * 60)

    # Run unit tests
    print("Running unit tests...")
    test_success = run_all_tests()

    if test_success:
        print("✅ All unit tests passed!")

        # Run performance benchmarks
        print("\nRunning performance benchmarks...")
        benchmark_results = run_benchmarks()

        print("\nBenchmark Summary:")
        for test_name, time_taken in benchmark_results.items():
            print(f"- {test_name}: {time_taken:.6f}s")

    else:
        print("❌ Some unit tests failed. Check the output for details.")


def cleanup_experiment_directories():
    """Clean up experiment directories created during examples."""
    import shutil

    directories_to_clean = [
        "example_experiments",
        "custom_experiments",
        "save_load_experiments",
        "comprehensive_experiments",
        "rl_experiments"
    ]

    for directory in directories_to_clean:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"Cleaned up {directory}")
            except Exception as e:
                print(f"Could not clean {directory}: {e}")


def main():
    """Run all examples."""
    print("COMPREHENSIVE RL TRADING SYSTEM EXAMPLES")
    print("=" * 80)
    print("This script demonstrates all features of the RL trading system.")
    print("Each example showcases different capabilities and use cases.")
    print("=" * 80)

    try:
        # Run all examples
        example_1_basic_usage()
        example_2_algorithm_comparison()
        example_3_custom_configuration()
        example_4_save_load_agents()
        example_5_quick_experiment()
        example_6_comprehensive_experiment()
        example_7_testing_and_benchmarking()

        print("\n" + "=" * 80)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nThe RL trading system is ready for use. Key features demonstrated:")
        print("✅ Multiple RL algorithms (DQN, Tabular, Policy Gradients, Actor-Critic)")
        print("✅ Sophisticated trading environment with realistic features")
        print("✅ Advanced training framework with monitoring and early stopping")
        print("✅ Comprehensive evaluation and comparison capabilities")
        print("✅ Save/load functionality for trained models")
        print("✅ Quick experiment and batch processing functions")
        print("✅ Extensive testing and benchmarking framework")
        print("\nYou can now use any of these components in your trading applications!")

    except Exception as e:
        rl_logger.error(f"Error in examples: {e}")
        print(f"❌ Error occurred: {e}")

    finally:
        # Clean up
        print("\nCleaning up experiment directories...")
        cleanup_experiment_directories()


if __name__ == "__main__":
    main()
