"""
Professional Algorithm Comparison and Benchmarking Tool.

Provides comprehensive comparison and benchmarking capabilities
for different RL algorithms in trading applications.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import util.loggers as loggers
from model.reinforcement.agents.managers.professional_agent_manager import (
    ProfessionalAgentManager,
)

# Import environment and agent managers
from model.reinforcement.environments.trading_environment import (
    OptimizedTradingEnvironment,
)

logger = loggers.setup_loggers()
benchmark_logger = logger['agent']


@dataclass
class BenchmarkConfig:
    """Configuration for algorithm benchmarks."""
    training_episodes: int = 500
    evaluation_episodes: int = 20
    num_runs: int = 3
    max_episode_steps: int = 1000
    save_models: bool = True
    generate_plots: bool = True
    random_seed: int = 42


@dataclass
class BenchmarkResult:
    """Results from algorithm benchmark."""
    algorithm_name: str
    config: Dict[str, Any]
    training_rewards: List[float]
    evaluation_rewards: List[float]
    training_time: float
    final_model_path: Optional[str]
    metrics: Dict[str, Any]


class TradingBenchmarkSuite:
    """
    Professional benchmark suite for trading RL algorithms.

    Provides standardized benchmarking and comparison of different
    RL algorithms on trading tasks with comprehensive metrics and analysis.
    """

    def __init__(self,
                 data_file: str,
                 benchmark_config: BenchmarkConfig = None,
                 output_dir: str = "benchmark_results"):
        """
        Initialize benchmark suite.

        Args:
            data_file: Path to trading data
            benchmark_config: Benchmark configuration
            output_dir: Directory for saving results
        """
        self.data_file = data_file
        self.config = benchmark_config or BenchmarkConfig()
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize components
        self.agent_manager = ProfessionalAgentManager(
            os.path.join(output_dir, "agents")
        )

        # Benchmark results
        self.results: Dict[str, BenchmarkResult] = {}
        self.comparison_data = {}

        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)

        benchmark_logger.info(f"Benchmark suite initialized with data: {data_file}")

    def setup_environment(self, **env_kwargs) -> OptimizedTradingEnvironment:
        """
        Setup standardized trading environment for benchmarks.

        Args:
            **env_kwargs: Additional environment arguments

        Returns:
            Configured trading environment
        """
        default_config = {
            'initial_balance': 10000,
            'lookback_window': 20,
            'transaction_cost_pct': 0.001,
            'time_cost_pct': 0.0001,
            'normalize_observations': True
        }
        default_config.update(env_kwargs)

        env = OptimizedTradingEnvironment(
            data_file=self.data_file,
            **default_config
        )

        return env

    def benchmark_algorithm(self,
                          algorithm_name: str,
                          algorithm_config: Dict[str, Any],
                          run_id: int = 0) -> BenchmarkResult:
        """
        Benchmark a single algorithm.

        Args:
            algorithm_name: Name of algorithm to benchmark
            algorithm_config: Algorithm configuration
            run_id: Run identifier for multiple runs

        Returns:
            Benchmark results
        """
        start_time = datetime.now()
        benchmark_logger.info(f"Starting benchmark for {algorithm_name} (run {run_id})")

        # Setup environment
        env = self.setup_environment()
        state_dim = env.observation_space.shape[0]

        # Determine action dimension based on algorithm
        if 'dqn' in algorithm_name.lower():
            action_dim = 3  # Hold, Buy, Sell
        else:
            action_dim = 1  # Continuous position sizing

        # Create agent
        agent_id = f"{algorithm_name}_run_{run_id}"
        agent = self.agent_manager.create_and_register_agent(
            agent_id=agent_id,
            agent_type=algorithm_name,
            state_dim=state_dim,
            action_dim=action_dim,
            experiment_name=f"Benchmark_{algorithm_name}",
            custom_config=algorithm_config
        )

        # Training phase
        training_rewards = []
        training_metrics = []

        for episode in range(self.config.training_episodes):
            state = env.reset()
            episode_reward = 0
            episode_metrics = {}

            for step in range(self.config.max_episode_steps):
                # Agent action
                if hasattr(agent, 'act'):
                    if 'dqn' in algorithm_name.lower():
                        action = agent.act(state, training=True)
                    else:
                        action = agent.act(state, add_noise=True)
                else:
                    # Fallback for different agent interfaces
                    action = env.action_space.sample()

                # Environment step
                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                # Store experience
                if hasattr(agent, 'remember'):
                    agent.remember(state, action, reward, next_state, done)

                # Learning step
                if hasattr(agent, 'learn') and step > 32:  # Start learning after some experiences
                    learn_metrics = agent.learn()
                    if learn_metrics:
                        episode_metrics.update(learn_metrics)

                state = next_state

                if done:
                    break

            training_rewards.append(episode_reward)
            training_metrics.append(episode_metrics)

            # Log progress
            if (episode + 1) % 50 == 0:
                mean_reward = np.mean(training_rewards[-50:])
                benchmark_logger.info(f"{algorithm_name} Episode {episode+1}: "
                                    f"Mean reward (last 50): {mean_reward:.2f}")

        # Evaluation phase
        evaluation_rewards = []

        for eval_episode in range(self.config.evaluation_episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(self.config.max_episode_steps):
                # Agent action (no exploration)
                if hasattr(agent, 'act'):
                    if 'dqn' in algorithm_name.lower():
                        action = agent.act(state, training=False)
                    else:
                        action = agent.act(state, add_noise=False)
                else:
                    action = env.action_space.sample()

                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state

                if done:
                    break

            evaluation_rewards.append(episode_reward)

        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()

        # Save model if requested
        model_path = None
        if self.config.save_models:
            model_path = self.agent_manager.save_agent(agent_id)

        # Compile metrics
        metrics = {
            'training_mean_reward': np.mean(training_rewards),
            'training_std_reward': np.std(training_rewards),
            'training_final_reward': training_rewards[-1] if training_rewards else 0,
            'evaluation_mean_reward': np.mean(evaluation_rewards),
            'evaluation_std_reward': np.std(evaluation_rewards),
            'convergence_episode': self._find_convergence_point(training_rewards),
            'sample_efficiency': self._calculate_sample_efficiency(training_rewards),
            'stability_score': self._calculate_stability_score(training_rewards),
            'final_metrics': training_metrics[-1] if training_metrics else {}
        }

        # Create result
        result = BenchmarkResult(
            algorithm_name=algorithm_name,
            config=algorithm_config,
            training_rewards=training_rewards,
            evaluation_rewards=evaluation_rewards,
            training_time=training_time,
            final_model_path=model_path,
            metrics=metrics
        )

        benchmark_logger.info(f"Benchmark completed for {algorithm_name} (run {run_id}): "
                            f"Eval reward = {metrics['evaluation_mean_reward']:.2f}")

        return result

    def run_comprehensive_benchmark(self,
                                  algorithms: Dict[str, Dict[str, Any]]) -> Dict[str, List[BenchmarkResult]]:
        """
        Run comprehensive benchmark across multiple algorithms and runs.

        Args:
            algorithms: Dictionary of {algorithm_name: config}

        Returns:
            Dictionary of {algorithm_name: [results_per_run]}
        """
        all_results = {}

        for algorithm_name, algorithm_config in algorithms.items():
            benchmark_logger.info(f"Benchmarking {algorithm_name} with {self.config.num_runs} runs")

            algorithm_results = []

            for run_id in range(self.config.num_runs):
                try:
                    result = self.benchmark_algorithm(
                        algorithm_name=algorithm_name,
                        algorithm_config=algorithm_config,
                        run_id=run_id
                    )
                    algorithm_results.append(result)

                except Exception as e:
                    benchmark_logger.error(f"Failed to benchmark {algorithm_name} run {run_id}: {e}")
                    continue

            all_results[algorithm_name] = algorithm_results

            # Log algorithm summary
            if algorithm_results:
                eval_rewards = [r.metrics['evaluation_mean_reward'] for r in algorithm_results]
                mean_eval = np.mean(eval_rewards)
                std_eval = np.std(eval_rewards)
                benchmark_logger.info(f"{algorithm_name} summary: "
                                    f"Eval reward = {mean_eval:.2f} ± {std_eval:.2f}")

        self.results = all_results
        return all_results

    def generate_comparison_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report.

        Returns:
            Comparison report dictionary
        """
        if not self.results:
            raise ValueError("No benchmark results available. Run benchmarks first.")

        report = {
            'benchmark_config': self.config.__dict__,
            'data_file': self.data_file,
            'timestamp': datetime.now().isoformat(),
            'algorithms': {},
            'rankings': {},
            'statistical_analysis': {}
        }

        # Aggregate results per algorithm
        for algorithm_name, results in self.results.items():
            if not results:
                continue

            # Aggregate metrics across runs
            eval_rewards = [r.metrics['evaluation_mean_reward'] for r in results]
            training_times = [r.training_time for r in results]
            convergence_episodes = [r.metrics['convergence_episode'] for r in results if r.metrics['convergence_episode'] is not None]

            algorithm_summary = {
                'num_runs': len(results),
                'evaluation_reward': {
                    'mean': np.mean(eval_rewards),
                    'std': np.std(eval_rewards),
                    'min': np.min(eval_rewards),
                    'max': np.max(eval_rewards)
                },
                'training_time': {
                    'mean': np.mean(training_times),
                    'std': np.std(training_times),
                    'total': np.sum(training_times)
                },
                'convergence': {
                    'mean_episodes': np.mean(convergence_episodes) if convergence_episodes else None,
                    'convergence_rate': len(convergence_episodes) / len(results)
                },
                'sample_efficiency': np.mean([r.metrics['sample_efficiency'] for r in results]),
                'stability_score': np.mean([r.metrics['stability_score'] for r in results])
            }

            report['algorithms'][algorithm_name] = algorithm_summary

        # Generate rankings
        algorithms = list(report['algorithms'].keys())

        report['rankings'] = {
            'by_evaluation_reward': sorted(algorithms,
                                         key=lambda x: report['algorithms'][x]['evaluation_reward']['mean'],
                                         reverse=True),
            'by_training_time': sorted(algorithms,
                                     key=lambda x: report['algorithms'][x]['training_time']['mean']),
            'by_sample_efficiency': sorted(algorithms,
                                         key=lambda x: report['algorithms'][x]['sample_efficiency'],
                                         reverse=True),
            'by_stability': sorted(algorithms,
                                 key=lambda x: report['algorithms'][x]['stability_score'],
                                 reverse=True)
        }

        # Best overall algorithm (weighted score)
        if algorithms:
            best_algorithm = self._calculate_overall_ranking(report['algorithms'])
            report['best_algorithm'] = best_algorithm

        return report

    def save_results(self, filepath: Optional[str] = None) -> str:
        """
        Save benchmark results to file.

        Args:
            filepath: Optional custom filepath

        Returns:
            Path where results were saved
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.json")

        # Generate comparison report
        report = self.generate_comparison_report()

        # Save detailed results
        detailed_results = {}
        for algorithm_name, results in self.results.items():
            detailed_results[algorithm_name] = []
            for i, result in enumerate(results):
                detailed_results[algorithm_name].append({
                    'run_id': i,
                    'training_rewards': result.training_rewards,
                    'evaluation_rewards': result.evaluation_rewards,
                    'training_time': result.training_time,
                    'metrics': result.metrics,
                    'config': result.config
                })

        # Combine report and detailed results
        full_results = {
            'summary_report': report,
            'detailed_results': detailed_results
        }

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)

        benchmark_logger.info(f"Benchmark results saved to {filepath}")
        return filepath

    def generate_plots(self, save_dir: Optional[str] = None) -> List[str]:
        """
        Generate visualization plots for benchmark results.

        Args:
            save_dir: Directory to save plots

        Returns:
            List of generated plot filepaths
        """
        if not self.results:
            raise ValueError("No benchmark results available")

        if save_dir is None:
            save_dir = os.path.join(self.output_dir, "plots")

        os.makedirs(save_dir, exist_ok=True)

        plot_files = []

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Training curves comparison
        plt.figure(figsize=(12, 8))
        for algorithm_name, results in self.results.items():
            if not results:
                continue

            # Average training curves across runs
            max_episodes = max(len(r.training_rewards) for r in results)
            avg_rewards = []
            std_rewards = []

            for episode in range(max_episodes):
                episode_rewards = []
                for result in results:
                    if episode < len(result.training_rewards):
                        episode_rewards.append(result.training_rewards[episode])

                if episode_rewards:
                    avg_rewards.append(np.mean(episode_rewards))
                    std_rewards.append(np.std(episode_rewards))
                else:
                    avg_rewards.append(np.nan)
                    std_rewards.append(np.nan)

            # Plot with confidence intervals
            episodes = range(len(avg_rewards))
            plt.plot(episodes, avg_rewards, label=algorithm_name, linewidth=2)
            plt.fill_between(episodes,
                           np.array(avg_rewards) - np.array(std_rewards),
                           np.array(avg_rewards) + np.array(std_rewards),
                           alpha=0.2)

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        training_curves_path = os.path.join(save_dir, "training_curves.png")
        plt.savefig(training_curves_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(training_curves_path)

        # 2. Evaluation performance comparison
        plt.figure(figsize=(10, 6))

        algorithm_names = []
        eval_means = []
        eval_stds = []

        for algorithm_name, results in self.results.items():
            if not results:
                continue

            eval_rewards = []
            for result in results:
                eval_rewards.extend(result.evaluation_rewards)

            algorithm_names.append(algorithm_name)
            eval_means.append(np.mean(eval_rewards))
            eval_stds.append(np.std(eval_rewards))

        bars = plt.bar(algorithm_names, eval_means, yerr=eval_stds, capsize=5)
        plt.xlabel('Algorithm')
        plt.ylabel('Evaluation Reward')
        plt.title('Evaluation Performance Comparison')
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, mean_val in zip(bars, eval_means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(eval_stds)*0.1,
                    f'{mean_val:.2f}', ha='center', va='bottom')

        eval_comparison_path = os.path.join(save_dir, "evaluation_comparison.png")
        plt.savefig(eval_comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(eval_comparison_path)

        # 3. Training time comparison
        plt.figure(figsize=(10, 6))

        training_times = []
        for algorithm_name in algorithm_names:
            results = self.results[algorithm_name]
            times = [r.training_time for r in results]
            training_times.append(np.mean(times))

        bars = plt.bar(algorithm_names, training_times)
        plt.xlabel('Algorithm')
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time Comparison')
        plt.xticks(rotation=45)

        # Add value labels
        for bar, time_val in zip(bars, training_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times)*0.02,
                    f'{time_val:.1f}s', ha='center', va='bottom')

        time_comparison_path = os.path.join(save_dir, "training_time_comparison.png")
        plt.savefig(time_comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(time_comparison_path)

        benchmark_logger.info(f"Generated {len(plot_files)} plots in {save_dir}")
        return plot_files

    def _find_convergence_point(self, rewards: List[float], window: int = 50, threshold: float = 0.05) -> Optional[int]:
        """Find episode where training converged."""
        if len(rewards) < window * 2:
            return None

        for i in range(window, len(rewards) - window):
            before_window = rewards[i-window:i]
            after_window = rewards[i:i+window]

            if abs(np.mean(after_window) - np.mean(before_window)) < threshold:
                return i

        return None

    def _calculate_sample_efficiency(self, rewards: List[float]) -> float:
        """Calculate sample efficiency metric."""
        if len(rewards) < 100:
            return 0.0

        # Area under the curve normalized by maximum possible area
        max_reward = max(rewards)
        if max_reward <= 0:
            return 0.0

        area_under_curve = np.sum(rewards)
        max_possible_area = max_reward * len(rewards)

        return area_under_curve / max_possible_area

    def _calculate_stability_score(self, rewards: List[float]) -> float:
        """Calculate stability score (lower variance is better)."""
        if len(rewards) < 10:
            return 0.0

        # Use coefficient of variation (std/mean) as stability metric
        mean_reward = np.mean(rewards)
        if mean_reward == 0:
            return 0.0

        cv = np.std(rewards) / abs(mean_reward)
        # Convert to score (higher is better)
        stability_score = 1 / (1 + cv)

        return stability_score

    def _calculate_overall_ranking(self, algorithms: Dict[str, Any]) -> str:
        """Calculate overall best algorithm using weighted scoring."""
        if not algorithms:
            return None

        # Weights for different criteria
        weights = {
            'evaluation_reward': 0.4,
            'sample_efficiency': 0.3,
            'stability': 0.2,
            'training_time': 0.1  # Lower is better for training time
        }

        scores = {}

        # Normalize metrics and calculate weighted scores
        eval_rewards = [alg['evaluation_reward']['mean'] for alg in algorithms.values()]
        sample_effs = [alg['sample_efficiency'] for alg in algorithms.values()]
        stabilities = [alg['stability_score'] for alg in algorithms.values()]
        train_times = [alg['training_time']['mean'] for alg in algorithms.values()]

        for alg_name, alg_data in algorithms.items():
            # Normalize metrics (0-1 scale)
            norm_eval = (alg_data['evaluation_reward']['mean'] - min(eval_rewards)) / (max(eval_rewards) - min(eval_rewards)) if max(eval_rewards) != min(eval_rewards) else 0.5
            norm_eff = (alg_data['sample_efficiency'] - min(sample_effs)) / (max(sample_effs) - min(sample_effs)) if max(sample_effs) != min(sample_effs) else 0.5
            norm_stab = (alg_data['stability_score'] - min(stabilities)) / (max(stabilities) - min(stabilities)) if max(stabilities) != min(stabilities) else 0.5
            norm_time = 1 - ((alg_data['training_time']['mean'] - min(train_times)) / (max(train_times) - min(train_times))) if max(train_times) != min(train_times) else 0.5

            # Calculate weighted score
            weighted_score = (
                weights['evaluation_reward'] * norm_eval +
                weights['sample_efficiency'] * norm_eff +
                weights['stability'] * norm_stab +
                weights['training_time'] * norm_time
            )

            scores[alg_name] = weighted_score

        # Return best algorithm
        best_algorithm = max(scores.keys(), key=lambda x: scores[x])
        return best_algorithm


# Predefined benchmark configurations
STANDARD_ALGORITHMS = {
    'modern_dqn': {
        'learning_rate': 0.0001,
        'memory_size': 100000,
        'batch_size': 32,
        'epsilon_decay': 0.995
    },
    'modern_td3': {
        'actor_lr': 0.0001,
        'critic_lr': 0.0001,
        'memory_size': 1000000,
        'batch_size': 32,
        'noise_std': 0.1
    }
}

QUICK_BENCHMARK_CONFIG = BenchmarkConfig(
    training_episodes=100,
    evaluation_episodes=10,
    num_runs=2,
    max_episode_steps=500
)

COMPREHENSIVE_BENCHMARK_CONFIG = BenchmarkConfig(
    training_episodes=1000,
    evaluation_episodes=50,
    num_runs=5,
    max_episode_steps=1000
)


# Example usage
if __name__ == "__main__":
    # Example: Run comprehensive benchmark
    print("=== Trading Algorithm Benchmark ===")

    # Sample data file (replace with actual path)
    data_file = "data/csv/BTC_1h.csv"

    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        print("Please ensure you have trading data available.")
    else:
        # Create benchmark suite
        benchmark = TradingBenchmarkSuite(
            data_file=data_file,
            benchmark_config=QUICK_BENCHMARK_CONFIG,
            output_dir="benchmark_output"
        )

        # Run benchmarks
        print("Running algorithm benchmarks...")
        results = benchmark.run_comprehensive_benchmark(STANDARD_ALGORITHMS)

        # Generate report
        print("Generating comparison report...")
        report = benchmark.generate_comparison_report()

        # Save results
        results_file = benchmark.save_results()
        print(f"Results saved to: {results_file}")

        # Generate plots
        if benchmark.config.generate_plots:
            plots = benchmark.generate_plots()
            print(f"Generated plots: {plots}")

        # Display summary
        print("\n=== Benchmark Summary ===")
        if 'best_algorithm' in report:
            print(f"Best overall algorithm: {report['best_algorithm']}")

        for alg_name in report['rankings']['by_evaluation_reward']:
            alg_data = report['algorithms'][alg_name]
            print(f"{alg_name}: {alg_data['evaluation_reward']['mean']:.2f} ± {alg_data['evaluation_reward']['std']:.2f}")

        print("\n=== Benchmark Complete ===")
