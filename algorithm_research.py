"""
RL Algorithm Comparison and Research Module

This module provides comprehensive tools for comparing RL algorithms,
conducting research experiments, and generating performance reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import asyncio
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor

# Import RL system components
from model.rl_system.integration.rl_system import RLSystemManager
from model.rl_system.environments.trading_env import TradingEnvironment

@dataclass
class ExperimentResult:
    """Results from a single algorithm experiment."""
    algorithm_name: str
    algorithm_config: Dict[str, Any]
    environment_config: Dict[str, Any]
    training_episodes: int
    training_time_seconds: float
    final_reward: float
    best_reward: float
    mean_reward: float
    std_reward: float
    convergence_episode: Optional[int]
    reward_history: List[float]
    loss_history: List[float]
    evaluation_rewards: List[float]
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class ComparisonReport:
    """Comprehensive comparison report for multiple algorithms."""
    experiment_id: str
    algorithms_tested: List[str]
    environment_description: str
    total_experiments: int
    results: List[ExperimentResult]
    best_algorithm: str
    performance_summary: Dict[str, Dict[str, float]]
    convergence_analysis: Dict[str, Any]
    statistical_analysis: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime

class PerformanceMetrics:
    """Calculate various performance metrics for RL algorithms."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for trading performance."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_maximum_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.min(drawdown)
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if len(returns) == 0:
            return 0.0
        annual_return = np.mean(returns) * 252
        max_drawdown = abs(PerformanceMetrics.calculate_maximum_drawdown(returns))
        return annual_return / max_drawdown if max_drawdown > 0 else 0.0
    
    @staticmethod
    def calculate_win_rate(returns: np.ndarray) -> float:
        """Calculate win rate (percentage of positive returns)."""
        if len(returns) == 0:
            return 0.0
        return np.sum(returns > 0) / len(returns)
    
    @staticmethod
    def calculate_profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor (total profits / total losses)."""
        if len(returns) == 0:
            return 0.0
        profits = np.sum(returns[returns > 0])
        losses = abs(np.sum(returns[returns < 0]))
        return profits / losses if losses > 0 else float('inf')

class AlgorithmComparator:
    """Compare multiple RL algorithms across different scenarios."""
    
    def __init__(self, output_dir: str = "comparison_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.rl_manager = RLSystemManager()
        self.logger = logging.getLogger(__name__)
        
    async def run_algorithm_comparison(
        self,
        algorithms: List[str],
        environment_configs: List[Dict[str, Any]],
        training_episodes: int = 1000,
        evaluation_episodes: int = 100,
        num_seeds: int = 3
    ) -> ComparisonReport:
        """Run comprehensive algorithm comparison."""
        
        experiment_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        all_results = []
        
        for env_config in environment_configs:
            for algorithm in algorithms:
                for seed in range(num_seeds):
                    try:
                        result = await self._run_single_experiment(
                            algorithm, env_config, training_episodes, 
                            evaluation_episodes, seed
                        )
                        all_results.append(result)
                        
                    except Exception as e:
                        self.logger.error(f"Error in experiment {algorithm} seed {seed}: {e}")
        
        # Generate comparison report
        report = self._generate_comparison_report(
            experiment_id, algorithms, environment_configs, all_results
        )
        
        # Save report
        await self._save_report(report)
        
        return report
    
    async def _run_single_experiment(
        self,
        algorithm: str,
        env_config: Dict[str, Any],
        training_episodes: int,
        evaluation_episodes: int,
        seed: int
    ) -> ExperimentResult:
        """Run a single algorithm experiment."""
        
        start_time = datetime.now()
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Create environment
        env = TradingEnvironment(**env_config)
        
        # Create agent
        agent = self.rl_manager.create_agent(algorithm)
        
        # Training
        reward_history = []
        loss_history = []
        best_reward = float('-inf')
        convergence_episode = None
        
        for episode in range(training_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.store_experience(state, action, reward, next_state, done)
                
                if hasattr(agent, 'train') and episode > 50:  # Start training after some exploration
                    loss = agent.train()
                    if loss is not None:
                        loss_history.append(loss)
                
                state = next_state
                episode_reward += reward
            
            reward_history.append(episode_reward)
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                if convergence_episode is None and episode > 100:
                    # Check for convergence (stable performance)
                    recent_rewards = reward_history[-50:]
                    if len(recent_rewards) == 50 and np.std(recent_rewards) < 0.1 * np.mean(recent_rewards):
                        convergence_episode = episode
        
        # Evaluation
        evaluation_rewards = []
        for _ in range(evaluation_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.act(state, explore=False)  # No exploration during evaluation
                state, reward, done, info = env.step(action)
                episode_reward += reward
            
            evaluation_rewards.append(episode_reward)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return ExperimentResult(
            algorithm_name=algorithm,
            algorithm_config=getattr(agent, 'config', {}),
            environment_config=env_config,
            training_episodes=training_episodes,
            training_time_seconds=training_time,
            final_reward=reward_history[-1] if reward_history else 0,
            best_reward=best_reward,
            mean_reward=np.mean(reward_history) if reward_history else 0,
            std_reward=np.std(reward_history) if reward_history else 0,
            convergence_episode=convergence_episode,
            reward_history=reward_history,
            loss_history=loss_history,
            evaluation_rewards=evaluation_rewards,
            metadata={'seed': seed, 'environment': str(env_config)},
            timestamp=datetime.now()
        )
    
    def _generate_comparison_report(
        self,
        experiment_id: str,
        algorithms: List[str],
        environment_configs: List[Dict[str, Any]],
        results: List[ExperimentResult]
    ) -> ComparisonReport:
        """Generate comprehensive comparison report."""
        
        # Group results by algorithm
        algorithm_results = {}
        for result in results:
            if result.algorithm_name not in algorithm_results:
                algorithm_results[result.algorithm_name] = []
            algorithm_results[result.algorithm_name].append(result)
        
        # Calculate performance summary
        performance_summary = {}
        for algorithm, alg_results in algorithm_results.items():
            eval_rewards = [r for result in alg_results for r in result.evaluation_rewards]
            
            performance_summary[algorithm] = {
                'mean_evaluation_reward': np.mean(eval_rewards),
                'std_evaluation_reward': np.std(eval_rewards),
                'best_evaluation_reward': np.max(eval_rewards),
                'worst_evaluation_reward': np.min(eval_rewards),
                'mean_training_time': np.mean([r.training_time_seconds for r in alg_results]),
                'convergence_rate': sum(1 for r in alg_results if r.convergence_episode is not None) / len(alg_results),
                'mean_convergence_episode': np.mean([r.convergence_episode for r in alg_results if r.convergence_episode is not None]) if any(r.convergence_episode for r in alg_results) else None
            }
        
        # Find best algorithm
        best_algorithm = max(
            performance_summary.keys(),
            key=lambda x: performance_summary[x]['mean_evaluation_reward']
        )
        
        # Convergence analysis
        convergence_analysis = self._analyze_convergence(algorithm_results)
        
        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(algorithm_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(performance_summary, statistical_analysis)
        
        return ComparisonReport(
            experiment_id=experiment_id,
            algorithms_tested=algorithms,
            environment_description=str(environment_configs[0]) if environment_configs else "Unknown",
            total_experiments=len(results),
            results=results,
            best_algorithm=best_algorithm,
            performance_summary=performance_summary,
            convergence_analysis=convergence_analysis,
            statistical_analysis=statistical_analysis,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _analyze_convergence(self, algorithm_results: Dict[str, List[ExperimentResult]]) -> Dict[str, Any]:
        """Analyze convergence patterns across algorithms."""
        convergence_analysis = {}
        
        for algorithm, results in algorithm_results.items():
            convergence_episodes = [r.convergence_episode for r in results if r.convergence_episode is not None]
            
            convergence_analysis[algorithm] = {
                'convergence_rate': len(convergence_episodes) / len(results),
                'mean_convergence_episode': np.mean(convergence_episodes) if convergence_episodes else None,
                'std_convergence_episode': np.std(convergence_episodes) if convergence_episodes else None,
                'fastest_convergence': np.min(convergence_episodes) if convergence_episodes else None,
                'slowest_convergence': np.max(convergence_episodes) if convergence_episodes else None
            }
        
        return convergence_analysis
    
    def _perform_statistical_analysis(self, algorithm_results: Dict[str, List[ExperimentResult]]) -> Dict[str, Any]:
        """Perform statistical analysis comparing algorithms."""
        from scipy import stats
        
        statistical_analysis = {}
        
        # Pairwise comparisons
        algorithms = list(algorithm_results.keys())
        pairwise_comparisons = {}
        
        for i, alg1 in enumerate(algorithms):
            for alg2 in algorithms[i+1:]:
                rewards1 = [r for result in algorithm_results[alg1] for r in result.evaluation_rewards]
                rewards2 = [r for result in algorithm_results[alg2] for r in result.evaluation_rewards]
                
                # Mann-Whitney U test (non-parametric)
                statistic, p_value = stats.mannwhitneyu(rewards1, rewards2, alternative='two-sided')
                
                pairwise_comparisons[f"{alg1}_vs_{alg2}"] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': abs(np.mean(rewards1) - np.mean(rewards2)) / np.sqrt((np.var(rewards1) + np.var(rewards2)) / 2)
                }
        
        statistical_analysis['pairwise_comparisons'] = pairwise_comparisons
        
        # ANOVA test
        if len(algorithms) > 2:
            all_rewards = [
                [r for result in algorithm_results[alg] for r in result.evaluation_rewards]
                for alg in algorithms
            ]
            f_statistic, p_value = stats.f_oneway(*all_rewards)
            
            statistical_analysis['anova'] = {
                'f_statistic': f_statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return statistical_analysis
    
    def _generate_recommendations(
        self, 
        performance_summary: Dict[str, Dict[str, float]], 
        statistical_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Best overall performer
        best_algorithm = max(
            performance_summary.keys(),
            key=lambda x: performance_summary[x]['mean_evaluation_reward']
        )
        recommendations.append(f"Best overall performer: {best_algorithm}")
        
        # Fastest convergence
        convergence_times = {
            alg: summary.get('mean_convergence_episode', float('inf'))
            for alg, summary in performance_summary.items()
            if summary.get('mean_convergence_episode') is not None
        }
        
        if convergence_times:
            fastest_converging = min(convergence_times.keys(), key=lambda x: convergence_times[x])
            recommendations.append(f"Fastest convergence: {fastest_converging}")
        
        # Most stable (lowest std)
        most_stable = min(
            performance_summary.keys(),
            key=lambda x: performance_summary[x]['std_evaluation_reward']
        )
        recommendations.append(f"Most stable performance: {most_stable}")
        
        # Time efficiency
        fastest_training = min(
            performance_summary.keys(),
            key=lambda x: performance_summary[x]['mean_training_time']
        )
        recommendations.append(f"Fastest training: {fastest_training}")
        
        # Statistical significance insights
        if 'pairwise_comparisons' in statistical_analysis:
            significant_pairs = [
                pair for pair, results in statistical_analysis['pairwise_comparisons'].items()
                if results['significant']
            ]
            if significant_pairs:
                recommendations.append(f"Statistically significant differences found in {len(significant_pairs)} pairwise comparisons")
        
        return recommendations
    
    async def _save_report(self, report: ComparisonReport):
        """Save comparison report to files."""
        
        # Save JSON report
        json_path = self.output_dir / f"{report.experiment_id}_report.json"
        with open(json_path, 'w') as f:
            # Convert to serializable format
            report_dict = asdict(report)
            # Convert datetime objects to strings
            report_dict['timestamp'] = report.timestamp.isoformat()
            for result in report_dict['results']:
                result['timestamp'] = result['timestamp'].isoformat() if isinstance(result['timestamp'], datetime) else result['timestamp']
            
            json.dump(report_dict, f, indent=2, default=str)
        
        # Generate and save visualizations
        await self._generate_visualizations(report)
        
        self.logger.info(f"Report saved to {json_path}")
    
    async def _generate_visualizations(self, report: ComparisonReport):
        """Generate visualization charts for the comparison."""
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Algorithm Comparison Report - {report.experiment_id}', fontsize=16)
        
        # 1. Mean evaluation rewards comparison
        algorithms = list(report.performance_summary.keys())
        mean_rewards = [report.performance_summary[alg]['mean_evaluation_reward'] for alg in algorithms]
        std_rewards = [report.performance_summary[alg]['std_evaluation_reward'] for alg in algorithms]
        
        axes[0, 0].bar(algorithms, mean_rewards, yerr=std_rewards, capsize=5)
        axes[0, 0].set_title('Mean Evaluation Rewards')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Training time comparison
        training_times = [report.performance_summary[alg]['mean_training_time'] for alg in algorithms]
        axes[0, 1].bar(algorithms, training_times)
        axes[0, 1].set_title('Mean Training Time')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Convergence analysis
        convergence_rates = [report.performance_summary[alg]['convergence_rate'] for alg in algorithms]
        axes[1, 0].bar(algorithms, convergence_rates)
        axes[1, 0].set_title('Convergence Rate')
        axes[1, 0].set_ylabel('Rate (0-1)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Reward distribution boxplot
        reward_data = []
        algorithm_labels = []
        for alg in algorithms:
            alg_results = [r for r in report.results if r.algorithm_name == alg]
            all_rewards = [reward for result in alg_results for reward in result.evaluation_rewards]
            reward_data.append(all_rewards)
            algorithm_labels.append(alg)
        
        axes[1, 1].boxplot(reward_data, labels=algorithm_labels)
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / f"{report.experiment_id}_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualizations saved to {viz_path}")

class ResearchExperimentRunner:
    """Run advanced research experiments with RL algorithms."""
    
    def __init__(self):
        self.comparator = AlgorithmComparator()
        self.logger = logging.getLogger(__name__)
    
    async def run_hyperparameter_study(
        self,
        algorithm: str,
        parameter_grid: Dict[str, List[Any]],
        environment_config: Dict[str, Any],
        training_episodes: int = 1000
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization study."""
        
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(product(*param_values))
        
        results = []
        
        for combination in param_combinations:
            param_dict = dict(zip(param_names, combination))
            
            try:
                # Create agent with specific parameters
                agent = self.comparator.rl_manager.create_agent(algorithm, **param_dict)
                
                # Run experiment
                result = await self.comparator._run_single_experiment(
                    algorithm, environment_config, training_episodes, 100, 0
                )
                result.algorithm_config.update(param_dict)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error in hyperparameter combination {param_dict}: {e}")
        
        # Find best parameters
        best_result = max(results, key=lambda x: np.mean(x.evaluation_rewards))
        
        return {
            'best_parameters': best_result.algorithm_config,
            'best_performance': np.mean(best_result.evaluation_rewards),
            'all_results': results,
            'parameter_grid': parameter_grid
        }
    
    async def run_stability_analysis(
        self,
        algorithms: List[str],
        environment_config: Dict[str, Any],
        num_runs: int = 10,
        training_episodes: int = 1000
    ) -> Dict[str, Any]:
        """Analyze algorithm stability across multiple runs."""
        
        stability_results = {}
        
        for algorithm in algorithms:
            runs = []
            
            for run in range(num_runs):
                try:
                    result = await self.comparator._run_single_experiment(
                        algorithm, environment_config, training_episodes, 100, run
                    )
                    runs.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Error in stability run {run} for {algorithm}: {e}")
            
            if runs:
                eval_rewards = [np.mean(r.evaluation_rewards) for r in runs]
                stability_results[algorithm] = {
                    'mean_performance': np.mean(eval_rewards),
                    'std_performance': np.std(eval_rewards),
                    'coefficient_of_variation': np.std(eval_rewards) / np.mean(eval_rewards) if np.mean(eval_rewards) != 0 else float('inf'),
                    'min_performance': np.min(eval_rewards),
                    'max_performance': np.max(eval_rewards),
                    'runs': runs
                }
        
        return stability_results
    
    async def run_scalability_analysis(
        self,
        algorithm: str,
        environment_sizes: List[int],
        training_episodes: int = 1000
    ) -> Dict[str, Any]:
        """Analyze algorithm scalability with different environment sizes."""
        
        scalability_results = {}
        
        for size in environment_sizes:
            env_config = {
                'observation_space_size': size,
                'action_space_size': min(5, size),  # Scale action space appropriately
                'max_steps': 1000
            }
            
            try:
                result = await self.comparator._run_single_experiment(
                    algorithm, env_config, training_episodes, 100, 0
                )
                
                scalability_results[size] = {
                    'training_time': result.training_time_seconds,
                    'final_performance': np.mean(result.evaluation_rewards),
                    'convergence_episode': result.convergence_episode,
                    'memory_efficiency': size / result.training_time_seconds  # Simple efficiency metric
                }
                
            except Exception as e:
                self.logger.error(f"Error in scalability test for size {size}: {e}")
        
        return scalability_results

# Global research tools
algorithm_comparator = AlgorithmComparator()
research_runner = ResearchExperimentRunner()
