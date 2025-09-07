"""
Comprehensive Training Framework for RL Agents.

This module provides a unified training framework that can handle all types of
RL agents, environments, and training configurations. It includes advanced
features like learning rate scheduling, early stopping, and performance tracking.
"""

import json
import os
import time
import warnings
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import util.loggers as loggers

from ..core.base_agents import BaseRLAgent
from ..environments.trading_env import TradingEnvironment

logger = loggers.setup_loggers()
rl_logger = logger['rl']

warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class TrainingConfig:
    """Configuration for training RL agents."""
    # Training parameters
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    eval_frequency: int = 50
    save_frequency: int = 100

    # Early stopping
    early_stopping: bool = True
    patience: int = 50
    min_improvement: float = 0.01

    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_decay_factor: float = 0.95
    lr_decay_frequency: int = 100
    min_lr: float = 1e-6

    # Performance tracking
    performance_window: int = 100
    target_reward: Optional[float] = None

    # Logging and saving
    log_level: str = 'INFO'
    save_best_only: bool = True
    checkpoint_dir: str = 'checkpoints'
    tensorboard_dir: Optional[str] = None

    # Evaluation
    eval_episodes: int = 10
    render_eval: bool = False


class LearningRateScheduler:
    """Learning rate scheduler for RL agents."""

    def __init__(self,
                 initial_lr: float,
                 decay_factor: float = 0.95,
                 decay_frequency: int = 100,
                 min_lr: float = 1e-6):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.decay_frequency = decay_frequency
        self.min_lr = min_lr
        self.current_lr = initial_lr
        self.step_count = 0

    def step(self) -> float:
        """Update learning rate and return current value."""
        self.step_count += 1

        if self.step_count % self.decay_frequency == 0:
            self.current_lr = max(
                self.min_lr,
                self.current_lr * self.decay_factor
            )

        return self.current_lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.current_lr = self.initial_lr
        self.step_count = 0


class EarlyStopping:
    """Early stopping mechanism for training."""

    def __init__(self,
                 patience: int = 50,
                 min_improvement: float = 0.01,
                 maximize: bool = True):
        self.patience = patience
        self.min_improvement = min_improvement
        self.maximize = maximize

        self.best_score = float('-inf') if maximize else float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop."""
        improved = False

        if self.maximize:
            if score > self.best_score + self.min_improvement:
                self.best_score = score
                self.counter = 0
                improved = True
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_improvement:
                self.best_score = score
                self.counter = 0
                improved = True
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return improved

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_score = float('-inf') if self.maximize else float('inf')
        self.counter = 0
        self.should_stop = False


class PerformanceTracker:
    """Track and analyze training performance."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.evaluation_scores = []

        # Performance statistics
        self.recent_rewards = deque(maxlen=window_size)
        self.best_reward = float('-inf')
        self.best_eval_score = float('-inf')

        # Timing
        self.episode_times = []
        self.training_start_time = None

    def start_training(self) -> None:
        """Mark the start of training."""
        self.training_start_time = time.time()

    def record_episode(self,
                      reward: float,
                      length: int,
                      loss: Optional[float] = None,
                      episode_time: Optional[float] = None) -> None:
        """Record episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.recent_rewards.append(reward)

        if loss is not None:
            self.episode_losses.append(loss)

        if episode_time is not None:
            self.episode_times.append(episode_time)

        # Update best reward
        if reward > self.best_reward:
            self.best_reward = reward

    def record_evaluation(self, score: float) -> None:
        """Record evaluation score."""
        self.evaluation_scores.append(score)

        if score > self.best_eval_score:
            self.best_eval_score = score

    def get_recent_performance(self) -> Dict[str, float]:
        """Get recent performance statistics."""
        if not self.recent_rewards:
            return {}

        return {
            'mean_reward': np.mean(self.recent_rewards),
            'std_reward': np.std(self.recent_rewards),
            'min_reward': np.min(self.recent_rewards),
            'max_reward': np.max(self.recent_rewards),
            'best_reward': self.best_reward
        }

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        total_time = time.time() - self.training_start_time if self.training_start_time else 0

        summary = {
            'total_episodes': len(self.episode_rewards),
            'total_training_time': total_time,
            'best_reward': self.best_reward,
            'best_eval_score': self.best_eval_score,
            'final_performance': self.get_recent_performance()
        }

        if self.episode_rewards:
            summary.update({
                'total_reward': np.sum(self.episode_rewards),
                'mean_episode_reward': np.mean(self.episode_rewards),
                'std_episode_reward': np.std(self.episode_rewards),
                'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0
            })

        if self.episode_times:
            summary['mean_episode_time'] = np.mean(self.episode_times)

        return summary

    def plot_training_progress(self, save_path: Optional[str] = None) -> None:
        """Plot training progress."""
        if not self.episode_rewards:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)

        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.7, label='Episode Reward')
        if len(self.episode_rewards) > 50:
            # Moving average
            window = min(50, len(self.episode_rewards) // 4)
            moving_avg = pd.Series(self.episode_rewards).rolling(window).mean()
            axes[0, 0].plot(moving_avg, label=f'Moving Average ({window})', linewidth=2)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Episode lengths
        if self.episode_lengths:
            axes[0, 1].plot(self.episode_lengths, alpha=0.7, color='orange')
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].grid(True)

        # Losses
        if self.episode_losses:
            axes[1, 0].plot(self.episode_losses, alpha=0.7, color='red')
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)

        # Evaluation scores
        if self.evaluation_scores:
            eval_episodes = [i * 50 for i in range(len(self.evaluation_scores))]  # Assuming eval every 50 episodes
            axes[1, 1].plot(eval_episodes, self.evaluation_scores, 'o-', color='green')
            axes[1, 1].set_title('Evaluation Scores')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            rl_logger.info(f"Training plot saved to {save_path}")

        plt.show()


class RLTrainer:
    """Comprehensive RL training framework."""

    def __init__(self,
                 agent: BaseRLAgent,
                 environment: TradingEnvironment,
                 config: TrainingConfig = None):
        """
        Initialize RL trainer.

        Args:
            agent: RL agent to train
            environment: Training environment
            config: Training configuration
        """
        self.agent = agent
        self.environment = environment
        self.config = config or TrainingConfig()

        # Initialize components
        self.performance_tracker = PerformanceTracker(self.config.performance_window)
        self.early_stopping = EarlyStopping(
            self.config.patience,
            self.config.min_improvement
        ) if self.config.early_stopping else None

        self.lr_scheduler = None
        if self.config.use_lr_scheduler and hasattr(agent, 'optimizer'):
            self.lr_scheduler = LearningRateScheduler(
                agent.config.learning_rate,
                self.config.lr_decay_factor,
                self.config.lr_decay_frequency,
                self.config.min_lr
            )

        # Setup directories
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        rl_logger.info(f"Initialized trainer for {agent.name} with {config.max_episodes} episodes")

    def train(self) -> Dict[str, Any]:
        """Train the RL agent."""
        rl_logger.info("Starting training...")
        self.performance_tracker.start_training()

        try:
            for episode in range(self.config.max_episodes):
                episode_start_time = time.time()

                # Run episode
                episode_metrics = self._run_episode(episode)

                episode_time = time.time() - episode_start_time

                # Record performance
                self.performance_tracker.record_episode(
                    episode_metrics['total_reward'],
                    episode_metrics['episode_length'],
                    episode_metrics.get('avg_loss'),
                    episode_time
                )

                # Update learning rate
                if self.lr_scheduler:
                    new_lr = self.lr_scheduler.step()
                    self._update_agent_lr(new_lr)

                # Evaluation
                if (episode + 1) % self.config.eval_frequency == 0:
                    eval_score = self._evaluate_agent()
                    self.performance_tracker.record_evaluation(eval_score)

                    # Early stopping check
                    if self.early_stopping:
                        improved = self.early_stopping(eval_score)

                        if improved:
                            rl_logger.info(f"Episode {episode + 1}: New best evaluation score: {eval_score:.4f}")

                        if self.early_stopping.should_stop:
                            rl_logger.info(f"Early stopping triggered at episode {episode + 1}")
                            break

                # Save checkpoint
                if (episode + 1) % self.config.save_frequency == 0:
                    self._save_checkpoint(episode + 1)

                # Log progress
                if (episode + 1) % 10 == 0:
                    recent_perf = self.performance_tracker.get_recent_performance()
                    rl_logger.info(f"Episode {episode + 1}/{self.config.max_episodes} - "
                                 f"Reward: {episode_metrics['total_reward']:.2f} - "
                                 f"Recent Mean: {recent_perf.get('mean_reward', 0):.2f}")

                # Check if target reached
                if self.config.target_reward is not None:
                    recent_perf = self.performance_tracker.get_recent_performance()
                    if recent_perf.get('mean_reward', 0) >= self.config.target_reward:
                        rl_logger.info(f"Target reward {self.config.target_reward} reached!")
                        break

            # Final evaluation and save
            final_eval_score = self._evaluate_agent()
            self.performance_tracker.record_evaluation(final_eval_score)
            self._save_checkpoint('final')

        except KeyboardInterrupt:
            rl_logger.info("Training interrupted by user")
        except Exception as e:
            rl_logger.error(f"Training failed with error: {e}")
            raise

        # Generate training summary
        summary = self.performance_tracker.get_training_summary()
        self._save_training_summary(summary)

        rl_logger.info("Training completed successfully")
        return summary

    def _run_episode(self, episode: int) -> Dict[str, Any]:
        """Run a single training episode."""
        state = self.environment.reset()
        self.agent.reset_episode()

        total_reward = 0
        episode_length = 0
        losses = []

        for step in range(self.config.max_steps_per_episode):
            # Select action
            action = self.agent.select_action(state, training=True)

            # Take step in environment
            next_state, reward, done, info = self.environment.step(action)

            # Update agent
            update_metrics = self.agent.update(state, action, reward, next_state, done)

            # Record metrics
            total_reward += reward
            episode_length += 1

            if 'loss' in update_metrics:
                losses.append(update_metrics['loss'])

            # Update state
            state = next_state

            if done:
                break

        # End episode
        self.agent.add_reward(total_reward)
        self.agent.end_episode()

        return {
            'total_reward': total_reward,
            'episode_length': episode_length,
            'avg_loss': np.mean(losses) if losses else None
        }

    def _evaluate_agent(self) -> float:
        """Evaluate agent performance."""
        self.agent.eval_mode()

        eval_rewards = []

        for _ in range(self.config.eval_episodes):
            state = self.environment.reset()
            total_reward = 0

            for _ in range(self.config.max_steps_per_episode):
                action = self.agent.select_action(state, training=False)
                next_state, reward, done, info = self.environment.step(action)

                total_reward += reward
                state = next_state

                if done:
                    break

            eval_rewards.append(total_reward)

        self.agent.train_mode()

        return np.mean(eval_rewards)

    def _update_agent_lr(self, new_lr: float) -> None:
        """Update agent learning rate."""
        if hasattr(self.agent, 'optimizer'):
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = new_lr
        elif hasattr(self.agent, 'policy_optimizer'):
            for param_group in self.agent.policy_optimizer.param_groups:
                param_group['lr'] = new_lr
        elif hasattr(self.agent, 'actor_optimizer'):
            for param_group in self.agent.actor_optimizer.param_groups:
                param_group['lr'] = new_lr

    def _save_checkpoint(self, episode: Union[int, str]) -> None:
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"{self.agent.name}_episode_{episode}"
        )

        try:
            self.agent.save_checkpoint(checkpoint_path)
            rl_logger.debug(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            rl_logger.error(f"Failed to save checkpoint: {e}")

    def _save_training_summary(self, summary: Dict[str, Any]) -> None:
        """Save training summary to file."""
        summary_path = os.path.join(
            self.config.checkpoint_dir,
            f"{self.agent.name}_training_summary.json"
        )

        # Add configuration to summary
        summary['training_config'] = asdict(self.config)
        summary['agent_config'] = asdict(self.agent.config)
        summary['timestamp'] = datetime.now().isoformat()

        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            rl_logger.info(f"Training summary saved: {summary_path}")
        except Exception as e:
            rl_logger.error(f"Failed to save training summary: {e}")

    def plot_training_progress(self, save_path: Optional[str] = None) -> None:
        """Plot training progress."""
        if save_path is None:
            save_path = os.path.join(
                self.config.checkpoint_dir,
                f"{self.agent.name}_training_progress.png"
            )

        self.performance_tracker.plot_training_progress(save_path)


def train_agent(agent: BaseRLAgent,
                environment: TradingEnvironment,
                config: TrainingConfig = None) -> Dict[str, Any]:
    """
    Convenience function to train an RL agent.

    Args:
        agent: RL agent to train
        environment: Training environment
        config: Training configuration

    Returns:
        Training summary
    """
    trainer = RLTrainer(agent, environment, config)
    return trainer.train()
