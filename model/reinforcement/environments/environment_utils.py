"""
Environment utilities for the trading reinforcement learning system.
Contains common classes and functions used across different environment implementations.
"""

import random
from typing import Dict, List

import numpy as np

import util.loggers as loggers

logger = loggers.setup_loggers()
env_logger = logger['env']


class ActionSpace:
    """
    Represents the space of possible actions.
    Actions: 0=sell, 1=hold, 2=buy, 3=buy_back_short, 4=sell_back_long
    """

    def __init__(self, n: int = 3):
        self.n = n
        self.allowed = n - 1

    def sample(self) -> int:
        """Sample a random action from the action space."""
        action = random.randint(0, self.allowed)
        if action > 4:
            env_logger.error(f'Invalid action sampled: {action}')
            raise ValueError(f"Invalid action {action}. Action should be between 0 and {self.allowed}.")
        env_logger.debug(f"Random action generated: {action}")
        return action

    def contains(self, action: int) -> bool:
        """Check if action is valid."""
        return 0 <= action <= self.allowed


class ObservationSpace:
    """Defines the observation space shape for the environment."""

    def __init__(self, shape: tuple):
        self.shape = shape
        self.size = np.prod(shape)

    def sample(self) -> np.ndarray:
        """Sample a random observation."""
        return np.random.random(self.shape).astype(np.float32)


class DynamicFeatureSelector:
    """
    Dynamically selects and manages features based on their performance in trading.
    """

    def __init__(self, initial_features: List[str]):
        self.features = list(initial_features)
        self.feature_performance: Dict[str, float] = {feature: 0.0 for feature in initial_features}
        self.exempted_features = {'dots', 'l_wave', 'rsi14', 'rsi40', 'ema_200'}
        self.performance_threshold = 0.1

    def evaluate_feature_performance(self, feature: str, trading_data, target_name: str) -> float:
        """
        Evaluate the performance of a feature based on correlation with target.

        Args:
            feature: Feature name to evaluate
            trading_data: DataFrame with trading data
            target_name: Target variable column name

        Returns:
            Performance score (correlation coefficient)
        """
        if feature not in trading_data.columns or target_name not in trading_data.columns:
            env_logger.warning(f"Feature {feature} or target {target_name} not found in data")
            return 0.0

        try:
            correlation = trading_data[feature].corr(trading_data[target_name])
            performance_score = abs(correlation) if not np.isnan(correlation) else 0.0
            self.feature_performance[feature] = performance_score
            return performance_score
        except Exception as e:
            env_logger.error(f"Error evaluating feature {feature}: {e}")
            return 0.0

    def update_feature_performance(self, feature: str, performance_impact: float):
        """Update the performance score for a specific feature."""
        if feature in self.feature_performance:
            self.feature_performance[feature] = performance_impact

    def adjust_features(self, threshold: float = None) -> List[str]:
        """
        Adjust the features list by removing underperforming features.

        Args:
            threshold: Minimum performance score to keep a feature

        Returns:
            Updated list of features
        """
        if threshold is None:
            threshold = self.performance_threshold

        self.features = [
            f for f in self.features
            if self.feature_performance.get(f, 0) > threshold or f in self.exempted_features
        ]

        env_logger.info(f"Features adjusted. Remaining: {len(self.features)}")
        return self.features

    def get_features(self) -> List[str]:
        """Get current list of features."""
        return self.features

    def get_feature_performance(self) -> Dict[str, float]:
        """Get performance scores for all features."""
        return self.feature_performance.copy()


class PerformanceTracker:
    """
    Tracks trading performance metrics for the environment.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all performance metrics."""
        self.total_rewards = 0.0
        self.correct_actions = 0
        self.total_actions = 0
        self.max_drawdown = 0.0
        self.max_portfolio_value = 0.0
        self.trade_count = 0
        self.episode_returns = []
        self.accuracy_history = []

    def update(self, reward: float, is_correct: bool, portfolio_value: float, is_trade: bool = False):
        """
        Update performance metrics.

        Args:
            reward: Reward received
            is_correct: Whether the action was correct
            portfolio_value: Current portfolio value
            is_trade: Whether a trade was executed
        """
        self.total_rewards += reward
        if is_correct:
            self.correct_actions += 1
        self.total_actions += 1

        if is_trade:
            self.trade_count += 1

        # Update max portfolio value and drawdown
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        if self.max_portfolio_value > 0:
            current_drawdown = 1 - (portfolio_value / self.max_portfolio_value)
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

    def get_accuracy(self) -> float:
        """Get current accuracy percentage."""
        return (self.correct_actions / self.total_actions * 100) if self.total_actions > 0 else 0.0

    def get_metrics(self) -> Dict[str, float]:
        """Get all performance metrics."""
        return {
            'total_rewards': self.total_rewards,
            'accuracy': self.get_accuracy(),
            'max_drawdown': self.max_drawdown,
            'trade_count': self.trade_count,
            'total_actions': self.total_actions
        }


class StateNormalizer:
    """
    Handles state normalization and preprocessing.
    """

    def __init__(self, method: str = 'standard'):
        self.method = method
        self.fitted = False
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None

    def fit(self, data: np.ndarray):
        """Fit the normalizer to the data."""
        if self.method == 'standard':
            self.mean_ = np.mean(data, axis=0)
            self.std_ = np.std(data, axis=0)
            # Avoid division by zero
            self.std_[self.std_ == 0] = 1.0
        elif self.method == 'minmax':
            self.min_ = np.min(data, axis=0)
            self.max_ = np.max(data, axis=0)
            # Avoid division by zero
            range_vals = self.max_ - self.min_
            range_vals[range_vals == 0] = 1.0
            self.max_ = self.min_ + range_vals

        self.fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform the data using fitted parameters."""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")

        if self.method == 'standard':
            return (data - self.mean_) / self.std_
        elif self.method == 'minmax':
            return (data - self.min_) / (self.max_ - self.min_)

        return data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform the data in one step."""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data back to original scale."""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse transform")

        if self.method == 'standard':
            return data * self.std_ + self.mean_
        elif self.method == 'minmax':
            return data * (self.max_ - self.min_) + self.min_

        return data


def validate_action(action: int, valid_actions: List[int]) -> bool:
    """
    Validate if an action is within the allowed set.

    Args:
        action: Action to validate
        valid_actions: List of valid actions

    Returns:
        True if valid, False otherwise
    """
    return action in valid_actions


def validate_price(price: float) -> bool:
    """
    Validate if a price is reasonable.

    Args:
        price: Price to validate

    Returns:
        True if valid, False otherwise
    """
    return isinstance(price, (int, float)) and price > 0


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Perform safe division avoiding division by zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero

    Returns:
        Division result or default value
    """
    return numerator / denominator if denominator != 0 else default


def clip_values(values: np.ndarray, min_val: float = -1e6, max_val: float = 1e6) -> np.ndarray:
    """
    Clip values to prevent overflow issues.

    Args:
        values: Array of values to clip
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clipped array
    """
    return np.clip(values, min_val, max_val)
