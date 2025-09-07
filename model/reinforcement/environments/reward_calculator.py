"""
Reward Calculator for the trading reinforcement learning system.
Centralizes and optimizes reward calculation logic.
"""

from typing import Dict, List, Tuple

import numpy as np

import util.loggers as loggers

logger = loggers.setup_loggers()
env_logger = logger['env']
rl_logger = logger['rl']


class RewardCalculator:
    """
    Centralized reward calculation system with multiple reward components.
    """

    def __init__(self, weights: Tuple[float, ...] = (0.2, 0.4, 0.2, 0.1, 0.1)):
        """
        Initialize the reward calculator.

        Args:
            weights: Weights for different reward components
                    (market_condition, financial_outcome, risk_adjusted, drawdown_penalty, trading_penalty)
        """
        self.weights = weights
        self.component_rewards = {}
        self.reward_history = []

    def calculate_combined_reward(self,
                                action: int,
                                current_price: float,
                                market_data: Dict,
                                trading_engine,
                                market_conditions: Dict) -> float:
        """
        Calculate the combined reward from all components.

        Args:
            action: Action taken (0=sell, 1=hold, 2=buy)
            current_price: Current market price
            market_data: Dictionary with market data
            trading_engine: Trading engine instance
            market_conditions: Dictionary with market condition flags

        Returns:
            Combined reward value
        """
        # Calculate individual reward components
        market_reward = self._calculate_market_condition_reward(action, market_conditions)
        financial_reward = self._calculate_financial_outcome_reward(action, trading_engine)
        risk_reward = self._calculate_risk_adjusted_reward(market_data.get('returns_history', []))
        drawdown_penalty = trading_engine.calculate_drawdown_penalty()
        trading_penalty = trading_engine.calculate_trading_penalty()

        # Apply weights
        weighted_components = {
            'market_condition': self.weights[0] * market_reward,
            'financial_outcome': self.weights[1] * financial_reward,
            'risk_adjusted': self.weights[2] * risk_reward,
            'drawdown_penalty': self.weights[3] * drawdown_penalty,
            'trading_penalty': self.weights[4] * trading_penalty
        }

        # Store for logging/analysis
        self.component_rewards = weighted_components

        # Calculate final reward
        combined_reward = sum(weighted_components.values())
        self.reward_history.append(combined_reward)

        # Log detailed breakdown
        self._log_reward_breakdown(weighted_components, combined_reward)

        return combined_reward

    def _calculate_market_condition_reward(self, action: int, conditions: Dict) -> float:
        """
        Calculate reward based on market conditions and action alignment.

        Args:
            action: Action taken
            conditions: Dictionary of market condition flags

        Returns:
            Market condition reward
        """
        # Extract condition flags
        bullish_conditions = [
            conditions.get('strong_buy_signal', False),
            conditions.get('super_buy', False),
            conditions.get('macd_buy', False),
            conditions.get('long_stochastic_signal', False),
            conditions.get('long_bollinger_outside', False),
            conditions.get('high_volatility', False),
            conditions.get('adx_signal', False),
            conditions.get('psar_signal', False),
            conditions.get('cdl_pattern', False),
            conditions.get('volume_break', False),
            conditions.get('resistance_break_signal', False)
        ]

        bearish_conditions = [
            conditions.get('strong_sell_signal', False),
            conditions.get('super_sell', False),
            conditions.get('short_stochastic_signal', False),
            conditions.get('short_bollinger_outside', False)
        ]

        neutral_conditions = [
            conditions.get('going_up_condition', False),
            conditions.get('going_down_condition', False),
            conditions.get('low_volatility', False)
        ]

        # Calculate reward based on action-condition alignment
        reward = 0.0

        if action == 2 and any(bullish_conditions):  # Buy on bullish signals
            if conditions.get('super_buy', False):
                reward = 2.0  # Higher reward for strong signals
            else:
                reward = 1.0
        elif action == 0 and any(bearish_conditions):  # Sell on bearish signals
            if conditions.get('super_sell', False):
                reward = 2.0
            else:
                reward = 1.0
        elif action == 1 and any(neutral_conditions):  # Hold on neutral signals
            if conditions.get('low_volatility', False):
                reward = 0.2
            elif conditions.get('going_up_condition', False) or conditions.get('going_down_condition', False):
                reward = 0.5
        else:
            reward = -0.2  # Penalty for misaligned actions

        return reward

    def _calculate_financial_outcome_reward(self, action: int, trading_engine) -> float:
        """
        Calculate reward based on financial outcomes.

        Args:
            action: Action taken
            trading_engine: Trading engine instance

        Returns:
            Financial outcome reward
        """
        pnl_info = trading_engine.calculate_pnl()
        current_pnl = pnl_info['unrealized_pnl_pct'] / 100.0  # Convert to ratio

        # Base reward from PnL
        financial_reward = current_pnl

        # Additional rewards for holding profitable positions
        if action == 1:  # Hold action
            if current_pnl > 0:
                # Reward for holding profitable positions
                holding_time_bonus = min(0.1, current_pnl * 0.1)
                financial_reward += holding_time_bonus
            elif current_pnl < -0.05:  # 5% loss threshold
                # Penalty for holding losing positions too long
                financial_reward += current_pnl * 0.5  # Additional penalty

        # Scale the reward to be more manageable
        financial_reward = np.tanh(financial_reward * 5)  # Scaled between -1 and 1

        return financial_reward

    def _calculate_risk_adjusted_reward(self, returns_history: List[float]) -> float:
        """
        Calculate risk-adjusted reward based on Sharpe ratio.

        Args:
            returns_history: List of historical returns

        Returns:
            Risk-adjusted reward
        """
        if len(returns_history) < 10:  # Need sufficient history
            return 0.0

        try:
            returns_array = np.array(returns_history[-20:])  # Use last 20 returns

            if len(returns_array) == 0:
                return 0.0

            # Calculate Sharpe ratio (assuming risk-free rate = 0)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)

            if std_return == 0:
                return mean_return if mean_return >= 0 else -1.0

            sharpe_ratio = mean_return / std_return

            # Scale and bound the Sharpe ratio
            risk_adjusted_reward = np.tanh(sharpe_ratio)  # Between -1 and 1

            return risk_adjusted_reward

        except Exception as e:
            env_logger.error(f"Error calculating risk-adjusted reward: {e}")
            return 0.0

    def calculate_action_correctness_reward(self, action: int, optimal_action: int) -> float:
        """
        Calculate reward for action correctness.

        Args:
            action: Action taken
            optimal_action: Optimal action based on conditions

        Returns:
            Correctness reward
        """
        if action == optimal_action:
            return 1.0
        elif abs(action - optimal_action) == 1:  # Close but not exact
            return 0.5
        else:
            return -0.5

    def calculate_consistency_reward(self, window_size: int = 10) -> float:
        """
        Calculate reward for consistent performance.

        Args:
            window_size: Number of recent rewards to consider

        Returns:
            Consistency reward
        """
        if len(self.reward_history) < window_size:
            return 0.0

        recent_rewards = self.reward_history[-window_size:]

        # Calculate coefficient of variation (std/mean) as consistency measure
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)

        if mean_reward == 0:
            return 0.0

        # Lower coefficient of variation = more consistent
        consistency_score = 1.0 / (1.0 + abs(std_reward / mean_reward))

        # Only reward positive consistency
        return consistency_score if mean_reward > 0 else 0.0

    def calculate_momentum_reward(self, recent_actions: List[int], recent_returns: List[float]) -> float:
        """
        Calculate reward based on momentum and action sequence.

        Args:
            recent_actions: List of recent actions
            recent_returns: List of recent returns

        Returns:
            Momentum reward
        """
        if len(recent_actions) < 3 or len(recent_returns) < 3:
            return 0.0

        # Check if actions are aligned with momentum
        momentum = np.mean(recent_returns[-3:])
        last_action = recent_actions[-1]

        momentum_reward = 0.0

        if momentum > 0.01 and last_action == 2:  # Positive momentum + buy
            momentum_reward = 0.5
        elif momentum < -0.01 and last_action == 0:  # Negative momentum + sell
            momentum_reward = 0.5
        elif abs(momentum) < 0.01 and last_action == 1:  # Neutral momentum + hold
            momentum_reward = 0.2

        return momentum_reward

    def get_reward_components(self) -> Dict[str, float]:
        """Get the latest reward component breakdown."""
        return self.component_rewards.copy()

    def get_reward_statistics(self) -> Dict[str, float]:
        """
        Get statistics about reward history.

        Returns:
            Dictionary with reward statistics
        """
        if not self.reward_history:
            return {}

        rewards = np.array(self.reward_history)

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'total_reward': np.sum(rewards),
            'positive_rewards': np.sum(rewards > 0),
            'negative_rewards': np.sum(rewards < 0),
            'reward_trend': np.polyfit(range(len(rewards)), rewards, 1)[0] if len(rewards) > 1 else 0.0
        }

    def update_weights(self, new_weights: Tuple[float, ...]):
        """
        Update the weights for reward components.

        Args:
            new_weights: New weights tuple
        """
        if len(new_weights) != len(self.weights):
            raise ValueError(f"New weights must have {len(self.weights)} components")

        if not np.isclose(sum(new_weights), 1.0):
            env_logger.warning("Weights don't sum to 1.0, normalizing...")
            total = sum(new_weights)
            new_weights = tuple(w / total for w in new_weights)

        self.weights = new_weights
        env_logger.info(f"Updated reward weights: {self.weights}")

    def reset(self):
        """Reset the reward calculator state."""
        self.component_rewards = {}
        self.reward_history = []
        env_logger.debug("Reward calculator reset")

    def _log_reward_breakdown(self, components: Dict[str, float], total: float):
        """
        Log detailed reward breakdown.

        Args:
            components: Dictionary of reward components
            total: Total combined reward
        """
        rl_logger.debug("=== Reward Breakdown ===")
        for component, value in components.items():
            rl_logger.debug(f"{component}: {value:.4f}")
        rl_logger.debug(f"Total Reward: {total:.4f}")
        rl_logger.debug("========================")

    def apply_reward_shaping(self, base_reward: float, shaping_factors: Dict) -> float:
        """
        Apply reward shaping techniques to improve learning.

        Args:
            base_reward: Base reward value
            shaping_factors: Dictionary with shaping parameters

        Returns:
            Shaped reward
        """
        shaped_reward = base_reward

        # Potential-based reward shaping
        if 'potential_prev' in shaping_factors and 'potential_curr' in shaping_factors:
            gamma = shaping_factors.get('gamma', 0.99)
            potential_diff = gamma * shaping_factors['potential_curr'] - shaping_factors['potential_prev']
            shaped_reward += potential_diff

        # Curiosity-based bonus
        if 'novelty_score' in shaping_factors:
            novelty_bonus = shaping_factors['novelty_score'] * 0.1
            shaped_reward += novelty_bonus

        # Exploration bonus
        if 'exploration_bonus' in shaping_factors:
            shaped_reward += shaping_factors['exploration_bonus']

        return shaped_reward

        return shaped_reward
