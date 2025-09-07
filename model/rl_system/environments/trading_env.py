"""
Enhanced Trading Environment for Reinforcement Learning.

This module provides a comprehensive trading environment designed specifically
for reinforcement learning agents. It includes market simulation, portfolio
management, realistic trading costs, and detailed performance metrics.
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import util.loggers as loggers

logger = loggers.setup_loggers()
rl_logger = logger['rl']

warnings.filterwarnings('ignore', category=FutureWarning)


class TradingAction(Enum):
    """Enumeration of possible trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2
    STRONG_BUY = 3
    STRONG_SELL = 4


@dataclass
class TradingConfig:
    """Configuration for the trading environment."""
    initial_balance: float = 10000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    max_position_size: float = 1.0   # Maximum position as fraction of portfolio
    lookback_window: int = 50        # Number of past observations
    reward_scaling: float = 1.0      # Scaling factor for rewards
    normalize_features: bool = True  # Whether to normalize input features
    include_technical_indicators: bool = True
    risk_penalty_factor: float = 0.1 # Penalty for high risk positions
    balance_reward_factor: float = 0.001  # Reward factor for portfolio balance


class TradingEnvironment:
    """
    Professional trading environment for RL agents.

    Features:
    - Realistic portfolio management
    - Transaction costs and slippage
    - Risk management metrics
    - Comprehensive state representation
    - Flexible reward functions
    - Multiple asset support
    """

    def __init__(self,
                 data: pd.DataFrame,
                 config: TradingConfig = None,
                 features: Optional[List[str]] = None):
        """
        Initialize the trading environment.

        Args:
            data: Market data with OHLCV columns
            config: Environment configuration
            features: List of feature columns to use
        """
        self.config = config or TradingConfig()
        self.data = data.copy()
        self.features = features or ['close', 'volume']

        # Validate data
        self._validate_data()

        # Prepare features
        self._prepare_features()

        # Environment state
        self.current_step = 0
        self.max_steps = len(self.data) - self.config.lookback_window - 1

        # Portfolio state
        self.initial_balance = self.config.initial_balance
        self.balance = self.initial_balance
        self.position = 0.0  # Current position (-1 to 1)
        self.shares_held = 0.0
        self.total_trades = 0
        self.transaction_costs = 0.0

        # Performance tracking
        self.portfolio_values = []
        self.returns = []
        self.actions_taken = []
        self.rewards_received = []

        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_value = self.initial_balance

        rl_logger.info(f"Trading environment initialized with {len(self.data)} data points")
        rl_logger.info(f"Max steps: {self.max_steps}, Features: {self.features}")

    def _validate_data(self) -> None:
        """Validate input data format."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if len(self.data) < self.config.lookback_window + 10:
            raise ValueError(f"Insufficient data: need at least {self.config.lookback_window + 10} rows")

        # Check for missing values
        if self.data[required_columns].isnull().any().any():
            rl_logger.warning("Found missing values in data, forward filling...")
            self.data[required_columns] = self.data[required_columns].fillna(method='ffill')

    def _prepare_features(self) -> None:
        """Prepare and engineer features for the environment."""
        # Calculate technical indicators if requested
        if self.config.include_technical_indicators:
            self._add_technical_indicators()

        # Calculate returns
        self.data['returns'] = self.data['close'].pct_change()
        self.data['volatility'] = self.data['returns'].rolling(20).std()

        # Add price ratios
        self.data['high_low_ratio'] = self.data['high'] / self.data['low']
        self.data['close_open_ratio'] = self.data['close'] / self.data['open']

        # Volume indicators
        self.data['volume_ma'] = self.data['volume'].rolling(20).mean()
        self.data['volume_ratio'] = self.data['volume'] / self.data['volume_ma']

        # Update features list
        self.features = [col for col in self.data.columns
                        if col not in ['open', 'high', 'low', 'close', 'volume']
                        and not self.data[col].isnull().all()]

        # Normalize features if requested
        if self.config.normalize_features:
            self._normalize_features()

        # Fill any remaining NaN values
        self.data = self.data.fillna(method='ffill').fillna(0)

    def _add_technical_indicators(self) -> None:
        """Add technical indicators to the data."""
        # Moving averages
        for window in [5, 10, 20, 50]:
            self.data[f'ma_{window}'] = self.data['close'].rolling(window).mean()
            self.data[f'ma_ratio_{window}'] = self.data['close'] / self.data[f'ma_{window}']

        # RSI
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = self.data['close'].ewm(span=12).mean()
        exp2 = self.data['close'].ewm(span=26).mean()
        self.data['macd'] = exp1 - exp2
        self.data['macd_signal'] = self.data['macd'].ewm(span=9).mean()
        self.data['macd_histogram'] = self.data['macd'] - self.data['macd_signal']

        # Bollinger Bands
        bb_ma = self.data['close'].rolling(20).mean()
        bb_std = self.data['close'].rolling(20).std()
        self.data['bb_upper'] = bb_ma + (2 * bb_std)
        self.data['bb_lower'] = bb_ma - (2 * bb_std)
        self.data['bb_position'] = (self.data['close'] - self.data['bb_lower']) / (self.data['bb_upper'] - self.data['bb_lower'])

    def _normalize_features(self) -> None:
        """Normalize features using rolling statistics."""
        for feature in self.features:
            if feature in self.data.columns:
                rolling_mean = self.data[feature].rolling(100, min_periods=10).mean()
                rolling_std = self.data[feature].rolling(100, min_periods=10).std()
                self.data[f'{feature}_normalized'] = (self.data[feature] - rolling_mean) / (rolling_std + 1e-8)

        # Update features list to use normalized versions
        normalized_features = [f'{feature}_normalized' for feature in self.features
                             if f'{feature}_normalized' in self.data.columns]
        if normalized_features:
            self.features = normalized_features

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.shares_held = 0.0
        self.total_trades = 0
        self.transaction_costs = 0.0

        # Clear tracking lists
        self.portfolio_values = [self.initial_balance]
        self.returns = []
        self.actions_taken = []
        self.rewards_received = []

        # Reset risk metrics
        self.max_drawdown = 0.0
        self.peak_value = self.initial_balance

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0=hold, 1=buy, 2=sell, etc.)

        Returns:
            observation: Next state
            reward: Reward for the action
            done: Whether episode is finished
            info: Additional information
        """
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0.0, True, self._get_info()

        # Execute action
        reward = self._execute_action(action)

        # Update step
        self.current_step += 1

        # Calculate portfolio value
        current_price = self._get_current_price()
        portfolio_value = self.balance + (self.shares_held * current_price)
        self.portfolio_values.append(portfolio_value)

        # Update peak and drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Calculate return
        if len(self.portfolio_values) > 1:
            period_return = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            self.returns.append(period_return)

        # Track action and reward
        self.actions_taken.append(action)
        self.rewards_received.append(reward)

        # Check if done
        done = (self.current_step >= self.max_steps or
                portfolio_value <= self.initial_balance * 0.1)  # Stop if 90% loss

        return self._get_observation(), reward, done, self._get_info()

    def _execute_action(self, action: int) -> float:
        """Execute trading action and return reward."""
        current_price = self._get_current_price()
        prev_portfolio_value = self.balance + (self.shares_held * current_price)

        # Map action to trading decision
        if action == TradingAction.HOLD.value:
            # No action taken
            pass
        elif action == TradingAction.BUY.value:
            self._execute_buy(current_price, 0.25)  # Buy 25% of available balance
        elif action == TradingAction.SELL.value:
            self._execute_sell(current_price, 0.25)  # Sell 25% of position
        elif action == TradingAction.STRONG_BUY.value:
            self._execute_buy(current_price, 0.5)   # Buy 50% of available balance
        elif action == TradingAction.STRONG_SELL.value:
            self._execute_sell(current_price, 0.5)  # Sell 50% of position

        # Calculate reward
        new_portfolio_value = self.balance + (self.shares_held * current_price)
        reward = self._calculate_reward(prev_portfolio_value, new_portfolio_value, action)

        return reward

    def _execute_buy(self, price: float, fraction: float) -> None:
        """Execute buy order."""
        if self.balance <= 0:
            return

        # Calculate transaction cost
        transaction_amount = self.balance * fraction
        transaction_cost = transaction_amount * self.config.transaction_cost

        if transaction_amount > transaction_cost:
            shares_to_buy = (transaction_amount - transaction_cost) / price

            self.shares_held += shares_to_buy
            self.balance -= transaction_amount
            self.transaction_costs += transaction_cost
            self.total_trades += 1

            # Update position
            portfolio_value = self.balance + (self.shares_held * price)
            self.position = min(1.0, (self.shares_held * price) / portfolio_value)

    def _execute_sell(self, price: float, fraction: float) -> None:
        """Execute sell order."""
        if self.shares_held <= 0:
            return

        shares_to_sell = self.shares_held * fraction
        transaction_amount = shares_to_sell * price
        transaction_cost = transaction_amount * self.config.transaction_cost

        self.shares_held -= shares_to_sell
        self.balance += (transaction_amount - transaction_cost)
        self.transaction_costs += transaction_cost
        self.total_trades += 1

        # Update position
        if self.balance + (self.shares_held * price) > 0:
            portfolio_value = self.balance + (self.shares_held * price)
            self.position = max(-1.0, (self.shares_held * price) / portfolio_value)
        else:
            self.position = 0.0

    def _calculate_reward(self, prev_value: float, new_value: float, action: int) -> float:
        """Calculate reward for the action taken."""
        # Base reward: portfolio value change
        value_change = new_value - prev_value
        reward = value_change * self.config.reward_scaling

        # Risk penalty for extreme positions
        risk_penalty = abs(self.position) * self.config.risk_penalty_factor
        reward -= risk_penalty

        # Balance reward for maintaining some cash
        if self.balance > 0:
            balance_reward = min(self.balance / new_value, 0.2) * self.config.balance_reward_factor
            reward += balance_reward

        # Transaction cost penalty
        if action != TradingAction.HOLD.value:
            # Penalty for unnecessary trading
            recent_actions = self.actions_taken[-5:] if len(self.actions_taken) >= 5 else self.actions_taken
            if recent_actions and all(a != TradingAction.HOLD.value for a in recent_actions):
                reward -= 0.001  # Penalty for overtrading

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current environment observation."""
        start_idx = self.current_step
        end_idx = start_idx + self.config.lookback_window

        # Get feature data
        feature_data = self.data[self.features].iloc[start_idx:end_idx].values

        # Flatten the lookback window
        market_features = feature_data.flatten()

        # Portfolio features
        current_price = self._get_current_price()
        portfolio_value = self.balance + (self.shares_held * current_price)

        portfolio_features = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position,                        # Current position
            self.shares_held * current_price / portfolio_value if portfolio_value > 0 else 0,  # Asset allocation
            self.total_trades / 100,             # Normalized trade count
            self.transaction_costs / self.initial_balance,  # Normalized costs
            len(self.portfolio_values) / self.max_steps,    # Progress through episode
        ])

        # Combine all features
        observation = np.concatenate([market_features, portfolio_features])

        # Ensure no NaN or inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

        return observation.astype(np.float32)

    def _get_current_price(self) -> float:
        """Get current market price."""
        idx = self.current_step + self.config.lookback_window
        if idx >= len(self.data):
            idx = len(self.data) - 1
        return float(self.data['close'].iloc[idx])

    def _get_info(self) -> Dict[str, Any]:
        """Get additional environment information."""
        current_price = self._get_current_price()
        portfolio_value = self.balance + (self.shares_held * current_price)

        return {
            'step': self.current_step,
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'position': self.position,
            'total_trades': self.total_trades,
            'transaction_costs': self.transaction_costs,
            'max_drawdown': self.max_drawdown,
            'current_price': current_price,
            'total_return': (portfolio_value - self.initial_balance) / self.initial_balance,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'win_rate': self._calculate_win_rate(),
            'avg_trade_return': self._calculate_avg_trade_return()
        }

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of returns."""
        if len(self.returns) < 2:
            return 0.0

        returns_array = np.array(self.returns)
        if np.std(returns_array) == 0:
            return 0.0

        return np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)  # Annualized

    def _calculate_win_rate(self) -> float:
        """Calculate win rate of trades."""
        if len(self.returns) == 0:
            return 0.0

        winning_trades = sum(1 for r in self.returns if r > 0)
        return winning_trades / len(self.returns)

    def _calculate_avg_trade_return(self) -> float:
        """Calculate average trade return."""
        if len(self.returns) == 0:
            return 0.0

        return np.mean(self.returns)

    def get_action_space_size(self) -> int:
        """Get the size of the action space."""
        return len(TradingAction)

    def get_observation_space_size(self) -> int:
        """Get the size of the observation space."""
        # Market features + portfolio features
        market_size = len(self.features) * self.config.lookback_window
        portfolio_size = 6  # Number of portfolio features
        return market_size + portfolio_size

    def render(self, mode: str = 'human') -> None:
        """Render environment state."""
        if mode == 'human':
            current_price = self._get_current_price()
            portfolio_value = self.balance + (self.shares_held * current_price)

            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Portfolio Value: ${portfolio_value:.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Shares: {self.shares_held:.4f}")
            print(f"Position: {self.position:.2f}")
            print(f"Current Price: ${current_price:.2f}")
            print(f"Total Trades: {self.total_trades}")
            print(f"Max Drawdown: {self.max_drawdown:.2%}")
            print("-" * 40)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.portfolio_values:
            return {}

        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.initial_balance) / self.initial_balance

        return {
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': self.total_trades,
            'transaction_costs': self.transaction_costs,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'win_rate': self._calculate_win_rate(),
            'avg_trade_return': self._calculate_avg_trade_return(),
            'portfolio_values': self.portfolio_values.copy(),
            'returns': self.returns.copy(),
            'actions_taken': self.actions_taken.copy(),
            'rewards_received': self.rewards_received.copy()
        }
