"""
Core Environment for the trading reinforcement learning system.
Enhanced with Context7 best practices for professional trading RL.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import model.reinforcement.utils.indicators as ind
import util.loggers as loggers
from util.utils import load_config

from .data_manager import DataManager

# Import our new modular components
from .environment_utils import (
    ActionSpace,
    DynamicFeatureSelector,
    ObservationSpace,
    PerformanceTracker,
    StateNormalizer,
    validate_price,
)
from .reward_calculator import RewardCalculator
from .trading_engine import TradingEngine

logger = loggers.setup_loggers()
env_logger = logger['env']
rl_logger = logger['rl']

# Constants
EARLY_STOP_REWARD_THRESHOLD = -5.0
DEFAULT_FEATURES = [
    'close', 'volume', 'rsi14', 'rsi40', 'ema_200', 'dots', 'l_wave', 'b_wave',
    'macd', 'macd_signal', 'macd_histogram', 'bb_upper', 'bb_lower', 'bb_middle',
    'atr', 'adx', 'stoch_k', 'stoch_d', 'williams_r', 'cci'
]

# Context7 Enhanced Constants
DEFAULT_TRADING_DAYS = 252  # Standard trading year
TRADING_COST_BPS = 1e-3     # 0.1% trading cost (Context7 standard)
TIME_COST_BPS = 1e-4        # 0.01% time cost per period
MAX_EPISODE_STEPS = 1000    # Professional episode length

# Context7 Trading Cost Structure
class TradingCosts:
    """Professional trading cost structure following Context7 patterns."""

    def __init__(self, trading_cost_bps: float = TRADING_COST_BPS,
                 time_cost_bps: float = TIME_COST_BPS):
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps

    def calculate_trading_cost(self, action: int, position_size: float, price: float) -> float:
        """Calculate trading costs based on Context7 patterns."""
        if action != 1:  # Not holding (action 1 = hold)
            return self.trading_cost_bps * position_size * price
        return self.time_cost_bps * position_size * price

    def get_cost_summary(self) -> str:
        """Get formatted cost summary."""
        return f'Trading costs: {self.trading_cost_bps:.2%} | Time costs: {self.time_cost_bps:.2%}'


class MultiAgentEnvironment:
    """
    Multi-agent wrapper for the trading environment.
    Supports both single and multi-agent scenarios.
    """

    def __init__(self, num_agents: int = 1, **kwargs):
        """
        Initialize multi-agent environment.

        Args:
            num_agents: Number of agents to create
            **kwargs: Arguments passed to individual environments
        """
        self.num_agents = num_agents
        self.single_agent_mode = (num_agents == 1)

        # Create individual environments
        self.agents = [TradingEnvironment(**kwargs) for _ in range(num_agents)]

        # Set common properties from first agent
        if self.agents:
            self.action_space = self.agents[0].action_space
            self.observation_space = self.agents[0].observation_space

    def reset(self):
        """Reset all agents."""
        if self.single_agent_mode:
            return self.agents[0].reset()
        else:
            return [agent.reset() for agent in self.agents]

    def step(self, actions):
        """Execute step for all agents."""
        if self.single_agent_mode:
            return self.agents[0].step(actions)
        else:
            results = []
            for agent, action in zip(self.agents, actions):
                results.append(agent.step(action))

            # Unpack results
            states, rewards, infos, dones = zip(*results)
            return list(states), list(rewards), list(infos), list(dones)

    def get_action_space(self):
        """Get action space."""
        return self.action_space

    def get_observation_space(self):
        """Get observation space."""
        return self.observation_space


class TradingEnvironment:
    """
    Enhanced trading environment following Context7 best practices.

    Implements professional trading patterns with:
    - Standardized state normalization
    - Professional cost structure
    - Enhanced performance tracking
    - Context7 observation/action spaces
    """

    def __init__(self,
                 symbol: str = 'BTC',
                 features: List[str] = None,
                 limit: int = MAX_EPISODE_STEPS,
                 time: str = '30m',
                 actions: int = 3,
                 min_accuracy: float = 0.6,
                 initial_balance: float = 10000.0,
                 trading_mode: str = 'spot',
                 leverage: float = 1.0,
                 transaction_costs: float = TRADING_COST_BPS,
                 config: Dict = None,
                 trading_days: int = DEFAULT_TRADING_DAYS):
        """
        Initialize the enhanced trading environment with Context7 patterns.

        Args:
            symbol: Trading symbol
            features: List of feature names to use
            limit: Maximum number of steps (Context7 default: 1000)
            time: Time frame
            actions: Number of possible actions
            min_accuracy: Minimum accuracy threshold
            initial_balance: Starting balance
            trading_mode: 'spot' or 'futures'
            leverage: Leverage for futures trading
            transaction_costs: Transaction cost ratio (Context7 default: 1e-3)
            config: Configuration dictionary
            trading_days: Number of trading days (Context7 default: 252)
        """
        self.config = config or load_config()
        self.symbol = symbol
        self.features = features or DEFAULT_FEATURES
        self.limit = limit
        self.time = time
        self.min_accuracy = min_accuracy
        self.trading_days = trading_days

        # Context7 Enhanced Components
        self.action_space = ActionSpace(actions)
        self.data_manager = DataManager(self.config, symbol)
        self.trading_engine = TradingEngine(
            initial_balance=initial_balance,
            leverage=leverage,
            transaction_costs=transaction_costs,
            trading_mode=trading_mode
        )
        self.reward_calculator = RewardCalculator()
        self.performance_tracker = PerformanceTracker()
        self.feature_selector = None
        self.state_normalizer = StateNormalizer()

        # Context7 Trading Costs
        self.trading_costs = TradingCosts(transaction_costs, TIME_COST_BPS)

        # Environment state tracking (Context7 enhanced)
        self.current_step = 0
        self.current_price = 0.0
        self.done = False
        self.episode_data = None
        self.observation_space = None

        # Enhanced state tracking with Context7 patterns
        self.recent_actions = []
        self.recent_rewards = []
        self.recent_trade_results = [0] * 5
        self.action_history = []

        # Context7 Performance metrics
        self.episode_start_balance = initial_balance
        self.max_episode_steps = limit
        self.nav_history = []  # Net Asset Value tracking
        self.market_nav_history = []  # Market comparison
        self.episode_returns = []

        # Context7 State dimensions (for compatibility)
        self.state_dim = None
        self.num_actions = actions

        # Initialize environment
        self._initialize_environment()

    def _initialize_environment(self):
        """Initialize the trading environment with Context7 enhanced patterns."""
        try:
            # Load and process data
            raw_data = self.data_manager.load_data()
            if raw_data.empty:
                raise ValueError("Failed to load trading data")

            # Create features
            processed_data = self.data_manager.create_features(self.features)
            if processed_data.empty:
                raise ValueError("Failed to create features")

            # Initialize feature selector
            self.feature_selector = DynamicFeatureSelector(self.features)

            # Set up observation space
            available_features = [f for f in self.features if f in processed_data.columns]
            self.features = available_features  # Update to only available features

            # Context7 Enhanced observation space with professional state information
            feature_count = len(self.features)
            # Additional Context7 state info: position, cash, nav, recent_performance, volatility, momentum, trend
            context7_info_count = 10
            total_obs_size = feature_count + context7_info_count

            self.observation_space = ObservationSpace((total_obs_size,))
            self.state_dim = total_obs_size  # Context7 compatibility

            # Normalize features using Context7 patterns
            self.episode_data = self.data_manager.normalize_features(self.features)

            # Fit state normalizer with Context7 enhanced normalization
            feature_data = self.episode_data[self.features].values
            self.state_normalizer.fit(feature_data)

            # Context7 logging with cost information
            env_logger.info(f"Context7 Environment initialized: {self.symbol}")
            env_logger.info(f"Features: {len(self.features)}, Data shape: {self.episode_data.shape}")
            env_logger.info(f"State dimension: {self.state_dim}, Actions: {self.num_actions}")
            env_logger.info(f"Max episode steps: {self.max_episode_steps}")
            env_logger.info(self.trading_costs.get_cost_summary())

        except Exception as e:
            env_logger.error(f"Failed to initialize environment: {e}")
            raise

    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.

        Returns:
            Initial observation
        """
        try:
            # Reset all components
            self.trading_engine.reset()
            self.reward_calculator.reset()
            self.performance_tracker.reset()

            # Reset state variables
            self.current_step = 0
            self.current_price = 0.0
            self.done = False
            self.recent_actions = []
            self.recent_rewards = []
            self.recent_trade_results = [0] * 5
            self.action_history = []

            # Get initial observation
            observation = self._get_observation()

            env_logger.info("Environment reset successfully")
            return observation

        except Exception as e:
            env_logger.error(f"Error during environment reset: {e}")
            raise

    def step(self, action: int) -> Tuple[np.ndarray, float, Dict, bool]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, info, done)
        """
        try:
            # Validate action
            if not self.action_space.contains(action):
                raise ValueError(f"Invalid action: {action}")

            # Check if episode is done
            if self.done or self.current_step >= len(self.episode_data) - 1:
                return self._get_observation(), 0.0, self._get_info(), True

            # Get current market data
            current_data = self.episode_data.iloc[self.current_step]
            self.current_price = current_data['close']

            # Validate price
            if not validate_price(self.current_price):
                env_logger.warning(f"Invalid price at step {self.current_step}: {self.current_price}")
                self.current_step += 1
                return self._get_observation(), -0.1, self._get_info(), False

            # Execute trade
            trade_result, updated_balance = self.trading_engine.execute_trade(action, self.current_price)

            # Get market conditions for reward calculation
            market_conditions = self._get_market_conditions(current_data)

            # Calculate reward
            market_data = {
                'returns_history': self.recent_rewards,
                'current_data': current_data
            }

            reward = self.reward_calculator.calculate_combined_reward(
                action, self.current_price, market_data,
                self.trading_engine, market_conditions
            )

            # Determine if action was correct
            optimal_action = self._get_optimal_action(market_conditions)
            is_correct = (action == optimal_action)

            # Update tracking
            self._update_tracking(action, reward, trade_result, is_correct, updated_balance)

            # Check if episode should end
            self.done = self._should_terminate()

            # Get next observation
            self.current_step += 1
            observation = self._get_observation()

            # Create info dictionary
            info = self._get_info()

            # Log step details
            self._log_step(action, reward, trade_result, is_correct)

            return observation, reward, info, self.done

        except Exception as e:
            env_logger.error(f"Error during step execution: {e}")
            # Return safe values to prevent crash
            return self._get_observation(), -1.0, self._get_info(), True

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation state.

        Returns:
            Observation array
        """
        try:
            if self.current_step >= len(self.episode_data):
                # Return zeros if beyond data
                return np.zeros(self.observation_space.shape, dtype=np.float32)

            # Get feature values
            current_data = self.episode_data.iloc[self.current_step]
            feature_values = current_data[self.features].values

            # Normalize features
            feature_values = self.state_normalizer.transform(feature_values.reshape(1, -1))[0]

            # Get additional state information
            portfolio_metrics = self.trading_engine.get_metrics()
            additional_info = np.array([
                portfolio_metrics['total_return_pct'] / 100.0,  # Normalized return
                portfolio_metrics['max_drawdown'],
                len(self.recent_trade_results) / 5.0,  # Normalized trade count
                np.mean(self.recent_trade_results) if self.recent_trade_results else 0.0,
                self._calculate_market_volatility(),
                self._calculate_momentum(),
                float(self.current_step) / len(self.episode_data)  # Progress
            ], dtype=np.float32)

            # Combine feature values and additional info
            observation = np.concatenate([feature_values, additional_info])

            # Ensure no infinite or NaN values
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

            return observation.astype(np.float32)

        except Exception as e:
            env_logger.error(f"Error getting observation: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_market_conditions(self, data_row: pd.Series) -> Dict[str, bool]:
        """
        Extract market conditions from current data.

        Args:
            data_row: Current data row

        Returns:
            Dictionary of market condition flags
        """
        try:
            conditions = {}

            # Basic conditions
            conditions['super_buy'] = (data_row.get('dots', 0) == 1) & (data_row.get('l_wave', 0) >= -50)
            conditions['super_sell'] = (data_row.get('dots', 0) == -1) & (data_row.get('l_wave', 0) >= 50)
            conditions['low_volatility'] = (45 <= data_row.get('rsi14', 50) <= 55)
            conditions['strong_upward_movement'] = (data_row.get('rsi14', 50) > 70)
            conditions['strong_downward_movement'] = (data_row.get('rsi14', 50) < 30)

            # Use indicator functions if available
            if self.current_step < len(self.episode_data) - 1:
                try:
                    # These require the full dataset and current index
                    full_data = self.data_manager.processed_data
                    if full_data is not None and self.current_step < len(full_data):
                        conditions['short_stochastic_signal'] = ind.short_stochastic_condition(full_data, self.current_step)
                        conditions['long_stochastic_signal'] = ind.long_stochastic_condition(full_data, self.current_step)
                        conditions['macd_buy'] = ind.macd_condition(full_data, self.current_step)
                        conditions['high_volatility'] = ind.atr_condition(full_data, self.current_step)
                        conditions['adx_signal'] = ind.adx_condition(full_data, self.current_step)
                except Exception as e:
                    env_logger.debug(f"Error calculating indicator conditions: {e}")
                    # Set default values if indicator calculations fail
                    for key in ['short_stochastic_signal', 'long_stochastic_signal',
                              'macd_buy', 'high_volatility', 'adx_signal']:
                        conditions[key] = False

            return conditions

        except Exception as e:
            env_logger.error(f"Error getting market conditions: {e}")
            return {}

    def _get_optimal_action(self, conditions: Dict[str, bool]) -> int:
        """
        Determine optimal action based on market conditions.

        Args:
            conditions: Market condition flags

        Returns:
            Optimal action (0=sell, 1=hold, 2=buy)
        """
        bullish_signals = sum([
            conditions.get('super_buy', False),
            conditions.get('strong_upward_movement', False),
            conditions.get('long_stochastic_signal', False),
            conditions.get('macd_buy', False),
        ])

        bearish_signals = sum([
            conditions.get('super_sell', False),
            conditions.get('strong_downward_movement', False),
            conditions.get('short_stochastic_signal', False),
        ])

        if bullish_signals >= 2:
            return 2  # Buy
        elif bearish_signals >= 2:
            return 0  # Sell
        else:
            return 1  # Hold

    def _update_tracking(self, action: int, reward: float, trade_result: float,
                        is_correct: bool, balance: float):
        """Update all tracking variables."""
        # Update action and reward history
        self.recent_actions.append(action)
        self.recent_rewards.append(reward)
        self.action_history.append(action)

        # Limit history size
        if len(self.recent_actions) > 20:
            self.recent_actions.pop(0)
        if len(self.recent_rewards) > 20:
            self.recent_rewards.pop(0)

        # Update trade results
        if action in [0, 2]:  # Only for actual trades
            self.recent_trade_results.append(trade_result)
            if len(self.recent_trade_results) > 5:
                self.recent_trade_results.pop(0)

        # Update performance tracker
        self.performance_tracker.update(reward, is_correct, balance, action in [0, 2])

    def _should_terminate(self) -> bool:
        """
        Check if episode should terminate.

        Returns:
            True if episode should end
        """
        # Check various termination conditions
        total_rewards = sum(self.recent_rewards)

        # Early stopping conditions
        if total_rewards <= EARLY_STOP_REWARD_THRESHOLD:
            env_logger.info("Early stopping: Low rewards")
            return True

        if self.current_step >= self.max_episode_steps:
            env_logger.info("Episode ended: Max steps reached")
            return True

        if self.current_step >= len(self.episode_data) - 1:
            env_logger.info("Episode ended: No more data")
            return True

        # Check if balance is too low
        balance_info = self.trading_engine.get_position_info()
        if balance_info['portfolio_balance'] < self.episode_start_balance * 0.1:
            env_logger.info("Early stopping: Portfolio too low")
            return True

        # Check accuracy threshold
        accuracy = self.performance_tracker.get_accuracy()
        if accuracy < self.min_accuracy * 100 and self.current_step > 50:
            env_logger.info(f"Early stopping: Low accuracy ({accuracy:.1f}%)")
            return True

        return False

    def _calculate_market_volatility(self) -> float:
        """Calculate current market volatility."""
        if len(self.recent_rewards) < 5:
            return 0.0
        return np.std(self.recent_rewards[-5:])

    def _calculate_momentum(self) -> float:
        """Calculate price momentum."""
        if len(self.recent_rewards) < 3:
            return 0.0
        return np.mean(self.recent_rewards[-3:])

    def _get_info(self) -> Dict:
        """
        Get information dictionary for the current state.

        Returns:
            Information dictionary
        """
        portfolio_metrics = self.trading_engine.get_metrics()
        performance_metrics = self.performance_tracker.get_metrics()

        return {
            'step': self.current_step,
            'portfolio_balance': portfolio_metrics['portfolio_balance'],
            'total_value': portfolio_metrics['total_value'],
            'total_return_pct': portfolio_metrics['total_return_pct'],
            'max_drawdown': portfolio_metrics['max_drawdown'],
            'trade_count': portfolio_metrics['trade_count'],
            'accuracy': performance_metrics['accuracy'],
            'total_rewards': performance_metrics['total_rewards'],
            'current_price': self.current_price,
            'recent_actions': self.recent_actions[-5:],
            'symbol': self.symbol
        }

    def _log_step(self, action: int, reward: float, trade_result: float, is_correct: bool):
        """Log step details."""
        if self.current_step % 100 == 0:  # Log every 100 steps
            info = self._get_info()
            env_logger.info(f"Step {self.current_step}: Action={action}, "
                          f"Reward={reward:.4f}, Correct={is_correct}, "
                          f"Balance=${info['portfolio_balance']:.2f}, "
                          f"Return={info['total_return_pct']:.2f}%")

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.features

    def get_metrics(self) -> Dict:
        """Get comprehensive environment metrics."""
        portfolio_metrics = self.trading_engine.get_metrics()
        performance_metrics = self.performance_tracker.get_metrics()
        reward_stats = self.reward_calculator.get_reward_statistics()

        return {
            **portfolio_metrics,
            **performance_metrics,
            **reward_stats,
            'current_step': self.current_step,
            'episode_progress': self.current_step / len(self.episode_data) if self.episode_data is not None else 0,
        }

    def update_features(self):
        """Update and adjust features based on performance."""
        if self.feature_selector and self.data_manager.processed_data is not None:
            # Evaluate and adjust features
            for feature in self.features:
                if feature in self.data_manager.processed_data.columns:
                    performance = self.feature_selector.evaluate_feature_performance(
                        feature, self.data_manager.processed_data, 'returns'
                    )
                    self.feature_selector.update_feature_performance(feature, performance)

            # Adjust feature list
            self.features = self.feature_selector.adjust_features()
            env_logger.info(f"Features updated. Active features: {len(self.features)}")

    # Context7 Enhanced Methods
    def get_episode_result(self) -> Dict:
        """
        Get episode results in Context7 format for professional analysis.

        Returns:
            Dictionary with episode performance metrics following Context7 patterns
        """
        portfolio_metrics = self.trading_engine.get_metrics()

        # Calculate Context7 standard metrics
        nav = portfolio_metrics['total_value'] / self.episode_start_balance
        market_nav = self.current_price / self.episode_data['close'].iloc[0] if len(self.episode_data) > 0 else 1.0
        strategy_return = (nav - 1.0) * 100
        market_return = (market_nav - 1.0) * 100

        result = {
            'nav': nav,
            'market_nav': market_nav,
            'strategy_return': strategy_return,
            'market_return': market_return,
            'difference': strategy_return - market_return,
            'total_value': portfolio_metrics['total_value'],
            'balance': portfolio_metrics['portfolio_balance'],
            'current_price': self.current_price,
            'current_position': portfolio_metrics.get('current_position', 0),
            'current_total_worth': portfolio_metrics['total_value'],
            'episode_steps': self.current_step,
            'trading_costs_paid': self._calculate_total_trading_costs()
        }

        return result

    def _calculate_total_trading_costs(self) -> float:
        """Calculate total trading costs paid during episode."""
        return sum([
            self.trading_costs.calculate_trading_cost(action, 1.0, price)
            for action, price in zip(self.action_history, self.nav_history)
        ])

    def seed(self, seed_value: int = 42):
        """Set random seed for reproducibility (Context7 standard)."""
        np.random.seed(seed_value)
        env_logger.info(f"Environment seed set to: {seed_value}")

    def get_state_dimensions(self) -> Tuple[int, int]:
        """
        Get state and action dimensions in Context7 format.

        Returns:
            Tuple of (state_dim, num_actions)
        """
        return self.state_dim, self.num_actions

    def get_max_episode_steps(self) -> int:
        """Get maximum episode steps (Context7 compatibility)."""
        return self.max_episode_steps

    def render(self, mode: str = 'human'):
        """
        Render environment state (Context7 compatible).

        Args:
            mode: Rendering mode ('human', 'rgb_array', etc.)
        """
        if mode == 'human':
            result = self.get_episode_result()
            print(f"Step: {self.current_step}/{self.max_episode_steps}")
            print(f"NAV: {result['nav']:.4f} | Market NAV: {result['market_nav']:.4f}")
            print(f"Strategy Return: {result['strategy_return']:.2f}% | Market Return: {result['market_return']:.2f}%")
            print(f"Balance: ${result['balance']:.2f} | Total Worth: ${result['current_total_worth']:.2f}")
            print(f"Current Price: ${result['current_price']:.2f}")
            print("-" * 50)


# Context7 Environment Registration
def register_trading_environment():
    """Register the trading environment with OpenAI Gym following Context7 patterns."""
    try:
        from gym.envs.registration import register

        register(
            id='trading-v2',
            entry_point='model.reinforcement.environments.trading_environment:TradingEnvironment',
            max_episode_steps=MAX_EPISODE_STEPS
        )

        # Also register with standard trading days
        register(
            id='trading-professional-v2',
            entry_point='model.reinforcement.environments.trading_environment:TradingEnvironment',
            max_episode_steps=DEFAULT_TRADING_DAYS,
            kwargs={'trading_days': DEFAULT_TRADING_DAYS}
        )

        env_logger.info("Context7 trading environments registered successfully")
        env_logger.info(f"Available: trading-v2 (max_steps={MAX_EPISODE_STEPS})")
        env_logger.info(f"Available: trading-professional-v2 (max_steps={DEFAULT_TRADING_DAYS})")

    except ImportError:
        env_logger.warning("OpenAI Gym not available. Environment registration skipped.")
    except Exception as e:
        env_logger.warning(f"Environment registration failed: {e}")


# Initialize Context7 registration on module import
register_trading_environment()


# Alias for backward compatibility
Environment = TradingEnvironment
