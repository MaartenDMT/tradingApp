import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta

import model.reinforcement.indicators as ind
import util.loggers as loggers
from util.utils import convert_df, features, load_config, tradex_features

logger = loggers.setup_loggers()
env_logger = logger['env']
rl_logger = logger['rl']

# Constants
EARLY_STOP_REWARD_THRESHOLD = -5.0


class ActionSpace:
    """
    Represents the space of possible actions.
    Returns:
    0 sell. 1 hold. 2 buy.
    """

    def __init__(self, n):
        self.n = n
        self.allowed: int = self.n - 1

    def sample(self):
        action = random.randint(0, self.allowed)
        if action > 2:
            env_logger.error(f'SAMPLE: not allowed action, action {action}')
            raise ValueError(
                f"Invalid action {action}. Action should be between 0 and 2.")
        env_logger.info(f"SAMPLE: random action generated: {action}")
        return action


class ObservationSpace:
    """Defines the observation space shape."""

    def __init__(self, n) -> None:
        self.shape = (n,)


class EnvironmentEx:
    def __init__(self, config, symbol, features, limit, time, actions, min_acc):
        self.config = config

        # Initialize other environment parameters

    def initialize_environment(self):
        self.data_manager.load_data()
        self.data_manager.preprocess_data()
        # Initialize other environment components


class Environment:
    """The main trading environment."""

    def __init__(self, symbol, features, limit, time, actions, min_acc):
        self.config = load_config()
        self.symbol, self.features = symbol, features
        self.limit, self.time = limit, time
        self.action_space = ActionSpace(actions)
        self.min_accuracy = min_acc
        self._initialize_environment()

    def _initialize_environment(self):
        self._set_initial_parameters()
        self.data_manager = DataManager(self.config)
        self.action_condition_logic = ActionConditionLogic()
        self.balance_management = BalanceManagement()
        self._setup_data()
        self._create_features()
        self._setup_observation_space()

    def _setup_data(self):
        self.data_manager.load_data()
        self.data_manager.preprocess_data()

    def _setup_observation_space(self):
        self.observation_space = ObservationSpace(
            len(self.env_data[self.features].columns))
        self.look_back = 1
        env_logger.info(
            f"Environment initialized with symbol {self.symbol} and features {self.features}. Observation space shape: {self.observation_space.shape}, lookback: {self.look_back}")

    def _set_initial_parameters(self):
        self.last_price, self.bar = None, 0
        self.trade_limit, self.threshold = 5, 0.2
        self.max_drawdown, self.trade_count = 0, 0
        self.last_action, self.bar = None, 0
        self.last_price, self.last_accuracy = None, None
        self.high_acc_counter, self.patience = 0, int(
            self.config['env']['patience'])
        self.wait, self.high_acc_threshold = 0, 5
        self.stocks_held, self.portfolio_balance = 0, 1000
        self.max_portfolio_balance, self.total_reward = 0, 0
        self.pnl, self.total_pnl = 0, 0
        self.accuracy, self.holding_time = 0, 0
        self.correct_actions = 0

    def _get_state(self):
        env_logger.info("STATE:get the state.")
        state = self._extract_state(self.bar)
        # print(f"the state shape {state.shape} & state {state}")
        return state

    def reset(self):
        env_logger.info("RESET:Environment reset.")

        self.env_data['action'] = 0
        self._set_initial_parameters()
        self.bar = self.look_back

        state = self._extract_state(self.bar)

        env_logger.info(
            f"RESET:the state shape {state.shape} & size is {state.size}")
        return state

    def _extract_state(self, bar):
        state_slice = self.env_data[self.features].iloc[bar -
                                                        self.look_back:bar].values
        return np.array(state_slice, dtype=np.float32)

    def step(self, action):
        env_logger.info(f"STEP: Action taken: {action}")

        # Extract the current row data from the raw dataframe
        df_row_raw = self.original_data.iloc[self.bar]

        if self.bar > 1:
            self.last_price = self.original_data.iloc[self.bar - 1]['close']
        else:
            # For the first step, initialize the last price to current price
            self.last_price = df_row_raw['close']

        # Updated for using df_row_raw
        current_price = df_row_raw['close']

        # Update balance and calculate drawdown and trading count
        self._update_balance(action, current_price)

        # Use the df_row_raw in check_conditions
        conditions = self._get_conditions(df_row_raw)

        # Assuming you have methods to compute each component
        market_condition_reward = self.compute_market_condition_reward(
            action, df_row_raw)
        financial_outcome_reward = self.compute_financial_outcome_reward(
            current_price, self.last_price, action, self.stocks_held)
        risk_adjusted_reward = self.compute_risk_adjusted_reward(
            self.env_data['r'])  # assuming self.env_data['r'] holds returns history

        # Calculate penalties
        drawdown_penalty = -1 * \
            max(0, self.max_drawdown - self.threshold)  # Define threshold
        trading_penalty = -1 * \
            max(0, self.trade_count - self.trade_limit) / \
            self.trade_limit  # Define trade_limit

        # Combine rewards and penalties
        reward = self.combined_reward(market_condition_reward, financial_outcome_reward,
                                      risk_adjusted_reward, drawdown_penalty, trading_penalty)

        correct = ind.is_correct_action(
            action, ind.get_optimal_action(reward))

        # Update the last action taken based on compute_action's result
        computed_action = self._compute_action_for_row(df_row_raw)
        if computed_action in [0, 2]:
            self.last_action = computed_action

        self._update_state_values(action, current_price, reward)
        done = self.is_episode_done()

        state = self._get_state()
        info = {}

        self._log_step_details(reward, done, *conditions, correct)

        if correct:
            self.correct_actions += 1

        return state, reward, info, done

    def _log_step_details(self, reward, done, *conditions, correct=False):

        # Log reward component details
        env_logger.info(
            f"LOG: Weighted Market Condition Reward: {self.market_condition_reward_weighted:.2f}")
        env_logger.info(
            f"LOG: Weighted Financial Outcome Reward: {self.financial_outcome_reward_weighted:.2f}")
        env_logger.info(
            f"LOG: Weighted Risk Adjusted Reward: {self.risk_adjusted_reward_weighted:.2f}")
        env_logger.info(
            f"LOG: Weighted Drawdown Penalty: {self.drawdown_penalty_weighted:.2f}")
        env_logger.info(
            f"LOG: Weighted Trading Penalty: {self.trading_penalty_weighted:.2f}")

        # Log general step details
        env_logger.info(
            f"LOG: Step {self.bar} | Action: {self.last_action} | Reward: {reward:.2f} | Correct: {correct}")
        env_logger.info(
            f"LOG: Portfolio Balance: ${self.portfolio_balance:.2f} | Stocks Held: {self.stocks_held:.2f} | PNL: {self.pnl:.2f}")
        env_logger.info(
            f"LOG: Total Reward: {self.total_reward} | Accuracy: {self.accuracy:.2%}")
        if done:
            env_logger.info("LOG: Episode done.")

        # Log market condition details
        condition_names = ["Short Stochastic", "Short Bollinger", "Long Stochastic", "Long Bollinger", "Low Volatility",
                           "Going Up", "Going Down", "Strong Buy", "Strong Sell", "Super Buy", "Super Sell",
                           "MACD Buy", "High Volatility", "ADX Signal", "PSAR Signal", "CDL Pattern", "Volume Break", "Resistance Break"]
        condition_details = dict(zip(condition_names, conditions))
        for name, value in condition_details.items():
            env_logger.info(f"LOG: Condition - {name}: {value}")

        env_logger.info("LOG: ===========================================")

    def is_episode_done(self):
        if self.total_reward <= EARLY_STOP_REWARD_THRESHOLD:
            env_logger.warning("DONE: Early stopping due to less rewards.")
            return True
        if self.bar >= len(self.env_data) - 1:
            env_logger.warning("DONE: Early stopping due to no more bars.")
            return True
        if self.wait >= self.patience:
            self.wait = 0
            env_logger.warning(
                "DONE: Early stopping due to lack of improvement.")
            return True
        if self.high_acc_counter >= self.high_acc_threshold:
            env_logger.warning(
                f"DONE: Stopping early due to sustained high accuracy: {self.accuracy}")
            return True
        if self.accuracy < self.min_accuracy and self.bar > self.look_back + 50:
            self.wait = 0
            env_logger.warning(
                f"DONE: Stopping early due to low accuracy: {self.accuracy}")
            return True
        return False

    def get_feature_names(self):
        return self.env_data.columns.tolist()


class DataManager:
    def __init__(self, config):
        self.config = config
        self.original_data = self.load_data()

    def load_data(self):
        # Implement data loading logic here
        # Get the file name from the configuration
        pickle_file_name = self.config['Path']['2020_30m_data']

        # Check if the file exists
        if not os.path.exists(pickle_file_name):
            env_logger.error('no data has been written')
            return None

        # Load the data from the pickle file
        with open(pickle_file_name, 'rb') as f:
            data_ = pickle.load(f)

        # Convert the dataframe (assuming `convert_df` is a function that processes your data)
        data = convert_df(data_)

        # Adjusting for the percentage
        percentage_to_keep = float(self.config['Data']['percentage']) / 100.0
        rows_to_keep = int(len(data) * percentage_to_keep)
        data = data.head(rows_to_keep)

        rl_logger.info(f'df shape:{data.shape}')

        return data

    def preprocess_data(self):
        if self.original_data.empty:
            env_logger.error("Data fetch failed or returned empty data.")
            return

        env_logger.info("Data fetched successfully.")

        # Generate features and tradex features
        feature = features(self.original_data.copy())

        processed_features = tradex_features(
            self.symbol, self.original_data.copy())

        # Combine processed_features and feature into self.original_data
        self.original_data = pd.concat([processed_features, feature], axis=1)
        self.original_data.dropna(inplace=True)

        self.original_data['last_price'] = self.original_data['close'].shift(1)
        self.env_data = self.original_data[self.features].copy()
        self.original_data['r'] = np.log(
            self.env_data['close'] / self.env_data['close'].shift(int(self.config['env']['shifts'])))

        self.env_data['d'] = self.compute_action(self.original_data)
        # Handle NaN values that will appear in the last 3 rows
        self.env_data['d'] = self.env_data['d'].ffill()
        self.env_data['r'] = self.original_data['r']

        self.env_data['b_wave'] = self.original_data['b_wave']
        self.env_data['l_wave'] = self.original_data['l_wave']

        self.env_data['rsi14'] = self.original_data['rsi14']
        self.env_data['rsi40'] = self.original_data['rsi40']
        self.env_data['ema_200'] = self.original_data['ema_200']
        self.env_data['s_dots'] = self.original_data['s_dots']
        self.env_data['dots'] = self.original_data['dots']

        self.env_data = (self.env_data - self.env_data.mean()
                         ) / self.env_data.std()
        self.env_data['action'] = 0

        env_logger.info(
            f"create features: Full PROCESSED data {self.original_data}.")
        env_logger.info(f"create features: observation Data {self.env_data}.")
        env_logger.info("Features created.")

    def get_feature_names(self):
        # Return the feature names
        return self.env_data.columns.tolist()

    def get_state(self, bar):
        # Implement state extraction logic
        env_logger.info("STATE:get the state.")
        state = self._extract_state(self.bar)
        # print(f"the state shape {state.shape} & state {state}")
        return state

    def _extract_state(self, bar):
        state_slice = self.env_data[self.features].iloc[bar -
                                                        self.look_back:bar].values
        return np.array(state_slice, dtype=np.float32)

    def compute_pnl(self):
        """
        Calculate the Profit and Loss (PnL) of the current portfolio as a percentage.

        Returns:
        pnl_percentage (float): The calculated profit or loss as a percentage.
        """
        # Check if last price and current price are available
        if self.last_price is None or self.last_action is None:
            return 0.0

        # Initial asset value
        initial_asset_value = self.stocks_held * self.last_price

        # Avoid division by zero
        if initial_asset_value == 0:
            return 0.0

        # Current asset value
        current_asset_value = self.stocks_held * \
            self.original_data.iloc[self.bar]['close']

        # Calculate PnL
        self.pnl = current_asset_value - initial_asset_value
        self.total_pnl += self.pnl

        # Calculate PnL as a percentage
        pnl_percentage = (self.pnl / initial_asset_value) * 100

        return pnl_percentage

    # Other data-related methods


class ActionConditionLogic:
    def __init__(self):
        pass

    def compute_action(self, df_row):
        return df.apply(lambda row: self._compute_action_for_row(row), axis=1)

    def _compute_action_for_row(self, df_row):
        """Compute action based on the data row and current price."""
        short_stochastic_signal, short_bollinger_outside, long_stochastic_signal, long_bollinger_outside, low_volatility, going_up_condition, going_down_condition, strong_buy_signal, strong_sell_signal, super_buy, super_sell, macd_buy, high_volatility, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal = self._get_conditions(
            df_row)

        bullish_conditions = [strong_buy_signal, super_buy, macd_buy, long_stochastic_signal, long_bollinger_outside,
                              high_volatility, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal]
        bearish_conditions = [strong_sell_signal, super_sell,
                              short_stochastic_signal, short_bollinger_outside]
        neutral_conditions = [going_up_condition,
                              going_down_condition, low_volatility]

        # Calculate PnL
        pnl = self.compute_pnl()  # You need to implement this method

        # Define PnL thresholds
        pnl_threshold_buy = -0.10  # e.g., 10% loss
        pnl_threshold_sell = 0.10  # e.g., 10% profit

        # Decision logic based on PnL
        if pnl <= pnl_threshold_buy:
            self.holding_time = 0
            return 2  # Buy action
        elif pnl >= pnl_threshold_sell:
            self.holding_time = 0
            return 0  # Sell action
        else:
            if any(neutral_conditions):
                self.holding_time += 1
                return 1  # Neutral action
            elif any(bullish_conditions):
                self.holding_time = 0
                return 2  # Bullish action
            elif any(bearish_conditions):
                self.holding_time = 0
                return 0  # Bearish action
            else:
                # Default logic if none of the above conditions are met
                self.holding_time += 1
                return 1

    def compute_market_condition_reward(self, action, df_row):
        """Compute and return the reward based on the current action, data row, and current price."""
        short_stochastic_signal, short_bollinger_outside, long_stochastic_signal, long_bollinger_outside, low_volatility, going_up_condition, going_down_condition, strong_buy_signal, strong_sell_signal, super_buy, super_sell, macd_buy, high_volatility, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal = self._get_conditions(
            df_row)

        bullish_conditions = [strong_buy_signal, super_buy, macd_buy, long_stochastic_signal, long_bollinger_outside,
                              high_volatility, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal]
        bearish_conditions = [strong_sell_signal, super_sell,
                              short_stochastic_signal, short_bollinger_outside]
        neutral_conditions = [going_up_condition,
                              going_down_condition, low_volatility]

        if action == 2 and any(bullish_conditions):
            if super_buy:
                return 2  # Higher reward for correctly identifying a strong buy signal
            else:
                return 1  # Reward for other bullish conditions
        elif action == 0 and any(bearish_conditions):
            if super_sell:
                return 2  # Higher reward for a strong sell signal
            else:
                return 1  # Reward for other bearish conditions
        elif action == 1 and any(neutral_conditions):
            if low_volatility:
                return 0.2
            elif going_up_condition:
                return 0.5
            elif going_down_condition:
                return 0.5
        else:
            return -0.2  # Penalize when no specific condition is met

    def compute_financial_outcome_reward(self, current_price, last_price, action, position_size):
        """
        Compute the reward based on action and market data.

        Parameters:
        - current_price: Current asset price.
        - last_price: Asset price at the last time step.
        - action: The action taken by the agent.
        - position_size: The size of the position.

        Returns:
        - Calculated reward.
        """

        p_or_l = ind.calculate_profit_or_loss(
            current_price, last_price, action, position_size)

        if action == 1 and p_or_l > 0:  # Only reward holding if in profit
            # Calculate holding reward, e.g., a function of holding time and PnL
            holding_reward = ind.calculate_holding_reward(
                self.holding_time, p_or_l)
            p_or_l += holding_reward

        return p_or_l

    def compute_risk_adjusted_reward(self, returns_history):
        """
        Compute the reward based on action and risk.

        Parameters:
        - returns_history: Historical returns for risk calculation.

        Returns:
        - Calculated reward.
        """
        # Calculate Sharpe Ratio for risk-adjusted return
        sharpe_ratio = ind.calculate_sharpe_ratio(returns_history)

        # Combine profit/loss and risk-adjusted return
        # Here you can decide how to weigh these components
        reward = sharpe_ratio  # Simple additive model as an example

        return reward

    def combined_reward(self, market_condition_reward, financial_outcome_reward, risk_adjusted_reward, drawdown_penalty, trading_penalty):
        """
        Combine different reward components into a single reward value.

        Parameters:
        - market_condition_reward: Reward based on market conditions.
        - financial_outcome_reward: Reward based on immediate financial outcome.
        - risk_adjusted_reward: Reward based on risk-adjusted returns.
        - weights: Tuple of weights for each component.

        Returns:
        - Combined reward.
        """
        self.market_condition_reward_weighted = weights[0] * \
            market_condition_reward
        self.financial_outcome_reward_weighted = weights[1] * \
            financial_outcome_reward
        self.risk_adjusted_reward_weighted = weights[2] * risk_adjusted_reward
        self.drawdown_penalty_weighted = weights[3] * drawdown_penalty
        self.trading_penalty_weighted = weights[4] * trading_penalty

        combined = (self.market_condition_reward_weighted +
                    self.financial_outcome_reward_weighted +
                    self.risk_adjusted_reward_weighted +
                    self.drawdown_penalty_weighted +
                    self.trading_penalty_weighted)

        return combined

    def get_conditions(self, df_row):
        """Helper method to centralize the conditions logic."""
        current_bar_index = self.bar

        # Adjusting to use self.original_data
        super_buy: bool = (df_row['dots'] == 1) & [df_row['l_wave'] >= -50]
        super_sell: bool = (df_row['dots'] == -1) & [df_row['l_wave'] >= 50]
        low_volatility: bool = (df_row['rsi14'] >= 45) & (
            df_row['rsi14'] <= 55)
        strong_upward_movement: bool = df_row['rsi14'] > 70
        strong_downward_movement: bool = df_row['rsi14'] < 30
        going_up_condition: bool = (df_row['close'] > df_row['last_price']) & (
            df_row['close'] > df_row['ema_200']) & (df_row['rsi40'] > 50)
        going_down_condition: bool = (df_row['close'] < df_row['last_price']) & (
            df_row['close'] < df_row['ema_200']) & (df_row['rsi40'] < 50)

        strong_buy_signal = strong_upward_movement & ~ind.is_increasing_trend(self.original_data,
                                                                              current_bar_index)
        strong_sell_signal = strong_downward_movement & ~ind.is_increasing_trend(self.original_data,
                                                                                 current_bar_index)  # ~ is the element-wise logical NOT

        # SHORT
        short_stochastic_signal = ~ind.short_stochastic_condition(
            self.original_data, current_bar_index)
        short_bollinger_outside = ~ind.short_bollinger_condition(
            self.original_data, current_bar_index)
        # LONG ONlY
        long_stochastic_signal = ~ind.long_stochastic_condition(
            self.original_data, current_bar_index)
        long_bollinger_outside = ~ind.long_bollinger_condition(
            self.original_data, current_bar_index)
        macd_buy = ~ind.macd_condition(self.original_data, current_bar_index)
        high_volatility = ~ind.atr_condition(
            self.original_data, current_bar_index)
        adx_signal = ~ind.adx_condition(self.original_data, current_bar_index)
        psar_signal = ~ind.parabolic_sar_condition(
            self.original_data, current_bar_index)
        cdl_pattern = ~ind.cdl_pattern(self.original_data, current_bar_index)
        volume_break = ~ind.volume_breakout(
            self.original_data, current_bar_index)
        resistance_break_signal = ~ind.resistance_break(
            self.original_data, current_bar_index)

        return short_stochastic_signal, short_bollinger_outside, long_stochastic_signal, long_bollinger_outside, low_volatility, going_up_condition, going_down_condition, strong_buy_signal, strong_sell_signal, super_buy, super_sell, macd_buy, high_volatility, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal

    # Other action and condition related methods


class BalanceManagement:
    def __init__(self):
        pass

    def update_balance(self, action, current_price):
        """Update portfolio balance based on the action taken.

        Parameters:
        - action (int): Action taken (0 for sell, 2 for buy).
        - current_price (float): Current stock price.
        - env_logger: Logger for logging information.

        Returns:
        None
        """

        env_logger.info(
            f"Starting balance: ${self.portfolio_balance:.2f}, Stocks held: {self.stocks_held:.2f}")

        # Calculate the amount to trade (10% of current balance)
        amount_to_trade = 0.1 * self.portfolio_balance

        # Update trading count if an action is taken
        if action in [0, 2]:
            self.trade_count += 1

        # Sell action
        if action == 0:
            # Sell only if you have enough stocks
            stocks_to_sell = min(
                self.stocks_held, amount_to_trade / current_price)
            self.portfolio_balance += stocks_to_sell * current_price
            self.stocks_held -= stocks_to_sell
            env_logger.info(
                f"Selling {stocks_to_sell:.2f} stocks at ${current_price:.2f} each.")

        # Buy action
        elif action == 2:
            # Buy only if balance allows
            stocks_to_buy = min(amount_to_trade / current_price,
                                self.portfolio_balance / current_price)
            self.portfolio_balance -= stocks_to_buy * current_price
            self.stocks_held += stocks_to_buy
            env_logger.info(
                f"Buying {stocks_to_buy:.2f} stocks at ${current_price:.2f} each.")

        else:
            env_logger.warning(f"Hold action: {action}. No action taken.")

        # Update the maximum portfolio balance
        self.max_portfolio_balance = max(
            self.max_portfolio_balance, self.portfolio_balance)

        # Calculate drawdown only when there's a decrease in value
        if self.portfolio_balance < self.max_portfolio_balance:
            current_drawdown = 1 - \
                (self.portfolio_balance / self.max_portfolio_balance)
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Log the updated balance
        env_logger.info(
            f"Updated balance: ${self.portfolio_balance:.2f}, Stocks held: {self.stocks_held:.2f}")
