import os
import pickle
import random

import numpy as np
import pandas as pd
import pandas_ta as ta

import util.loggers as loggers
from util.utils import convert_df, features, load_config, tradex_features

logger = loggers.setup_loggers()
env_logger = logger['env']
rl_logger = logger['rl']

# Constants
EARLY_STOP_REWARD_THRESHOLD = -12.0
HIGH_ACC_REWARD_VALUES = [1.5, 1, 0.5]


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


class Environment:
    """The main trading environment."""

    def __init__(self, symbol, features, limit, time, actions, min_acc):
        self.config = load_config()
        self.limit = limit
        self.time = time
        self.symbol = symbol
        self.features = features
        self.action_space = ActionSpace(actions)
        self.min_accuracy = min_acc
        self.last_price = None
        self._setup_initial_params()

    def _setup_initial_params(self):
        self.last_action = None
        self.bar = 0
        self.last_price = None
        self.last_accuracy = None
        self.high_acc_counter = 0
        self.patience = int(self.config['env']['patience'])
        self.wait = 0
        self.high_acc_threshold = 5
        self.data_ = self._get_data()
        self._create_features()
        self.observation_space = ObservationSpace(len(self.data.columns))
        self.look_back = self.observation_space.shape[0]

        # portfolio:
        self.stocks_held = 0
        self.portfolio_balance = 1000

        env_logger.info(
            f"Environment initialized with symbol {self.symbol} and features {self.features}. Observation space shape: {self.observation_space.shape} and lookback {self.look_back}")

    def _get_data(self):
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

    def _create_features(self):
        if self.data_.empty:
            env_logger.error("Data fetch failed or returned empty data.")
            return

        env_logger.info("Data fetched successfully.")

        # Generate features and tradex features
        feature = features(self.data_.copy())

        processed_features = tradex_features(self.symbol, self.data_.copy())

        # Combine processed_features and feature into self.data_
        self.data_ = pd.concat([processed_features, feature], axis=1)
        self.data_.dropna(inplace=True)

        self.data_['last_price'] = self.data_['close'].shift(1)
        self.data = self.data_[self.features].copy()
        self.data_['r'] = np.log(
            self.data['close'] / self.data['close'].shift(1))
        self.data['d'] = self.compute_action(self.data_)
        self.data['r'] = self.data_['r']

        self.data['b_wave'] = self.data_['b_wave']
        self.data['l_wave'] = self.data_['l_wave']

        self.data['rsi14'] = self.data_['rsi14']
        self.data['rsi40'] = self.data_['rsi40']
        self.data['ema_200'] = self.data_['ema_200']
        self.data['s_dots'] = self.data_['s_dots']
        self.data['dots'] = self.data_['dots']

        self.data = (self.data - self.data.mean()) / self.data.std()
        self.data['action'] = 0

        env_logger.info(f"create features: Full PROCESSED data {self.data_ }.")
        env_logger.info(f"create features: observation Data {self.data}.")
        env_logger.info("Features created.")

    def get_feature_names(self):
        return self.data.columns.tolist()

    def _get_state(self):
        env_logger.info("STATE:get the state.")
        state = self._extract_state(self.bar)
        return state

    def reset(self):
        env_logger.info("RESET:Environment reset.")

        self.data['action'] = 0
        self.total_reward = 0
        self.accuracy = 0
        self.bar = self.look_back

        state = self._extract_state(self.bar)

        env_logger.info(
            f"RESET:the state shape {state.shape} & size is {state.size}")
        return state

    def _extract_state(self, bar):
        state_slice = self.data[self.features].iloc[bar -
                                                    self.look_back:bar].values
        return np.array(state_slice, dtype=np.float32)

    def step(self, action):
        env_logger.info(f"STEP: Action taken: {action}")

        # Extract the current row data from the standardized dataframe
        df_row_standardized = self.data.iloc[self.bar]

        # Extract the current row data from the raw dataframe
        df_row_raw = self.data_.iloc[self.bar]

        if self.bar > 0:
            self.last_price = self.data_.iloc[self.bar - 1]['close']
        else:
            # For the first step, initialize the last price to current price
            self.last_price = df_row_raw['close']

        # Updated for using df_row_raw
        current_price = df_row_raw['close']
        correct = action == df_row_standardized['d'] and action in [0, 2, 1]
        self._update_balance(action, current_price)

        # Use the df_row_raw in check_conditions
        conditions = self._get_conditions(df_row_raw)

        # Use the df_row_raw in compute_reward
        reward = self.compute_reward(action, df_row_raw)

        # Update the last action taken based on compute_action's result
        computed_action = self._compute_action_for_row(df_row_raw)
        if computed_action in [0, 1]:
            self.last_action = computed_action

        self._update_state_values(action, current_price, reward)
        done = self.is_episode_done()

        state = self._get_state()
        info = {}

        self._log_step_details(reward, done, *conditions, correct)

        return state, reward, info, done

    def _update_balance(self, action, current_price):
        """Update portfolio balance based on the action taken.

        Parameters:
        - action (int): Action taken (0 for sell, 2 for buy).
        - current_price (float): Current stock price.

        Returns:
        None
        """

        env_logger.info(
            f"Starting balance: ${self.portfolio_balance:.2f}, Stocks held: {self.stocks_held:.2f}")

        # Calculate the amount to trade (10% of current balance)
        amount_to_trade = 0.1 * self.portfolio_balance

        if action == 0:  # Sell
            stocks_to_sell = self.stocks_held  # Selling all stocks currently held
            self.portfolio_balance += stocks_to_sell * current_price
            self.stocks_held -= stocks_to_sell

            env_logger.info(f"Selling {stocks_to_sell:.2f} stocks at ${current_price:.2f} each. \
                            Updated balance: ${self.portfolio_balance:.2f}")

        elif action == 2:  # Buy
            # Calculating stocks to buy based on amount and current price
            stocks_to_buy = amount_to_trade / current_price
            self.portfolio_balance -= stocks_to_buy * current_price
            self.stocks_held += stocks_to_buy

            env_logger.info(f"Buying {stocks_to_buy:.2f} stocks at ${current_price:.2f} each. \
                            Updated balance: ${self.portfolio_balance:.2f}")

        else:
            env_logger.warning(f"Unknown action: {action}. No action taken.")

    def _update_state_values(self, action, current_price, reward):
        """Update internal state values based on the action, current price, and received reward."""

        # Log update information
        env_logger.info(
            f"Updating the state values based on action: {action}, current_price: {current_price}, reward: {reward}")

        # Update last action and price
        if action in [0, 2] and reward in [1, 1.5]:
            self.last_action = action
        self.last_price = current_price

        # Update data and bar counter
        self.data['action'].iloc[self.bar] = action
        self.bar += 1

        # Update reward and accuracy
        self.total_reward = round(self.total_reward + reward, 2)
        self.accuracy = round(self.total_reward /
                              (self.bar - self.look_back), 2)
        self.last_accuracy = self.accuracy

        # Update counters based on accuracy and reward values
        if reward in HIGH_ACC_REWARD_VALUES and self.accuracy == self.last_accuracy:
            self.wait += 1
        elif self.accuracy >= 1.0:
            self.high_acc_counter += 1
        else:
            self.high_acc_counter = 0

    def _log_step_details(self, reward, done, low_volatility, going_up_condition, going_down_condition, strong_buy_signal, strong_sell_signal, super_buy, super_sell, macd_buy, bollinger_outside, high_volatility, stochastic_signal, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal, correct):
        env_logger.info(
            f"LOG: volatility: {low_volatility}, going Up: {going_up_condition}, going down: {going_down_condition}\nstrong buy: {strong_buy_signal}, strong Sell: {strong_sell_signal}")
        env_logger.info(
            f"LOG: Super buy: {super_buy}, Super Sell: {super_sell}")
        env_logger.info(
            f"LOG: MACD Buy: {macd_buy}, Bollinger Outside: {bollinger_outside}, High Volatility: {high_volatility}, Stochastic Signal: {stochastic_signal}, ADX Signal: {adx_signal}, PSAR Signal: {psar_signal}, cdl_pattern: {cdl_pattern}, Volume Break: {volume_break}, Resistance Break: {resistance_break_signal}")
        env_logger.info(
            f"LOG: Total reward is {self.total_reward} | Accuracy is {self.accuracy}")
        env_logger.info(f'LOG: Last action taken {self.last_action}')
        env_logger.info(f"LOG: Reward: {reward}")
        if done:
            env_logger.info("LOG: Episode done.")
        elif reward in HIGH_ACC_REWARD_VALUES:
            env_logger.info("LOG: Action was correct.")
        else:
            env_logger.info("LOG: Action was False.")
        # Added this line
        env_logger.info(f"LOG: Action correctness: {correct}")

        env_logger.info(
            f"PORTFOLIO: Portfolio Balance: ${self.portfolio_balance}")
        env_logger.info(f"PORTFOLIO: Stocks Held: {self.stocks_held}")
        env_logger.info(f"LOG: ===========================================")

    def is_episode_done(self):
        if self.total_reward <= EARLY_STOP_REWARD_THRESHOLD:
            env_logger.warning("STEP: Early stopping due to less rewards.")
            return True
        if self.bar >= len(self.data) - 1:
            env_logger.warning("STEP: Early stopping due to no more bars.")
            return True
        if self.wait >= self.patience:
            self.wait = 0
            env_logger.warning(
                "STEP: Early stopping due to lack of improvement.")
            return True
        if self.high_acc_counter >= self.high_acc_threshold:
            env_logger.warning(
                f"STEP: Stopping early due to sustained high accuracy: {self.accuracy}")
            return True
        if self.accuracy < self.min_accuracy and self.bar > self.look_back + 50:
            self.wait = 0
            env_logger.warning(
                f"STEP: Stopping early due to low accuracy: {self.accuracy}")
            return True
        return False

    def _get_conditions(self, df_row):
        """Helper method to centralize the conditions logic."""
        current_bar_index = self.bar

        # Adjusting to use self.data_
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

        strong_buy_signal = strong_upward_movement & ~self.is_increasing_trend(
            current_bar_index)
        strong_sell_signal = strong_downward_movement & ~self.is_increasing_trend(
            current_bar_index)  # ~ is the element-wise logical NOT

        # New Conditions
        macd_buy = ~self.macd_condition(current_bar_index)
        bollinger_outside = ~self.bollinger_condition(current_bar_index)
        high_volatility = ~self.atr_condition(current_bar_index)
        stochastic_signal = ~self.stochastic_condition(current_bar_index)
        adx_signal = ~self.adx_condition(current_bar_index)
        psar_signal = ~self.parabolic_sar_condition(current_bar_index)
        cdl_pattern = ~self.cdl_pattern(current_bar_index)
        volume_break = ~self.volume_breakout(current_bar_index)
        resistance_break_signal = ~self.resistance_break(current_bar_index)

        return low_volatility, going_up_condition, going_down_condition, strong_buy_signal, strong_sell_signal, super_buy, super_sell, macd_buy, bollinger_outside, high_volatility, stochastic_signal, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal

    def compute_action(self, df):
        return df.apply(lambda row: self._compute_action_for_row(row), axis=1)

    def _compute_action_for_row(self, df_row):
        """Compute action based on the data row and current price."""
        low_volatility, going_up_condition, going_down_condition, strong_buy_signal, strong_sell_signal, super_buy, super_sell, macd_buy, bollinger_outside, high_volatility, stochastic_signal, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal = self._get_conditions(
            df_row)

        bullish_conditions = [strong_buy_signal, super_buy, macd_buy, bollinger_outside, high_volatility,
                              stochastic_signal, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal]
        bearish_conditions = [strong_sell_signal, super_sell]
        neutral_conditions = [going_up_condition,
                              going_down_condition, low_volatility]

        if any(bullish_conditions):
            return 2  # Bullish action
        elif any(bearish_conditions):
            return 0  # Bearish action
        elif any(neutral_conditions):
            return 1  # Neutral action
        else:
            # Default logic if none of the above conditions are met
            return 2 if df_row['r'] > 0 else (0 if df_row['r'] < 0 else 1)

    def compute_reward(self, action, df_row):
        """Compute and return the reward based on the current action, data row, and current price."""
        low_volatility, going_up_condition, going_down_condition, strong_buy_signal, strong_sell_signal, super_buy, super_sell, macd_buy, bollinger_outside, high_volatility, stochastic_signal, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal = self._get_conditions(
            df_row)

        bullish_conditions = [strong_buy_signal, super_buy, macd_buy, bollinger_outside, high_volatility,
                              stochastic_signal, adx_signal, psar_signal, cdl_pattern, volume_break, resistance_break_signal]
        bearish_conditions = [strong_sell_signal, super_sell]
        neutral_conditions = [going_up_condition,
                              going_down_condition, low_volatility]

        if action == 2 and any(bullish_conditions):
            if strong_buy_signal:
                return 1.5  # Higher reward for correctly identifying a strong buy signal
            else:
                return 1  # Reward for other bullish conditions
        elif action == 0 and any(bearish_conditions):
            if strong_sell_signal:
                return 1.5  # Higher reward for a strong sell signal
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

    def macd_condition(self, current_bar_index):
        """Checks if the MACD line is above the signal line."""
        if current_bar_index < 15:
            return False
        macd = self.data_.ta.macd(
            fast=14, slow=28, signal=9).iloc[current_bar_index]
        return macd['MACD_14_28_9'] > macd['MACDs_14_28_9']

    def bollinger_condition(self, current_bar_index):
        """Checks for a breakout from the Bollinger Bands."""
        if current_bar_index < 21:
            return False
        bollinger = self.data_.ta.bbands().iloc[current_bar_index]
        return self.data_.iloc[current_bar_index]['close'] > bollinger['BBU_5_2.0'] or self.data_.iloc[current_bar_index]['close'] < bollinger['BBL_5_2.0']

    def atr_condition(self, current_bar_index):
        """Checks if the current ATR is greater than the mean ATR."""
        if current_bar_index < 15:
            return False
        atr = self.data_.ta.atr().iloc[current_bar_index]
        return atr > atr.mean()

    def stochastic_condition(self, current_bar_index):
        """Checks for a bullish crossover in overbought territory in stochastic oscillator."""
        if current_bar_index < 15:
            return False
        stochastic = self.data_.ta.stoch().iloc[current_bar_index]
        return stochastic['STOCHk_14_3_3'] > stochastic['STOCHd_14_3_3'] and stochastic['STOCHk_14_3_3'] > 80

    def adx_condition(self, current_bar_index):
        """Checks if the strength of the trend is strong with ADX > 25."""
        if current_bar_index < 15:
            return False
        adx = self.data_.ta.adx().iloc[current_bar_index]
        return adx['ADX_14'] > 25

    def parabolic_sar_condition(self, current_bar_index):
        """Checks if the close price is above the Parabolic SAR."""
        if current_bar_index < 21:
            return False
        psar = self.data_.ta.psar().iloc[current_bar_index]
        return self.data_.iloc[current_bar_index]['close'] > psar

    def cdl_pattern(self, current_bar_index):
        """Checks for the presence of a cdl_pattern candlestick pattern."""
        if current_bar_index < 2:
            return False
        cdl_pattern = self.data_.ta.cdl_pattern(
            name=["doji", "hammer"]).iloc[current_bar_index]
        return cdl_pattern != 0

    def volume_breakout(self, current_bar_index):
        """Checks for the presence of a cdl_pattern candlestick pattern."""
        if current_bar_index < 21:
            return False
        # Average volume over the last 20 bars.
        avg_volume = self.data_[
            'volume'].iloc[current_bar_index-20:current_bar_index].mean()
        # Current volume is 150% of the average volume.
        return self.data_.iloc[current_bar_index]['volume'] > 1.5 * avg_volume

    def resistance_break(self, current_bar_index):
        """Checks for the presence of a cdl_pattern candlestick pattern."""
        if current_bar_index < 21:
            return False
        # Maximum high over the last 20 bars.
        resistance = self.data_[
            'high'].iloc[current_bar_index-20:current_bar_index].max()

        # Current close is above the resistance.
        return self.data_.iloc[current_bar_index]['close'] > resistance

    def is_increasing_trend(self, current_bar_index):
        """Checks if there's an increasing trend for the past 3 bars."""
        if current_bar_index < 2:
            return False
        return (self.data_.iloc[current_bar_index]['close'] > self.data_.iloc[current_bar_index - 1]['close']) and (self.data_.iloc[current_bar_index - 1]['close'] > self.data_.iloc[current_bar_index - 2]['close'])
