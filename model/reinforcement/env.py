import os
import pickle
import random

import numpy as np
import pandas as pd

import util.loggers as loggers
from util.utils import convert_df, features, load_config, tradex_features

logger = loggers.setup_loggers()
env_logger = logger['env']
rl_logger = logger['rl']

# Constants
EARLY_STOP_REWARD_THRESHOLD = -12.0
HIGH_ACC_REWARD_VALUES = [1, 0.8, 0.5]


class ActionSpace:
    """
    Represents the space of possible actions.
    Returns:
    - 0 sell.
    - 1 hold.
    - 2 buy.

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
        env_logger.info(f"create features: normal data {self.data_ }.")

        # Generate features and tradex features
        feature = features(self.data_.copy())
        env_logger.info(f"create features: extra feature {feature}.")

        processed_features = tradex_features(self.symbol, self.data_.copy())
        env_logger.info(
            f"create features: processed_features {processed_features}.")

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
        state = self._extract_state(self.bar)
        env_logger.info(f"STATE: the state is {state}.")
        return state

    def reset(self):
        env_logger.info("RESET:Environment reset.")

        self.data['action'] = 0
        self.total_reward = 0
        self.accuracy = 0
        self.bar = self.look_back

        state = self._extract_state(self.bar)

        env_logger.info(f"RESET:the data is {self.data[self.features]}")
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
        correct = action == df_row_standardized['d'] and action in [0, 2, 0]

        # Use the df_row_raw in check_conditions
        low_volatility, going_up_condition, going_down_condition, strong_buy_signal, strong_sell_signal = self._get_conditions(
            df_row_raw)

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

        self._log_step_details(
            reward, done, low_volatility, going_up_condition, going_down_condition, strong_buy_signal, strong_sell_signal, correct)

        return state, reward, info, done

    def _update_state_values(self, action, current_price, reward):
        if action in [0, 2] and reward == 1:
            self.last_action = action
        self.last_price = current_price
        self.data['action'].iloc[self.bar] = action
        self.total_reward = round(self.total_reward + reward, 2)
        self.bar += 1
        self.accuracy = round(self.total_reward /
                              (self.bar - self.look_back), 2)
        self.last_accuracy = self.accuracy

        if reward in HIGH_ACC_REWARD_VALUES and self.accuracy == self.last_accuracy:
            self.wait += 1
        elif self.accuracy >= 1.0:
            self.high_acc_counter += 1
        else:
            self.high_acc_counter = 0

    def _log_step_details(self, reward, done, is_low_volatility, is_going_up_hold, is_going_down_hold, strong_buy_signal, strong_sell_signal, correct):
        env_logger.info(
            f"STEP: volatility: {is_low_volatility}, going Up: {is_going_up_hold}, going down: {is_going_down_hold}\nstrong buy: {strong_buy_signal}, strong Sell: {strong_sell_signal}")

        env_logger.info(
            f"STEP: Total reward is {self.total_reward} | Accuracy is {self.accuracy}")
        env_logger.info(f'STEP: Last action taken {self.last_action}')
        env_logger.info(f"STEP: Reward: {reward}")
        if done:
            env_logger.info("STEP: Episode done.")
        elif reward in HIGH_ACC_REWARD_VALUES:
            env_logger.info("STEP: Action was correct.")
        else:
            env_logger.info("STEP: Action was False.")
        # Added this line
        env_logger.info(f"STEP: Action correctness: {correct}")
        env_logger.info(f"STEP: ===========================================")

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

    def is_increasing_trend(self, current_bar_index):
        """Checks if there's an increasing trend for the past 3 bars."""
        if current_bar_index < 2:
            return False
        return (self.data_.iloc[current_bar_index]['close'] > self.data_.iloc[current_bar_index - 1]['close']) and (self.data_.iloc[current_bar_index - 1]['close'] > self.data_.iloc[current_bar_index - 2]['close'])

    def _get_conditions(self, df_row):
        """Helper method to centralize the conditions logic."""
        current_bar_index = self.bar

        # Adjusting to use self.data_
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

        return low_volatility, going_up_condition, going_down_condition, strong_buy_signal, strong_sell_signal

    def compute_action(self, df):
        return df.apply(lambda row: self._compute_action_for_row(row), axis=1)

    def _compute_action_for_row(self, df_row):
        """Compute action based on the data row and current price."""
        low_volatility, going_up_condition, going_down_condition, strong_buy_signal, strong_sell_signal = self._get_conditions(
            df_row)

        if low_volatility:
            return 1
        elif going_up_condition or strong_buy_signal:
            return 2
        elif going_down_condition or strong_sell_signal:
            return 0
        else:
            return 2 if df_row['r'] > 0 else (0 if df_row['r'] < 0 else 1)

    def compute_reward(self, action, df_row):
        """Compute and return the reward based on the current action, data row, and current price."""
        is_low_volatility, is_going_up_hold, is_going_down_hold, strong_buy_signal, strong_sell_signal = self._get_conditions(
            df_row)

        if action == 2 and strong_buy_signal:
            return 2  # For example, a higher reward for correctly identifying a strong buy signal
        elif action == 0 and strong_sell_signal:
            return 1.5  # Similarly, a higher reward for a strong sell signal
        elif action == 1 and is_going_up_hold:
            return 1
        elif action == 0 and is_low_volatility:
            return 0.1
        elif action == 0 and is_going_up_hold:
            return 0.2
        elif action == 0 and is_going_down_hold:
            return 0.2
        else:
            return -0.2
