import os
import pickle
import random

import numpy as np
import pandas as pd
import pandas_ta as ta

import util.loggers as loggers
from model.features import Tradex_indicator
from util.utils import load_config

logger = loggers.setup_loggers()
pd.set_option('mode.chained_assignment', None)
rl_logger = logger['env']


class Actions:
    def __init__(self, n):
        self.n = n
        self.allowed: int = self.n - 2

    def sample(self):
        action = random.randint(-1, self.allowed)
        if action > 1:
            rl_logger.warning(f'SAMPLE: not allowed action, action  {action}')

        rl_logger.info(f"SAMPLE: random action generated: {action}")
        return action


class observation_space:
    def __init__(self, n) -> None:
        self.shape = (n,)


class Environment:

    def __init__(self, symbol, features, limit, time, actions, min_acc):
        self.config = load_config()
        self.limit = limit
        self.time = time
        self.symbol = symbol
        self.features = features
        self.action_space = Actions(actions)
        self.min_accuracy = min_acc

        self.last_action = None  # To keep track of the last action
        self.last_price = None   # To keep track of the last price
        self.last_accuracy = None  # To keep track of the last accuracy
        self.high_acc_counter = 0

        # steps params
        self.patience = 20
        self.wait = 0
        self.high_acc_threshold = 5

        self._get_data(True)
        self._create_features()
        self.observation_space = observation_space(len(self.data.columns))
        self.look_back = self.observation_space.shape[0]
        rl_logger.info(
            f"Environment initialized with symbol {self.symbol} and features {self.features}. Observation space shape: {self.observation_space.shape}")

    def _get_data(self, none=True):
        if none:
            # Define the name of the pickle file based on days, ticker, and time frame
            pickle_file_name = self.config['Path']['2020_30m_data']

            # Check if the pickle file already exists
            if os.path.exists(pickle_file_name):
                # If it exists, read from it
                with open(pickle_file_name, 'rb') as f:
                    self.data_ = pickle.load(f)

                # Calculate the maximum starting index to get 10,000 rows
                max_start_index = len(self.data_) - 1000

                # Generate a random starting index
                random_start_index = random.randint(0, max_start_index)

                # Get 10,000 rows starting from the random index
                self.data_ = self.data_.iloc[random_start_index:random_start_index + 1000]

                return self.data_
            else:
                rl_logger.error('no data has been writen')

    def _create_features(self):
        if self.data_ is None or self.data_.empty:
            rl_logger.error("Data fetch failed or returned empty data.")
        else:
            rl_logger.info("Data fetched successfully.")

        # Moving Average
        self.data_['ma_10'] = self.data_['close'].rolling(window=10).mean()

        # Exponential Moving Average
        self.data_['ema_10'] = self.data_['close'].ewm(
            span=10, adjust=False).mean()
        self.data_['ema_200'] = self.data_['close'].ewm(
            span=200, adjust=False).mean()

        # Momentum
        # Stochastic Oscillator (STOCH)
        stoch = self.data_.ta.stoch()
        # RSI
        rsi = self.data_.ta.rsi(length=14)
        rsi_40 = self.data_.ta.rsi(length=40)

        # MACD
        macd = self.data_.ta.macd(fast=14, slow=28)

        # Volatility
        # ATR
        atr = self.data_.ta.atr()

        # TREND
        # ADX
        adx = self.data_.ta.adx()

        # Volume
        # CMF
        cmf = self.data_.ta.cmf()

        # Statistics
        # KURT
        kurt = self.data_.ta.kurtosis()

        self.data_ = pd.concat(
            [self.data_, stoch, rsi, rsi_40, macd, atr, adx, cmf, kurt], axis=1)
        self.data_['last_price'] = self.data_['close'].shift(1)
        self.data_.dropna(inplace=True)
        rl_logger.info(self.data_)

        self.data = self.data_[self.features].copy()
        self.data = (self.data - self.data.mean()) / self.data.std()

        self.data['r'] = np.log(self.data['close'] /
                                self.data['close'].shift(1))

        self.data_['r'] = self.data['r']

        # Create a new column 'd' in the dataframe to store the "ideal" action
        self.data['d'] = compute_action(self.data_)

        self.data['action'] = 0
        rl_logger.info("Features created.")

    def get_feature_names(self):
        return self.data.columns.tolist()

    def _get_state(self):
        state = self.data[self.features].iloc[self.bar -
                                              self.look_back:self.bar].values

        return np.array(state, dtype=np.float32)

    def reset(self):
        rl_logger.info("RESET:Environment reset.")
        self.data['action'] = 0
        self.total_reward = 0
        self.accuracy = 0
        self.bar = self.look_back
        state = self.data[self.features].iloc[self.bar -
                                              self.look_back:self.bar].values
        rl_logger.info(f"RESET:the reset state is {state}")
        state = np.array(state, dtype=np.float32)
        return state

    def step(self, action):
        rl_logger.info(f"STEP: Action taken: {action}")

        current_price = self.data['close'].iloc[self.bar]
        correct = action == self.data['d'].iloc[self.bar] and action == 1 or action == -1

        is_low_volatility, is_going_up_hold, is_going_down_hold = self.check_conditions(
            current_price)
        reward = self.compute_reward(
            action, is_low_volatility, is_going_up_hold, is_going_down_hold, correct)

        if action in [1, -1] and reward in [1]:
            self.last_action = action  # Update the last action

        self.last_price = current_price  # Update the last price

        self.data['action'].iloc[self.bar] = action
        self.total_reward += reward
        self.total_reward = round(self.total_reward, 2)
        self.bar += 1
        self.accuracy = round(self.total_reward /
                              (self.bar - self.look_back), 2)
        self.last_accuracy = self.accuracy
        done = False
        rl_logger.info(
            f"STEP: Total reward is {self.total_reward} | Accuracy is {self.accuracy}")
        rl_logger.info(f'STEP: Last action taken {self.last_action}')

        if self.is_episode_done():  # Use custom episode termination condition
            rl_logger.info("STEP: Episode done.")
            done = True

        elif reward in [1, 0.8, 0.5]:
            rl_logger.info("STEP: Action was correct.")
            if self.accuracy == self.last_accuracy:
                self.wait += 1
        elif self.accuracy >= 1.0:
            self.high_acc_counter += 1  # Increment the counter when accuracy is 1.0
        else:
            self.high_acc_counter = 0
            rl_logger.info("STEP: Action was False.")

        state = self._get_state()
        info = {}
        rl_logger.info(
            f"STEP: Reward: {reward} Info: {info}")
        rl_logger.info(
            f"STEP: ===========================================")

        return state, reward, info, done

    def is_episode_done(self):
        # Define your custom episode termination condition for testing
        if self.total_reward <= -50.0:
            rl_logger.warning(
                "STEP: Early stopping due to less rewards.")
            return True
        elif self.bar >= len(self.data) - 1:
            rl_logger.warning(
                "STEP: Early stopping due to no more bars.")
            return True

        elif self.wait >= self.patience:
            rl_logger.warning(
                "STEP: Early stopping due to lack of improvement.")
            self.wait = 0
            return True
        elif self.high_acc_counter >= self.high_acc_threshold:  # Check if counter has reached the threshold
            rl_logger.warning(
                f"STEP: Stopping early due to sustained high accuracy: {self.accuracy}")
            return True
        elif (self.accuracy < self.min_accuracy and self.bar > self.look_back + 50):
            self.wait = 0
            rl_logger.warning(
                f"STEP: Stopping early due to low accuracy: {self.accuracy}")
            return True
        return False

    def check_conditions(self, current_price):
        # All your market condition checks can be done here
        # Return all the boolean flags

        if self.last_price == None:
            self.last_price = current_price
        if self.last_action == None:
            self.last_action = 0

        is_low_volatility = (
            (float(self.data_['ADX_14'].iloc[self.bar]) < 25) &
            (float(self.data_['ADX_14'].iloc[self.bar]) < 20) &
            (self.data_['RSI_14'].iloc[self.bar] > 45) &
            (self.data_['RSI_14'].iloc[self.bar] < 55)
        )

        is_going_up_hold = (
            (self.last_action == 1) &
            (current_price > self.last_price) &
            (current_price > self.data_['ema_200'].iloc[self.bar]) &
            (self.data_['RSI_40'].iloc[self.bar] > 50)
        )
        # (self.data['RSI_14'].iloc[self.bar] > self.data['RSI_40'].iloc[self.bar])
        is_going_down_hold = (
            (self.last_action == -1) &
            (current_price < self.last_price) &
            (current_price < self.data_['ema_200'].iloc[self.bar]) &
            (self.data_['RSI_40'].iloc[self.bar] < 50)
        )

        return is_low_volatility, is_going_up_hold, is_going_down_hold

    def compute_reward(self, action, is_low_volatility, is_going_up_hold, is_going_down_hold, correct):
        # Compute and return the reward based on the current action and market conditions
        if correct:
            reward = 1
        elif action == 0 and is_low_volatility:  # If the agent chooses to hold in a low-volatility market
            reward = 0.5  # Give some reward, less than the reward for a correct action
        elif action == 0 and is_going_up_hold:
            reward = 0.8
        elif action == 0 and is_going_down_hold:
            reward = 0.8
        else:
            reward = -0.5  # Incorrect action

        return reward


def compute_action(df):
    # Verify that DataFrame is not empty
    if df.empty:
        return pd.Series()

    # Conditions for low volatility, going up and going down
    is_low_volatility = (df['ADX_14'] < 25) & (df['ADX_14'] < 20) & (
        df['RSI_14'] > 45) & (df['RSI_14'] < 55)
    is_going_up_hold = (df['close'] > df['last_price']) & (
        df['close'] > df['ema_200']) & (df['RSI_40'] > 50)
    is_going_down_hold = (df['close'] < df['last_price']) & (
        df['close'] < df['ema_200']) & (df['RSI_40'] < 50)

    rl_logger.info(f"Length of DataFrame: {len(df)}")
    rl_logger.info(f"Length of is_low_volatility: {len(is_low_volatility)}")
    rl_logger.info(f"Length of is_going_up_hold: {len(is_going_up_hold)}")
    rl_logger.info(f"Length of is_going_down_hold: {len(is_going_down_hold)}")

    # Create an array to store actions
    actions = np.zeros(len(df))

    # Populate the actions array based on the conditions
    actions[is_low_volatility] = 0
    actions[is_going_up_hold] = 0
    actions[is_going_down_hold] = 0

    mask = ~(is_low_volatility | is_going_up_hold | is_going_down_hold)
    rl_logger.info(f"Length of mask: {len(mask)}")

    actions[mask] = np.where(df[mask]['r'] > 0, 1,
                             np.where(df[mask]['r'] < 0, -1, 0))

    return pd.Series(actions, index=df.index)
