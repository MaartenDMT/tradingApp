import numpy as np
import pandas as pd
import os
import pickle
import gym
from gym import spaces
from datetime import datetime
import mplfinance as mpf
import util.loggers as loggers
from util.utils import convert_df, tradex_features

logger = loggers.setup_loggers()
env_logger = logger['env']


class TradingEnvironment(gym.Env):
    def __init__(self, candlestick_data, initial_balance=1000, max_position=1, penalty_threshold=10, penalty_amount=0.10, transaction_cost=0.04):
        super(TradingEnvironment, self).__init__()

        self.logger = env_logger
        if candlestick_data is None:
            self.candlestick_data = tradex_features('custom', pd.DataFrame(
                self._get_data(), columns=['open', 'high', 'low', 'close', 'volume']))
        else:
            self.candlestick_data = tradex_features('custom', pd.DataFrame(
                candlestick_data, columns=['open', 'high', 'low', 'close', 'volume']))

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.current_position = 0
        self.max_position = max_position
        self.candlestick_data.dropna(inplace=True)

        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([3, 5])
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(
            self.candlestick_data.columns),), dtype=np.float32)

        # Store data for the returns plot
        self.balances_history = []
        self.prices_history = []

        # Penalty
        self.penalty_threshold = penalty_threshold
        self.penalty_amount = penalty_amount
        self.buy_count = 0
        self.sell_count = 0

        # Transaction cost
        self.transaction_cost = transaction_cost

    def _get_info(self):
        return {
            'balance': self.balance,
            'current_position': self.current_position,
            'current_price': self.candlestick_data.iloc[self.current_step]['close'],
            'current_total_worth': self.balance + self.current_position * self.candlestick_data.iloc[self.current_step]['close']
        }

    def reset(self, seed=None):
        self.balance = self.initial_balance
        self.current_step = 0
        self.current_position = 0
        self.balances_history = []
        self.prices_history = []
        self.buy_count = 0
        self.sell_count = 0

        info = self._get_info()
        next_observation = self._next_observation()
        return next_observation, info

    def _next_observation(self):
        return self.candlestick_data.iloc[self.current_step].values

    def step(self, action):
        # Check if action is a numpy array with shape (1, 2)
        if isinstance(action, np.ndarray) and action.shape == (1, 2):
            action_type = action[0, 0]
            percentage = action[0, 1]
        # Check if action is a numpy array with shape (2,)
        elif isinstance(action, np.ndarray) and action.shape == (2,):
            action_type, percentage = action  # Directly unpack the numpy array
        # Check if action is a tuple with exactly two elements
        elif isinstance(action, tuple) and len(action) == 2:
            action_type, percentage = action
        else:
            raise ValueError(
                f"Invalid action format: {action}. Expected a numpy array with shape (1, 2), (2,), or a tuple with two elements.")

        current_price = self.candlestick_data.iloc[self.current_step]['close']
        amount_to_invest = self.balance * (0.01 * (percentage + 1))

        if action_type == 0:  # Buy
            if self.balance >= amount_to_invest * (1 + self.transaction_cost) and self.current_position < self.max_position:
                self.current_position += amount_to_invest / current_price
                self.balance -= amount_to_invest * (1 + self.transaction_cost)
                self.buy_count += 1
        elif action_type == 1:  # Sell
            if self.current_position > 0:
                self.balance += amount_to_invest * (1 - self.transaction_cost)
                self.current_position -= amount_to_invest / current_price
                self.sell_count += 1

        self.current_step += 1
        terminated = self.balance <= 100
        done = self.current_step == len(
            self.candlestick_data) - 1 or terminated

        # Save balance and price for returns plot
        self.balances_history.append(self.balance)
        self.prices_history.append(current_price)

        reward = self.balance - self.initial_balance

        # Apply penalties if thresholds are exceeded
        if self.buy_count > self.penalty_threshold:
            reward -= reward * self.penalty_amount
        if self.sell_count > self.penalty_threshold:
            reward -= reward * self.penalty_amount

         # Additional rewards for good actions
        if action_type == 0 and self.current_position > 0:  # Reward for buying at a good price
            next_price = self.candlestick_data.iloc[self.current_step + 1]['close'] if self.current_step + 1 < len(
                self.candlestick_data) else current_price
            if next_price > current_price:
                reward += (next_price - current_price) * self.current_position

        if action_type == 1 and self.current_position < self.max_position:  # Reward for selling at a good price
            next_price = self.candlestick_data.iloc[self.current_step + 1]['close'] if self.current_step + 1 < len(
                self.candlestick_data) else current_price
            if next_price < current_price:
                reward += (current_price - next_price) * self.current_position

        info = self._get_info()
        observation = self._next_observation()
        return observation, reward, terminated, done, info

    def plot_returns(self):
        # Calculate returns
        returns = np.diff(self.balances_history, prepend=0) / \
            np.array(self.prices_history)
        cumulative_returns = np.cumsum(returns)

        # Add 0 to the first element of cumulative returns
        cumulative_returns = np.insert(cumulative_returns, 0, 0)

        # Build candlestick chart using mplfinance
        start_date = datetime.now()
        candlestick_data = pd.DataFrame(self.candlestick_data, columns=[
                                        'open', 'high', 'low', 'close', 'volume'], index=pd.date_range(start_date, periods=len(self.candlestick_data), freq='30min'))

        # Build the cumulative returns plot and add it to the candlestick chart
        apdict = mpf.make_addplot(
            cumulative_returns, title='Cumulative returns', color='lightblue', secondary_y=False)
        mpf.plot(candlestick_data, type='candle', mav=(3, 6, 9),
                 volume=True, show_nontrading=True, style='yahoo', addplot=apdict)

    def _get_data(self):
        pickle_file_name = 'data/pickle/all/30m_data_all.pkl'

        if not os.path.exists(pickle_file_name):
            self.logger.error('No data has been written')
            return pd.DataFrame()  # Return an empty DataFrame instead of None for consistency

        with open(pickle_file_name, 'rb') as f:
            data_ = pickle.load(f)

        if data_.empty:
            self.logger.error("Loaded data is empty.")
            return pd.DataFrame()

        data = convert_df(data_)

        if data.empty or data.isnull().values.any():
            self.logger.error(
                "Converted data is empty or contains NaN values.")
            return pd.DataFrame()

        percentage_to_keep = 15 / 100.0
        rows_to_keep = int(len(data) * percentage_to_keep)
        data = data.head(rows_to_keep)

        self.logger.info(f'Dataframe shape: {data.shape}')
        return data
