from datetime import datetime

import gymnasium as gym
import mplfinance as mpf
import numpy as np
import pandas as pd
from gymnasium import spaces

from model.reinforcement.rl_util import generate_random_candlestick_data


class TradingEnvironment(gym.Env):
    def __init__(self, candlestick_data, initial_balance=1000, max_position=1):
        super(TradingEnvironment, self).__init__()

        self.candlestick_data = candlestick_data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.current_position = 0
        self.max_position = max_position

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Buy, sell, Hold
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(
            5,), dtype=np.float32)  # open, high, low, close, volume

        # Variables to store data for the returns plot
        self.balances_history = []
        self.prices_history = []

    def _get_info(self):
        return {
            'balance': self.balance,
            'current_position': self.current_position,
            'current_price': self.candlestick_data[self.current_step][3],
            'current_total_worth': self.balance + self.current_position * self.candlestick_data[self.current_step][3]
        }

    def reset(self, seed=None):
        self.balance = self.initial_balance
        self.current_step = 0
        self.current_position = 0
        self.balances_history = []
        self.prices_history = []

        info = self._get_info()
        return self._next_observation(), info

    def _next_observation(self):
        return self.candlestick_data[self.current_step]

    def step(self, action):
        executed_action = action
        current_price = self.candlestick_data[self.current_step][3]

        if executed_action == 0:  # Buy
            if self.balance >= current_price and self.current_position < self.max_position:
                self.current_position += 1
                self.balance -= current_price
        elif executed_action == 1:  # Sell
            if self.current_position > 0:
                self.current_position -= 1
                self.balance += current_price

        self.current_step += 1
        done = self.current_step == len(self.candlestick_data) - 1
        terminated = self.balance <= 0

        # Save balance and price for returns plot
        self.balances_history.append(self.balance)
        self.prices_history.append(current_price)

        reward = self.balance - self.initial_balance
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
                                        'open', 'high', 'low', 'close', 'volume'], index=pd.date_range(start_date, periods=len(self.candlestick_data), freq='D'))

        # Build the cumulative returns plot and add it to the candlestick chart
        apdict = mpf.make_addplot(
            cumulative_returns, title='Cumulative returns', color='lightblue', secondary_y=False)
        mpf.plot(candlestick_data, type='candle', mav=(3, 6, 9),
                 volume=True, show_nontrading=True, style='yahoo', addplot=apdict)


if __name__ == '__main__':
    data, start_data = generate_random_candlestick_data(100)
    env = TradingEnvironment(data, 100)
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # random agent action
        obs, reward, terminated, done, info = env.step(action)
        print(
            f"action: {action}, Balance: {info['balance']}, shares: {info['current_position']}, Price:{info['current_price']}, Total Worth: {info['current_total_worth']}")

    env.plot_returns()
