import unittest

import numpy as np
import pandas as pd

from model.reinforcement.env import Environment
from model.reinforcement.rl_env.env_uttils import (ObservationSpace,
                                                   TradingEnvironment)


class TestEnvironment(unittest.TestCase):

    def setUp(self):
        # Initialize the Environment object with test parameters
        self.test_env = Environment(
            symbol="BTCUSDT",
            features=["close", "open"],
            limit=10,
            time="30m",
            actions=5,
            min_acc=0.5
        )

    def test_get_data(self):
        data = self.test_env._get_data()
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        # Check for expected columns, data types, and no missing values
        # Other checks...

    def test_initialize_environment(self):
        # Test the _initialize_environment method
        self.test_env._initialize_environment()
        self.assertIsNotNone(self.test_env.tradingEnv)
        self.assertIsNotNone(self.test_env.original_data)
        self.assertIsInstance(self.test_env.tradingEnv, TradingEnvironment)
        self.assertIsInstance(self.test_env.original_data, pd.DataFrame)
        self.assertIn("close", self.test_env.original_data.columns)
        self.assertIn("open", self.test_env.original_data.columns)

    def test_set_initial_parameters(self):
        # Test the _set_initial_parameters method
        self.test_env._set_initial_parameters()
        self.assertEqual(self.test_env.last_price, None)
        self.assertEqual(self.test_env.bar, 0)
        self.assertIsNone(self.test_env.last_price)
        self.assertEqual(self.test_env.trade_limit, 5)
        self.assertEqual(self.test_env.threshold, 0.2)

    def test_setup_observation_space(self):
        # Test the _setup_observation_space method
        self.test_env._setup_observation_space()
        self.assertIsNotNone(self.test_env.observation_space)
        self.assertIsInstance(
            self.test_env.observation_space, ObservationSpace)
        self.assertEqual(
            self.test_env.observation_space.shape, (9,))

    def test_get_data(self):
        # Test the _get_data method
        data = self.test_env._get_data()
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)

    def test_create_features(self):
        # Test the _create_features method
        self.test_env._create_features()
        self.assertIsNotNone(self.test_env.env_data)
        self.assertIn("action", self.test_env.env_data.columns)
        self.assertFalse(self.test_env.env_data.isnull().values.any())

    def test_step_function(self):
        # Test the step function
        state, reward, info, done = self.test_env.step(2)
        self.assertIsNotNone(state)
        self.assertIsNotNone(reward)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(info, dict)
        self.assertIsInstance(done, bool)

    def test_reset_functionality(self):
        initial_state = self.test_env.reset()
        self.assertIsNotNone(initial_state)
        self.assertEqual(self.test_env.bar, self.test_env.look_back)
        # Check if the portfolio balance is reset
        self.assertEqual(self.test_env.portfolio_balance, 10_000)
        # Other checks...

    def test_execute_trade(self):
        initial_balance = self.test_env.portfolio_balance
        # Example action and price
        self.test_env.execute_trade(action=2, current_price=100)
        self.assertNotEqual(self.test_env.portfolio_balance, initial_balance)
        # Other checks...

    def test_get_state(self):
        state = self.test_env._get_state()
        self.assertIsNotNone(state)
        self.assertIsInstance(state, np.ndarray)
        # Check the state's shape or content
        # Other checks...

    def test_update_and_adjust_features(self):
        self.test_env.update_and_adjust_features()
        # Verify changes in features, if applicable
        # Other checks...

    def test_calculate_reward(self):
        action = 2  # Example action
        df_row_raw = self.test_env.original_data.iloc[0]
        reward = self.test_env.calculate_reward(action, df_row_raw)
        self.assertIsInstance(reward, (int, float))
        # Verify reward calculation logic
        # Other checks...

    def test_handling_different_actions(self):
        valid_actions = [0, 1, 2, 3, 4]
        for action in valid_actions:
            state, reward, info, done = self.test_env.step(action)
            self.assertIsNotNone(state)
            # Assert other conditions based on action
        with self.assertRaises(ValueError):
            self.test_env.step(5)  # Test with an invalid action

    def test_create_features(self):
        self.test_env._create_features()
        self.assertIn('r', self.test_env.env_data.columns)
        # Check for correct data types, absence of NaNs, etc.
        # Other checks...


if __name__ == '__main__':
    unittest.main()
