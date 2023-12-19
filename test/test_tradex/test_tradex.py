import os
import pickle
import unittest

import pandas as pd

import util.loggers as loggers
from model.features import Tradex_indicator
from util.utils import convert_df, load_config

logger = loggers.setup_loggers()
env_logger = logger['env']
rl_logger = logger['rl']


class TestTradexIndicator(unittest.TestCase):

    def setUp(self):
        # Setup for tests - instantiate the Tradex_indicator class with required parameters
        self.symbol = "TEST_SYMBOL"
        self.timeframe = "30m"
        self.config = load_config()
        for section in self.config.sections():
            rl_logger.info(f'Section: {section}')
            for key, value in self.config.items(section):
                rl_logger.info(f'{key} = {value}')

        self.tradex_indicator = Tradex_indicator(
            symbol=self.symbol, timeframe=self.timeframe, t=None, get_data=False, data=self._get_data())

    def _get_data(self):

        pickle_file_name = 'data/pickle/all/30m_data_all.pkl'
        if not os.path.exists(pickle_file_name):
            env_logger.error('No data has been written')
            return pd.DataFrame()  # Return an empty DataFrame instead of None for consistency

        with open(pickle_file_name, 'rb') as f:
            data_ = pickle.load(f)

        if data_.empty:
            env_logger.error("Loaded data is empty.")
            return pd.DataFrame()

        data = convert_df(data_)

        if data.empty or data.isnull().values.any():
            env_logger.error("Converted data is empty or contains NaN values.")
            return pd.DataFrame()

        percentage_to_keep = float(0.20) / 100.0
        rows_to_keep = int(len(data) * percentage_to_keep)
        data = data.head(rows_to_keep)
        rl_logger.info(data)

        rl_logger.info(f'Dataframe shape: {data.shape}')
        return data

    def test_run(self):
        # Test if run method properly initializes trend, screener, real_time, scanner
        self.tradex_indicator.run()
        self.assertIsNotNone(self.tradex_indicator.trend)
        self.assertIsNotNone(self.tradex_indicator.screener)
        self.assertIsNotNone(self.tradex_indicator.real_time)
        self.assertIsNotNone(self.tradex_indicator.scanner)

    # Add tests for Trend, Screener, Real_time, Scanner here
    # Each test should create an instance of the respective class and check if the methods work as expected

    def test_trend(self):
        # Example of testing Trend class
        trend = self.tradex_indicator.trend(
            self.tradex_indicator.data, self.tradex_indicator.tradex_logger)
        self.assertIsNotNone(trend.df_trend)
        # Additional assertions to validate the behavior of Trend

    # Similar tests for Screener, Real_time, and Scanner


if __name__ == '__main__':
    unittest.main()
