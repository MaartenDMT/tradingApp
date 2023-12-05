import unittest
from model.features import Tradex_indicator


class TestTradexIndicator(unittest.TestCase):

    def setUp(self):
        # Setup for tests - instantiate the Tradex_indicator class with required parameters
        self.symbol = "TEST_SYMBOL"
        self.timeframe = "1h"
        self.tradex_indicator = Tradex_indicator(
            symbol=self.symbol, timeframe=self.timeframe)

    def test_get_data(self):
        # Test if get_data method returns a DataFrame with expected columns
        data = self.tradex_indicator.get_data()
        expected_columns = ['date', 'open', 'high',
                            'low', 'close', 'volume', 'symbol']
        self.assertIsNotNone(data)
        self.assertTrue(
            all(column in data.columns for column in expected_columns))

    def test_changeTime(self):
        # Test changeTime method
        new_timeframe = '30m'
        self.tradex_indicator.changeTime(new_timeframe)
        # Assuming data is a time-series DataFrame
        self.assertEqual(
            self.tradex_indicator.data.index.freqstr, new_timeframe)

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
