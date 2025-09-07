
import unittest

from model.reinforcement.environments.trading_engine import TradingEngine


class TestTradingEngine(unittest.TestCase):

    def setUp(self):
        # This sets up a TradingEngine instance before each test
        self.env = TradingEngine(initial_balance=10000, leverage=10, transaction_costs=0.01,
                                trading_mode='futures')

    def test_initial_conditions(self):
        """Test if the environment initializes correctly."""
        self.assertEqual(self.env.initial_balance, 10000)
        self.assertEqual(self.env.leverage, 10)
        self.assertEqual(self.env.transaction_costs, 0.01)
        # Add more assertions as necessary

    def test_invalid_action(self):
        """Test handling of invalid actions."""
        with self.assertRaises(ValueError):
            self.env.execute_trade(5, 100)  # Invalid action

    def test_invalid_price(self):
        """Test handling of invalid price values."""
        with self.assertRaises(ValueError):
            self.env.execute_trade(1, -50)  # Negative price is invalid

    def test_buy_action(self):
        """Test the buy action in futures trading."""
        trade_result, portfolio_balance = self.env.future_trading(2, 100)
        # Balance should change
        self.assertNotEqual(portfolio_balance, self.env.initial_balance)
        # Long positions should increase
        self.assertGreater(self.env.open_positions['long'], 0)

    def test_sell_action_no_stock(self):
        """Test sell action when no stocks are held."""
        with self.assertRaises(ValueError):
            self.env.spot_trading(0, 100)  # Selling when no stocks are held

    def test_drawdown_calculation(self):
        """Test drawdown calculation after a series of trades."""
        # Perform a series of trades to induce a drawdown
        # Assert that the drawdown is calculated correctly

    def test_reset_functionality(self):
        """Test if the reset method works correctly."""
        self.env.future_trading(2, 100)
        self.env.reset()
        self.assertEqual(self.env.portfolio_balance, self.env.initial_balance)
        self.assertEqual(self.env.stocks_held, 0.0)

    # Add more tests to cover each method and edge cases

    def test_transaction_cost_application(self):
        """Test if transaction costs are applied correctly."""
        initial_balance = self.env.portfolio_balance
        self.env.future_trading(2, 100)  # Assume this is a buy action
        self.env.future_trading(4, 110)  # Assume this is a sell back action
        expected_cost = (self.env.pos_to_trade * initial_balance *
                         self.env.transaction_costs) * 2  # Buy and Sell
        self.assertAlmostEqual(self.env.portfolio_balance,
                               initial_balance - expected_cost, places=2)

    # def test_leverage_impact(self):
    #     """Test the impact of leverage on profits and losses."""
    #     self.env.future_trading(2, 100)  # Buy at 100
    #     self.env.future_trading(4, 110)  # Sell at 110
    #     profit_without_leverage = (
    #         110 - 100) * (self.env.pos_to_trade * self.env.initial_balance)
    #     profit_with_leverage = profit_without_leverage * self.env.leverage
    #     self.assertAlmostEqual(self.env.portfolio_balance -
    #                            self.env.initial_balance, profit_with_leverage, delta=0.01)

    # def test_trade_limit_enforcement(self):
    #     """Test enforcement of trade limits."""
    #     for _ in range(self.env.trade_limit + 10):  # Exceed trade limit
    #         self.env.future_trading(2, 100)  # Buy
    #         self.env.future_trading(4, 100)  # Sell
    #     self.assertGreater(self.env.trade_count, self.env.trade_limit)
    #     penalty = self.env.calculate_trading_penalty()
    #     self.assertLessEqual(penalty, 0)

    # def test_drawdown_threshold_enforcement(self):
    #     """Test if drawdown threshold is enforced."""
    #     # Perform trades to cause a significant drawdown
    #     # Assert that the drawdown exceeds the threshold and check if penalty is applied

    # def test_position_tracking_accuracy(self):
    #     """Test accuracy of position tracking."""
    #     self.env.future_trading(2, 100)  # Buy
    #     self.assertEqual(self.env.open_positions['long'], 1)
    #     self.env.future_trading(4, 110)  # Sell
    #     self.assertEqual(self.env.open_positions['long'], 0)

    # def test_pnl_calculation(self):
    #     """Test accuracy of PnL calculation."""
    #     self.env.future_trading(2, 100)  # Buy
    #     self.env.future_trading(4, 110)  # Sell
    #     expected_pnl = (110 - 100) * self.env.leverage * \
    #         (self.env.pos_to_trade * self.env.initial_balance)
    #     self.assertAlmostEqual(self.env.shortterm_pnl(
    #     ), expected_pnl / self.env.initial_balance, delta=0.01)


if __name__ == '__main__':
    unittest.main()
