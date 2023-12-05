
from model.reinforcement.rl_env.env_uttils import TradingEnvironment
import unittest


class TestTradingEnvironment(unittest.TestCase):

    def setUp(self):
        # Initialize TradingEnvironment with some test data
        self.env = TradingEnvironment(
            initial_balance=10000,
            leverage=10,
            transaction_costs=0.01,
            trade_limit=100,
            drawdown_threshold=0.2,
            symbol="TEST",
            trading_mode='futures'
        )
    # ENV ------------------------------------------------------

    def test_balance_limit(self):
        # Test balance limit check
        with self.assertRaises(ValueError):
            self.env.check_balance_limit(20000)  # Exceeds balance

    def test_drawdown_calculation(self):
        # Test drawdown calculation
        self.env.max_drawdown = 0.3
        penalty = self.env.calculate_drawdown_penalty()
        print(penalty)
        self.assertLessEqual(penalty, 0)

    def test_trade_limit_penalty(self):
        # Test if trade limit penalty is applied correctly
        for _ in range(105):  # Exceed the trade limit
            self.env.spot_trading(2, 100)
            self.env.spot_trading(4, 100)
        penalty = self.env.calculate_trading_penalty()
        self.assertTrue(penalty < 0)

    def test_invalid_trading_mode(self):
        # Test handling of invalid trading mode
        original_mode = self.env.trading_mode
        self.env.trading_mode = 'invalid_mode'  # Set an invalid trading mode

        with self.assertRaises(ValueError):
            # Attempt to execute trade with invalid mode
            self.env.execute_trade(2, 100)

        self.env.trading_mode = original_mode  # Reset to original mode after test

    def test_drawdown_behavior(self):
        # Test behavior when drawdown limit is exceeded
        self.env.max_drawdown = 0.05  # 5% drawdown limit
        self.env.portfolio_balance -= 600  # induce a drawdown
        drawdown = 1 - (self.env.portfolio_balance / self.env.initial_balance)
        self.assertTrue(drawdown > self.env.max_drawdown)
        penalty = self.env.calculate_drawdown_penalty()
        self.assertTrue(penalty < 0)

    # FUTURE -----------------------------------------------------------

    def test_initial_setup(self):
        # Test the initial setup of the trading environment
        self.assertEqual(self.env.initial_balance, 10000)
        self.assertEqual(self.env.portfolio_balance, 10000)
        self.assertEqual(self.env.leverage, 10)
        self.assertEqual(self.env.trade_limit, 100)

    def test_buy_and_sell_futures(self):
        # Test buying and selling futures
        current_price = 100
        self.env.future_trading(2, current_price)  # Buy action
        self.assertEqual(self.env.open_positions['long'], 1)

        self.env.future_trading(4, current_price + 20)  # sell back action
        self.assertEqual(self.env.open_positions['long'], 0)
        self.assertFalse(self.env.portfolio_balance > 10000)

    def test_invalid_price(self):
        # Test handling of invalid price
        with self.assertRaises(ValueError):
            self.env.future_trading(2, -100)  # Invalid price

    def test_short_position_futures(self):
        # Test handling short position in futures trading
        current_price = 200
        self.env.future_trading(0, current_price)  # Short action
        self.assertEqual(self.env.open_positions['short'], 1)

        self.env.future_trading(3, current_price - 10)  # Buy back action
        self.assertEqual(self.env.open_positions['short'], 0)

        # Adjust the expected balance calculation for transaction costs
        # Assuming no transaction costs for simplicity
        expected_profit = 10 * self.env.leverage
        expected_balance = self.env.initial_balance + \
            expected_profit - self.transaction_costs
        # Allowing a small margin for rounding errors
        self.assertAlmostEqual(self.env.portfolio_balance,
                               expected_balance, delta=1)

    def test_leverage_impact(self):
        # Test the impact of leverage on PnL
        current_price = 150
        self.env.future_trading(2, current_price)  # Buy action
        self.env.future_trading(4, current_price + 50)  # Sell back action
        expected_pnl = 50 * self.env.leverage
        actual_pnl = self.env.portfolio_balance - self.env.initial_balance
        # Allowing a small margin for rounding errors
        self.assertAlmostEqual(actual_pnl, expected_pnl, delta=1)

    def test_portfolio_balance_update(self):
        # Test if the portfolio balance updates correctly after a trade
        initial_balance = self.env.portfolio_balance
        self.env.future_trading(2, 100)  # Buy action
        self.env.future_trading(4, 110)  # Sell back action

        # Expected profit calculation considering leverage and transaction costs
        expected_profit = (110 - 100) * self.position_size * \
            self.env.leverage  # Simplified calculation
        expected_balance = initial_balance + expected_profit - self.transaction_costs

        self.assertTrue(self.env.portfolio_balance > initial_balance,
                        msg=f"Actual: {self.env.portfolio_balance}, Expected: {expected_balance}")

    def test_stress_with_random_actions(self):
        # Random actions to stress test the environment
        import random
        for _ in range(1000):
            action = random.choice([0, 1, 2, 3, 4])
            price = random.uniform(50, 150)
            self.env.future_trading(action, price)
        # You can assert certain conditions after this random sequence

    def test_position_sizes_and_entry_prices(self):
        # Testing updates to position sizes and entry prices
        initial_price = 100
        self.env.future_trading(2, initial_price)  # Buy at initial price
        second_price = 110
        self.env.future_trading(2, second_price)  # Buy at higher price
        expected_avg_price = (initial_price + second_price) / 2
        self.assertEqual(self.env.entry_prices['long'], expected_avg_price)
        self.assertTrue(self.env.position_size_l > 0)

    def test_proper_closure_of_positions(self):
        # Ensuring positions are closed correctly
        self.env.future_trading(2, 100)  # Open long position
        self.env.future_trading(4, 110)  # Close long position
        self.assertEqual(self.env.open_positions['long'], 0)
        self.assertEqual(self.env.entry_prices['long'], 0)

    def test_leverage_effect(self):
        # Testing the impact of leverage on trade outcomes
        initial_balance = self.env.portfolio_balance
        self.env.future_trading(2, 100)  # Buy action
        self.env.future_trading(4, 105)  # Sell action
        expected_balance_change = (105 - 100) * self.env.leverage
        self.assertAlmostEqual(self.env.portfolio_balance,
                               initial_balance + expected_balance_change)
    # SPOT ------------------------------------------------------------------------

    def test_buy_and_sell_spot(self):
        # Test buying and selling in spot trading
        current_price = 50
        self.env.trading_mode = 'spot'
        self.env.spot_trading(2, current_price)  # Buy action
        self.assertGreater(self.env.stocks_held, 0)

        self.env.spot_trading(4, current_price + 5)  # Sell back action
        self.assertEqual(self.env.stocks_held, 0)
        self.assertTrue(self.env.portfolio_balance > 10000)

    def test_transaction_costs(self):
        # Test if transaction costs are being applied
        current_price = 100
        self.env.spot_trading(2, current_price)  # Buy action
        self.env.spot_trading(4, current_price)  # Sell action
        expected_balance_after_costs = self.env.initial_balance - \
            (2 * (current_price * self.env.pos_to_trade * self.env.transaction_costs))
        self.assertAlmostEqual(self.env.portfolio_balance,
                               expected_balance_after_costs, places=2)

    def test_no_trade_on_hold(self):
        # Test that no trade occurs when the hold action is chosen
        initial_balance = self.env.portfolio_balance
        self.env.spot_trading(1, 100)  # Hold action
        self.assertEqual(self.env.portfolio_balance, initial_balance)
        self.assertEqual(self.env.open_positions['long'], 0)
        self.assertEqual(self.env.open_positions['short'], 0)

    def test_price_validation(self):
        # Test if price validation is working correctly
        with self.assertRaises(ValueError):
            self.env.spot_trading(2, 0)  # Invalid zero price
        with self.assertRaises(ValueError):
            self.env.spot_trading(2, -10)  # Invalid negative price

    def test_max_trade_limit(self):
        # Test the enforcement of the maximum trade limit
        initial_price = 10000
        for _ in range(self.env.trade_limit + 1):
            try:
                self.env.spot_trading(2, initial_price)
                self.env.spot_trading(0, initial_price)
            except ValueError:
                # Ignore balance errors to focus on trade limit
                pass

        print(f'trade count {self.env.trade_count}')
        self.assertGreaterEqual(self.env.trade_count, 26)
        penalty = self.env.calculate_trading_penalty()
        self.assertLessEqual(penalty, 0)

    def test_pnl_calculation_accuracy(self):
        # Test accuracy of PnL calculations
        current_price = 100
        self.env.spot_trading(2, current_price)  # Buy action
        new_price = 110
        self.env.spot_trading(4, new_price)  # Sell action
        expected_pnl = (new_price - current_price) * self.env.stocks_held
        actual_pnl = self.env.shortterm_pnl()
        self.assertAlmostEqual(actual_pnl, expected_pnl /
                               self.env.initial_balance * 100, places=2)

    def test_reset_functionality(self):
        # Test if reset function resets the environment correctly
        self.env.spot_trading(2, 100)  # Some trading activity
        self.env.reset()
        self.assertEqual(self.env.portfolio_balance, self.env.initial_balance)
        self.assertEqual(self.env.trade_count, 0)
        self.assertEqual(self.env.open_positions['long'], 0)
        self.assertEqual(self.env.open_positions['short'], 0)

    def test_invalid_action_handling(self):
        # Test handling of invalid trading action
        with self.assertRaises(ValueError):
            self.env.spot_trading(5, 100)  # Non-existent action

    def test_multiple_concurrent_positions(self):
        # Opening multiple positions and checking the correct handling
        self.env.future_trading(2, 100)  # Open long position
        self.env.future_trading(0, 110)  # Open short position
        self.assertEqual(self.env.open_positions['long'], 1)
        self.assertEqual(self.env.open_positions['short'], 1)

    def test_invalid_transactions_exceeding_balance(self):
        # Attempting a trade that exceeds the available balance
        high_amount = self.env.initial_balance * 100  # Excessively high amount
        with self.assertRaises(ValueError):
            self.env.spot_trading(2, high_amount)


    # Add more tests for
if __name__ == '__main__':
    unittest.main()
