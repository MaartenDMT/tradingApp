"""
Unit tests for AutoBot functionality.
"""
import asyncio
import os
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.machinelearning.autobot import (AutoBot, ConnectionManager,
                                           RiskManager)


class TestRiskManager(unittest.TestCase):
    """Test cases for RiskManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.risk_manager = RiskManager(
            max_daily_loss=0.05,
            max_position_size=0.1,
            max_open_positions=3
        )

    def test_initialization(self):
        """Test RiskManager initialization."""
        self.assertEqual(self.risk_manager.max_daily_loss, 0.05)
        self.assertEqual(self.risk_manager.max_position_size, 0.1)
        self.assertEqual(self.risk_manager.max_open_positions, 3)
        self.assertEqual(self.risk_manager.daily_pnl, 0.0)
        self.assertIsNone(self.risk_manager.session_start_balance)

    def test_can_open_position_new_session(self):
        """Test position opening in new session."""
        can_open, reason = self.risk_manager.can_open_position(
            current_balance=1000,
            position_size=50,
            open_positions_count=1
        )
        self.assertTrue(can_open)
        self.assertEqual(reason, "OK")

    def test_can_open_position_max_loss_exceeded(self):
        """Test position blocking when max daily loss exceeded."""
        self.risk_manager.session_start_balance = 1000
        self.risk_manager.daily_pnl = -60  # 6% loss

        can_open, reason = self.risk_manager.can_open_position(
            current_balance=940,
            position_size=50,
            open_positions_count=1
        )
        self.assertFalse(can_open)
        self.assertEqual(reason, "Daily loss limit exceeded")

    def test_can_open_position_size_too_large(self):
        """Test position blocking when position size too large."""
        can_open, reason = self.risk_manager.can_open_position(
            current_balance=1000,
            position_size=150,  # 15% of balance
            open_positions_count=1
        )
        self.assertFalse(can_open)
        self.assertEqual(reason, "Position size too large")

    def test_can_open_position_max_positions_reached(self):
        """Test position blocking when max positions reached."""
        can_open, reason = self.risk_manager.can_open_position(
            current_balance=1000,
            position_size=50,
            open_positions_count=3  # At max limit
        )
        self.assertFalse(can_open)
        self.assertEqual(reason, "Maximum open positions reached")

    def test_update_pnl(self):
        """Test P&L update functionality."""
        initial_pnl = self.risk_manager.daily_pnl
        self.risk_manager.update_pnl(25.5)
        self.assertEqual(self.risk_manager.daily_pnl, initial_pnl + 25.5)

        self.risk_manager.update_pnl(-10.0)
        self.assertEqual(self.risk_manager.daily_pnl, initial_pnl + 25.5 - 10.0)


class TestConnectionManager(unittest.TestCase):
    """Test cases for ConnectionManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.ws_url = "wss://test.example.com"
        self.mock_callback = Mock()
        self.connection_manager = ConnectionManager(
            ws_url=self.ws_url,
            on_message_callback=self.mock_callback,
            max_retries=3,
            retry_delay=0.1  # Short delay for testing
        )

    def test_initialization(self):
        """Test ConnectionManager initialization."""
        self.assertEqual(self.connection_manager.ws_url, self.ws_url)
        self.assertEqual(self.connection_manager.on_message_callback, self.mock_callback)
        self.assertEqual(self.connection_manager.max_retries, 3)
        self.assertEqual(self.connection_manager.retry_delay, 0.1)
        self.assertEqual(self.connection_manager.connection_attempts, 0)
        self.assertFalse(self.connection_manager.is_connected)

    def test_on_open(self):
        """Test WebSocket open handler."""
        mock_ws = Mock()
        self.connection_manager._on_open(mock_ws)

        self.assertTrue(self.connection_manager.is_connected)
        self.assertEqual(self.connection_manager.connection_attempts, 0)

    def test_on_close(self):
        """Test WebSocket close handler."""
        mock_ws = Mock()
        self.connection_manager.is_connected = True

        self.connection_manager._on_close(mock_ws, 1000, "Normal closure")

        self.assertFalse(self.connection_manager.is_connected)

    def test_on_error(self):
        """Test WebSocket error handler."""
        mock_ws = Mock()
        self.connection_manager.is_connected = True

        self.connection_manager._on_error(mock_ws, "Connection error")

        self.assertFalse(self.connection_manager.is_connected)

    @patch('websocket.WebSocketApp')
    def test_reconnect_if_needed_already_connected(self, mock_websocket_app):
        """Test reconnect when already connected."""
        self.connection_manager.is_connected = True

        async def run_test():
            result = await self.connection_manager.reconnect_if_needed()
            self.assertTrue(result)
            mock_websocket_app.assert_not_called()

        asyncio.run(run_test())


class TestAutoBot(unittest.TestCase):
    """Test cases for AutoBot functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_exchange = Mock()
        self.mock_ml = Mock()
        self.mock_trade_x = Mock()

        # Create sample DataFrame
        self.sample_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [104, 105, 106],
            'volume': [1000, 1100, 1200]
        })

        self.autobot = AutoBot(
            exchange=self.mock_exchange,
            symbol="BTC/USDT",
            amount=0.01,
            stop_loss=2.0,
            take_profit=5.0,
            model=Mock(),
            timeframe="1h",
            ml=self.mock_ml,
            trade_x=self.mock_trade_x,
            df=self.sample_df
        )

    def test_initialization(self):
        """Test AutoBot initialization."""
        self.assertEqual(self.autobot.symbol, "BTC/USDT")
        self.assertEqual(self.autobot.amount, 0.01)
        self.assertEqual(self.autobot.stop_loss, 2.0)
        self.assertEqual(self.autobot.take_profit, 5.0)
        self.assertEqual(self.autobot.timeframe, "1h")
        self.assertFalse(self.autobot.auto_trade)
        self.assertIsInstance(self.autobot.risk_manager, RiskManager)
        self.assertEqual(self.autobot.trade_count, 0)
        self.assertEqual(self.autobot.successful_trades, 0)
        self.assertEqual(self.autobot.failed_trades, 0)

    def test_str_representation(self):
        """Test string representation of AutoBot."""
        expected = "Autobot-Mock: BTC/USDT-1h-Mock- Trades: 0 -> "
        actual = str(self.autobot)
        self.assertEqual(actual, expected)

    def test_getnormal_symbol(self):
        """Test symbol normalization."""
        result = self.autobot.getnormal_symbol()
        self.assertEqual(result, "btcusdt")

    def test_getnormal_count(self):
        """Test timeframe to seconds conversion."""
        test_cases = [
            ("1m", 60),
            ("5m", 300),
            ("30m", 1800),
            ("1h", 3600),
            ("3h", 10800)
        ]

        for timeframe, expected_seconds in test_cases:
            self.autobot.timeframe = timeframe
            result = self.autobot.getnormal_count()
            self.assertEqual(result, expected_seconds)

    def test_get_performance_metrics_no_trades(self):
        """Test performance metrics with no trades."""
        metrics = self.autobot.get_performance_metrics()
        self.assertEqual(metrics["message"], "No trades executed yet")

    def test_get_performance_metrics_with_trades(self):
        """Test performance metrics with some trades."""
        self.autobot.trade_count = 10
        self.autobot.successful_trades = 7
        self.autobot.failed_trades = 3

        metrics = self.autobot.get_performance_metrics()

        self.assertEqual(metrics["total_trades"], 10)
        self.assertEqual(metrics["successful_trades"], 7)
        self.assertEqual(metrics["failed_trades"], 3)
        self.assertEqual(metrics["success_rate"], "70.00%")
        self.assertEqual(metrics["open_positions"], 0)
        self.assertIn("risk_manager_stats", metrics)

    @patch('asyncio.create_task')
    def test_start_auto_trading_enable(self, mock_create_task):
        """Test enabling auto trading."""
        with patch.object(self.autobot, 'start_async') as mock_start_async:
            # Mock asyncio.get_event_loop to return a running loop
            mock_loop = Mock()
            mock_loop.is_running.return_value = True

            with patch('asyncio.get_event_loop', return_value=mock_loop):
                self.autobot.start_auto_trading(True)
                mock_create_task.assert_called_once()

    def test_get_model_type_classifier(self):
        """Test model type detection for classifier."""
        mock_model = Mock()
        mock_model.__str__ = Mock(return_value="RandomForestClassifier")

        result = self.autobot.get_model_type(mock_model)
        self.assertEqual(result, "classifier")

    def test_get_model_type_regression(self):
        """Test model type detection for regression."""
        mock_model = Mock()
        mock_model.__str__ = Mock(return_value="LinearRegression")

        result = self.autobot.get_model_type(mock_model)
        self.assertEqual(result, "regression")

    def test_get_model_type_unknown(self):
        """Test model type detection for unknown model."""
        mock_model = Mock()
        mock_model.__str__ = Mock(return_value="UnknownModel")

        result = self.autobot.get_model_type(mock_model)
        self.assertEqual(result, "UnknownModel")

    @patch('time.time')
    def test_execute_trade_async_buy_success(self, mock_time):
        """Test successful buy trade execution."""
        mock_time.return_value = 1234567890

        # Mock exchange responses
        self.mock_exchange.fetch_balance.return_value = {
            'total': {'USDT': 1000}
        }
        self.mock_exchange.fetch_ticker.return_value = {'last': 50000}
        self.mock_exchange.create_order.side_effect = [
            {'id': 'order123'},  # Main order
            {'id': 'stop123'}    # Stop loss order
        ]

        async def run_test():
            result = await self.autobot.execute_trade_async('buy', confidence_score=0.8)
            self.assertTrue(result)
            self.assertEqual(self.autobot.trade_count, 1)
            self.assertIn('order123', self.autobot.open_orders)

            order_info = self.autobot.open_orders['order123']
            self.assertEqual(order_info['side'], 'buy')
            self.assertEqual(order_info['confidence'], 0.8)

        asyncio.run(run_test())

    def test_execute_trade_async_risk_blocked(self):
        """Test trade execution blocked by risk management."""
        # Set up risk manager to block trades
        self.autobot.risk_manager.session_start_balance = 1000
        self.autobot.risk_manager.daily_pnl = -60  # Exceeds 5% limit

        self.mock_exchange.fetch_balance.return_value = {
            'total': {'USDT': 940}
        }
        self.mock_exchange.fetch_ticker.return_value = {'last': 50000}

        async def run_test():
            result = await self.autobot.execute_trade_async('buy', confidence_score=0.8)
            self.assertFalse(result)
            self.assertEqual(self.autobot.trade_count, 0)
            self.assertEqual(len(self.autobot.open_orders), 0)

        asyncio.run(run_test())


class TestAutoBotIntegration(unittest.TestCase):
    """Integration tests for AutoBot."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.mock_exchange = Mock()
        self.mock_ml = Mock()
        self.mock_trade_x = Mock()

        # Create more realistic sample data
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)

        self.sample_df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + np.random.rand(100) * 100,
            'low': prices - np.random.rand(100) * 100,
            'close': prices + np.random.randn(100) * 50,
            'volume': np.random.rand(100) * 1000 + 500
        })

        self.autobot = AutoBot(
            exchange=self.mock_exchange,
            symbol="BTC/USDT",
            amount=0.01,
            stop_loss=2.0,
            take_profit=5.0,
            model=Mock(),
            timeframe="1h",
            ml=self.mock_ml,
            trade_x=self.mock_trade_x,
            df=self.sample_df
        )

    def test_full_trading_cycle_classifier(self):
        """Test complete trading cycle for classifier model."""
        # Setup mock responses
        self.mock_ml.predict.return_value = 2  # Buy signal (2-1=1)
        self.mock_trade_x.run.return_value = {'signal': 'BUY'}
        self.mock_exchange.fetch_ticker.return_value = {'last': 50000}
        self.mock_exchange.fetch_balance.return_value = {'total': {'USDT': 1000}}
        self.mock_exchange.create_order.side_effect = [
            {'id': 'order123'},
            {'id': 'stop123'}
        ]

        # Mock model type detection
        self.autobot.model.__str__ = Mock(return_value="RandomForestClassifier")

        async def run_test():
            await self.autobot._run_auto_trading_classifier_async()

            # Verify trade was executed
            self.assertEqual(self.autobot.trade_count, 1)
            self.assertIn('order123', self.autobot.open_orders)

            # Verify exchange calls
            self.mock_exchange.fetch_ticker.assert_called_with("BTC/USDT")
            self.mock_exchange.create_order.assert_called()

        asyncio.run(run_test())

    def test_full_trading_cycle_regression(self):
        """Test complete trading cycle for regression model."""
        # Setup mock responses
        current_price = 50000
        predicted_price = 51000  # 2% higher

        self.mock_ml.predict.return_value = predicted_price
        self.mock_exchange.fetch_ticker.return_value = {'last': current_price}
        self.mock_exchange.fetch_balance.return_value = {'total': {'USDT': 1000}}
        self.mock_exchange.create_order.side_effect = [
            {'id': 'order123'},
            {'id': 'stop123'}
        ]

        # Mock model type detection
        self.autobot.model.__str__ = Mock(return_value="LinearRegression")

        async def run_test():
            await self.autobot._run_auto_trading_regression_async()

            # Verify trade was executed (should be a buy since predicted > current)
            self.assertEqual(self.autobot.trade_count, 1)
            self.assertIn('order123', self.autobot.open_orders)

        asyncio.run(run_test())


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestRiskManager))
    test_suite.addTest(unittest.makeSuite(TestConnectionManager))
    test_suite.addTest(unittest.makeSuite(TestAutoBot))
    test_suite.addTest(unittest.makeSuite(TestAutoBotIntegration))

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    if result.wasSuccessful():
        print(f"\n✅ All {result.testsRun} tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests")

        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")

        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
                print(f"- {test}: {traceback}")
