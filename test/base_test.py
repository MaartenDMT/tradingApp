"""
Base Test Class for Trading Application

This module provides a base test class with common functionality
for all tests in the trading application.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class BaseTestCase(unittest.TestCase):
    """Base test case with common functionality for all tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures."""
        # Ensure test directories exist
        test_dirs = [
            'data/test',
            'data/test/logs',
            'data/test/cache',
            'data/test/ml',
            'data/test/rl'
        ]
        
        for dir_path in test_dirs:
            full_path = project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Set test environment variables
        os.environ['TESTING'] = 'true'
        os.environ['DEV_MODE'] = 'true'
        
        # Mock configuration for tests
        self.mock_config = Mock()
        self.mock_config.window_size = "800x600"
        self.mock_config.symbol = "BTC/USDT"
        self.mock_config.timeframes = ["1m", "5m", "15m"]
        self.mock_config.portfolio_balance = 10000.0
        
        # Mock logger for tests
        self.mock_logger = Mock()
        
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Clean up environment variables
        if 'TESTING' in os.environ:
            del os.environ['TESTING']
            
    def create_mock_dataframe(self, rows=100, columns=None):
        """Create a mock pandas DataFrame for testing.
        
        Args:
            rows: Number of rows
            columns: List of column names
            
        Returns:
            Mock DataFrame
        """
        import pandas as pd
        import numpy as np
        
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']
            
        data = {}
        for col in columns:
            if col == 'timestamp':
                data[col] = pd.date_range('2023-01-01', periods=rows, freq='1min')
            else:
                data[col] = np.random.random(rows) * 100000
            
        return pd.DataFrame(data)
        
    def create_mock_exchange(self):
        """Create a mock exchange for testing.
        
        Returns:
            Mock exchange object
        """
        mock_exchange = Mock()
        mock_exchange.name = "MockExchange"
        mock_exchange.load_markets.return_value = None
        mock_exchange.fetch_ohlcv.return_value = [
            [1640995200000, 40000, 41000, 39000, 40500, 1000],
            [1640995260000, 40500, 41500, 40000, 41000, 1200],
        ]
        mock_exchange.create_market_buy_order.return_value = {
            'id': 'test_order_id',
            'symbol': 'BTC/USDT',
            'type': 'market',
            'side': 'buy',
            'amount': 0.001,
            'price': 41000,
            'status': 'closed'
        }
        return mock_exchange
        
    def assertLogMessage(self, mock_logger, level, message_contains):
        """Assert that a log message was called with specific content.
        
        Args:
            mock_logger: Mock logger object
            level: Log level (e.g., 'info', 'error')
            message_contains: String that should be contained in the message
        """
        # Get the appropriate method call
        method_calls = getattr(mock_logger, level).call_args_list
        
        # Check if any call contains the expected message
        found = False
        for call in method_calls:
            if call and len(call) > 0 and len(call[0]) > 0:
                if message_contains in str(call[0][0]):
                    found = True
                    break
                    
        self.assertTrue(
            found, 
            f"Expected log message containing '{message_contains}' at level '{level}' not found"
        )