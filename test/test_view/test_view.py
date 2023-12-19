import unittest
from tkinter import Tk
from unittest.mock import MagicMock, Mock

import presenters
from view.views import LoginView, MainView, WindowView


class TestWindowView(unittest.TestCase):
    def setUp(self):
        self.window = WindowView()

    def test_initialization(self):
        self.assertEqual(self.window.title(), "Trading App")

    # ... Additional tests for WindowView ...


class TestLoginView(unittest.TestCase):
    def setUp(self):
        self.parent = Mock()
        self.login_view = LoginView(self.parent)
        self.presenter = Mock()
        self.login_view.create_ui(self.presenter)

    def test_ui_creation(self):
        self.assertIsNone(self.login_view.username_entry)
        self.assertIsNone(self.login_view.password_entry)

    def test_get_credentials(self):
        self.login_view._username_var.set("testuser")
        self.login_view._password_var.set("password123")
        self.assertEqual(self.login_view.get_username(), "testuser")
        self.assertEqual(self.login_view.get_password(), "password123")

    # ... Additional tests for LoginView ...


class TestMainView(unittest.TestCase):
    def setUp(self):
        self.root = Tk()  # Create a real Tkinter root window
        self.main_view = MainView(self.root)

        # Mock the presenter and configure it to return a mock exchange with a fake market
        self.presenter = Mock()
        mock_exchange = MagicMock()
        mock_exchange.load_markets.return_value = {
            'BTC/USDT': {}, 'ETH/USDT': {}}  # Fake market data
        self.presenter.get_exchange.return_value = mock_exchange

        self.main_view.create_ui(self.presenter)

    def test_ui_creation(self):
        self.assertIsNotNone(self.main_view.notebook)
        self.assertEqual(len(self.main_view.notebook.tabs()),
                         6)  # Check if 6 tabs are added

    def tearDown(self):
        self.root.destroy()  # Clean up the root window


# Run the test suite
if __name__ == '__main__':
    unittest.main()
