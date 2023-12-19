import unittest
from unittest.mock import Mock, patch

# Replace 'your_module' with the actual module name
from model.models import TradeTabModel, Trading


class TestTrading(unittest.TestCase):

    def setUp(self):
        self.mock_exchange = Mock()
        self.trading = Trading(
            self.mock_exchange, 'BTC/USDT', 10, 0.001, 50, 2, 'spot')

    def test_place_trade(self):
        self.mock_exchange.create_order.return_value = Mock()
        result = self.trading.place_trade(
            'buy', 'limit', 1, 10000, 9500, 10500)
        self.mock_exchange.create_order.assert_called_with(
            symbol='BTC/USDT', side='buy', type='limit', quantity=1, price=10000,
            stopPrice=9500, takeProfitPrice=10500)
        self.assertIsNotNone(result)

    def test_modify_takeprofit_stoploss(self):
        self.mock_exchange.edit_order.return_value = Mock()
        trade_id = '12345'
        self.trading.modify_takeprofit_stoploss(
            trade_id, take_profit=10500, stop_loss=9500)
        self.mock_exchange.edit_order.assert_called_with(
            trade_id, takeProfitPrice=10500, stopLossPrice=9500)

    def test_get_balance(self):
        self.mock_exchange.fetch_balance.return_value = {
            'USDT': 1000, 'BTC': 0.1}
        usdt, btc = self.trading.get_balance()
        self.assertEqual(usdt, 1000)
        self.assertEqual(btc, 0.1)

    def test_get_bid_ask(self):
        self.mock_exchange.fetch_ticker.return_value = {
            'bid': 9500, 'ask': 9600}
        bid, ask = self.trading.getBidAsk()
        self.assertEqual(bid, 9500)
        self.assertEqual(ask, 9600)

    def test_fetch_open_trades(self):
        self.mock_exchange.fetch_open_orders.return_value = [
            'trade1', 'trade2']
        result = self.trading.fetch_open_trades()
        self.assertEqual(result, ['trade1', 'trade2'])

    def test_get_ticker_price(self):
        self.mock_exchange.fetch_ticker.return_value = {'last': 9550}
        last_price = self.trading.get_ticker_price()
        self.assertEqual(last_price, 9550)


class TestTradeTabModel(unittest.TestCase):

    def setUp(self):
        self.mock_presenter = Mock()
        self.mock_presenter.get_exchange.return_value = Mock()
        self.trade_tab_model = TradeTabModel(self.mock_presenter)

    # Replace 'your_module' with the actual module name
    @patch('model.Trading')
    def test_get_trading(self, mock_trading_class):
        result = self.trade_tab_model.get_trading()
        mock_trading_class.assert_called_with(
            self.mock_presenter.get_exchange(), 'BTC/USDT', 10, 0.001, 50, 2, 'spot')
        self.assertIsInstance(result, Trading)

    def test_get_balance(self):
        self.trade_tab_model._trading.get_balance.return_value = (1000, 0.1)
        usdt, btc = self.trade_tab_model.get_balance()
        self.assertEqual(usdt, 1000)
        self.assertEqual(btc, 0.1)

    def test_update_stoploss(self):
        self.trade_tab_model._trading.trade_id = '12345'
        self.trade_tab_model.update_stoploss(9500)
        self.trade_tab_model._trading.modify_takeprofit_stoploss.assert_called_with(
            '12345', stop_loss=9500)

    def test_update_takeprofit(self):
        self.trade_tab_model._trading.trade_id = '12345'
        self.trade_tab_model.update_takeprofit(10500)
        self.trade_tab_model._trading.modify_takeprofit_stoploss.assert_called_with(
            '12345', take_profit=10500)

    def test_place_trade(self):
        self.trade_tab_model.place_trade('buy', 'limit', 1, 10000, 9500, 10500)
        self.trade_tab_model._trading.place_trade.assert_called_with(
            'buy', 'limit', 1, 10000, 9500, 10500)


if __name__ == '__main__':
    unittest.main()
