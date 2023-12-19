from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import util.loggers as loggers
from util.websocket_util import WebSocketClient

logger = loggers.setup_loggers()


class Trading:
    def __init__(self, exchange, symbol, position_size, max_drawdown,
                 moving_average_period, decision_tree_depth, trade_type) -> None:

        self.exchange = exchange  # phemex
        self.trade_type = trade_type
        self.getBalancUSDT, self.getBalancBTC = self.get_balance()
        self.manual_logger = logger['manual']
        self.symbol = symbol
        self.stoploss = 0
        self.max_drawdown = max_drawdown
        self.moving_average_period = moving_average_period
        self.decision_tree_depth = decision_tree_depth
        self.real_time_data = []  # To store real-time data
        self.analyzer = SentimentIntensityAnalyzer()

    def get_balance(self):
        # Modified to get USDT balance specifically
        balance = self.exchange.fetch_balance({'type': self.trade_type})
        return balance['USDT'], balance['BTC']

    def place_trade(self, symbol, side, order_type, amount, price=None, stop_loss=None, take_profit=None):
        self.symbol = symbol
        # Fetch the ticker price to calculate the equivalent quantity in the quote currency
        ticker = self.exchange.fetch_ticker(symbol)
        last_price = ticker['last']
        params = {}
        if order_type == 'limit':
            if side == 'buy':
                quantity = amount / price
            else:
                quantity = amount / price
        else:
            quantity = amount / last_price

        # Include stop loss and take profit if they are provided
        if stop_loss is not None:
            params['stopPrice'] = stop_loss / price
            if side == 'buy':
                params['stop'] = 'loss'  # Adjust based on your exchange's API
            else:
                params['stop'] = 'entry'  # Adjust based on your exchange's API

        if take_profit is not None:
            params['takeProfitPrice'] = take_profit / price

        # Create and return the order
        return self.exchange.create_order(symbol, order_type, side, quantity, price, params)

    def modify_takeprofit_stoploss(self, trade_id, take_profit=None, stop_loss=None):
        ticker = self.exchange.fetch_ticker(self.symbol)
        last_price = ticker['last']
        # Method to modify take profit and stop loss
        order_params = {}
        if take_profit:
            order_params['takeProfitPrice'] = take_profit / last_price
        if stop_loss:
            order_params['stopLossPrice'] = stop_loss / last_price
        return self.exchange.edit_order(trade_id, **order_params)

    def getBidAsk(self):

        # get orderbook the bid/ask prices
        orderbook = self.exchange.fetch_ticker(self.symbol)

        bid = orderbook['bid']
        ask = orderbook['ask']

        return bid, ask

    def fetch_open_trades(self, symbol):
        # Method to fetch open trades
        return self.exchange.fetch_open_orders(symbol)

    def get_ticker_price(self):
        # Method to get the real-time ticker price
        ticker = self.exchange.fetch_ticker(self.symbol)
        return ticker['last']

    def on_real_time_data(self, data):
        # This method will be called every time new data is received
        self.real_time_data.append(data)
        if len(self.real_time_data) > 200:
            self.real_time_data.pop(0)  # Remove the oldest element

    def start_real_time_updates(self):
        # Modified to handle real-time data
        self.real_time_client = WebSocketClient(
            self.symbol, self.on_real_time_data)
        self.real_time_client.start()

    def get_real_time_data(self):
        # Method to access the latest real-time data
        return self.real_time_data

    def stop_real_time_updates(self):
        if hasattr(self, 'real_time_client'):
            self.real_time_client.stop()

    def scale_in_out(self, amount) -> None:
        # Check if we are in a trade
        if self.check_trade_status():
            # If we are, update the size of the trade
            self.exchange.update_order(self.trade_id, {'amount': amount})

    def set_symbol(self, symbol):
        self.symbol = symbol
