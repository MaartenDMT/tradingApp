import re
import threading
import traceback

import websocket

import util.loggers as loggers
from model.features import Tradex_indicator
from model.machinelearning.ml_util import classifier, regression
from util.candlestick_parser import *

logger = loggers.setup_loggers()
autobot_logger = logger['autobot']

'''
#TODO: Make when you start a bot, 
that you automatically get data loaded, after the data is loaded.
you make the

'''


class AutoBot:
    def __init__(self, exchange, symbol, amount, stop_loss, take_profit, model, time, ml, trade_x, df) -> None:
        # Initialize class variables as before
        self.exchange = exchange
        self.symbol: str = symbol
        self.amount: float = amount
        self.stop_loss: float = stop_loss
        self.take_profit: float = take_profit
        self.model = model
        self.time = time
        self.ml = ml
        self.trade_x: Tradex_indicator = trade_x
        self.df = df
        self.open_orders = {}
        self.indicators = None

        self.auto_trade = False

        # Set up logging
        self.logger = autobot_logger
        self.logger.info(
            f"the amount of autobot:, {self.amount} and {self.model}:")

        stream_names = [
            f'{self.symbol.lower()}@kline_{self.time}']
        self.websocket_url = 'wss://stream.binance.com:9443/stream?streams=' + \
            "/".join(stream_names)

    def __str__(self):
        return f"Autobot-{self.exchange}: {self.symbol}-{self.time}-{self.model}- Trades: {len(self.open_orders)} -> "

    def getnormal_symbol(self) -> str:
        symbol = self.symbol.replace('/', '').lower()
        return symbol

    def getnormal_count(self) -> int:
        if self.time == "1m":
            return 60
        if self.time == "5m":
            return 300
        if self.time == "30m":
            return 1_800
        if self.time == "1h":
            return 3_600
        if self.time == "3h":
            return 10_800

    def start_auto_trading(self, event) -> None:
        self.auto_trade = event

        # Create single WebSocket connection
        self.ws = websocket.WebSocketApp(
            self.websocket_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close)

        self.ws.on_open = self.on_open

        # Run WebSocket connection in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def on_message(self, ws, message):
        try:
            data = decode_json_message(message)
            if not data:
                return

            if not validate_message_data(data):
                return
            self.handle_candlestick_data(data)

            if self.get_model_type(self.model) in ["classifier"] or classifier:
                self._run_auto_trading_classifier()

            elif self.get_model_type(self.model) in ["regression"] or regression:
                self._run_auto_trading_regression()
            else:
                self.logger.error("no model to use!")

        except Exception as e:
            self.logger.error(
                f'Unexpected error: {e}\n{traceback.format_exc()}')

    def on_error(self, ws, error):
        try:
            # Handle specific WebSocket error here
            raise websocket.WebSocketException(f'WebSocket error: {error}')
        except websocket.WebSocketException as e:
            self.logger.error(e)
        except Exception as e:
            self.logger.error(
                f'Unexpected error: {e}\n{traceback.format_exc()}')

    def on_close(self, ws):
        try:
            # Handle specific WebSocket close event here
            self.logger.info('WebSocket connection closed')
        except Exception as e:
            self.logger.error(
                f'Unexpected error: {e}\n{traceback.format_exc()}')

    def on_open(self, ws):
        # and one for the processing of the data
        self.logger.info('WebSocket connection opened')
        self.create_tradex_signals()

    def create_tradex_signals(self):
        try:
            # Check if the timeframe is in self.candlestick_data before creating an Indicator instance
            if self.df is not None:
                self.logger.info(
                    f'the create tradex data: {self.df}')
                try:
                    self.indicators = Indicator(self.trade_x)
                except KeyError as e:
                    self.logger.error(
                        f"Key error creating tradex signals for: {e}")
                except ValueError as e:  # replace with the actual type of exception you expect
                    self.logger.error(
                        f"Value error creating tradex signals for: {e}")
                except Exception as e:
                    self.logger.error(
                        f"Error creating tradex signals for: {e}\n{traceback.format_exc()}")

            else:
                self.logger.error(
                    f'data: does not exist in self.candlestick_data')
                return

        except KeyError as e:
            self.logger.error(
                f"Key error creating tradex signals for: {e}")
        except ValueError as e:  # replace with the actual type of exception you expect
            self.logger.error(
                f"Value error creating tradex signals for : {e}")
        except Exception as e:
            self.logger.error(
                f"Error creating tradex signals for: {e}\n{traceback.format_exc()}")

        threading.Thread(target=self.indicators.run).start()

    def stop_auto_trading(self, event) -> None:
        self.auto_trade = event

    def _run_auto_trading_classifier(self) -> None:
        if self.auto_trade == False:
            return
        # Get the current price of the asset
        current_price = self.exchange.fetch_ticker(self.symbol)['last']

        # Use the model to predict the next price
        prediction = self.ml.predict(self.df,
                                     self.model, self.time, self.symbol) - 1

        self.logger.info(
            f"{self.exchange} exchange, the {self.model} predicted:{prediction}")

        # Use the trade_x to get the signal
        trade_x_signal = self.trade_x.run()
        self.logger.info(trade_x_signal)

        # Check if the prediction is above the take profit or below the stop loss
        if prediction == -1:

            open_orders = self.exchange.fetch_open_orders(
                symbol=self.symbol)
            for order in open_orders:
                if order['side'] == 'sell':
                    self.logger.info(
                        f'sell order for {self.symbol} already exists with id: {order["id"]}')
                    if self.check_order_status(order['id']):
                        self.open_orders.pop(order['id'], None)
                    else:
                        return
            # Place a sell order
            try:
                order = self.exchange.create_order(
                    symbol=self.symbol, type='limit', side='sell', amount=self.amount, price=current_price)
                self.logger.info(
                    f'Sold {self.amount} {self.symbol} at {current_price}')
                stop_loss_order = self.exchange.create_order(
                    symbol=self.symbol, type='stoploss', side='sell', amount=self.amount, price=current_price / (1 + self.stop_loss / 100))
                self.open_orders[order['id']] = {
                    'symbol': self.symbol, 'side': 'sell', 'amount': self.amount, 'price': current_price}
            except Exception as e:
                self.logger.error(f"autobot Sell order:{e}")

        elif prediction == 1:

            open_orders = self.exchange.fetch_open_orders(
                symbol=self.symbol)
            for order in open_orders:
                if order['side'] == 'buy':
                    self.logger.info(
                        f'buy order for {self.symbol} already exists with id: {order["id"]}')
                    if self.check_order_status(order['id']):
                        self.open_orders.pop(order['id'], None)
                    else:
                        return

            # Place a buy order
            try:
                order = self.exchange.create_order(
                    symbol=self.symbol, type='limit', side='buy', amount=self.amount, price=current_price)
                self.logger.info(
                    f'Bought {self.amount} {self.symbol} at {current_price}')
                stop_loss_order = self.exchange.create_order(
                    symbol=self.symbol, type='stoploss', side='sell', amount=self.amount, price=current_price * (1 + self.stop_loss / 100))
                self.open_orders[order['id']] = {
                    'symbol': self.symbol, 'side': 'buy', 'amount': self.amount, 'price': current_price}
            except Exception as e:
                self.logger.error(f"autobot Buy order:{e}")

        elif prediction == 0:
            self.logger.info(
                f"the prediction gave {prediction} waiting for a trading oppertunity!")
        else:
            self.logger.error(f"there was no prediction!")

    def _run_auto_trading_regression(self) -> None:
        if self.auto_trade == False:
            return
        # Get the current price of the asset
        current_price = self.exchange.fetch_ticker(self.symbol)['last']

        # Use the model to predict the next price
        prediction = self.ml.predict(self.model, self.time, self.symbol)
        autobot_logger.info(
            f"{self.exchange} exchange, the {self.model} predicted:{prediction}")

        # Use the trade_x to get the signal
        trade_x_signal = self.trade_x.run()
        self.info(trade_x_signal)

        # Check if the prediction is above the take profit or below the stop loss
        if prediction < current_price:

            open_orders = self.exchange.fetch_open_orders(
                symbol=self.symbol)
            for order in open_orders:
                if order['side'] == 'sell':
                    self.logger.info(
                        f'sell order for {self.symbol} already exists with id: {order["id"]}')
                    if self.check_order_status(order['id']):
                        self.open_orders.pop(order['id'], None)
                    else:
                        return
            # Place a sell order
            try:
                order = self.exchange.create_order(
                    symbol=self.symbol, type='limit', side='sell', amount=self.amount, price=current_price)
                self.logger.info(
                    f'Sold {self.amount} {self.symbol} at {current_price}')
                stop_loss_order = self.exchange.create_order(
                    symbol=self.symbol, type='stoploss', side='sell', amount=self.amount, price=current_price / (1 + self.stop_loss / 100))
                self.open_orders[order['id']] = {
                    'symbol': self.symbol, 'side': 'sell', 'amount': self.amount, 'price': current_price}
            except Exception as e:
                self.logger.error(e)

        elif prediction > current_price:

            open_orders = self.exchange.fetch_open_orders(
                symbol=self.symbol)
            for order in open_orders:
                if order['side'] == 'buy':
                    self.logger.info(
                        f'buy order for {self.symbol} already exists with id: {order["id"]}')
                    if self.check_order_status(order['id']):
                        self.open_orders.pop(order['id'], None)
                    else:
                        return

            # Place a buy order
            try:
                order = self.exchange.create_order(
                    symbol=self.symbol, type='limit', side='buy', amount=self.amount, price=current_price)
                self.logger.info(
                    f'Bought {self.amount} {self.symbol} at {current_price}')
                stop_loss_order = self.exchange.create_order(
                    symbol=self.symbol, type='stoploss', side='sell', amount=self.amount, price=current_price * (1 + self.stop_loss / 100))
                self.open_orders[order['id']] = {
                    'symbol': self.symbol, 'side': 'buy', 'amount': self.amount, 'price': current_price}
            except Exception as e:
                self.logger.error(e)
        else:
            self.logger.error(f"there was no prediction!")

    def check_order_status(self, order_id) -> bool:
        order = self.exchange.fetch_order(order_id, self.symbol)
        if order['status'] == 'closed' or order['status'] == 'filled':
            return True
        else:
            return False

    def orders(self, order):
        ...

    def get_model_type(self, model) -> str:
        model_string = str(model)
        regex = r"(regression|classifier)"

        # Use regular expressions to extract the word "Regression"
        match = re.search(regex, model_string, re.IGNORECASE)

        if match:
            return match.group(0).lower()
        else:
            return model_string

    def handle_candlestick_data(self, data):
        try:
            candlestick, timeframe = parse_candlestick(data)
        except Exception as e:
            self.logger.error(
                f"Failed to parse candlestick: {e}\n{traceback.format_exc()}")
            return

        if not candlestick.get('x'):
            return

        self.logger.info('New candlestick received from websocket')

        try:
            new_row = extract_candlestick_data(candlestick)
        except Exception as e:
            self.logger.error(
                f"Failed to extract candlestick data: {e}\n{traceback.format_exc()}")
            return

        self.df = add_row_and_maintain_size(new_row, self.df)

        self.indicators.run(self.df)

        if self.ws.sock.connected:
            ...
        else:
            self.logger.warning(
                'WebSocket connection is not alive. Attempting to reconnect...')
            self.reconnect()


class Indicator:
    def __init__(self, tradex: Tradex_indicator):
        self.instance = tradex
        self.trend = pd.DataFrame()
        self.screener = pd.DataFrame()
        self.real_time = pd.DataFrame()
        self.scanner = pd.DataFrame()
        self.logger = autobot_logger

    def run(self, data=None):
        self.logger.info("making the indicators")
        try:
            self.update_data(data)
            self.instance.run()
            self.trend = self.instance.trend.df_trend.tail(50)
            self.screener = self.instance.screener.df_screener.tail(50)
            self.real_time = self.instance.real_time.df_real_time.tail(50)
            self.scanner = self.instance.scanner.df_scanner.tail(50)
            self.data = self.get_indicators_df()
        except Exception as e:
            self.logger.error(
                f"Error creating tradex signals: {e}\n{traceback.format_exc()}")

    def get_indicators_df(self):
        # Combine all DataFrames into one
        df_list = [self.trend, self.screener, self.real_time, self.scanner]
        combined_df = pd.concat(df_list, axis=1, keys=[
                                'Trend', 'Screener', 'Real_Time', 'Scanner'])

        # If you want to handle duplicated columns across the DataFrames
        combined_df.columns = combined_df.columns.map(
            lambda x: f'{x[0]}_{x[1]}' if x[1] != '' else x[0])

        return combined_df

    def update_data(self, new_data):
        if new_data is not None:
            self.instance = Tradex_indicator(
                self.instance.symbol, self.instance.timeframe, None, False, new_data)
