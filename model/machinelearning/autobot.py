import asyncio
import re
import threading
import time
import traceback

import pandas as pd
import websocket

from model.features import Tradex_indicator
from util import loggers
from util.candlestick_parser import (decode_json_message, parse_candlestick,
                                     validate_message_data)

from .ml_util import classifier, regression

# Import optimized utilities - only import what we actually use
try:
    from ...util.cache import HybridCache
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False

# Async utilities for enhanced trading
try:
    from ...util.async_client import AsyncCCXTClient
    from ...util.async_trading import Order, Position, TradingEngine
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

logger = loggers.setup_loggers()
autobot_logger = logger['autobot']


class RiskManager:
    """Advanced risk management for trading operations."""

    def __init__(self, max_daily_loss=0.05, max_position_size=0.1, max_open_positions=3):
        self.max_daily_loss = max_daily_loss  # 5% max daily loss
        self.max_position_size = max_position_size  # 10% max position size
        self.max_open_positions = max_open_positions
        self.daily_pnl = 0.0
        self.session_start_balance = None

    def can_open_position(self, current_balance, position_size, open_positions_count):
        """Check if a new position can be opened based on risk parameters."""
        if self.session_start_balance is None:
            self.session_start_balance = current_balance

        # Check daily loss limit
        daily_loss_pct = abs(self.daily_pnl) / self.session_start_balance if self.session_start_balance > 0 else 0
        if daily_loss_pct >= self.max_daily_loss:
            return False, "Daily loss limit exceeded"

        # Check position size limit
        position_size_pct = position_size / current_balance if current_balance > 0 else 0
        if position_size_pct > self.max_position_size:
            return False, "Position size too large"

        # Check max open positions
        if open_positions_count >= self.max_open_positions:
            return False, "Maximum open positions reached"

        return True, "OK"

    def update_pnl(self, pnl_change):
        """Update the daily P&L."""
        self.daily_pnl += pnl_change


class ConnectionManager:
    """Manage WebSocket connections with reconnection logic."""

    def __init__(self, ws_url, on_message_callback, max_retries=5, retry_delay=1.0):
        self.ws_url = ws_url
        self.on_message_callback = on_message_callback
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_attempts = 0
        self.ws = None
        self.is_connected = False

    async def connect(self):
        """Establish WebSocket connection with retry logic."""
        while self.connection_attempts < self.max_retries:
            try:
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_message=self.on_message_callback,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open
                )

                # Start connection in a separate thread
                wst = threading.Thread(target=self.ws.run_forever)
                wst.daemon = True
                wst.start()

                # Wait for connection to establish
                await asyncio.sleep(2)

                if self.is_connected:
                    autobot_logger.info(f"WebSocket connected successfully after {self.connection_attempts + 1} attempts")
                    return True

            except Exception as e:
                self.connection_attempts += 1
                autobot_logger.error(f"Connection attempt {self.connection_attempts} failed: {e}")

                if self.connection_attempts < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** self.connection_attempts))  # Exponential backoff

        autobot_logger.error(f"Failed to connect after {self.max_retries} attempts")
        return False

    def _on_open(self, ws):
        self.is_connected = True
        self.connection_attempts = 0
        autobot_logger.info("WebSocket connection opened")

    def _on_close(self, ws, close_status_code, close_msg):
        self.is_connected = False
        autobot_logger.warning(f"WebSocket connection closed: {close_status_code} - {close_msg}")

    def _on_error(self, ws, error):
        self.is_connected = False
        autobot_logger.error(f"WebSocket error: {error}")

    async def reconnect_if_needed(self):
        """Check connection and reconnect if necessary."""
        if not self.is_connected:
            autobot_logger.info("Attempting to reconnect WebSocket...")
            return await self.connect()
        return True

'''
#TODO: Make when you start a bot,
that you automatically get data loaded, after the data is loaded.
you make the

'''


class AutoBot:
    def __init__(self, exchange, symbol, amount, stop_loss, take_profit, model, timeframe, ml, trade_x, df) -> None:
        # Initialize class variables
        self.exchange = exchange
        self.symbol: str = symbol
        self.amount: float = amount
        self.stop_loss: float = stop_loss
        self.take_profit: float = take_profit
        self.model = model
        self.timeframe = timeframe  # Renamed to avoid shadowing built-in time
        self.ml = ml
        self.trade_x: Tradex_indicator = trade_x
        self.df = df
        self.open_orders = {}
        self.indicators = None

        self.auto_trade = False

        # Set up logging
        self.logger = autobot_logger

        # Initialize modern components
        self.risk_manager = RiskManager()
        self.connection_manager = None
        self._trading_task = None
        self._is_running = False

        # Performance tracking
        self.trade_count = 0
        self.successful_trades = 0
        self.failed_trades = 0

        self.logger.info(f"AutoBot initialized - Amount: {self.amount}, Model: {self.model}, Timeframe: {self.timeframe}")

        stream_names = [f'{self.symbol.lower()}@kline_{self.timeframe}']
        self.websocket_url = 'wss://stream.binance.com:9443/stream?streams=' + "/".join(stream_names)

        # Initialize connection manager
        self.connection_manager = ConnectionManager(
            self.websocket_url,
            self.on_message,
            max_retries=5,
            retry_delay=1.0
        )

    async def start_async(self):
        """Start the AutoBot with async operations."""
        self.logger.info("Starting AutoBot in async mode...")
        self._is_running = True

        # Connect to WebSocket
        connected = await self.connection_manager.connect()
        if not connected:
            self.logger.error("Failed to establish WebSocket connection")
            return False

        # Start the main trading loop
        self._trading_task = asyncio.create_task(self._trading_loop())

        return True

    async def stop_async(self):
        """Stop the AutoBot gracefully."""
        self.logger.info("Stopping AutoBot...")
        self._is_running = False

        if self._trading_task:
            self._trading_task.cancel()
            try:
                await self._trading_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket connection
        if self.connection_manager and self.connection_manager.ws:
            self.connection_manager.ws.close()

        self.logger.info("AutoBot stopped")

    async def _trading_loop(self):
        """Main trading loop with error handling and reconnection."""
        while self._is_running:
            try:
                # Check connection health
                await self.connection_manager.reconnect_if_needed()

                # Perform any periodic tasks
                await self._periodic_health_check()

                # Wait before next iteration
                await asyncio.sleep(5)  # Check every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)  # Wait longer after error

    async def _periodic_health_check(self):
        """Perform periodic health checks and maintenance."""
        try:
            # Log performance metrics
            if self.trade_count > 0:
                success_rate = (self.successful_trades / self.trade_count) * 100
                self.logger.info(f"Performance - Trades: {self.trade_count}, Success Rate: {success_rate:.1f}%")

            # Check open orders status
            await self._check_open_orders_status()

        except Exception as e:
            self.logger.error(f"Health check error: {e}")

    async def _check_open_orders_status(self):
        """Check the status of open orders asynchronously."""
        for order_id, order_info in list(self.open_orders.items()):
            try:
                order = self.exchange.fetch_order(order_id, self.symbol)
                if order['status'] in ['closed', 'filled', 'canceled']:
                    self.logger.info(f"Order {order_id} completed with status: {order['status']}")
                    del self.open_orders[order_id]

                    if order['status'] == 'filled':
                        self.successful_trades += 1

            except Exception as e:
                self.logger.error(f"Error checking order {order_id}: {e}")
                # Remove problematic orders after several failures
                del self.open_orders[order_id]

    async def execute_trade_async(self, side, confidence_score=None):
        """
        Execute a trade asynchronously with enhanced risk management.

        :param side: 'buy' or 'sell'
        :param confidence_score: ML model confidence (0.0 to 1.0)
        """
        try:
            # Get current balance and price
            balance = self.exchange.fetch_balance()
            current_price = self.exchange.fetch_ticker(self.symbol)['last']

            # Calculate position size based on confidence
            position_size = self.amount
            if confidence_score is not None:
                # Adjust position size based on confidence (50% to 100% of base amount)
                position_size = self.amount * (0.5 + 0.5 * confidence_score)

            # Risk management check
            can_trade, reason = self.risk_manager.can_open_position(
                balance['total']['USDT'] if 'USDT' in balance['total'] else 1000,
                position_size,
                len(self.open_orders)
            )

            if not can_trade:
                self.logger.warning(f"Trade blocked by risk management: {reason}")
                return False

            # Execute the trade
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='market',  # Use market orders for faster execution
                side=side,
                amount=position_size,
                price=current_price
            )

            # Create stop loss order
            stop_price = current_price * (1 - self.stop_loss / 100) if side == 'buy' else current_price * (1 + self.stop_loss / 100)
            stop_loss_order = self.exchange.create_order(
                symbol=self.symbol,
                type='stop_loss_limit',
                side='sell' if side == 'buy' else 'buy',
                amount=position_size,
                price=stop_price * 0.99 if side == 'buy' else stop_price * 1.01,  # Slight offset for limit price
                stopPrice=stop_price
            )

            # Store order information
            self.open_orders[order['id']] = {
                'symbol': self.symbol,
                'side': side,
                'amount': position_size,
                'price': current_price,
                'stop_loss_id': stop_loss_order['id'] if stop_loss_order else None,
                'confidence': confidence_score,
                'timestamp': time.time()  # Use timestamp instead of datetime for simplicity
            }

            self.trade_count += 1
            self.logger.info(f"✓ {side.upper()} order executed: {position_size:.4f} {self.symbol} at {current_price:.4f} (Confidence: {confidence_score:.2f})")

            return True

        except Exception as e:
            self.failed_trades += 1
            self.logger.error(f"✗ Trade execution failed: {e}")
            return False

    def __str__(self):
        return f"Autobot-{self.exchange}: {self.symbol}-{self.timeframe}-{self.model}- Trades: {len(self.open_orders)} -> "

    def getnormal_symbol(self) -> str:
        symbol = self.symbol.replace('/', '').lower()
        return symbol

    def getnormal_count(self) -> int:
        if self.timeframe == "1m":
            return 60
        if self.timeframe == "5m":
            return 300
        if self.timeframe == "30m":
            return 1_800
        if self.timeframe == "1h":
            return 3_600
        if self.timeframe == "3h":
            return 10_800

    def start_auto_trading(self, event) -> None:
        """Start auto trading - now using the modern async approach."""
        self.auto_trade = event

        if event:
            # Use the new async start method
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running, create a task
                    asyncio.create_task(self.start_async())
                else:
                    # If no loop is running, run it
                    loop.run_until_complete(self.start_async())
            except Exception as e:
                self.logger.error(f"Failed to start async trading: {e}")
                # Fallback to old method
                self._start_legacy_websocket()
        else:
            # Stop trading
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.stop_async())
                else:
                    loop.run_until_complete(self.stop_async())
            except Exception as e:
                self.logger.error(f"Failed to stop async trading: {e}")

    def _start_legacy_websocket(self):
        """Legacy WebSocket start method as fallback."""
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

            if self.get_model_type(self.model) in ["classifier"]:
                asyncio.create_task(self._run_auto_trading_classifier_async())
            elif self.get_model_type(self.model) in ["regression"]:
                asyncio.create_task(self._run_auto_trading_regression_async())
            else:
                # Fallback to synchronous methods
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
                    'data: does not exist in self.candlestick_data')
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
        if not self.auto_trade:
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
                    'symbol': self.symbol, 'side': 'buy', 'amount': self.amount, 'price': current_price, 'stop_loss_id': stop_loss_order['id'] if stop_loss_order else None}
            except Exception as e:
                self.logger.error(f"autobot Buy order:{e}")

        elif prediction == 0:
            self.logger.info(
                f"the prediction gave {prediction} waiting for a trading oppertunity!")
        else:
            self.logger.error("there was no prediction!")

    def _run_auto_trading_regression(self) -> None:
        if not self.auto_trade:
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
                    'symbol': self.symbol, 'side': 'buy', 'amount': self.amount, 'price': current_price, 'stop_loss_id': stop_loss_order['id'] if stop_loss_order else None}
            except Exception as e:
                self.logger.error(e)
        else:
            self.logger.error("there was no prediction!")

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

    async def _run_auto_trading_classifier_async(self) -> None:
        """Async version of classifier trading with enhanced features."""
        if not self.auto_trade:
            return

        try:
            # Get the current price of the asset
            current_price = self.exchange.fetch_ticker(self.symbol)['last']

            # Use the model to predict the next price with confidence
            prediction = self.ml.predict(self.df, self.model, self.timeframe, self.symbol) - 1

            # Calculate confidence score (you can enhance this based on your ML model output)
            confidence_score = 0.7  # Default confidence, can be enhanced with model probability outputs

            self.logger.info(f"{self.exchange} exchange, {self.model} predicted: {prediction} (confidence: {confidence_score:.2f})")

            # Use the trade_x to get the signal
            trade_x_signal = self.trade_x.run()
            self.logger.info(f"TradeX signal: {trade_x_signal}")

            # Enhanced decision making with TradeX signal confirmation
            if prediction == 1 and trade_x_signal.get('signal') == 'BUY':
                await self.execute_trade_async('buy', confidence_score)
            elif prediction == -1 and trade_x_signal.get('signal') == 'SELL':
                await self.execute_trade_async('sell', confidence_score)
            elif prediction == 0:
                self.logger.info("Neutral prediction - no trade executed")
            else:
                self.logger.info("Prediction and TradeX signal do not align - no trade executed")

        except Exception as e:
            self.logger.error(f"Error in async classifier trading: {e}")

    async def _run_auto_trading_regression_async(self) -> None:
        """Async version of regression trading with enhanced features."""
        if not self.auto_trade:
            return

        try:
            # Get the current price of the asset
            current_price = self.exchange.fetch_ticker(self.symbol)['last']

            # Use the model to predict the next price
            prediction = self.ml.predict(self.model, self.timeframe, self.symbol)

            # Calculate confidence based on prediction distance from current price
            price_change_pct = abs(prediction - current_price) / current_price
            confidence_score = min(0.9, max(0.5, price_change_pct * 10))  # Scale confidence

            self.logger.info(f"{self.exchange} exchange, {self.model} predicted: {prediction:.4f} vs current: {current_price:.4f} (confidence: {confidence_score:.2f})")

            # Enhanced trading logic with minimum price difference threshold
            price_difference_threshold = current_price * 0.001  # 0.1% minimum difference

            if prediction < current_price - price_difference_threshold:
                # Prediction suggests price will drop - sell
                await self.execute_trade_async('sell', confidence_score)
            elif prediction > current_price + price_difference_threshold:
                # Prediction suggests price will rise - buy
                await self.execute_trade_async('buy', confidence_score)
            else:
                self.logger.info("Price prediction too close to current price - no trade executed")

        except Exception as e:
            self.logger.error(f"Error in async regression trading: {e}")

    def get_performance_metrics(self):
        """Get trading performance metrics."""
        if self.trade_count == 0:
            return {"message": "No trades executed yet"}

        success_rate = (self.successful_trades / self.trade_count) * 100
        return {
            "total_trades": self.trade_count,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "success_rate": f"{success_rate:.2f}%",
            "open_positions": len(self.open_orders),
            "risk_manager_stats": {
                "daily_pnl": self.risk_manager.daily_pnl,
                "max_daily_loss": self.risk_manager.max_daily_loss,
                "max_position_size": self.risk_manager.max_position_size
            }
        }


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
