import time
import traceback
from pprint import pprint

import numpy as np
import pandas_ta as ta
import yfinance as yf
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import util.loggers as loggers
from util.websocket_util import WebSocketClient

logger = loggers.setup_loggers()


class Trading:
    def __init__(self, exchange, symbol='BTC/USD:USD', trade_type='swap') -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.trade_type = trade_type
        self.manual_logger = logger['manual']
        self.balance = self.get_balance()  # Store the balance
        self.analyzer = SentimentIntensityAnalyzer()
        self.leverage = None

    # Utility Methods
    @staticmethod
    def extract_currencies_with_separator(trading_pair):
        # Split the trading pair string using the ':' delimiter
        parts = trading_pair.split(':')

        # Take the first part before ':' as the base currency
        base_currency = parts[0]

        currencies_with_separator = base_currency

        return currencies_with_separator

    def get_default_min_qty(self):
        # Return a default minimum quantity value or implement logic to determine it
        return 0.001  # Example default value, adjust as needed

    def is_order_value_sufficient(self, amount, price):
        min_cost = self.exchange.market(self.symbol).get(
            'limits', {}).get('cost', {}).get('min', 0)
        order_cost = self.calculate_order_cost(amount, price)
        return order_cost >= min_cost

    def update_settings(self, settings):
        """
        Update multiple trading settings at once.

        :param settings: A dictionary of settings to update.
                        Keys can include 'symbol', 'accountType', 'leverage', etc.
        """
        try:
            for key, value in settings.items():
                if key == 'symbol':
                    self.symbol = value
                elif key == 'accountType':
                    self.exchange.options['defaultType'] = value
                elif key == 'leverage':
                    result = self.set_leverage(value)
                    if not result:
                        self.manual_logger.error(
                            f"Failed to set leverage: {value}")
                # Add more settings as needed

            # Log the updated settings
            self.manual_logger.info(f"Updated settings: {settings}")
        except Exception as e:
            self.manual_logger.error(
                f'there was an error with updating the settings {e}\n{traceback.format_exc()}')

    def set_symbol(self, symbol):
        self.symbol = symbol

    def set_leverage(self, leverage):
        # Check if leverage setting is supported by the exchange
        if not hasattr(self.exchange, 'set_leverage'):
            self.manual_logger.info(
                "Leverage setting not supported for this exchange.")
            return False

        # Check if the desired leverage is within the allowed limits
        leverage_limits = self.exchange.market(
            self.symbol).get('limits', {}).get('leverage', {})
        min_leverage = leverage_limits.get('min', 0)
        max_leverage = leverage_limits.get('max', float('inf'))

        if min_leverage is not None and max_leverage is not None:
            if not (min_leverage <= leverage <= max_leverage):
                self.manual_logger.error(
                    f"Leverage value {leverage} is not within allowed limits.")
                return False

        # Set the leverage
        try:
            self.exchange.set_leverage(leverage, self.symbol)
            self.leverage = leverage
            self.manual_logger.info(
                f"Leverage set to {leverage} for {self.symbol}.")
            return True
        except Exception as e:
            self.manual_logger.error(f"Error setting leverage: {e}")
            return False

        # Example usage:
        # trading_instance = Trading(your_exchange_object)
        # trading_instance.update_settings({'symbol': 'ETH/USD', 'accountType': 'swap', 'leverage': 10})

    # 2. Trade Execution and Management

    def place_trade(self, symbol, side, order_type, amount, price=None, stop_loss=None, take_profit=None):
        self.symbol = symbol
        try:
            # Fetch open trades for the symbol
            open_trades = self.fetch_open_trades(symbol)

            # Check if a similar trade exists
            similar_trade_exists = any(
                trade for trade in open_trades if trade['symbol'] == symbol and trade['side'] == side and trade['status'] == 'open'
            )

            # If a similar trade exists, log it and return without creating a new order
            if similar_trade_exists:
                self.manual_logger.info(
                    f"Similar trade already exists for {symbol} on the {side} side. No new trade placed.")
                return None

            # Continue with placing the trade if no similar trade exists
            last_price = self.fetch_market_data('ticker')['last_price']
            quantity = self.calculate_quantity(
                amount, price, last_price, order_type, side)

            if quantity is None:
                self.manual_logger.error("Failed to calculate quantity.")
                return None

            params = self.prepare_order_params(
                order_type, price, last_price, stop_loss, take_profit, side)
            if price is not None:
                price = self.adjust_for_tick_size(price, last_price)

            return self.exchange.create_order(symbol, order_type, side, quantity, price, params)
        except Exception as e:
            self.manual_logger.error(
                f"Error placing trade: {e}\n{traceback.format_exc()}")
            return None

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

    def scale_in_out(self, amount) -> None:
        # Check if we are in a trade
        if self.check_trade_status():
            # If we are, update the size of the trade
            self.exchange.update_order(self.trade_id, {'amount': amount})

    def check_order_status(self, order_id):
        return self.exchange.fetch_order(order_id, self.symbol)

    def get_position_info(self):
        # This method depends on the specific API provided by the exchange
        return self.exchange.private_get_position({'symbol': self.symbol})

    # 3. Risk Management and Analysis
    def calculate_drawdown(self, peak_balance, current_balance):
        drawdown = (peak_balance - current_balance) / peak_balance
        return drawdown

    def calculate_var(self, portfolio, confidence_level=.95):
        """
        Calculate the Value at Risk (VaR) of a portfolio.

        :param portfolio: A dictionary with symbols as keys and weights as values.
        :param confidence_level: The confidence level for VaR (e.g., 95%).
        :return: The VaR value.
        """
        portfolio_returns = []
        for symbol, weight in portfolio.items():
            # Fetch historical data (e.g., closing prices for the last 100 days)
            historical_data = self.fetch_historical_data(
                symbol, '1d', None, 100)
            # Calculate daily returns
            prices = [data[4] for data in historical_data]  # Closing prices
            returns = np.diff(prices) / prices[:-1]
            weighted_returns = returns * weight
            portfolio_returns.append(weighted_returns)

        # Calculate portfolio daily return
        portfolio_daily_return = np.sum(portfolio_returns, axis=0)

        # Calculate VaR
        var = -np.percentile(portfolio_daily_return, 100 - confidence_level)
        return var
    # Example usage:
    # trading_instance = Trading(your_exchange_object)
    # portfolio = {'BTC/USD': 0.6, 'ETH/USD': 0.4}
    # var = trading_instance.calculate_var(portfolio, 95)
    # print("Value at Risk (95% confidence):", var)

    def update_dynamic_stop_loss(self, symbol, entry_price, current_price, initial_stop_loss, trailing_percent):
        # Fetch open trades and find the relevant trade_id
        open_trades = self.fetch_open_trades(symbol)
        relevant_trade = next(
            (trade for trade in open_trades if trade['symbol'] == symbol and trade['status'] == 'open'), None)

        if relevant_trade is not None:
            trade_id = relevant_trade['id']
            if current_price > entry_price:  # Assuming a long position
                new_stop_loss = current_price - \
                    (current_price * trailing_percent / 100)
                new_stop_loss = max(new_stop_loss, initial_stop_loss)
                self.modify_takeprofit_stoploss(
                    trade_id, stop_loss=new_stop_loss)
        else:
            self.manual_logger.info(f"No open trade found for {symbol}.")

    def calculate_liquidation_price(self, leverage, balance, position_size):
        # This is a simplified calculation; the actual formula depends on the exchange's margin policy
        return balance / (position_size * leverage)

    def check_risk_limits(self, position_size):
        risk_limits = self.exchange.market(self.symbol).get(
            'info', {}).get('riskLimits', [])
        for limit in risk_limits:
            if position_size <= float(limit.get('limit', '0')):
                return limit
        return None

    # 4. Advanced Trading Strategies

    def execute_twap_order(self, symbol, total_amount, duration, side):
        start_time = self.exchange.milliseconds()
        end_time = start_time + duration
        total_slices = duration / (60 * 1000)  # Example: slice every minute

        while self.exchange.milliseconds() < end_time:
            slice_size = total_amount / total_slices
            self.place_trade(symbol, side, 'market', slice_size)
            time.sleep(duration / total_slices)  # Wait for the next slice

    # 5. Market Data and Analysis

    def fetch_market_data(self, data_type, depth=5):
        """
        Fetches various types of market data based on the specified data_type.
        :param data_type: The type of market data to fetch (e.g., 'ticker', 'bidask', 'depth').
        :param depth: Used for order book depth, default is 5.
        :return: The requested market data or None in case of an error.
        """
        try:
            if data_type == 'ticker':
                # Fetch and return ticker data
                ticker = self.exchange.fetch_ticker(self.symbol)
                # pprint(ticker)
                return {'last_price': ticker['last'], 'timestamp': ticker['timestamp']}

            elif data_type == 'bidask':
                # Fetch the order book data for the symbol
                orderbook = self.exchange.fetch_order_book(self.symbol)

                # Get the best bid and ask prices from the order book
                bid = orderbook['bids'][0][0] if len(
                    orderbook['bids']) > 0 else None
                ask = orderbook['asks'][0][0] if len(
                    orderbook['asks']) > 0 else None

                # Return the bid and ask prices
                return {'bid': bid, 'ask': ask}

            elif data_type == 'depth':
                # Fetch and return order book depth
                order_book = self.exchange.fetch_order_book(
                    self.symbol, limit=depth)
                return order_book

            else:
                self.manual_logger.error(
                    f"Unknown market data type: {data_type}")
                return None

        except Exception as e:
            self.manual_logger.error(f"Error fetching market data: {e}")
            return None

    def fetch_open_trades(self, symbol):
        # Method to fetch open trades
        return self.exchange.fetch_open_orders(symbol)

    def fetch_historical_data(self, timeframe='1d', since=None, limit=100):
        return self.exchange.fetch_ohlcv(self.symbol, timeframe, since, limit)

    def calculate_pnl(self, entry_price, exit_price, contract_quantity, is_long):
        contract_value = self.contract_to_underlying(contract_quantity)
        pnl = (exit_price - entry_price) * contract_value
        return pnl if is_long else -pnl

    def calculate_breakeven_price(self, entry_price, fee_percent, contract_quantity, is_long):
        total_fee = entry_price * contract_quantity * fee_percent / 100
        breakeven_move = total_fee / contract_quantity
        return entry_price + breakeven_move if is_long else entry_price - breakeven_move

    def calculate_order_cost(self, amount, price):
        contract_size = self.exchange.market(
            self.symbol).get('contractSize', 1)
        return amount * price * contract_size

    def get_funding_rate(self):
        info = self.exchange.market(self.symbol).get('info', {})
        return info.get('fundingRateSymbol')

    def get_tick_size(self):
        return self.exchange.market(self.symbol).get('precision', {}).get('price')

    def contract_to_underlying(self, contract_quantity):
        contract_size = self.exchange.market(
            self.symbol).get('contractSize', 1)
        return contract_quantity * contract_size

    # Additional utility and helper methods can be added as needed

    def transfer_funds(self, amount, currency_code, from_account_type, to_account_type):
        """
        Transfer funds between account types.

        :param amount: The amount to transfer.
        :param currency_code: The currency code, e.g., 'USDT'.
        :param from_account_type: The account type to transfer from, e.g., 'spot'.
        :param to_account_type: The account type to transfer to, e.g., 'swap'.
        :return: The response from the exchange or None in case of an error.
        """
        try:
            # Set the account type to the source account
            self.set_accountType(from_account_type)

            # Perform the transfer
            params = {
                'currency': currency_code,
                'amount': amount,
                'fromAccount': from_account_type,
                'toAccount': to_account_type
            }
            response = self.exchange.sapi_post_asset_transfer(params)
            self.manual_logger.info(f"Transfer successful: {response}")
            return response
        except Exception as e:
            self.manual_logger.error(f"Error transferring funds: {e}")
            return None
        # trading_instance = Trading(your_exchange_object)
        # response = trading_instance.transfer_funds(10, 'USDT', 'spot', 'swap')

    def get_balance(self):
        try:
            # Adjust parameters based on your needs (spot or swap)
            params = {"code": "USD"}
            usd_balance = self.exchange.fetch_balance({'code': 'USD'})
            btc_balance = self.exchange.fetch_balance({'code': 'BTC'})
            balance = self.exchange.deep_extend(usd_balance, btc_balance)
            # pprint(balance)
            return balance['total'].get('USD', 0)

        except Exception as e:
            self.manual_logger.error(f"Error fetching balance: {e}")
            return {'total': 0, 'free': 0}

    def calculate_quantity(self, amount, price, last_price, order_type, side):
        try:
            market = self.exchange.market(self.symbol)

            # Extract contract size
            contract_size = float(market.get('contractSize'))

            # Ensure precision is an integer
            precision = int(market.get('precision', {}).get('amount', 1))

            # Determine minimum quantity (use a default value or a method to calculate it)
            min_qty = self.get_default_min_qty()

            # Calculate the quantity in terms of contracts
            if order_type == 'limit' and price is not None:
                quantity = (amount / price) / contract_size
            else:  # For market orders
                quantity = (amount / last_price) / contract_size

            # Ensure quantity meets the minimum requirement and round to precision
            quantity = max(quantity, min_qty)
            quantity = round(quantity, precision)

            self.manual_logger.info(f'the quantity is {quantity}')

            return quantity
        except Exception as e:
            self.manual_logger.error(
                f"Error calculating quantity: {e}\n{traceback.format_exc()}")
            return None

    def prepare_order_params(self, order_type, price, last_price, stop_loss, take_profit, side):
        params = {}
        if stop_loss is not None:
            stop_price = self.adjust_for_tick_size(stop_loss, last_price)
            params['stopPrice'] = stop_price

        if take_profit is not None:
            take_profit_price = self.adjust_for_tick_size(
                take_profit, last_price)
            params['takeProfitPrice'] = take_profit_price

        # Additional parameters based on the exchange's requirements can be added here

        return params

    def adjust_for_tick_size(self, price, last_price):
        try:
            market = self.exchange.market(self.symbol)
            tick_size = market['precision']['price']
            tick_price = round(price / tick_size) * tick_size
            self.manual_logger.info(f'the adjusted tick price is {tick_price}')
            return tick_price
        except Exception as e:
            self.manual_logger.error(f"Error adjusting for tick size: {e}")
            return last_price  # Fallback to last_price in case of an error

    def perform_sentiment_analysis(self, text):
        """
        Perform sentiment analysis on the given text.

        Args:
            text (str): The text to analyze for sentiment.

        Returns:
            dict: A dictionary containing sentiment scores (e.g., 'compound', 'neg', 'neu', 'pos').
        """
        sentiment_scores = self.analyzer.polarity_scores(text)
        return sentiment_scores

    # Example usage:
    # trading_instance = Trading(your_exchange_object)
    # text_to_analyze = "I love trading cryptocurrencies!"
    # sentiment_scores = trading_instance.perform_sentiment_analysis(text_to_analyze)
    # print(sentiment_scores)


class RealTimeDataHandler:
    def __init__(self, exchange, symbol, data_callback):
        self.exchange = exchange
        self.symbol = symbol
        self.data_callback = data_callback
        self.real_time_data = []
        self.real_time_client = None

    def on_real_time_data(self, data):
        """Callback method for handling real-time data."""
        self.real_time_data.append(data)
        if len(self.real_time_data) > 200:
            self.real_time_data.pop(0)  # Keep the list size manageable

    def start_real_time_updates(self):
        """Starts real-time data updates."""
        if not self.real_time_client:
            self.real_time_client = WebSocketClient(
                self.symbol, self.on_real_time_data)
        self.real_time_client.start()

    def stop_real_time_updates(self):
        """Stops real-time data updates."""
        if self.real_time_client:
            self.real_time_client.stop()
            self.real_time_client = None

    def get_real_time_data(self):
        """Retrieves the latest real-time data."""
        return self.real_time_data


class MarketAnalyzer:
    def __init__(self):
        self.crypto_analyzer = SentimentIntensityAnalyzer()

    def analyze_crypto_sentiment(self, crypto_text):
        """
        Analyze sentiment for cryptocurrency-related text.

        Args:
            crypto_text (str): Text related to cryptocurrencies.

        Returns:
            dict: A dictionary containing sentiment scores (e.g., 'compound', 'neg', 'neu', 'pos').
        """
        sentiment_scores = self.crypto_analyzer.polarity_scores(crypto_text)
        return sentiment_scores

    def analyze_stock_technicals(self, stock_symbol, start_date, end_date):
        """
        Analyze technical indicators for a stock.

        Args:
            stock_symbol (str): The stock symbol (e.g., AAPL for Apple Inc.).
            start_date (str): Start date for historical data (YYYY-MM-DD).
            end_date (str): End date for historical data (YYYY-MM-DD).

        Returns:
            dict: A dictionary containing technical indicators (e.g., 'SMA', 'RSI', 'MACD').
        """
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        close_prices = stock_data['Close']

        sma_50 = ta.sma(close_prices, length=50)
        rsi_14 = ta.rsi(close_prices, length=14)
        macd, signal, _ = ta.macd(close_prices)

        technical_indicators = {
            'SMA_50': sma_50.iloc[-1],
            'RSI_14': rsi_14.iloc[-1],
            'MACD': macd[-1] - signal[-1]
        }

        return technical_indicators

    def analyze_forex_sentiment(self, forex_text):
        """
        Analyze sentiment for forex-related text.

        Args:
            forex_text (str): Text related to forex markets.

        Returns:
            float: Sentiment polarity score (-1.0 to 1.0).
        """
        blob = TextBlob(forex_text)
        sentiment_score = blob.sentiment.polarity
        return sentiment_score


# # Example usage:
# analyzer = MarketAnalyzer()
# crypto_text = "Bitcoin is showing strong growth in the market."
# crypto_sentiment = analyzer.analyze_crypto_sentiment(crypto_text)
# print("Crypto Sentiment:", crypto_sentiment)

# stock_symbol = "AAPL"
# start_date = "2022-01-01"
# end_date = "2022-12-31"
# stock_technicals = analyzer.analyze_stock_technicals(
#     stock_symbol, start_date, end_date)
# print("Stock Technicals:", stock_technicals)

# forex_text = "The USD is weakening against the Euro."
# forex_sentiment = analyzer.analyze_forex_sentiment(forex_text)
# print("Forex Sentiment:", forex_sentiment)
