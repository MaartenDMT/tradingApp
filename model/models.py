import logging
import os
import pickle
import threading
import traceback

import ccxt
import pandas as pd
import psycopg2
from dotenv import load_dotenv

import util.loggers as loggers
from model.autobot import AutoBot
from model.features import Tradex_indicator
from model.machinelearning import MachineLearning
from model.trading import Trading

# Set the path to the .env file
dotenv_path = r'.env'
# Load the environment variables from the .env file located two directories above
load_dotenv(dotenv_path)

logger = loggers.setup_loggers()


class Models:
    def __init__(self) -> None:
        self.model_logger = logger['model']

        # create a handler to log to the Listbox widget
        self.login_model = LoginModel()

    def create_tabmodels(self, presenter) -> None:
        self._presenter = presenter
        self.model_logger.info("Starting loading the models")
        self.mainview_model = MainviewModel(self.model_logger, self._presenter)
        self.tradetab_model = TradeTabModel(self.model_logger, self._presenter)
        self.exchangetab_model = ExchangeTabModel(
            self.model_logger, self._presenter)
        self.bottab_model = BotTabModel(self.model_logger, self._presenter)
        self.charttab_model = ChartTabModel(self.model_logger, self._presenter)
        self.model_logger.info("Finished loading the models")

    def get_ML(self) -> MachineLearning:
        ml = MachineLearning(self.tradetab_model.exchange,
                             self.tradetab_model.symbol, self.model_logger)
        return ml

    def get_exchange(self):
        try:
            # Create an instance of the exchange using the CCXT library
            exchange = ccxt.binance({
                # 'options': {
                #     'adjustForTimeDifference': True,
                # },
                'apiKey': os.environ.get('API_KEY_BIN_TEST'),
                'secret': os.environ.get('API_SECRET_BIN_TEST'),
                'enableRateLimit': True,
            })
            exchange.set_sandbox_mode(True)
            # self.model_logger.info(exchange.fetch_free_balance())
            if not exchange.check_required_credentials():
                self.model_logger.info('THERE ARE NOT CREDENTIALS')
        except Exception as e:
            self.model_logger.error(
                f"Error creating exchange {exchange} {e}\n{traceback.format_exc()}")

        return exchange


class LoginModel:
    def __init__(self) -> None:
        self._username = None
        self._password = None
        self._table_name = "User"

    def set_credentials(self, username: str, password: str) -> None:
        self._username = username
        self._password = password

    def check_credentials(self) -> bool:
        self.connect()
        try:
            # connect to the postgresql database
            self.cursor.execute(
                f"SELECT * FROM \"{self._table_name}\" WHERE username = '{self._username}' AND password = '{self._password}'")
            user = self.cursor.fetchone()
        except psycopg2.errors.UniqueViolation as e:
            # Display an error message if a unique constraint violation occurs
            return False

         # If the connection is successful, close the login screen and open the TradingApp window
        if user[1] == self._username and user[2] == self._password:
            self.user = user[1]
            self.conn.close()
            return True
        return False

    def register(self) -> bool:
        self.connect()
        # Retrieve the username and password from the entry fields
        try:
            # Connect to the database and insert the new record
            self.cursor.execute(
                f"INSERT INTO \"{self._table_name}\" (username, password) VALUES ('{self._username}', '{self._password}')")
            self.conn.commit()
        except psycopg2.errors.UniqueViolation as e:
            # code to handle the exception
            self.conn.close()
            return False

        self.conn.close()
        return True

    def connect(self) -> None:
        try:
            # Connect to the database using the provided username and password
            self.conn = psycopg2.connect(
                host=os.environ.get('PGHOST'),
                port=os.environ.get('PGPORT'),
                dbname=os.environ.get('PGDATABASE'),
                user=os.environ.get('PGUSER'),
                password=os.environ.get('PGPASSWORD')
            )
            self.cursor = self.conn.cursor()
        except Exception as e:
            print(e)


class MainviewModel:
    def __init__(self, model_logger, presenter) -> None:
        self.model_logger = model_logger
        self.presenter = presenter
        self.model_logger.info("loading the Main view model")


class TradeTabModel:
    def __init__(self, model_logger, presenter) -> None:
        self.model_logger = model_logger
        self.presenter = presenter
        self.model_logger.info("loading the Trade tab model")
        self._trading = self.get_trading()

    def get_trading(self, symbol='BTC/USDT') -> Trading:

        self.exchange = self.presenter.get_exchange()
        # self.exchange.set_sandbox_mode(True)
        self.symbol = symbol
        self.position_size = 10
        self.max_drawdown = 0.001
        self.moving_average_period = 50
        self.decision_tree_depth = 2

        trading = Trading(self.exchange, self.symbol, self.position_size, self.max_drawdown,
                          self.moving_average_period, self.decision_tree_depth)
        return trading

    def get_balance(self):
        return self._trading.getBalance()

    def update_stoploss(self, stop_loss) -> None:

        # Modify the stop loss level of the trade
        try:
            self._trading.exchange.update_order(
                self._trading.trade_id, {'stopLoss': stop_loss})
            self.model_logger.info(
                f"Trade ID: {self._trading.trade_id} Modify Stoploss {stop_loss} ")
        except Exception as e:
            self.model_logger.error(f"in the modify_stop_loss Error: {e}")

    def update_takeprofit(self, take_profit) -> None:

        # Modify the take profit level of the trade
        try:
            self._trading.exchange.update_order(
                self._trading.trade_id, {'takeProfit': take_profit})
            self.model_logger.info(
                f"Trade ID: {self._trading.trade_id} Modify TakeProfit {take_profit} ")
        except Exception as e:
            self.model_logger.error(f"in the modify_take_profit Error: {e}")

    def place_trade(self, trade_type, amount, price, stoploss, takeprofit, file) -> None:
        ml = self.presenter.get_ml()
        model = ml.load_model(file)

        # Make predictions using the scaled test data
        y_pred = ml.predict(model)

        # check and place the trade using the specified parameters
        self._trading.check_and_place_trade(
            trade_type, amount, price, takeprofit, stoploss, y_pred)


class ExchangeTabModel:
    def __init__(self, model_logger, presenter) -> None:
        self.model_logger = model_logger
        self._presenter = presenter
        self.model_logger.info("loading the Exchange tab model")
        self.exchange = []

    def set_first_exchange(self):
        exchange = getattr(ccxt, "phemex")({
            'apiKey': os.environ.get('API_KEY_PHE_TEST'),
            'secret': os.environ.get('API_SECRET_PHE_TEST'),
            'rateLimit': 2000,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            },
            'enableRateLimit': True})
        self.exchange.append(exchange)
        return exchange

    def create_exchange(self, exchange_name, api_key, api_secret):
        exchange = getattr(ccxt, exchange_name)({
            'apiKey': api_key,
            'secret': api_secret,
            'rateLimit': 2000,
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
            },
        })
        self.exchange.append(exchange)
        self.model_logger.info(f'creating the exchange {exchange}')
        return exchange

    def get_exchange(self, index):
        return self.exchange[index]

    def remove_exchange(self, index) -> None:
        self.model_logger.info(f'Removing exchange {self.exchange[index]}')
        del self.exchange[index]


class BotTabModel:
    def __init__(self, model_logger, presenter) -> None:
        self.model_logger = model_logger
        self._presenter = presenter
        self.model_logger.info("loading the Bot tab model")
        self.bots = []
        self.auto_trade_threads = []
        self.stop_event_trade = threading.Event()

    def get_data_ml_files(self) -> list:
        path = r'data/ml/'
        # Get a list of files in the directory
        files = os.listdir(path)

        return files

    def start_bot(self, index: int) -> bool:
        print(f"the index is {index}")

        if index < 0 or index >= len(self.bots):
            self.model_logger.error(f"Index {index} is out of bounds.")
            return False

        if index >= len(self.auto_trade_threads):
            self.model_logger.info(f"Creating a new thread for index {index}")
            self.stop_event_trade.clear()
            thread = threading.Thread(
                target=self.bots[index].start_auto_trading, args=(self.stop_event_trade,))
            self.auto_trade_threads.append(thread)
        else:
            self.auto_trade_threads[index] = threading.Thread(
                target=self.bots[index].start_auto_trading, args=(self.stop_event_trade,))

        if not self.auto_trade_threads[index].is_alive():
            self.auto_trade_threads[index].setDaemon(True)
            self.auto_trade_threads[index].start()

        self.bots[index].auto_trade = True
        self.model_logger.info(
            f"Starting a auto trading bot {self.bots[index]}")
        return True

    def stop_bot(self, index: int) -> bool:
        if index <= len(self.auto_trade_threads) and self.auto_trade_threads[index].is_alive():
            self.stop_event_trade.set()
            self.auto_trade_threads[index].join()
            self.bots[index].auto_trade = False
            self.model_logger.info(
                f"Stopping the auto trading bot called: {self.bots[index]}")
            return True
        else:
            self.model_logger.error(f"there is no bot to stop! ")
            return False

    def create_bot(self) -> None:
        bot = self._presenter.get_auto_bot()
        self.bots.append(bot)
        print(f"name of the bot: {self.bots.__iter__}")
        self.model_logger.info(f"Creating a auto trading bot {bot}")

    def destroy_bot(self, index: int) -> bool:
        if index <= len(self.bots):
            self.stop_bot(index)
            del self.auto_trade_threads[index]
            del self.bots[index]
            self.model_logger.info(
                f"Destroying the auto trading bot called: {self.bots[index]}")
            return True
        else:
            self.model_logger.error(f"there is no bot to destroy! ")
            return False

    def get_autobot(self, exchange, symbol, amount, stop_loss, take_profit, file, time):

        ml = MachineLearning(exchange, symbol, self.model_logger)
        model = ml.load_model(file)

        with open(f'data/pickle/{time}.p', 'rb') as f:
            df = pickle.load(f)
        df = df.dropna()
        trade_x = Tradex_indicator(
            symbol=symbol, timeframe=time, t=None, get_data=True, data=df.copy())
        # # Make predictions using the scaled test data
        # y_pred = ml.predict(model)
        autobot = AutoBot(exchange, symbol, amount, stop_loss,
                          take_profit, model, time, ml, trade_x, self.model_logger)
        # self._presenter.save_autobot(autobot)
        return autobot


class ChartTabModel:
    def __init__(self, model_logger, presenter) -> None:
        self.model_logger = model_logger
        self.presenter = presenter
        self.model_logger.info("loading the Chart tab model")
        self.stop_event_chart = threading.Event()
        self.auto_chart = False

    def toggle_auto_charting(self) -> bool:

        self.bot = self.presenter.get_bot()
        # Toggle the automatic trading flag
        self.auto_chart = not self.auto_chart

        # Update the button text
        if self.auto_chart:

            self.model_logger.info(
                f"Starting auto charting with symbol {self.bot.symbol}")

            # Create a new thread object
            self.auto_update_chart = threading.Thread(
                target=self.presenter.update_chart, args=(self.stop_event_chart,))

            # Reset the stop event
            self.stop_event_chart.clear()

            # Start the thread
            self.auto_update_chart.start()

            return True

        else:
            self.model_logger.info(
                f"Stopping auto charting with symbol {self.bot.symbol}")

            # Set the stop event to signal the thread to stop
            self.stop_event_chart.set()

            # Wait for the thread to finish
            self.auto_update_chart.join()

            return False

    def get_data(self) -> pd.DataFrame:
        # Get the ticker data for the symbol
        ticker = self.bot.exchange.fetch_ohlcv(
            self.bot.symbol, limit=20, timeframe='1m')

        df = pd.DataFrame(
            ticker, columns=['date', 'open', 'high', 'low', 'close', 'volume'])

        # Convert the 'Date' column to a DatetimeIndex
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df.set_index(df['date'], inplace=True)
        self.model_logger.info(
            f"getting data to plot the chart symbol {self.bot.symbol}")

        return df
