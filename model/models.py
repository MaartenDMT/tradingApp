import os
import pickle
import threading
import traceback
from time import sleep

import ccxt
import pandas as pd
import psycopg2
from dotenv import load_dotenv

import util.loggers as loggers
from model.features import Tradex_indicator
from model.machinelearning.autobot import AutoBot
from model.machinelearning.machinelearning import MachineLearning
from model.manualtrading.trading import Trading
from model.reinforcement.rl_models import TensorflowModel, TorchModel, StablebaselineModel
from util.utils import load_config

config = load_config()

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
        self.mainview_model = MainviewModel(self._presenter)
        self.tradetab_model = TradeTabModel(self._presenter)
        self.exchangetab_model = ExchangeTabModel(self._presenter)
        self.bottab_model = BotTabModel(self._presenter)
        self.charttab_model = ChartTabModel(self._presenter)
        self.rltab_model = ReinforcementTabModel(self._presenter)
        self.model_logger.info("Finished loading the models")

    def get_ML(self) -> MachineLearning:
        ml = MachineLearning(self.tradetab_model.exchange,
                             self.tradetab_model.symbol)
        return ml

    def get_exchange(self, exchange_name="phemex", api_key=None, api_secret=None, test_mode=True):
        try:
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({
                'apiKey': api_key or os.environ.get(f'API_KEY_{exchange_name.upper()}_TEST'),
                'secret': api_secret or os.environ.get(f'API_SECRET_{exchange_name.upper()}_TEST'),
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })

            if test_mode:
                exchange.set_sandbox_mode(True)

            return exchange
        except Exception as e:
            self.model_logger.error(f"Error creating exchange: {e}")
            return None


class LoginModel:
    def __init__(self) -> None:
        self._username = None
        self._password = None
        self._table_name = "User"
        self.logged_in = False  # Initialize the logged_in attribute

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
            self.model_logger.error(e)

    def set_logged_in(self, value):
        self.logged_in = value

    def get_logged_in_status(self):
        return self.logged_in


class MainviewModel:
    def __init__(self, presenter) -> None:
        self.model_logger = logger['model']
        self.presenter = presenter
        self.model_logger.info("loading the Main view model")


class TradeTabModel:
    def __init__(self, presenter) -> None:
        self.model_logger = logger['model']
        self.presenter = presenter
        self.model_logger.info("Loading the Trade tab model")
        self._trading = self.get_trading()

    def get_trading(self, symbol='BTC/USD:USD', trade_type='swap') -> Trading:
        self.exchange = self.presenter.get_exchange()
        self.symbol = symbol
        self.trade_type = trade_type
        trading = Trading(self.exchange, self.symbol, self.trade_type)
        return trading

    # Trading actions
    def place_trade(self, symbol, side, trade_type, amount, price, stoploss, takeprofit):
        self._trading.place_trade(
            symbol, side, trade_type, amount, price, stoploss, takeprofit)
        sleep(3)
        self._trading.fetch_open_trades(symbol)

    def update_stoploss_takeprofit(self, symbol, stop_loss=None, take_profit=None):
        open_trades = self._trading.fetch_open_trades(symbol)
        for trade in open_trades:
            if trade['status'] == 'open':
                self._trading.modify_takeprofit_stoploss(
                    trade['id'], take_profit, stop_loss)
                self.model_logger.info(
                    f"Updated SL/TP for Trade ID: {trade['id']}")

    def scale_in_out(self, amount):
        self._trading.scale_in_out(amount)

    # Market data
    def get_market_data(self, data_type, depth=5):
        return self._trading.fetch_market_data(data_type, depth)

    def get_ticker_price(self):
        return self.get_market_data('ticker')['last_price']

    def set_symbol(self, symbol):
        self._trading.set_symbol(symbol)

    # Real-time updates
    def start_stop_real_time_updates(self, start=True):
        if start:
            self._trading.start_real_time_updates()
        else:
            self._trading.stop_real_time_updates()

    def get_real_time_data(self):
        return self._trading.get_real_time_data()

    # Settings and configuration
    def update_settings(self, settings):
        self._trading.update_settings(settings)

    # Analysis and calculations
    def get_position_info(self):
        return self._trading.get_position_info()

    def execute_advanced_orders(self, symbol, total_amount, duration, side, order_type):
        if order_type == 'twap':
            self._trading.execute_twap_order(
                symbol, total_amount, duration, side)
        elif order_type == 'dynamic_stop_loss':
            entry_price, current_price, initial_stop_loss, trailing_percent = self.get_dynamic_stop_loss_params()
            self._trading.update_dynamic_stop_loss(
                symbol, entry_price, current_price, initial_stop_loss, trailing_percent)

    def calculate_financial_metrics(self, metric_type, **kwargs):
        if metric_type == 'var':
            return self._trading.calculate_var(kwargs['portfolio'], kwargs['confidence_level'])
        elif metric_type == 'drawdown':
            return self._trading.calculate_drawdown(kwargs['peak_balance'], kwargs['current_balance'])
        elif metric_type == 'pnl':
            return self._trading.calculate_pnl(kwargs['entry_price'], kwargs['exit_price'], kwargs['contract_quantity'], kwargs['is_long'])
        elif metric_type == 'breakeven_price':
            return self._trading.calculate_breakeven_price(kwargs['entry_price'], kwargs['fee_percent'], kwargs['contract_quantity'], kwargs['is_long'])

    # Data fetching and fund transfers
    def fetch_data_and_transfer_funds(self, fetch_type, **kwargs):
        if fetch_type == 'open_trades':
            return self._trading.fetch_open_trades(kwargs['symbol'])
        elif fetch_type == 'historical_data':
            return self._trading.fetch_historical_data(kwargs['symbol'], kwargs['timeframe'], kwargs['since'], kwargs['limit'])
        elif fetch_type == 'transfer_funds':
            return self._trading.transfer_funds(kwargs['amount'], kwargs['currency_code'], kwargs['from_account_type'], kwargs['to_account_type'])

    # Helper methods for specific calculations
    def calculate_liquidation_price(self, leverage, balance, position_size):
        return self._trading.calculate_liquidation_price(leverage, balance, position_size)

    def check_risk_limits(self, position_size):
        return self._trading.check_risk_limits(position_size)

    def calculate_order_cost(self, amount, price):
        return self._trading.calculate_order_cost(amount, price)

    def get_funding_rate(self):
        return self._trading.get_funding_rate()

    def get_tick_size(self):
        return self._trading.get_tick_size()

    def contract_to_underlying(self, contract_quantity):
        return self._trading.contract_to_underlying(contract_quantity)

    def get_balance(self):
        return self._trading.get_balance()


class ExchangeTabModel:
    def __init__(self, presenter) -> None:
        self.model_logger = logger['model']
        self._presenter = presenter
        self.exchanges = {}
        self.model_logger.info("loading the Exchange tab model")

    def set_first_exchange(self, test_mode=True):
        return self._presenter.get_exchange(test_mode=test_mode)

    def create_exchange(self, exchange_name, api_key, api_secret):
        exchange = self._presenter.get_exchange(
            exchange_name, api_key, api_secret)
        if exchange:
            self.exchanges[exchange_name] = exchange
            self.model_logger.info(f'Created exchange: {exchange_name}')
        return exchange

    def get_exchange(self, exchange_name):
        return self.exchanges.get(exchange_name)

    def remove_exchange(self, exchange_name) -> None:
        if exchange_name in self.exchanges:
            self.model_logger.info(f'Removing exchange: {exchange_name}')
            del self.exchanges[exchange_name]


class BotTabModel:
    def __init__(self, presenter) -> None:
        self.model_logger = logger['model']
        self._presenter = presenter
        self.model_logger.info("loading the Bot tab model")
        self.bots = []
        self.auto_trade_threads = []
        self.stop_event_trade = threading.Event()

    def get_data_ml_files(self) -> list:
        path = r'data/ml/2020'
        # Get a list of files in the directory
        files = os.listdir(path)

        return files

    def start_bot(self, index: int) -> bool:
        self.model_logger.info(f"the index is {index}")

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
        bot = self._presenter.bot_tab.get_auto_bot()
        self.bots.append(bot)
        self.model_logger.info(f"name of the bot: {self.bots.__iter__}")
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

        ml = MachineLearning(exchange, symbol)
        model = ml.load_model(file)

        # with open(f'data/pickle/{time}.p', 'rb') as f:
        #     df = pickle.load(f)

        df = pd.read_pickle(f'data/pickle/{time}.p')
        df = df.dropna()

        trade_x = Tradex_indicator(
            symbol=symbol, timeframe=time, t=None, get_data=False, data=df.copy())
        # # Make predictions using the scaled test data
        # y_pred = ml.predict(model)
        autobot = AutoBot(exchange, symbol, amount, stop_loss,
                          take_profit, model, time, ml, trade_x, df)
        # self._presenter.save_autobot(autobot)
        return autobot


class ChartTabModel:
    def __init__(self, presenter) -> None:
        self.model_logger = logger['model']
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


class ReinforcementTabModel:
    def __init__(self, presenter) -> None:
        self.model_logger = logger['model']
        self.rl_logger = logger['rl']
        self.presenter = presenter

        self.features = ['open', 'high', 'low',
                         'close', 'volume', 'portfolio_balance']
        self.result = None
        self.model_logger.info("loading the reinforcement tab model")
        # Define a parameter grid with hyperparameters and their possible values
        self.params = {
            'gamma': float(config['Params']['gamma']),
            'learning_rate': float(config['Params']['learning_rate']),
            'batch_size': int(config['Params']['batch_size']),
            'epsilon_min': float(config['Params']['epsilon_min']),
            'epsilon_decay': float(config['Params']['epsilon_decay']),
            'episodes': int(config['Params']['episodes']),
            'env_actions': int(config['Params']['env_actions']),
            'test_episodes': int(config['Params']['test_episodes']),
            'min_acc': float(config['Params']['min_acc']),
            'features': self.features
        }

    # Define a function for training and evaluating the DQL agent

    def train_and_evaluate(self, params, logger):
        StablebaselineModel(params, logger)

    def start(self):
        try:
            evaluation_thread = threading.Thread(
                target=self.train_and_evaluate, args=(self.params, self.rl_logger,))
            evaluation_thread.setDaemon(True)
            evaluation_thread.start()
        except Exception as e:
            self.model_logger.error(f"{e}\n{traceback.format_exc()}")
