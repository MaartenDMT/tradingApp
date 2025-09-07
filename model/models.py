import os
import threading
import traceback
from time import sleep

import ccxt
import pandas as pd
import psycopg2
import psycopg2.pool
from dotenv import load_dotenv

import util.loggers as loggers
from model.features import Tradex_indicator
from model.machinelearning.autobot import AutoBot
from model.machinelearning.machinelearning import MachineLearning
from model.manualtrading.trading import Trading
from model.reinforcement.agents.agent_manager import StablebaselineModel
from util.utils import load_config

config = load_config()

# Set the path to the .env file
dotenv_path = r'.env'
# Load the environment variables from the .env file located two directories above
load_dotenv(dotenv_path)

logger = loggers.setup_loggers()

# Create a connection pool for better database performance
db_pool = None

def initialize_db_pool():
    """Initialize the database connection pool"""
    global db_pool
    try:
        db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=os.environ.get('PGHOST'),
            port=os.environ.get('PGPORT'),
            dbname=os.environ.get('PGDATABASE'),
            user=os.environ.get('PGUSER'),
            password=os.environ.get('PGPASSWORD')
        )
        logger['model'].info("Database connection pool created successfully")
        return True
    except Exception as e:
        logger['model'].error(f"Error creating database connection pool: {e}")
        db_pool = None
        return False

# Initialize the database pool when the module is loaded
initialize_db_pool()


class Models:
    def __init__(self) -> None:
        self.model_logger = logger['model']
        self.model_logger.info("Initializing Models class")

        # create a handler to log to the Listbox widget
        self.login_model = LoginModel()
        self.model_logger.info("LoginModel initialized successfully")

    def create_tabmodels(self, presenter) -> None:
        self._presenter = presenter
        self.model_logger.info("Starting loading the models")
        # Lazy initialization - models will be created only when accessed
        self._mainview_model = None
        self._tradetab_model = None
        self._exchangetab_model = None
        self._bottab_model = None
        self._charttab_model = None
        self._rltab_model = None
        self.model_logger.info("Finished setting up model placeholders")

    def _ensure_mainview_model(self):
        if self._mainview_model is None:
            self.model_logger.debug("Initializing MainviewModel")
            self._mainview_model = MainviewModel(self._presenter)

    def _ensure_tradetab_model(self):
        if self._tradetab_model is None:
            self.model_logger.debug("Initializing TradeTabModel")
            self._tradetab_model = TradeTabModel(self._presenter)

    def _ensure_exchangetab_model(self):
        if self._exchangetab_model is None:
            self.model_logger.debug("Initializing ExchangeTabModel")
            self._exchangetab_model = ExchangeTabModel(self._presenter)

    def _ensure_bottab_model(self):
        if self._bottab_model is None:
            self.model_logger.debug("Initializing BotTabModel")
            self._bottab_model = BotTabModel(self._presenter)

    def _ensure_charttab_model(self):
        if self._charttab_model is None:
            self.model_logger.debug("Initializing ChartTabModel")
            self._charttab_model = ChartTabModel(self._presenter)

    def _ensure_rltab_model(self):
        if self._rltab_model is None:
            self.model_logger.debug("Initializing ReinforcementTabModel")
            self._rltab_model = ReinforcementTabModel(self._presenter)

    def get_ML(self) -> MachineLearning:
        self._ensure_tradetab_model()
        self.model_logger.debug("Creating MachineLearning instance")
        ml = MachineLearning(self._tradetab_model.exchange,
                             self._tradetab_model.symbol)
        return ml

    # Properties to ensure lazy initialization
    @property
    def mainview_model(self):
        self._ensure_mainview_model()
        return self._mainview_model

    @property
    def tradetab_model(self):
        self._ensure_tradetab_model()
        return self._tradetab_model

    @property
    def exchangetab_model(self):
        self._ensure_exchangetab_model()
        return self._exchangetab_model

    @property
    def bottab_model(self):
        self._ensure_bottab_model()
        return self._bottab_model

    @property
    def charttab_model(self):
        self._ensure_charttab_model()
        return self._charttab_model

    @property
    def rltab_model(self):
        self._ensure_rltab_model()
        return self._rltab_model

    def get_exchange(self, exchange_name="phemex", api_key=None, api_secret=None, test_mode=True):
        try:
            self.model_logger.info(f"Creating exchange: {exchange_name}")
            exchange_class = getattr(ccxt, exchange_name)

            # Use provided keys or fallback to environment variables
            actual_api_key = api_key or os.environ.get(f'API_KEY_{exchange_name.upper()}_TEST')
            actual_api_secret = api_secret or os.environ.get(f'API_SECRET_{exchange_name.upper()}_TEST')

            # Log a masked version of the API key for security using the secure utility
            from util.secure_credentials import mask_sensitive_data
            masked_key = mask_sensitive_data(actual_api_key) if actual_api_key else 'None'
            self.model_logger.debug(f"Using API key (masked): {masked_key}")

            exchange = exchange_class({
                'apiKey': actual_api_key,
                'secret': actual_api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })

            if test_mode:
                exchange.set_sandbox_mode(True)
                self.model_logger.info(f"Sandbox mode enabled for {exchange_name}")

            self.model_logger.info(f"Exchange {exchange_name} created successfully")
            return exchange
        except Exception as e:
            from util.error_handling import handle_exception
            return handle_exception(self.model_logger, f"creating exchange {exchange_name}", e,
                                  rethrow=False, default_return=None)


class LoginModel:
    def __init__(self) -> None:
        self._username = None
        self._password = None
        self._table_name = "User"
        self.logged_in = False  # Initialize the logged_in attribute
        self.model_logger = logger['model']
        self.model_logger.info("LoginModel initialized")

    def set_credentials(self, username: str, password: str) -> None:
        self.model_logger.debug(f"Setting credentials for user: {username}")
        self._username = username
        self._password = password

    def _execute_query_with_pool(self, query, params=None):
        """Execute a database query using the connection pool"""
        if not db_pool:
            raise Exception("Database connection pool is not available")

        conn = None
        cursor = None
        try:
            conn = db_pool.getconn()
            cursor = conn.cursor()
            self.model_logger.debug("Using connection from pool")

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            return cursor
        except Exception as e:
            # If there's an error, make sure to return the connection to the pool
            if conn:
                db_pool.putconn(conn)
            raise e

    def _execute_query_with_direct_connection(self, query, params=None):
        """Execute a database query using a direct connection"""
        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(
                host=os.environ.get('PGHOST'),
                port=os.environ.get('PGPORT'),
                dbname=os.environ.get('PGDATABASE'),
                user=os.environ.get('PGUSER'),
                password=os.environ.get('PGPASSWORD')
            )
            cursor = conn.cursor()
            self.model_logger.debug("Direct database connection established")

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            return conn, cursor
        except Exception as e:
            # If there's an error, make sure to close the connection
            if cursor:
                cursor.close()
            if conn:
                conn.close()
            raise e

    def check_credentials(self) -> bool:
        # For development purposes, allow automatic login with test credentials
        # without requiring a database connection
        if self._username == 'test' and self._password == 't':
            self.user = self._username
            self.model_logger.info(f"User {self._username} authenticated successfully (dev mode)")
            return True

        try:
            self.model_logger.info(f"Checking credentials for user: {self._username}")

            query = f"SELECT * FROM \"{self._table_name}\" WHERE username = %s AND password = %s"
            params = (self._username, self._password)

            if db_pool:
                # Use connection pool
                cursor = self._execute_query_with_pool(query, params)
                user = cursor.fetchone()

                # Return connection to pool
                cursor.close()
                db_pool.putconn(cursor.connection)
                self.model_logger.debug("Connection returned to pool")
            else:
                # Use direct connection
                conn, cursor = self._execute_query_with_direct_connection(query, params)
                user = cursor.fetchone()

                # Close direct connection
                cursor.close()
                conn.close()
                self.model_logger.debug("Direct connection closed")

            # Check if user exists and credentials match
            if user and user[1] == self._username and user[2] == self._password:
                self.user = user[1]
                self.model_logger.info(f"User {self._username} authenticated successfully")
                return True
            else:
                self.model_logger.warning(f"Authentication failed for user: {self._username}")
                return False

        except psycopg2.errors.UniqueViolation as e:
            self.model_logger.error(f"Unique violation error: {e}")
            return False
        except Exception as e:
            self.model_logger.error(f"Error checking credentials: {e}")
            self.model_logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def register(self) -> bool:
        try:
            self.model_logger.info(f"Registering new user: {self._username}")

            query = f"INSERT INTO \"{self._table_name}\" (username, password) VALUES (%s, %s)"
            params = (self._username, self._password)

            if db_pool:
                # Use connection pool
                cursor = self._execute_query_with_pool(query, params)
                cursor.connection.commit()

                # Return connection to pool
                cursor.close()
                db_pool.putconn(cursor.connection)
                self.model_logger.debug("Connection returned to pool")
            else:
                # Use direct connection
                conn, cursor = self._execute_query_with_direct_connection(query, params)
                conn.commit()

                # Close direct connection
                cursor.close()
                conn.close()
                self.model_logger.debug("Direct connection closed")

            self.model_logger.info(f"User {self._username} registered successfully")
            return True

        except psycopg2.errors.UniqueViolation as e:
            self.model_logger.error(f"User already exists: {e}")
            return False
        except Exception as e:
            self.model_logger.error(f"Error registering user: {e}")
            self.model_logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def set_logged_in(self, value):
        self.logged_in = value
        self.model_logger.debug(f"Logged in status set to: {value}")

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
        self._trading = None
        self.exchange = None
        self.symbol = 'BTC/USD:USD'
        self.trade_type = 'swap'

    def get_trading(self, symbol='BTC/USD:USD', trade_type='swap') -> Trading:
        # Only create the trading object when it's actually needed
        if self._trading is None or self.symbol != symbol or self.trade_type != trade_type:
            self.exchange = self.presenter.get_exchange()
            self.symbol = symbol
            self.trade_type = trade_type
            self._trading = Trading(self.exchange, self.symbol, self.trade_type)
        return self._trading

    def _ensure_trading_object(self):
        """Ensure the trading object is initialized before using it"""
        if self._trading is None:
            self._trading = self.get_trading(self.symbol, self.trade_type)

    # Trading actions
    def place_trade(self, symbol, side, trade_type, amount, price, stoploss, takeprofit):
        # Validate inputs
        from util.error_handling import handle_exception
        from util.validation import validate_number, validate_symbol

        if not validate_symbol(symbol):
            handle_exception(self.model_logger, "placing trade",
                           ValueError(f"Invalid symbol: {symbol}"),
                           rethrow=True)

        if side not in ['buy', 'sell']:
            handle_exception(self.model_logger, "placing trade",
                           ValueError(f"Invalid side: {side}"),
                           rethrow=True)

        if not validate_number(amount, min_value=0):
            handle_exception(self.model_logger, "placing trade",
                           ValueError(f"Invalid amount: {amount}"),
                           rethrow=True)

        if price is not None and not validate_number(price, min_value=0):
            handle_exception(self.model_logger, "placing trade",
                           ValueError(f"Invalid price: {price}"),
                           rethrow=True)

        if stoploss is not None and not validate_number(stoploss, min_value=0):
            handle_exception(self.model_logger, "placing trade",
                           ValueError(f"Invalid stoploss: {stoploss}"),
                           rethrow=True)

        if takeprofit is not None and not validate_number(takeprofit, min_value=0):
            handle_exception(self.model_logger, "placing trade",
                           ValueError(f"Invalid takeprofit: {takeprofit}"),
                           rethrow=True)

        self._ensure_trading_object()
        self._trading.place_trade(
            symbol, side, trade_type, amount, price, stoploss, takeprofit)
        # Wait for trade execution before fetching open trades
        sleep(1)  # Reduced from 3 seconds to 1 second
        self._trading.fetch_open_trades(symbol)

    def update_stoploss_takeprofit(self, symbol, stop_loss=None, take_profit=None):
        self._ensure_trading_object()
        open_trades = self._trading.fetch_open_trades(symbol)
        for trade in open_trades:
            if trade['status'] == 'open':
                self._trading.modify_takeprofit_stoploss(
                    trade['id'], take_profit, stop_loss)
                self.model_logger.info(
                    f"Updated SL/TP for Trade ID: {trade['id']}")

    def scale_in_out(self, amount):
        self._ensure_trading_object()
        self._trading.scale_in_out(amount)

    # Market data
    def get_market_data(self, data_type, depth=5):
        self._ensure_trading_object()
        return self._trading.fetch_market_data(data_type, depth)

    def get_ticker_price(self):
        self._ensure_trading_object()
        return self.get_market_data('ticker')['last_price']

    def set_symbol(self, symbol):
        self._ensure_trading_object()
        self._trading.set_symbol(symbol)

    # Real-time updates
    def start_stop_real_time_updates(self, start=True):
        self._ensure_trading_object()
        if start:
            self._trading.start_real_time_updates()
        else:
            self._trading.stop_real_time_updates()

    def get_real_time_data(self):
        self._ensure_trading_object()
        return self._trading.get_real_time_data()

    # Settings and configuration
    def update_settings(self, settings):
        self._ensure_trading_object()
        self._trading.update_settings(settings)

    # Analysis and calculations
    def get_position_info(self):
        self._ensure_trading_object()
        return self._trading.get_position_info()

    def execute_advanced_orders(self, symbol, total_amount, duration, side, order_type):
        self._ensure_trading_object()
        if order_type == 'twap':
            self._trading.execute_twap_order(
                symbol, total_amount, duration, side)
        elif order_type == 'dynamic_stop_loss':
            entry_price, current_price, initial_stop_loss, trailing_percent = self.get_dynamic_stop_loss_params()
            self._trading.update_dynamic_stop_loss(
                symbol, entry_price, current_price, initial_stop_loss, trailing_percent)

    def calculate_financial_metrics(self, metric_type, **kwargs):
        self._ensure_trading_object()
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
        self._ensure_trading_object()
        if fetch_type == 'open_trades':
            return self._trading.fetch_open_trades(kwargs['symbol'])
        elif fetch_type == 'historical_data':
            return self._trading.fetch_historical_data(kwargs['symbol'], kwargs['timeframe'], kwargs['since'], kwargs['limit'])
        elif fetch_type == 'transfer_funds':
            return self._trading.transfer_funds(kwargs['amount'], kwargs['currency_code'], kwargs['from_account_type'], kwargs['to_account_type'])

    # Helper methods for specific calculations
    def calculate_liquidation_price(self, leverage, balance, position_size):
        self._ensure_trading_object()
        return self._trading.calculate_liquidation_price(leverage, balance, position_size)

    def check_risk_limits(self, position_size):
        self._ensure_trading_object()
        return self._trading.check_risk_limits(position_size)

    def calculate_order_cost(self, amount, price):
        self._ensure_trading_object()
        return self._trading.calculate_order_cost(amount, price)

    def get_funding_rate(self):
        self._ensure_trading_object()
        return self._trading.get_funding_rate()

    def get_tick_size(self):
        self._ensure_trading_object()
        return self._trading.get_tick_size()

    def contract_to_underlying(self, contract_quantity):
        self._ensure_trading_object()
        return self._trading.contract_to_underlying(contract_quantity)

    def get_balance(self):
        self._ensure_trading_object()
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
        try:
            # Check cache first
            from util.cache import get_cache
            cache = get_cache()
            cached_files = cache.get("ml_files_list")
            if cached_files is not None:
                self.model_logger.debug("Returning ML files list from cache")
                return cached_files

            path = r'data/ml/2020'
            # Check if directory exists before accessing
            if os.path.exists(path):
                # Get a list of files in the directory
                files = os.listdir(path)
                # Cache the result for 60 seconds
                cache.set("ml_files_list", files, ttl=60)
                return files
            else:
                self.model_logger.warning(f"ML data directory not found: {path}")
                return []
        except Exception as e:
            self.model_logger.error(f"Error accessing ML files: {e}")
            return []

    def start_bot(self, index: int) -> bool:
        self.model_logger.info(f"the index is {index}")

        if index < 0 or index >= len(self.bots):
            self.model_logger.error(f"Index {index} is out of bounds.")
            return False

        if index >= len(self.auto_trade_threads):
            self.model_logger.info(f"Creating a new thread for index {index}")
            self.stop_event_trade.clear()
            thread = threading.Thread(
                target=self.bots[index].start_auto_trading, args=(self.stop_event_trade,), daemon=True)
            self.auto_trade_threads.append(thread)
        else:
            self.auto_trade_threads[index] = threading.Thread(
                target=self.bots[index].start_auto_trading, args=(self.stop_event_trade,), daemon=True)

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
            self.model_logger.error("there is no bot to stop! ")
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
            self.model_logger.error("there is no bot to destroy! ")
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
                target=self.presenter.update_chart, args=(self.stop_event_chart,), daemon=True)

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
        # Check cache first (cache for 10 seconds since this is chart data)
        from util.cache import get_cache
        cache = get_cache()
        cache_key = f"chart_data_{self.bot.symbol}"
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            self.model_logger.debug(f"Returning chart data for {self.bot.symbol} from cache")
            return cached_data

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

        # Cache the result for 10 seconds
        cache.set(cache_key, df, ttl=10)

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
                target=self.train_and_evaluate, args=(self.params, self.rl_logger,), daemon=True)
            evaluation_thread.start()
        except Exception as e:
            self.model_logger.error(f"{e}\n{traceback.format_exc()}")
