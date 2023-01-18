import logging
import os
import pickle
import threading
from pathlib import Path

import ccxt
import pandas as pd
import psycopg2
from dotenv import load_dotenv

from model.autobot import AutoBot
from model.machinelearning import MachineLearning
from model.trading import Trading
from model.features import Tradex_indicator

# Set the path to the .env file
dotenv_path = r'.env'
# Load the environment variables from the .env file located two directories above
load_dotenv(dotenv_path)

class Models:
    def __init__(self) -> None:
        
        # create a logger for the application & extra tools
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # create a handler to log to the Listbox widget
        self.add_log_info()
        self.login_model = LoginModel()
                
    def create_tabmodels(self, presenter) -> None:
        self._presenter = presenter
        self.mainview_model = MainviewModel()
        self.tradetab_model = TradeTabModel(self.logger, self._presenter)
        self.exchangetab_model = ExchangeTabModel()
        self.bottab_model = BotTabModel(self.logger, self._presenter)
        self.charttab_model = ChartTabModel(self.logger, self._presenter)
    

    def add_log_info(self) -> None:
        file = r'data/logs/'
        file_handler = logging.FileHandler(f'{file}logg-info.log')
        # create a stream handler to log to the console
        stream_handler = logging.StreamHandler()
        # create a formatter for the log messages
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # add the formatter to the handlers
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        
    def get_ML(self) -> MachineLearning:
        ml = MachineLearning(self.tradetab_model.exchange, self.tradetab_model.symbol, self.logger)
        return ml
    
    def get_exchange(self):
        # Create an instance of the exchange using the CCXT library
        exchange =  getattr(ccxt, "phemex")({'apiKey': os.environ.get('API_KEY_PHE_TEST'),
                        'secret': os.environ.get('API_SECRET_PHE_TEST'),
                        'rateLimit': 2000,
                        'enableRateLimit': True})
        if not exchange.check_required_credentials():
            print('THERE ARE NOT CREDENTIALS')
            
        return exchange
    
    
    
class LoginModel:
    def __init__(self):
        self._username = None
        self._password = None
        self._table_name = "User"

    def set_credentials(self, username: str, password: str):
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
    
    def register(self) ->bool:
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
    
    def connect(self) ->None:
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
    def __init__(self):
        pass

class TradeTabModel:
    def __init__(self, logger, presenter):
        self.logger = logger
        self.presenter = presenter
        self._trading = self.get_trading()
    
    def get_trading(self, symbol='BTC/USDT') -> Trading:

        self.exchange = self.presenter.get_exchange()      
        self.exchange.set_sandbox_mode(True)
        self.symbol = symbol
        self.position_size = 10
        self.max_drawdown = 0.001
        self.moving_average_period = 50
        self.decision_tree_depth = 2

        trading = Trading(self.exchange, self.symbol, self.position_size, self.max_drawdown,
                         self.moving_average_period, self.decision_tree_depth, self.logger)
        return trading
    
    
    def update_stoploss(self, stop_loss) -> None:

        # Modify the stop loss level of the trade
        try:
            self._trading.exchange.update_order(
                self._trading.trade_id, {'stopLoss': stop_loss})
            self.logger.info(
                f"Trade ID: {self._trading.trade_id} Modify Stoploss {stop_loss} ")
        except Exception as e:
            self.logger.error(f"in the modify_stop_loss Error: {e}")

    def update_takeprofit(self,take_profit) -> None:

        # Modify the take profit level of the trade
        try:
            self._trading.exchange.update_order(
                self._trading.trade_id, {'takeProfit': take_profit})
            self.logger.info(
                f"Trade ID: {self._trading.trade_id} Modify TakeProfit {take_profit} ")
        except Exception as e:
            self.logger.error(f"in the modify_take_profit Error: {e}")
            
    def place_trade(self, trade_type, amount,price, stoploss, takeprofit, file) -> None:
        ml = self.presenter.get_ml()
        model = ml.load_model(file)

        # Make predictions using the scaled test data
        y_pred = ml.predict(model)
        
        # check and place the trade using the specified parameters
        self._trading.check_and_place_trade(
            trade_type, amount, price, takeprofit, stoploss, y_pred)

class ExchangeTabModel:
    def __init__(self):
        pass
    
    def get_exchange(self):
        pass

class BotTabModel:
    def __init__(self, logger, presenter):
        self.logger = logger
        self._presenter = presenter
        self.bots = []
        self.auto_trade_threads = []
        self.stop_event_trade = threading.Event()
    
    def get_data_ml_files(self) -> list:
        path = r'data/ml/'
        # Get a list of files in the directory
        files = os.listdir(path)
        
        return files
    
    def start_bot(self, index: int) -> None:
        if len(self.auto_trade_threads) <= index:
            self.logger.info(f"no bot detected, creating a bot")
            self.bots.append(self._presenter.get_auto_bot())
            self.auto_trade_threads.append(threading.Thread(target=self.bots[-1].start_auto_trading, args=(self.stop_event_trade,)))
            self.stop_event_trade.clear()
            self.auto_trade_threads[-1].setDaemon(True)
            self.auto_trade_threads[-1].start()
            self.bots[-1].auto_trade = True

        elif self.auto_trade_threads[index] and not self.auto_trade_threads[index].is_alive():
            self.stop_event_trade.clear()
            self.auto_trade_threads[index] = threading.Thread(target=self.bots[index].start_auto_trading, args=(self.stop_event_trade,))
            self.auto_trade_threads[index].setDaemon(True)
            self.auto_trade_threads[index].start()
            self.bots[index].auto_trade = True
            self.logger.info(f"Starting a auto trading bot {self.bots[index]}")

    def stop_bot(self, index: int) -> None:
        if index < len(self.auto_trade_threads) and self.auto_trade_threads[index].is_alive():
            self.stop_event_trade.set()
            self.auto_trade_threads[index].join()
            self.bots[index].auto_trade = False
            self.logger.info(f"Stopping the auto trading bot called: {self.bots[index]}")
        else:
            self.logger.error(f"there is no bot to stop! ")

    def create_bot(self) -> None:
        self.bots.append(self._presenter.get_auto_bot())
        self.logger.info(f"Creating a auto trading bot {self._presenter.get_auto_bot()}")


    def destroy_bot(self, index: int) -> None:
        if index < len(self.auto_trade_threads):
            self.stop_bot(index)
            del self.auto_trade_threads[index]
            del self.bots[index]
            self.logger.info(f"Destroying the auto trading bot called: {self.bots[index]}")
            
            
    def get_autobot(self, exchange,symbol,amount, stop_loss,take_profit, file, time):
        
        ml = MachineLearning(exchange, symbol,self.logger)
        model = ml.load_model(file)
        
        with open(f'data/pickle/{time}.p', 'rb') as f:
            df = pickle.load(f)
        df = df.dropna()
        trade_x = Tradex_indicator(symbol=symbol, t=None, get_data=True ,data=df)
        # # Make predictions using the scaled test data
        # y_pred = ml.predict(model)
        autobot = AutoBot(exchange,symbol, amount, stop_loss, take_profit, model, time, ml, trade_x, self.logger )
        #self._presenter.save_autobot(autobot)
        return autobot
            
class ChartTabModel:
    def __init__(self, logger, presenter):
        self.logger = logger
        self.presenter = presenter
        self.stop_event_chart = threading.Event()
        self.auto_chart = False
    
    def toggle_auto_charting(self) -> bool:
        
        self.bot = self.presenter.get_bot()
        # Toggle the automatic trading flag
        self.auto_chart = not self.auto_chart

        # Update the button text
        if self.auto_chart:

            self.logger.info(f"Starting auto charting with symbol {self.bot.symbol}")

            # Create a new thread object
            self.auto_update_chart = threading.Thread(
                target=self.presenter.update_chart, args=(self.stop_event_chart,))
                
            # Reset the stop event
            self.stop_event_chart.clear()

            # Start the thread
            self.auto_update_chart.start()
            
            return True

        else:
            self.logger.info(f"Stopping auto charting with symbol {self.bot.symbol}")

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
        self.logger.info(
            f"getting data to plot the chart symbol {self.bot.symbol}")

        return df

   