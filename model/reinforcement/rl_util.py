import glob
import logging
import random
from datetime import datetime

import mplfinance as mpf
import numpy as np
import pandas as pd


def next_available_filename(base_filename, extension, directory='data/'):
    directory = f"{directory}{extension}"
    existing_files = glob.glob(f"{directory}/{base_filename}*.{extension}")
    max_number = 0
    for file in existing_files:
        try:
            file_number = int(file.split(base_filename)[
                              1].split(f".{extension}")[0])
            if file_number > max_number:
                max_number = file_number
        except ValueError:
            print(f"Error in the value for {extension}")

    next_number = max_number + 1
    next_filename = f"{directory}/{base_filename}{next_number}.{extension}"
    return next_filename


# Set up root logger
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create separate loggers for Environment and Agent
env_logger = logging.getLogger('Environment')
agent_logger = logging.getLogger('Agent')
main_logger = logging.getLogger('Main')


def env_logging():

    # Optionally, you can customize them further, e.g., set different log levels
    env_logger.setLevel(logging.DEBUG)

    # Optionally, set up file handlers to log to separate files
    env_handler = logging.FileHandler('data/logs/environment.log')

    # Format the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    env_handler.setFormatter(formatter)

    # Add the handlers
    env_logger.addHandler(env_handler)

    return env_logger


def agent_logging():

    # Optionally, you can customize them further, e.g., set different log levels
    agent_logger.setLevel(logging.DEBUG)

    # Optionally, set up file handlers to log to separate files
    agent_handler = logging.FileHandler('data/logs/agent.log')

    # Format the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    agent_handler.setFormatter(formatter)

    # Add the handlers
    agent_logger .addHandler(agent_handler)

    return agent_logger


def main_logging():

    # Optionally, you can customize them further, e.g., set different log levels
    main_logger.setLevel(logging.INFO)

    # Optionally, set up file handlers to log to separate files
    main_handler = logging.FileHandler('data/logs/main.log')

    # Format the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_handler.setFormatter(formatter)

    # Add the handlers
    main_logger.addHandler(main_handler)

    return main_logger


def best_csv(b_reward, acc) -> bool:
    filename = "data/best_model/best_model.csv"
    if glob.glob(filename):
        df = pd.read_csv(filename)
        csv_reward = df['reward']
        csv_accuracy = df["accuracy"]
        if b_reward > csv_reward and acc > csv_accuracy:
            dict_ar = {'accuracy': acc, 'reward': b_reward}
            # save the model and overwrite the csv
            df = pd.DataFrame(dict_ar)
            df.to_csv(filename)
            return True
        return False
    return True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def generate_random_candlestick_data(
    num_candles, initial_price=100, min_volatility=0.01, max_volatility=0.05, 
    max_shadow_amplitude=0.5, max_volume_multiplier=5.0, start_date=None):
    
    """
    Generate random candlestick data with varying volatility, shadow amplitude, and volume using numpy

    Args:
        num_candles (int): Number of candlesticks to generate.
        initial_price (int, optional): Initial price. Defaults to 100.
        min_volatility (float, optional): Minimum volatility. Defaults to 0.01.
        max_volatility (float, optional): Maximum volatility. Defaults to 0.05.
        max_shadow_amplitude (float, optional): Maximum shadow amplitude. Defaults to 0.5.
        max_volume_multiplier (float, optional): Maximum volume multiplier. Defaults to 5.0.
        start_date (datetime, optional): Start date. Defaults to None.
    
    Returns:
        tuple: Numpy array containing candlestick data and the start date.
    """
    
    open_prices = np.zeros(num_candles)
    high_prices = np.zeros(num_candles)
    low_prices = np.zeros(num_candles)
    close_prices = np.zeros(num_candles)    
    volume = np.zeros(num_candles)

    if start_date is None:
        start_date = datetime.now()
        
    volatilities = np.random.uniform(min_volatility, max_volatility, num_candles)
    
    for i in range(num_candles):
        open_prices[i] = initial_price
        price_movement = np.random.normal(0, volatilities[i] * initial_price)
        close_prices[i] = open_prices[i] + price_movement
        
        # Calculate shadow amplitude as a percentage of the candle body
        shadow_amplitude = np.random.uniform(0, max_shadow_amplitude) * abs(price_movement)
        high_prices[i] = max(open_prices[i], close_prices[i]) + shadow_amplitude
        low_prices[i] = min(open_prices[i], close_prices[i]) - shadow_amplitude
        
        # Generate a base volume and apply a multiplier to create variations
        base_volume = random.randint(100, 1000)
        volume[i] = base_volume * np.random.uniform(1.0, max_volume_multiplier)
        
        # Set the initial price for the next candle to the close price of the current candle
        initial_price = close_prices[i]
        
    candlestick_data = np.column_stack((open_prices, high_prices, low_prices, close_prices, volume))
    return candlestick_data, start_date
        
        
def plot_candlestick_chart(candlestick_data, start_date, title='Candlestick Chart'):
    """
    Plots the candlestick chart using matplotlib

    Args:
        candlestick_data (numpy array): Numpy array containing candlestick data
        start_date (datetime): Start date for the candlestick
        title (str, optional): Title of the chart. Defaults to 'Candlestick Chart'.
    """
    
    # dates = [start_date + timedelta(days=i) for i in range(len(candlestick_data))]
    candlestick_data = pd.DataFrame(candlestick_data, columns=['open', 'high', 'low', 'close', 'volume'], index=pd.date_range(start_date, periods=len(candlestick_data), freq="30min"))
    mpf.plot(candlestick_data, type='candle', mav=(3, 6, 9), volume=True, show_nontrading=True, style='yahoo', title=title)
    

# print("Loading...")
# candlestick_data, start_date = generate_random_candlestick_data(num_candles=100)
# plot_candlestick_chart(candlestick_data, start_date, title="Candlestick Chart")