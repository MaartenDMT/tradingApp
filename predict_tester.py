import datetime
import os
import random
import traceback
from queue import Queue

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from matplotlib.animation import FuncAnimation

from model.features import Tradex_indicator


def future_score_reg(y_true, y_pred, n_candles=2, starting_capital=100, threshold_percentage=0.0):
    """
    Calculate the future score based on true and predicted values.
    Args:
    y_true (list): List of true values.
    y_pred (list): List of predicted values.
    n_candles (int): Number of candles to look ahead.
    starting_capital (float): The starting capital for the investment.

    Returns:
    float: The return percentage.
    """
    if not y_true or not y_pred:
        print("Error: y_true or y_pred is empty.")
        return 0

    if starting_capital <= 0:
        print("Error: starting_capital should be greater than 0.")
        return 0

    total_return = 0
    position = 0  # -1 for short, 0 for no position, 1 for long
    entry_price = 0

    # Adjust the length to iterate based on available data
    length_to_iterate = min(len(y_pred), len(y_true) - n_candles)

    # Convert length_to_iterate to an integer
    length_to_iterate = int(length_to_iterate)

    # Warn if there's not enough future data
    if length_to_iterate < len(y_pred):
        ...

    for i in range(length_to_iterate):
        if position != 0:  # if holding a position, long or short
            total_return += (y_true[i + n_candles] - entry_price) * position
            position = 0  # close the position

        predicted_return = y_pred[i] - y_true[i]

        threshold = threshold_percentage * y_true[i]
        if predicted_return > threshold:  # Predicted price increase -> go long
            entry_price = y_true[i]
            position = 1
        elif predicted_return < -threshold:  # Predicted price decrease -> go short
            entry_price = y_true[i]
            position = -1

    # Close any open position at the end
    if position != 0:
        total_return += (y_true[-1] - entry_price) * position

    # Calculate and return the return as a percentage of the starting capital
    return_percentage = (total_return / starting_capital) * 100
    return return_percentage


def get_random_model_file():
    # Get a list of model files in the "data/ml/" directory
    model_files = [file for file in os.listdir(
        'data/ml/2020/3h/') if file.endswith('.p')]

    if not model_files:
        raise ValueError("No model files found in the 'data/ml/' directory.")

    # Select a random model file
    random_model_file = random.choice(model_files)

    # Return the full path to the selected model file
    return os.path.join('data/ml/2020/3h/', random_model_file)


def load_random_model():
    try:
        # Get a random model file
        model_file = get_random_model_file()
        # Load the model using joblib
        with open(model_file, 'rb') as file:
            loaded_model = load(file)

        return loaded_model
    except Exception as e:
        print(f"Error loading the random model: {e}\n{traceback.format_exc()}")
        return None


# Example usage:
random_loaded_model = load_random_model()

if random_loaded_model:
    print("Randomly selected model loaded successfully.")
    # Use the loaded model for predictions or any other tasks
    # For example: random_loaded_model.predict(...)
else:
    print("Failed to load a random model.")


column_names = []


def process_features(df, symbol):

    tradex = Tradex_indicator(
        symbol=symbol, timeframe='1h', t=None, get_data=False, data=df.copy())
    tradex.run()
    trend = tradex.trend.get_trend()
    screener = tradex.screener.get_screener()
    real_time = tradex.real_time.get_real_time()
    scanner = tradex.scanner.get_scanner()

    processed_features = pd.concat(
        [df[['open', 'high', 'low', 'volume']], trend, screener, real_time, scanner], axis=1)

    column_names = processed_features.columns.tolist()

    # Fill missing values in the entire DataFrame
    processed_features = processed_features.fillna(0)

    # Return the processed features as a NumPy array
    return processed_features.to_numpy()


def load_pretrained_scaler(self):
    return joblib.load('data/scaler/scaler.p')


def predict(model, df, t='30m', symbol='BTC/USDT') -> (int, datetime.datetime):
    try:

        # Use the process_features method to get the features for prediction
        features = process_features(df, symbol)

        # If a scaler was used during training, load the pre-fitted scaler and transform the features
        # Make sure the scaler is saved after fitting on training data
        # Load the scaler fitted on training data
        scaler = load_pretrained_scaler(features)
        features = scaler.transform(features)

        # Make a prediction using the model
        prediction = model.predict(features[-1].reshape(1, -1))[0]

        # Get the current timestamp
        current_timestamp = datetime.datetime.now()

        # Logging the prediction result
        print(
            f"Prediction for {symbol} at timeframe {t}: {prediction}")

        return prediction, current_timestamp
    except Exception as e:
        print(f"Error during prediction: {e}\n{traceback.format_exc()}")
        return 0, datetime.datetime.now()


def plot_real_time_predictions(model, symbol='BTC/USDT', timeframe='30m', interval=0.1):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    ax_pred = axs[0]
    ax_true = axs[1]

    predictions = []
    true_values = []
    timestamps = []
    # starting_capital
    money_count_container = [100]  # Use a list as a mutable container

    # Define the update function

    def update(i):
        global sampled_df, money_count
        # If the queue is not empty, get the next row of data and add it to sampled_df
        if not data_queue.empty():
            new_data = data_queue.get()
            sampled_df = sampled_df.append(new_data)

        # Retain only the latest 500 rows of sampled_df
        if len(sampled_df) >= 500:
            sampled_df = sampled_df.tail(500)

        # Assuming that true value is present in the 'sampled_df' dataframe,
        true_value = sampled_df.iloc[-1]['close']
        true_values.append(true_value)

        # Perform the prediction using the updated sampled_df
        prediction, timestamp = predict(
            model, sampled_df, t='30m', symbol='BTC/USDT')
        predictions.append(prediction)
        timestamps.append(timestamp)

        # Retain only the latest 500 data points
        if len(timestamps) > 100:
            timestamps.pop(0)
            predictions.pop(0)
            true_values.pop(0)

        # Calculate the future score
        return_percentage = future_score_reg(
            true_values, predictions, threshold_percentage=0.03)
        money_count_container[0] = money_count_container[0] * \
            (1 + return_percentage / 100)

        # Update the plots
        ax_pred.clear()
        ax_pred.plot(timestamps, predictions,
                     label=f'Predictions\nCapital: ${money_count_container[0]:.2f}')
        ax_pred.scatter(timestamps[-1], predictions[-1],
                        c='green' if return_percentage >= 0 else 'red')
        # if not hasattr(update, "last_return_percentage") or np.sign(return_percentage) != np.sign(update.last_return_percentage):
        #     ax_pred.scatter(timestamps[-1], predictions[-1],
        #                     c='red' if return_percentage > 0 else 'green')
        # update.last_return_percentage = return_percentage
        ax_pred.set_xlabel('Timestamp')
        ax_pred.set_ylabel('Predicted Value')
        ax_pred.set_title(f'Real-time Predictions for {symbol} ({timeframe})')
        ax_pred.legend()

        ax_true.clear()
        ax_true.plot(timestamps, true_values,
                     label='True Values', color='orange')
        ax_true.set_xlabel('Timestamp')
        ax_true.set_ylabel('True Value')
        ax_true.set_title(f'True Values for {symbol} ({timeframe})')
        ax_true.legend()

    # Start the animation
    ani = FuncAnimation(fig, update, interval=interval * 1000)
    plt.show()


pickle_file_path = f'data/pickle/30m.p'
with open(pickle_file_path, 'rb') as file:
    df = load(file)


# Define the percentage of data to be sampled and used initially
sample_percent = 20
num_rows = len(df)
num_samples = int((sample_percent / 100) * num_rows)

# Sample the initial data
sampled_df = df.sample(n=num_samples, random_state=42)

# Get the remaining 80% of the data and put it in a queue
remaining_df = df.drop(sampled_df.index)
data_queue = Queue()
for index, row in remaining_df.iterrows():
    data_queue.put(row)

# Load a random model
loaded_model = load_random_model()

# Usage
plot_real_time_predictions(loaded_model)
