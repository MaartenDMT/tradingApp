import json
import traceback
from typing import Any, Dict

import pandas as pd

import util.loggers as loggers

logger = loggers.setup_loggers()
app_logger = logger['app']

MAX_SIZE = 1000
# On Message: get the date from Websocket --------------------------------- #


def decode_json_message(message):
    try:
        data = json.loads(message)
        if not isinstance(data, dict):
            raise ValueError(f'Expected dict, got {type(data).__name__}')
        return data
    except KeyError as e:
        app_logger.error(f'Key Error processing message: {e}')
    except json.JSONDecodeError as e:
        app_logger.error(f"JSONDecodeError: {e}")
    except ValueError as e:
        app_logger.error(f'Value Error processing message: {e}')
    return None


def validate_message_data(data):
    if 'data' not in data or 'k' not in data['data']:
        app_logger.warning("Invalid WebSocket message format.")
        return False
    return True


def add_row_and_maintain_size(new_row, candlestick_data):

    app_logger.info(f'1: {candlestick_data}')

    # Convert new_row to a DataFrame if it's a dict
    if isinstance(new_row, dict):
        new_row = pd.DataFrame([new_row])
    try:
        if not set(new_row.columns).issubset(set(candlestick_data.columns)):
            raise ValueError(
                "Columns mismatch between new_row and existing DataFrame.")

        if len(candlestick_data) >= MAX_SIZE:
            # Drop the first/oldest row
            candlestick_data.drop(
                candlestick_data.index[0], inplace=True
            )
            app_logger.info(
                f'2: {candlestick_data}')
        # Append the new row
        candlestick_data = pd.concat(
            [candlestick_data, new_row]
        ).reset_index(drop=True)
        app_logger.info(
            f'3: {candlestick_data}')

    except IndexError as e:
        # add logging here
        app_logger.error(f"IndexError occurred: {e}")
    except ValueError as e:
        app_logger.error(
            f"ValueError occurred: {e}.\nNew Row: {new_row}\nData: {candlestick_data}")
    except Exception as e:
        app_logger.error(
            f"Error when adding a row occurred: {e}\n{traceback.format_exc()}")
    return candlestick_data


def parse_candlestick(data: Dict[str, Any]) -> Dict[str, Any]:
    candlestick = data['data']['k']
    if not isinstance(candlestick, dict) or 'x' not in candlestick:
        raise ValueError("Invalid candlestick data in WebSocket message.")

    # Figure out the timeframe from the stream name
    stream_name = data['stream']
    try:
        timeframe = stream_name.split('@')[-1].split('_')[-1]
    except IndexError:
        app_logger.error("Error parsing timeframe from stream name.")
        return

    return candlestick, timeframe


def extract_candlestick_data(candlestick: Dict[str, Any]) -> pd.Series:
    date = pd.to_datetime(candlestick['t'], unit='ms')
    open_price = float(candlestick['o'])
    high_price = float(candlestick['h'])
    low_price = float(candlestick['l'])
    close_price = float(candlestick['c'])
    volume = float(candlestick['v'])
    new_row = {'date': date,
               'open': open_price,
               'high': high_price,
               'low': low_price,
               'close': close_price,
               'volume': volume}
    return new_row
