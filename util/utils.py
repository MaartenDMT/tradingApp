import logging
from configparser import ConfigParser
from time import sleep
from tkinter import END

import numpy as np
import pandas as pd

from model.features import Tradex_indicator
from util.secure_config import load_secure_config


def load_config():
    config = load_secure_config()
    return config


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


def is_float(string) -> bool:
    """Returns True if the string is a valid floating-point number, False otherwise."""
    try:
        float(string)
        return True
    except ValueError:
        return False


def validate_float(d, i, P, S, T) -> bool:
    """Validation function for the Entry widget. Only allows floating-point numbers to be inserted."""
    if S == '-':
        # Allow the hyphen character to be inserted
        return True
    elif S == '.':
        return True
    elif not S:
        # Allow deletion of characters
        return True
    elif is_float(S):
        # Allow valid floating-point numbers to be inserted
        return True
    else:
        # Reject all other characters
        return False


class ListboxHandler(logging.Handler):
    def __init__(self, listbox_widget) -> None:
        # store a reference to the Listbox widget
        self.listbox_widget = listbox_widget
        # run the superclass constructor
        logging.Handler.__init__(self)

    def emit(self, record) -> None:
        # append the log message to the Listbox widget
        self.listbox_widget.insert(END, self.format(record) + "\n")


def tradex_features(symbol, df):

    tradex = Tradex_indicator(
        symbol=symbol, t=None, get_data=False, data=df.copy())

    done = tradex.run()

    if done:
        # Assuming tradex.trend, tradex.screener, etc., are instances of their respective classes
        # Call get_trend on the instance of Trend class
        trend = tradex.trend.get_trend()
        screener = tradex.screener.get_screener()  # Similarly for other attributes
        real_time = tradex.real_time.get_real_time()
        scanner = tradex.scanner.get_scanner()

        processed_features = pd.concat(
            [df, trend, real_time, scanner], axis=1)
    else:
        # Assuming tradex.trend, tradex.screener, etc., are instances of their respective classes
        # Call get_trend on the instance of Trend class
        trend = tradex.trend.get_trend()
        screener = tradex.screener.get_screener()  # Similarly for other attributes
        real_time = tradex.real_time.get_real_time()
        scanner = tradex.scanner.get_scanner()

        processed_features = pd.concat(
            [df, trend, screener, real_time, scanner], axis=1)

    return processed_features.drop_duplicates()


def features(df):

    # Volume
    # CMF
    cmf = df.ta.cmf()

    # Statistics
    # KURT
    kurt = df.ta.kurtosis()

    data = pd.concat([cmf, kurt], axis=1)

    return data


def convert_df(df):
    print(df)
    # List of columns to be converted to float
    columns_to_convert = ['open', 'high', 'low', 'close', 'volume']

    # Apply the conversion to the specified columns
    df[columns_to_convert] = df[columns_to_convert].astype(float)

    return df
