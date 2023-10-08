import logging
from configparser import ConfigParser
from tkinter import END

import numpy as np


def load_config():
    config = ConfigParser()
    config.read('config.ini')
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
