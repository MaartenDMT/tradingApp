from tkinter import Listbox
import ccxt
from ttkbootstrap import Button, Entry, Frame, Label, OptionMenu, StringVar


class ExchangeTab(Frame):
    def __init__(self, parent, presenter)-> None:
        super().__init__(parent)
        self._parent = parent
        self._presenter = presenter
        # ... GUI elements for the trade page go here ...
        
         # Create a dropdown menu for selecting the exchange
        self.exchange_var = StringVar(self)
        self.exchange_var.set("phemex")
        self.exchange_menu = OptionMenu(
            self, self.exchange_var, *ccxt.exchanges, direction='flush')
        self.api_key_label = Label(self, text='API Key:')
        self.api_key_entry = Entry(self)
        self.api_secret_label = Label(
            self, text='API Secret:')
        self.api_secret_entry = Entry(self, show='*')
        self.submit_exchange = Button(self, text='Submit', command='')
        
        self.history_autobot = Listbox(self)

        # self exchange page -------------------------------------------------

        self.exchange_menu.grid(row=0, column=0,pady=5,padx=5)
        self.api_key_label.grid(row=1, column=0,pady=5)
        self.api_key_entry.grid(row=1, column=1,pady=5)
        self.api_secret_label.grid(row=2, column=0,pady=5)
        self.api_secret_entry.grid(row=2, column=1,pady=5)
        self.submit_exchange.grid(row=3, column=0, columnspan=2,pady=5)
        
        self.history_autobot.grid(row=4, column=0, columnspan=2,pady=5, sticky='NSEW')