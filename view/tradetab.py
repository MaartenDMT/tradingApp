from tkinter import END, BooleanVar, Listbox

from ttkbootstrap import (Button, Checkbutton, Entry, Frame, Label, OptionMenu,
                          Scale, StringVar)
from ttkbootstrap.constants import *

from util.utils import validate_float


class TradeTab(Frame):
    def __init__(self, parent, presenter) -> None:
        super().__init__(parent)
        self._parent = parent
        self._presenter = presenter
        self.exchange = self._presenter.get_exchange()
        self.update_id = None
        self.symbol = "BTC"
        self.symbol_names = []

        # Initialize and place GUI elements
        self.initialize_gui_elements()
        self.place_gui_elements()

    def initialize_gui_elements(self):
        # Create all GUI elements here
        self.create_symbol_elements()
        self.create_trade_elements()
        self.create_balance_elements()
        self.create_open_trades_elements()
        self.create_data_display_elements()
        self.create_refresh_button()

    def place_gui_elements(self):
        # Organize and place all GUI elements here
        # First column
        self.type_menu.grid(row=0, column=0, pady=5)
        self.usdt_label.grid(row=1, column=0, pady=5)
        self.symbol_label.grid(row=2, column=0, pady=5)
        self.amount_label.grid(row=3, column=0, pady=5)
        self.amount_slider.grid(row=4, column=0, pady=5)
        self.stoploss_slider.grid(row=5, column=0, pady=5)
        self.stop_loss_button.grid(row=6, column=0, pady=5)

        # Second column
        self.symbol_select.grid(row=0, column=1, pady=5)
        self.btc_label.grid(row=1, column=1, pady=5)
        self.price_label.grid(row=2, column=1, pady=5)
        self.price_entry.grid(row=3, column=1, pady=5)
        self.buy_long.grid(row=4, column=1, pady=5)
        self.takeprofit_slider.grid(row=5, column=1, pady=5)
        self.take_profit_button.grid(row=6, column=1, pady=5)

        # Third column
        self.bid_label.grid(row=0, column=2, pady=5)
        self.ask_label.grid(row=1, column=2, pady=5)
        self.ticker_price_label.grid(row=2, column=2, pady=5)
        self.trade_button.grid(row=3, column=2, pady=5)
        self.refresh_button.grid(row=4, column=2, pady=5)

        # Listview for trades information
        self.open_trades_listbox.grid(
            row=7, column=0, columnspan=3, pady=5, padx=5, sticky='nsew')

    def create_symbol_elements(self):
        self.symbol_select_var = StringVar(self, name='BTC/USDT')
        self.symbol_select_var.set("BTC/USDT")
        self.symbol = self.symbol_select_var.get()
        self.symbol_select = OptionMenu(
            self, self.symbol_select_var, *self.exchange.load_markets().keys())

    def create_slider_elements(self):
        return Scale(self, from_=0, to=100, orient='horizontal', value=15)

    def create_trade_elements(self):
        # Dropdown menu for selecting trade type
        self.type_var = StringVar(self, name='market')
        self.type_menu = OptionMenu(
            self, self.type_var, "market", "limit", "trailing", "stop")

        # Labels and entry fields for amount and price
        self.symbol_label = Label(
            self, text='Symbol: ' + self.symbol_select_var.get())
        self.amount_label = Label(self, text='Amount', font=('Arial', 12))
        self.price_label = Label(self, text='Price', font=('Arial', 12))
        self.amount_slider = self.create_slider_elements()
        self.price_entry = Entry(self, validate='key', validatecommand=(
            self.register(validate_float), '%d', '%i', '%P', '%S', '%T'))

        # Sliders for setting stop loss and take profit levels
        self.stoploss_slider = self.create_slider_elements()
        self.takeprofit_slider = self.create_slider_elements()

        # Buttons for trading and modifying stop loss/take profit
        self.trade_button = Button(
            self, text="Trade", command=self._presenter.trading_presenter.place_trade)
        self.stop_loss_button = Button(
            self, text="Set Stop Loss", command=self._presenter.trading_presenter.update_stoploss)
        self.take_profit_button = Button(
            self, text="Set Take Profit", command=self._presenter.trading_presenter.update_takeprofit)
        self.buy_var = BooleanVar(self, value=True)
        self.buy_long = Checkbutton(
            self, text='buy?', variable=self.buy_var)

    def create_balance_elements(self):
        # Labels for displaying balance
        self.usdt_label = Label(self, text='USDT:', font=('Arial', 10))
        self.btc_label = Label(self, text='BTC:', font=('Arial', 10))
        self.ticker_price_label = Label(
            self, text='Ticker Price:', font=('Arial', 10))

    def create_open_trades_elements(self):
        # Treeview for open trades
        self.open_trades_listbox = Listbox(self)

    def create_data_display_elements(self):
        # Treeview for displaying bid and ask prices
        self.bid_label = Label(self, text="Bid:")
        self.ask_label = Label(self, text="Ask:")

    def create_refresh_button(self):
        # Refresh button
        self.refresh_button = Button(
            self, text="Refresh Data", command=self._presenter.trading_presenter.start_refresh_data_thread)

    def select_exchange(self):
        open_trades = self.open_trades_listbox.curselection()[0]
        return open_trades

    def remove_exchange_from_optionmenu(self) -> None:
        index = self.open_trades_listbox.curselection()[0]
        self.open_trades_listbox.delete(index)
        return index
