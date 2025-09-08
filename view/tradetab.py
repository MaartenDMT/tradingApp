from tkinter import Listbox

try:
    from ttkbootstrap import (BooleanVar, Button, Checkbutton, Entry, Frame,
                              IntVar, Label, OptionMenu, Scale, StringVar)
    from ttkbootstrap.constants import *
    HAS_TTKBOOTSTRAP = True
except Exception:
    # Fallback to tkinter widgets when ttkbootstrap is not available
    from tkinter import (BooleanVar, Button, Checkbutton, Entry, Frame, IntVar,
                         Label, OptionMenu, Scale, StringVar)
    HAS_TTKBOOTSTRAP = False

from util.utils import validate_float


class TradeTab(Frame):
    def __init__(self, parent, presenter) -> None:
        super().__init__(parent)
        self._parent = parent
        self._presenter = presenter
        self.exchange = self._presenter.get_exchange()
        self.update_id = None
        self.symbol = "BTC/USD:USD"
        self.leverage = 16
        self.symbols = ['BTC/USD:USD', 'ETH/USD:USD', 'OP/USD:USD']
        self.accountTypes = ['swap', 'spot', 'future', 'margin', 'delivery']

        # Initialize and place GUI elements
        self.initialize_gui_elements()
        self.place_gui_elements()

        # Additional UI elements and functions
        self.initialize_additional_elements()
        self.place_additional_elements()

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
        self.type_menu.grid(row=0, column=0, pady=5, padx=5,)
        self.usdt_label.grid(row=1, column=0, pady=5, padx=5,)
        self.symbol_label.grid(row=2, column=0, pady=5, padx=5,)
        self.amount_label.grid(row=3, column=0, pady=5, padx=5,)
        self.amount_slider.grid(row=4, column=0, pady=5, padx=5,)
        self.stoploss_slider.grid(row=5, column=0, pady=5, padx=5,)
        self.stop_loss_button.grid(row=6, column=0, pady=5, padx=5,)

        # Second column
        self.symbol_select.grid(row=0, column=1, pady=5, padx=5,)
        self.price_label.grid(row=2, column=1, pady=5, padx=5,)
        self.price_entry.grid(row=3, column=1, pady=5, padx=5,)
        self.buy_long.grid(row=4, column=1, pady=5, padx=5,)
        self.takeprofit_slider.grid(row=5, column=1, pady=5, padx=5,)
        self.take_profit_button.grid(row=6, column=1, pady=5, padx=5,)

        # Third column
        self.leverage_opt.grid(row=0, column=2, pady=5, padx=5,)
        self.bid_label.grid(row=1, column=2, pady=5, padx=5,)
        self.ask_label.grid(row=2, column=2, pady=5, padx=5,)
        self.ticker_price_label.grid(row=3, column=2, pady=5, padx=5,)
        self.trade_button.grid(row=4, column=2, pady=5, padx=5,)
        self.refresh_button.grid(row=5, column=2, pady=5, padx=5,)

        # Fourth Column
        self.accountTypes_opt.grid(row=0, column=3, pady=5, padx=5,)

    def initialize_additional_elements(self):
        # Additional UI components initialization
        # For example, buttons for executing advanced orders and calculating financial metrics
        self.execute_advanced_order_button = Button(
            self, text="Execute Advanced Order",
            command=self.on_execute_advanced_order)

        self.calculate_metric_button = Button(
            self, text="Calculate Metric",
            command=self.on_calculate_metric)

        # Entry fields for additional parameters
        # Instead of OptionMenu for total amount, use Scale
        self.total_amount_var = IntVar(self)
        self.total_amount_scale = self.create_slider_elements()
        self.total_amount_label = Label(self, text="Total Amount:")

        # Set an initial value for the scale (e.g., 100)
        self.total_amount_var.set(15)

        # Instead of Entry for duration, use OptionMenu
        self.duration_var = StringVar(self)
        self.duration_options = {
            '1m': '60',
            '3m': '180',
            '5m': '300',
            '15m': '900',
            '30m': '1800',
            '1h': '3600',
            '2h': '7200',
            '3h': '10800',
            '4h': '14400',
            '6h': '21600',
            '12h': '43200',
            '1d': '86400'
        }
        self.duration_menu = OptionMenu(
            self, self.duration_var, '30m', *self.duration_options.keys())
        self.duration_label = Label(self, text="Duration:")

        self.side_var = StringVar(self)
        self.side_menu = OptionMenu(self, self.side_var, "buy",  "buy", "sell")
        self.side_label = Label(self, text="Side:")

        self.order_type_var = StringVar(self)
        self.order_type_menu = OptionMenu(
            self, self.order_type_var, "twap", "twap", "dynamic_stop_loss")
        self.order_type_label = Label(self, text="Order Type:")

        self.metric_type_var = StringVar(self)
        self.metric_type_menu = OptionMenu(
            self, self.metric_type_var, "var", "var", "drawdown", "pnl", "breakeven_price")
        self.metric_type_label = Label(self, text="Metric Type:")

        # Additional fields for metric calculation
        self.entry_price_label = Label(self, text="Entry Price:")
        self.entry_price_entry = Entry(self)

        self.exit_price_label = Label(self, text="Exit Price:")
        self.exit_price_entry = Entry(self)

        self.contract_quantity_label = Label(self, text="Contract Quantity:")
        self.contract_quantity_entry = Entry(self)

        self.is_long_var = BooleanVar(self)
        self.is_long_checkbutton = Checkbutton(
            self, text="Is Long", variable=self.is_long_var)

    def place_additional_elements(self):
        # Adjusted placement of UI components to fit in columns 3, 4, and 5
        self.execute_advanced_order_button.grid(
            row=1, column=3, pady=5, padx=5,)
        self.total_amount_label.grid(row=2, column=3, pady=5, padx=5,)
        self.total_amount_scale.grid(
            row=2, column=4, pady=5, padx=5,)  # Adjusted column

        self.duration_label.grid(row=3, column=3, pady=5, padx=5,)
        self.duration_menu.grid(row=3, column=4, pady=5,
                                padx=5,)  # Adjusted column

        self.side_label.grid(row=4, column=3, pady=5, padx=5,)
        self.side_menu.grid(row=4, column=4, pady=5,
                            padx=5,)  # Adjusted column

        self.order_type_label.grid(row=5, column=3, pady=5, padx=5,)
        self.order_type_menu.grid(row=5, column=4, pady=5,
                                  padx=5,)  # Adjusted column

        self.calculate_metric_button.grid(
            row=1, column=5, pady=5, padx=5,)  # Adjusted column
        self.metric_type_label.grid(
            row=2, column=5, pady=5, padx=5,)  # Adjusted column
        self.metric_type_menu.grid(
            row=2, column=6, pady=5, padx=5,)  # Adjusted column
        self.entry_price_label.grid(
            row=3, column=5, pady=5, padx=5,)  # Adjusted column
        self.entry_price_entry.grid(
            row=3, column=6, pady=5, padx=5,)  # Adjusted column
        self.exit_price_label.grid(
            row=4, column=5, pady=5, padx=5,)  # Adjusted column
        self.exit_price_entry.grid(
            row=4, column=6, pady=5, padx=5,)  # Adjusted column
        self.contract_quantity_label.grid(
            row=5, column=5, pady=5, padx=5,)  # Adjusted column
        self.contract_quantity_entry.grid(
            row=5, column=6, pady=5, padx=5,)  # Adjusted column

        self.is_long_checkbutton.grid(
            row=1, column=7, pady=5, padx=5,)  # Adjusted column

        # Listview for trades information
        self.open_trades_listbox.grid(
            row=7, column=0, columnspan=8, pady=5, padx=5, sticky='nsew')  # Adjusted columnspan

    def create_symbol_elements(self):
        self.symbol_select_var = StringVar(self, value=self.symbol)
        self.symbol = self.symbol_select_var.get()
        self.symbol_select = OptionMenu(
            self, self.symbol_select_var, 'BTC/USD:USD', *self.symbols)

    def create_slider_elements(self):
        return Scale(self, from_=0, to=100, orient='horizontal', value=15)

    def create_trade_elements(self):
        # Dropdown menu for selecting trade type
        self.type_var = StringVar(self)
        self.type_menu = OptionMenu(
            self, self.type_var, "market", "market", "limit", "trailing", "stop")

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
        self.leverage_var = IntVar(self)
        self.leverage_opt = OptionMenu(
            self, self.leverage_var, 10, 5, 10, 16, 20)
        self.leverage_label = Label(self, text=f'Leverage: {self.leverage}')

        self.accountTypes_var = StringVar(self)
        self.accountTypes_opt = OptionMenu(
            self, self.accountTypes_var, 'swap', *self.accountTypes)
        self.accountTypes_label = Label(
            self, text='Account Type', font=('Arial', 12))

    def create_balance_elements(self):
        # Labels for displaying balance
        self.usdt_label = Label(self, text='USDT:', font=('Arial', 10))
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

    def open_trades(self):
        open_trades = self.open_trades_listbox.curselection()[0]
        return open_trades

    def remove_open_trades_from_optionmenu(self) -> None:
        index = self.open_trades_listbox.curselection()[0]
        self.open_trades_listbox.delete(index)
        return index

    def set_accountType(self):
        return self.accountTypes_var.get()

    def on_execute_advanced_order(self):
        try:
            # Handle advanced order execution
            # Retrieve values from UI components and call appropriate presenter method
            symbol = self.symbol
            balance = self._presenter.get_balance()  # Get the available balance
            percentage = float(self.total_amount_entry.get()) / 100.0
            # Calculate the total amount based on the percentage
            total_amount = int(balance['free'] * percentage)
            duration = self.duration_entry.get()
            side = self.side_var.get()
            order_type = self.order_type_var.get()
            # Call presenter method
            self._presenter.execute_advanced_orders(
                symbol, total_amount, duration, side, order_type)
        except ValueError as e:
            # Handle invalid input (e.g., non-numeric percentage)
            print(f"Invalid percentage input: {e}")

    def on_calculate_metric(self):
        # Handle metric calculation
        # Retrieve values and call presenter method
        metric_type = self.metric_type_var.get()
        entry_price = self.entry_price_entry.get()
        exit_price = self.exit_price_entry.get()
        contract_quantity = self.contract_quantity_entry.get()
        is_long = self.is_long_var.get()
        # Call presenter method
        self._presenter.calculate_financial_metrics(
            metric_type, entry_price=entry_price, exit_price=exit_price, contract_quantity=contract_quantity, is_long=is_long)
