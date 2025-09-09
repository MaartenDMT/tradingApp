import os
import pprint
import threading
import traceback
from tkinter import messagebox

import pandas as pd
import numpy as np

try:
    from ttkbootstrap import Frame
    HAS_TTKBOOTSTRAP = True
except Exception:
    # Fallback to tkinter Frame if ttkbootstrap is not installed (tests / headless env)
    from tkinter import Frame
    HAS_TTKBOOTSTRAP = False

import util.loggers as loggers

MAX_POSITION_SIZE = 0.01
MIN_STOP_LOSS_LEVEL = 0.10

logger = loggers.setup_loggers()
app_logger = logger['app']


class Presenter:
    def __init__(self, model, view) -> None:
        self._model = model
        self._view = view
        self.get_frames()
        # Initialize placeholders for tab presenters
        self._trading_presenter = None
        self._bot_tab = None
        self._chart_tab = None
        self._exchange_tab = None
        self._ml_tab = None
        self._rl_tab = None
        
        # New optimized system presenters
        self._trading_system_presenter = None
        self._ml_system_presenter = None
        self._rl_system_presenter = None

    def run(self) -> None:
        self._view.mainloop()

    def get_exchange(self, test_mode=True):
        exchange = self._model.get_exchange(test_mode=test_mode)
        return exchange

    # Lazy initialization methods for tab presenters
    def _ensure_trading_presenter(self):
        if self._trading_presenter is None:
            app_logger.debug("Initializing TradePresenter")
            self._trading_presenter = TradePresenter(self._model, self.main_view, self)

    def _ensure_bot_tab(self):
        if self._bot_tab is None:
            app_logger.debug("Initializing BotPresenter")
            self._bot_tab = BotPresenter(self._model, self.main_view, self)

    def _ensure_chart_tab(self):
        if self._chart_tab is None:
            app_logger.debug("Initializing ChartPresenter")
            self._chart_tab = ChartPresenter(self._model, self.main_view, self)

    def _ensure_exchange_tab(self):
        if self._exchange_tab is None:
            app_logger.debug("Initializing ExchangePresenter")
            self._exchange_tab = ExchangePresenter(self._model, self.main_view, self)

    def _ensure_ml_tab(self):
        if self._ml_tab is None:
            app_logger.debug("Initializing MLPresenter")
            self._ml_tab = MLPresenter(self._model, self.main_view, self)

    def _ensure_rl_tab(self):
        if self._rl_tab is None:
            app_logger.debug("Initializing RLPresenter")
            self._rl_tab = RLPresenter(self._model, self.main_view, self)
    
    # New optimized systems lazy initialization
    def _ensure_trading_system_presenter(self):
        if self._trading_system_presenter is None:
            app_logger.debug("Initializing TradingSystemPresenter")
            self._trading_system_presenter = TradingSystemPresenter(self._model, self.main_view, self)
    
    def _ensure_ml_system_presenter(self):
        if self._ml_system_presenter is None:
            app_logger.debug("Initializing MLSystemPresenter")
            self._ml_system_presenter = MLSystemPresenter(self._model, self.main_view, self)
    
    def _ensure_rl_system_presenter(self):
        if self._rl_system_presenter is None:
            app_logger.debug("Initializing RLSystemPresenter")
            self._rl_system_presenter = RLSystemPresenter(self._model, self.main_view, self)

    # Properties to ensure lazy initialization
    @property
    def trading_presenter(self):
        self._ensure_trading_presenter()
        return self._trading_presenter

    @property
    def bot_tab(self):
        self._ensure_bot_tab()
        return self._bot_tab

    @property
    def chart_tab(self):
        self._ensure_chart_tab()
        return self._chart_tab

    @property
    def exchange_tab(self):
        self._ensure_exchange_tab()
        return self._exchange_tab

    @property
    def ml_tab(self):
        self._ensure_ml_tab()
        return self._ml_tab

    @property
    def rl_tab(self):
        self._ensure_rl_tab()
        return self._rl_tab
    
    # New optimized systems properties
    @property
    def trading_system_presenter(self):
        self._ensure_trading_system_presenter()
        return self._trading_system_presenter
    
    @property
    def ml_system_presenter(self):
        self._ensure_ml_system_presenter()
        return self._ml_system_presenter
    
    @property
    def rl_system_presenter(self):
        self._ensure_rl_system_presenter()
        return self._rl_system_presenter

    # Login view -------------------------------------------------------------------

    def on_login_button_clicked(self) -> None:
        username = self.loginview.get_username() or 'test'
        password = self.loginview.get_password() or 't'
        self._model.login_model.set_credentials(username, password)

        # For development, automatically log in with test credentials
        if username == 'test' and password == 't':
            self.get_main_view()
        elif self._model.login_model.check_credentials():
            self.get_main_view()
        else:
            self.loginview.login_failed()

    def on_register_button_clicked(self) -> None:
        username = self.loginview.get_username()
        password = self.loginview.get_password()

        self._model.login_model.set_credentials(username, password)
        registered = self._model.login_model.register()
        if registered and self._model.login_model.check_credentials():
            self.get_main_view()
        else:
            self.loginview.login_failed()

    def get_frames(self) -> None:
        self._view.frames = {}
        self.loginview = self._view.loginview(self)
        self._view.show_frame(self.loginview, self)

    # Main view  ----------------------------------------------------------------

    def get_main_view(self) -> None:
        self.loginview.destroy()
        self.main_view = self._view.main_view(self._view)
        self._model.create_tabmodels(self)
        self.get_tabs(self.main_view)
        self._view.show_frame(self.main_view, self)
        self.main_listbox = MainListBox(self._model, self.main_view, self)

    def get_tabs(self, main_view) -> None:
        # Don't initialize tab presenters here - they will be initialized lazily when accessed
        app_logger.info("Tab presenters will be initialized lazily when accessed")
        pass

    def get_bot(self):
        return self.bot_tab.get_auto_bot()

    def update_chart(self, stop_event):
        return self.chart_tab.update_chart(stop_event)

    def calculate_financial_metrics(self, metric_type, **kwargs):
        return self.trading_presenter.calculate_financial_metrics(metric_type, **kwargs)


class MainListBox:
    def __init__(self, model, view, presenter) -> None:
        self._model = model
        self.main_view = view
        self.presenter = presenter
        self.load()

    def load(self):
        self.main_view.list_box("Welcome to the trade-x Bot")

    def set_text(self, text):
        self.main_view.list_box(text)


class TradePresenter:
    def __init__(self, model, view, presenter) -> None:
        self._model = model.tradetab_model
        self.main_view = view
        self.presenter = presenter

    def trade_tab(self) -> Frame:
        trade_tab_view = self.main_view.trade_tab
        return trade_tab_view

    def get_real_time_date(self):
        return self._model.get_real_time_data()

    def get_balance(self):
        return self._model.get_balance()

    # === Trade Execution and Parameters ===

    def place_trade(self):
        try:
            trade_tab = self.trade_tab()
            trade_params = self.extract_trade_parameters(trade_tab)
            amount = self.calculate_trade_amount(
                trade_params['percentage_amount'])

            # Place the trade through the model
            self._model.place_trade(
                symbol=trade_params['symbol'],
                side=trade_params['side'],
                trade_type=trade_params['trade_type'],
                amount=amount,
                price=trade_params['price'],
                stoploss=trade_params['stop_loss'],
                takeprofit=trade_params['take_profit']
            )
            self.presenter.main_listbox.set_text(
                f"Placing trade: {trade_params}")
        except Exception as e:  # Replace with specific exception types
            app_logger.error(
                f"Error with placing trade: {e}\n{traceback.format_exc()}")

    def update_stoploss(self):
        try:
            trade_tab = self.trade_tab()
            leverage = int(trade_tab.leverage.get())
            sl_percentage = trade_tab.stoploss_slider.get()

            # Fetch the current market price
            last_price = self._model.get_ticker_price()

            # Adjust the stop loss percentage based on leverage
            adjusted_sl_percentage = sl_percentage / leverage

            # Calculate the new stop loss price based on the trade's side
            side = "buy" if bool(trade_tab.buy_var) else "sell"
            if side == "buy":
                new_stoploss = last_price * \
                    (1 - adjusted_sl_percentage / 100.0)
            else:
                new_stoploss = last_price * \
                    (1 + adjusted_sl_percentage / 100.0)

            # Update the stop loss in the trading model
            self._model.update_stoploss(new_stoploss)
            self.presenter.main_listbox.set_text(
                f"Updating stop loss to: {new_stoploss}")

        except Exception as e:
            app_logger.error(
                f"Error updating stop loss: {e}\n{traceback.format_exc()}")

    def update_takeprofit(self):
        try:
            trade_tab = self.trade_tab()
            leverage = int(trade_tab.leverage.get())
            tp_percentage = trade_tab.takeprofit_slider.get()

            # Fetch the current market price
            last_price = self._model.get_ticker_price()

            # Adjust the take profit percentage based on leverage
            adjusted_tp_percentage = tp_percentage / leverage

            # Calculate the new take profit price based on the trade's side
            side = "buy" if bool(trade_tab.buy_var) else "sell"
            if side == "buy":
                new_takeprofit = last_price * \
                    (1 + adjusted_tp_percentage / 100.0)
            else:
                new_takeprofit = last_price * \
                    (1 - adjusted_tp_percentage / 100.0)

            # Update the take profit in the trading model
            self._model.update_takeprofit(new_takeprofit)
            self.presenter.main_listbox.set_text(
                f"Updating take profit to: {new_takeprofit}")

        except Exception as e:
            app_logger.error(
                f"Error updating take profit: {e}\n{traceback.format_exc()}")

    # 1. Execute Advanced Orders
    def execute_advanced_orders(self, symbol, total_amount, duration, side, order_type):
        if order_type == 'twap':
            self._model.execute_advanced_orders(
                symbol, total_amount, duration, side, 'twap')
        elif order_type == 'dynamic_stop_loss':
            # Assuming you have a way to get these parameters
            entry_price, current_price, initial_stop_loss, trailing_percent = self.get_dynamic_stop_loss_params()
            self._model.execute_advanced_orders(symbol, total_amount, duration, side, 'dynamic_stop_loss',
                                                entry_price=entry_price, current_price=current_price,
                                                initial_stop_loss=initial_stop_loss, trailing_percent=trailing_percent)
            self.presenter.main_listbox.set_text(
                f"Executing Dynamic Stop Loss order: Symbol={symbol}, Total Amount={total_amount}, Duration={duration}, Side={side}, Entry Price={entry_price}, Current Price={current_price}, Initial Stop Loss={initial_stop_loss}, Trailing Percent={trailing_percent}")

    # === Data Display and Updates ===

    def update_balance(self):
        balance = self._model.get_balance()  # Get the balance from the model
        trade_tab = self.trade_tab()
        trade_tab.usdt_label.config(text=f"USDT: Total {balance}")
        self.presenter.main_listbox.set_text(
            f"Getting the Balance: USDT free {balance}")

    def calculate_trade_amount(self, percentage_amount):
        balance = self._model.get_balance()  # Fetch the current balance
        return (percentage_amount / 100.0) * balance * self.get_leverage()

    def calculate_stop_loss(self, trade_tab, price):
        # Validate inputs
        from util.validation import validate_number, validate_percentage

        if not validate_number(price, min_value=0):
            raise ValueError(f"Invalid price: {price}")

        stop_loss_percentage = float(trade_tab.stoploss_slider.get())

        # Validate that the stop loss percentage is between 0 and 100 using our utility
        if not validate_percentage(stop_loss_percentage):
            raise ValueError(f"Stop loss percentage must be between 0 and 100, got: {stop_loss_percentage}")

        # Calculate the stop loss value based on the percentage
        # Your stop loss calculation logic here
        stop_loss_value = price - (1 + (stop_loss_percentage / 100))

        return stop_loss_value

    def calculate_take_profit(self, trade_tab, price):
        # Validate inputs
        from util.validation import validate_number, validate_percentage

        if not validate_number(price, min_value=0):
            raise ValueError(f"Invalid price: {price}")

        take_profit_percentage = float(trade_tab.takeprofit_slider.get())

        # Validate that the take profit percentage is between 0 and 100 using our utility
        if not validate_percentage(take_profit_percentage):
            raise ValueError(f"Take profit percentage must be between 0 and 100, got: {take_profit_percentage}")

        # Calculate the take profit value based on the percentage
        # Your take profit calculation logic here
        take_profit_value = price + (1 + (take_profit_percentage / 100))

        return take_profit_value

    # Bid and ask
    def update_bid_ask(self):
        bid, ask = self._model.get_market_data('bidask')
        pprint.pprint(bid)
        trade_tab = self.trade_tab()
        trade_tab.bid_label.config(text=f"Bid: {bid}")
        trade_tab.ask_label.config(text=f"Ask: {ask}")

    def update_open_trades(self):
        trade_tab = self.trade_tab()
        symbol = trade_tab.symbol  # You need to fetch the symbol from the trade tab
        open_trades = self._model.fetch_data_and_transfer_funds(
            'open_trades', symbol=symbol)

        # Clear existing entries
        trade_tab.open_trades_listbox.delete(0, 'end')

        for trade in open_trades:
            # Extract relevant information from each trade dictionary
            trade_id = trade.get('id', 'Unknown ID')[:4]
            symbol = trade.get('symbol', 'Unknown Symbol')
            # 'buy' or 'sell'
            side = trade.get('side', 'Unknown Side').capitalize()
            status = trade.get('status', 'Unknown Status')
            amount = trade.get('amount', 0)
            # Use triggerPrice if available, else price
            price = trade.get('triggerPrice', trade.get('price', 'N/A'))

            # Format the display string
            display_text = f"ID: {trade_id}, Symbol: {symbol}, Side: {side}, Status: {status}, Amount: {amount}, Price: {price}"
            trade_tab.open_trades_listbox.insert('end', display_text)

    # Update ticker prices

    def update_ticker_price(self):
        last_price = self._model.get_ticker_price()
        trade_tab = self.trade_tab()
        trade_tab.ticker_price_label.config(text=f"Ticker Price: {last_price}")

    def refresh_data(self, settings=None):
        try:
            if settings is None:
                settings = {}  # Initialize an empty settings dictionary

            # Update relevant settings in the settings dictionary as needed
            settings['symbol'] = self.get_symbol()
            settings['leverage'] = self.get_leverage()
            settings['accountType'] = self.get_accountype()

            # Call the update_settings method of the trading class with the updated settings
            self._model.update_settings(settings)

            # The rest of your refresh_data method remains unchanged
            self.update_bid_ask()
            self.update_open_trades()
            self.update_ticker_price()
            self.update_balance()
            self.presenter.main_listbox.set_text("refreshing the data")
        except Exception as e:
            app_logger.error(f"Error refreshing data: {e}")
            self.presenter.main_listbox.set_text(f"Error refreshing data: {str(e)}")

    def start_refresh_data_thread(self):
        refresh_thread = threading.Thread(target=self.refresh_data, daemon=True)
        refresh_thread.start()

    def scale_in_out(self, amount):
        self._model.scale_in_out(amount)
        self.presenter.main_listbox.set_text(f"scaling the data: {amount}")

    # === Symbol and Account Type and leverage===

    def get_symbol(self):
        trade_tab = self.trade_tab()
        symbol = trade_tab.symbol_select_var.get()  # Directly fetch the selected symbol
        # Update the label in the view
        trade_tab.symbol_label.config(text='Symbol: ' + symbol)
        self.presenter.main_listbox.set_text(f"Setting the symbol: {symbol}")
        return symbol

    def get_leverage(self):
        trade_tab = self.trade_tab()
        leverage = trade_tab.leverage_var.get()  # Directly fetch the leverage value
        # Update the label in the view
        trade_tab.leverage_label.config(text=f'Leverage: {leverage}')
        self.presenter.main_listbox.set_text(
            f"Setting the leverage: {leverage}")
        return leverage

    def get_accountype(self):
        trade_tab = self.trade_tab()
        type = trade_tab.set_accountType()
        return type

    # === Metric Calculation ===

    def calculate_financial_metrics(self, metric_type, **kwargs):
        if metric_type == 'var':
            return self._model.calculate_financial_metrics('var', portfolio=kwargs['portfolio'],
                                                           confidence_level=kwargs['confidence_level'])
        elif metric_type == 'drawdown':
            return self._model.calculate_financial_metrics('drawdown', peak_balance=kwargs['peak_balance'],
                                                           current_balance=kwargs['current_balance'])
        elif metric_type == 'pnl':
            return self._model.calculate_financial_metrics('pnl', entry_price=kwargs['entry_price'],
                                                           exit_price=kwargs['exit_price'],
                                                           contract_quantity=kwargs['contract_quantity'],
                                                           is_long=kwargs['is_long'])
        elif metric_type == 'breakeven_price':
            return self._model.calculate_financial_metrics('breakeven_price', entry_price=kwargs['entry_price'],
                                                           fee_percent=kwargs['fee_percent'],
                                                           contract_quantity=kwargs['contract_quantity'],
                                                           is_long=kwargs['is_long'])

    # === Data Fetch and Transfer Funds ===

    def fetch_data_and_transfer_funds(self, fetch_type, **kwargs):
        if fetch_type == 'open_trades':
            return self._model.fetch_data_and_transfer_funds('open_trades', symbol=kwargs['symbol'])
        elif fetch_type == 'historical_data':
            return self._model.fetch_data_and_transfer_funds('historical_data', symbol=kwargs['symbol'],
                                                             timeframe=kwargs['timeframe'],
                                                             since=kwargs['since'], limit=kwargs['limit'])
        elif fetch_type == 'transfer_funds':
            return self._model.fetch_data_and_transfer_funds('transfer_funds', amount=kwargs['amount'],
                                                             currency_code=kwargs['currency_code'],
                                                             from_account_type=kwargs['from_account_type'],
                                                             to_account_type=kwargs['to_account_type'])
    # 3. Fetch Data and Transfer Funds

    def extract_trade_parameters(self, trade_tab):
        parameters = {
            'symbol': trade_tab.symbol,
            'side': "buy" if trade_tab.buy_var.get() else "sell",
            'trade_type': trade_tab.type_var.get(),
            'price': None,
            'stop_loss': None,
            'take_profit': None
        }

        # Validate and extract price
        if trade_tab.type_var.get() == 'limit':
            price_str = trade_tab.price_entry.get()
            try:
                parameters['price'] = float(price_str)
            except ValueError:
                # Handle invalid price input, e.g., non-numeric input
                self.presenter.main_listbox.set_text("Invalid price input")
                # You can add further error handling logic here

        last_price = self._model.get_ticker_price()
        # Validate and extract stop loss
        try:
            parameters['stop_loss'] = self.calculate_stop_loss(
                trade_tab, last_price)
        except Exception as e:
            # Handle exceptions raised by calculate_stop_loss
            self.presenter.main_listbox.set_text(
                f"Error in stop loss calculation: {str(e)}")
            # You can add further error handling logic here

        # Validate and extract take profit
        try:
            parameters['take_profit'] = self.calculate_take_profit(
                trade_tab, last_price)
        except Exception as e:
            # Handle exceptions raised by calculate_take_profit
            self.presenter.main_listbox.set_text(
                f"Error in take profit calculation: {str(e)}")
            # You can add further error handling logic here

         # Extract percentage_amount
        try:
            parameters['percentage_amount'] = float(
                trade_tab.amount_slider.get())
        except ValueError:
            self.presenter.main_listbox.set_text(
                "Invalid percentage amount input")
            # Handle the invalid input, for example, by setting a default value or alerting the user

        return parameters


class BotPresenter:
    def __init__(self, model, view, presenter) -> None:
        self._model = model
        self.main_view = view
        self.presenter = presenter
        self.bot_count = 0
        self.exchange_count = 0
    # Implement the methods related to the Bot Tab here

    def bot_tab_view(self) -> Frame:
        bot_tab_view = self.main_view.bot_tab
        return bot_tab_view

    


class ChartPresenter:
    def __init__(self, model, view, presenter) -> None:
        self._model = model
        self.main_view = view
        self.presenter = presenter
        self.app_logger = app_logger
    # Implement the methods related to the Chart Tab here

    def chart_tab_view(self) -> Frame:
        chart_tab_view = self.main_view.chart_tab
        return chart_tab_view

    def update_chart(self, stop_event) -> None:
        # Check if the stop event has been set
        if not stop_event.is_set():
            try:
                # Check if automatic trading is enabled
                self.model_logger.debug('AUTO CHART TRADING --> TESTER')
                # Clear the figure
                self.chart_tab.axes.clear()
                # Get the data for the chart
                data = self._model.charttab_model.get_data()
                # Plot the data on the figure
                self.chart_tab.axes.plot(data.index, data.close)
                # Redraw the canvas
                self.chart_tab.canvas.draw()
            except Exception as e:
                self.app_logger.error(f"Error updating chart: {e}")
            finally:
                # Call the update_chart function again after 60 seconds
                self.main_view.after(60_000, self.update_chart, stop_event)

    def toggle_auto_charting(self) -> None:
        self.auto_chart = self._model.charttab_model.toggle_auto_charting()
        self.chart_tab = self.chart_tab_view()
        if self.auto_chart:
            # Clear the figure
            self.chart_tab.axes.clear()
            # Update the status label
            self.chart_tab.start_autochart_button.config(
                text="Stop Auto Charting")

        else:
            # Clear the figure
            self.chart_tab.axes.clear()
            # Update the status label
            self.chart_tab.start_autochart_button.config(
                text="Start Auto Charting")


class ExchangePresenter:
    def __init__(self, model, view, presenter) -> None:
        self._model = model
        self.main_view = view
        self.presenter = presenter
    # Implement the methods related to the Exchange Tab here

    def exchange_tab_view(self) -> Frame:
        exchange_tab_view = self.main_view.exchange_tab
        return exchange_tab_view

    def save_first_exchange(self) -> None:
        exchange_tab = self.exchange_tab_view()
        value = exchange_tab.text_exchange_var.get()
        exchange = self._model.exchangetab_model.set_first_exchange(
            test_mode=value)

        exchange_tab.add_exchange_optionmenu(exchange)

    def create_exchange(self) -> None:
        exchange_tab = self.exchange_tab_view()
        exchange_name = exchange_tab.exchange_var.get()
        api_key = exchange_tab.api_key_entry.get()
        api_secret = exchange_tab.api_secret_entry.get()
        exchange = self._model.exchangetab_model.create_exchange(
            exchange_name, api_key, api_secret)
        if exchange:
            exchange_tab.add_exchange_optionmenu(exchange)
            self.presenter.main_listbox.set_text(
                f"Exchange {exchange_name} has been created")

    def remove_exchange(self) -> None:
        exchange_tab = self.exchange_tab_view()
        index = exchange_tab.remove_exchange_from_optionmenu()
        self._model.exchangetab_model.remove_exchange(index)
        self.presenter.main_listbox.set_text(
            f"exchange {index}:has been removed")

    def select_exchange(self):
        exchange_tab = self.exchange_tab_view()
        index = exchange_tab.select_exchange()
        if index is not None:
            exchange = self._model.exchangetab_model.get_exchange(index)
            self.presenter.main_listbox.set_text(
                f"exchange {exchange}:has been selected")
            return exchange
        return None


class MLPresenter:
    def __init__(self, model, view, presenter) -> None:
        self._model = model
        self.main_view = view
        self.presenter = presenter
    # Implement the methods related to the ML Tab here

    def ml_tab_view(self) -> Frame:
        ml_tab_view = self.main_view.ml_tab
        return ml_tab_view

    def get_ML_model(self) -> str:
        self.ml_tab = self.ml_tab_view()
        selected_algorithm = self.ml_tab.type_var.get()
        self.ml_tab.current_ml_label.config(text=f"{selected_algorithm}")
        self._model.model_logger.info(
            f"machine learning model {selected_algorithm} selected")
        self.presenter.main_listbox.set_text(
            f"ML {selected_algorithm}: has been selected")
        return selected_algorithm

    def train_evaluate_save_model(self) -> None:
        selected_algorithm = self.get_ML_model()
        ml_system_name = selected_algorithm

        # Run ML training in a separate thread to avoid blocking the GUI
        training_thread = threading.Thread(
            target=self._run_ml_training_in_thread,
            args=(ml_system_name, selected_algorithm),
            daemon=True
        )
        training_thread.start()

    def _run_ml_training_in_thread(self, ml_system_name: str, selected_algorithm: str):
        try:
            # 1. Create ML System
            ml_system = self._model.ml_system_model.create_ml_system(
                name=ml_system_name,
                algorithm=selected_algorithm,
                target_type='classification'
            )

            if not ml_system:
                self.presenter.main_listbox.set_text(f"Failed to create ML System {ml_system_name}")
                return

            # 2. Prepare data for training (dummy for now)
            X_train = pd.DataFrame(np.random.rand(100, 10))
            y_train = pd.Series(np.random.randint(0, 2, 100))

            # 3. Train ML System
            train_results = self._model.ml_system_model.train_ml_system(
                name=ml_system_name,
                X_train=X_train,
                y_train=y_train
            )

            if train_results.get('success'):
                self.presenter.main_listbox.set_text(f"ML System {ml_system_name} trained successfully.")
            else:
                self.presenter.main_listbox.set_text(f"ML System {ml_system_name} training failed: {train_results.get('message')}")
        except Exception as e:
            self.app_logger.error(f"Error during ML training thread: {e}\n{traceback.format_exc()}")
            self.presenter.main_listbox.set_text(f"Error during ML training: {e}")

    def predict_with_model(self) -> int:
        selected_algorithm = self.get_ML_model()
        ml_system_name = selected_algorithm # Assuming the same name as created during training

        # Run ML prediction in a separate thread
        prediction_thread = threading.Thread(
            target=self._run_ml_prediction_in_thread,
            args=(ml_system_name,),
            daemon=True
        )
        prediction_thread.start()
        return 0 # Return immediately, prediction result will be updated via main_listbox

    def _run_ml_prediction_in_thread(self, ml_system_name: str):
        try:
            # 1. Get ML System
            ml_system = self._model.ml_system_model.get_ml_system(ml_system_name)
            if not ml_system:
                self.presenter.main_listbox.set_text(f"ML System {ml_system_name} not found for prediction.")
                return

            # 2. Prepare data for prediction (dummy for now).
            X_predict = pd.DataFrame(np.random.rand(1, 10))

            # 3. Predict with ML System
            predict_results = self._model.ml_system_model.predict_with_ml_system(
                name=ml_system_name,
                X=X_predict
            )

            if predict_results.get('success'):
                predictions = predict_results.get('predictions', [])
                if predictions:
                    predict = predictions[0] # Assuming single prediction
                    self.presenter.main_listbox.set_text(f"ML has predicted {predict}")
                else:
                    self.presenter.main_listbox.set_text("ML prediction returned no results.")
            else:
                self.presenter.main_listbox.set_text(f"ML prediction failed: {predict_results.get('message')}")
        except Exception as e:
            self.app_logger.error(f"Error during ML prediction thread: {e}\n{traceback.format_exc()}")
            self.presenter.main_listbox.set_text(f"Error during ML prediction: {e}")

    def get_ml(self):
        # This method seems to be a getter for the ML instance.
        # It should probably return the MLSystemModel instance itself, or a specific MLSystem.
        # For now, let's return the MLSystemModel.
        return self._model.ml_system_model


class RLPresenter:
    def __init__(self, model, view, presenter) -> None:
        self._model = model
        self.main_view = view
        self.presenter = presenter

    # Implement the methods related to the ML Tab here

    def rl_tab_view(self) -> Frame:
        rl_tab_view = self.main_view.rl_tab
        return rl_tab_view

    def train_evaluate_save_rlmodel(self) -> None:
        # Run RL training and evaluation in a separate thread
        training_thread = threading.Thread(
            target=self._run_rl_training_in_thread,
            daemon=True
        )
        training_thread.start()

    def _run_rl_training_in_thread(self):
        try:
            # 1. Get selected algorithm (assuming there's a way to select it in the UI)
            # For now, let's hardcode a simple agent type
            agent_type = 'dqn'
            agent_name = f"{agent_type}_agent"

            # 2. Create RL agent
            # Need state_dim and action_dim. For now, use dummy values.
            state_dim = 10
            action_dim = 2
            agent = self._model.rl_system_model.create_agent(
                name=agent_name,
                agent_type=agent_type,
                state_dim=state_dim,
                action_dim=action_dim
            )

            if not agent:
                self.presenter.main_listbox.set_text(f"Failed to create RL Agent {agent_name}")
                return

            # 3. Prepare environment data (dummy for now)
            environment_data = {"data": "dummy_data"}

            # 4. Train the agent
            train_results = self._model.rl_system_model.train_agent(
                name=agent_name,
                environment_data=environment_data
            )

            if train_results.get('success'):
                self.presenter.main_listbox.set_text(f"RL Agent {agent_name} trained successfully.")
                # 5. Evaluate the agent
                evaluate_results = self._model.rl_system_model.evaluate_agent(
                    name=agent_name,
                    environment_data=environment_data
                )
                if evaluate_results.get('success'):
                    mean_reward = evaluate_results.get('results', {}).get('mean_reward', 'N/A')
                    self.presenter.main_listbox.set_text(f"RL Agent {agent_name} evaluated - Mean reward: {mean_reward}")
                else:
                    self.presenter.main_listbox.set_text(f"RL Agent {agent_name} evaluation failed: {evaluate_results.get('message')}")
            else:
                self.presenter.main_listbox.set_text(f"RL Agent {agent_name} training failed: {train_results.get('message')}")
        except Exception as e:
            self.app_logger.error(f"Error during RL training thread: {e}\n{traceback.format_exc()}")
            self.presenter.main_listbox.set_text(f"Error during RL training: {e}")

    def start_rlmodel(self) -> None:
        self.presenter.main_listbox.set_text(
            "Starting the DQRL Model")
        self.train_evaluate_save_rlmodel()


# New Optimized System Presenters

class TradingSystemPresenter:
    """Presenter for the optimized trading system."""
    
    def __init__(self, model, view, presenter) -> None:
        self._model = model
        self.main_view = view
        self.presenter = presenter
        self.app_logger = app_logger
        self.app_logger.info("TradingSystemPresenter initialized")
    
    def get_trading_system(self, config_dict=None):
        """Get the trading system instance."""
        if hasattr(self._model, 'trading_system_model') and self._model.trading_system_model:
            return self._model.trading_system_model.get_trading_system(config_dict)
        return None
    
    def get_system_status(self):
        """Get trading system status for display."""
        try:
            if hasattr(self._model, 'trading_system_model') and self._model.trading_system_model:
                status = self._model.trading_system_model.get_system_status()
                self.presenter.main_listbox.set_text(f"Trading System Status: {status.get('is_running', 'Unknown')}")
                return status
            return {'status': 'not_available'}
        except Exception as e:
            self.app_logger.error(f"Error getting trading system status: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def execute_signal(self, signal_params):
        """Execute a trading signal through the optimized system."""
        try:
            if hasattr(self._model, 'trading_system_model') and self._model.trading_system_model:
                result = await self._model.trading_system_model.execute_trade_signal(signal_params)
                
                if result.get('success'):
                    self.presenter.main_listbox.set_text(f"Trade executed successfully: {result.get('order_id', 'N/A')}")
                else:
                    self.presenter.main_listbox.set_text(f"Trade failed: {result.get('message', 'Unknown error')}")
                
                return result
            return {'success': False, 'message': 'Trading system not available'}
            
        except Exception as e:
            self.app_logger.error(f"Error executing trade signal: {e}")
            self.presenter.main_listbox.set_text(f"Trading error: {e}")
            return {'success': False, 'message': str(e)}
    
    def start_trading_system(self, config_dict=None):
        """Start the trading system."""
        try:
            trading_system = self.get_trading_system(config_dict)
            if trading_system:
                # Since start() is async, we'd need to handle this appropriately
                self.presenter.main_listbox.set_text("Trading system started (async operation)")
                return True
            else:
                self.presenter.main_listbox.set_text("Failed to start trading system - not available")
                return False
        except Exception as e:
            self.app_logger.error(f"Error starting trading system: {e}")
            self.presenter.main_listbox.set_text(f"Error starting trading system: {e}")
            return False
    
    def stop_trading_system(self):
        """Stop the trading system."""
        try:
            if hasattr(self._model, 'trading_system_model') and self._model.trading_system_model:
                # This would need async handling as well
                self.presenter.main_listbox.set_text("Trading system stopped (async operation)")
                return True
            return False
        except Exception as e:
            self.app_logger.error(f"Error stopping trading system: {e}")
            return False


class MLSystemPresenter:
    """Presenter for the ML system integration."""
    
    def __init__(self, model, view, presenter) -> None:
        self._model = model
        self.main_view = view
        self.presenter = presenter
        self.app_logger = app_logger
        self.app_logger.info("MLSystemPresenter initialized")
    
    def create_ml_system(self, name, algorithm='random_forest', target_type='classification', config_dict=None):
        """Create a new ML system."""
        try:
            if hasattr(self._model, 'ml_system_model') and self._model.ml_system_model:
                ml_system = self._model.ml_system_model.create_ml_system(name, algorithm, target_type, config_dict)
                
                if ml_system:
                    self.presenter.main_listbox.set_text(f"ML System '{name}' created with {algorithm}")
                    return ml_system
                else:
                    self.presenter.main_listbox.set_text(f"Failed to create ML System '{name}'")
                    return None
            
            self.presenter.main_listbox.set_text("ML System not available")
            return None
            
        except Exception as e:
            self.app_logger.error(f"Error creating ML system: {e}")
            self.presenter.main_listbox.set_text(f"ML System error: {e}")
            return None
    
    def list_ml_systems(self):
        """List all created ML systems."""
        try:
            if hasattr(self._model, 'ml_system_model') and self._model.ml_system_model:
                systems = self._model.ml_system_model.list_ml_systems()
                
                if systems:
                    system_names = list(systems.keys())
                    self.presenter.main_listbox.set_text(f"ML Systems: {', '.join(system_names)}")
                else:
                    self.presenter.main_listbox.set_text("No ML systems created yet")
                
                return systems
            return {}
            
        except Exception as e:
            self.app_logger.error(f"Error listing ML systems: {e}")
            return {}
    
    def train_ml_system(self, name, X_train, y_train, **kwargs):
        """Train a specific ML system."""
        try:
            if hasattr(self._model, 'ml_system_model') and self._model.ml_system_model:
                result = self._model.ml_system_model.train_ml_system(name, X_train, y_train, **kwargs)
                
                if result.get('success'):
                    self.presenter.main_listbox.set_text(f"ML System '{name}' trained successfully")
                else:
                    self.presenter.main_listbox.set_text(f"ML System '{name}' training failed: {result.get('message')}")
                
                return result
            
            return {'success': False, 'message': 'ML System not available'}
            
        except Exception as e:
            self.app_logger.error(f"Error training ML system: {e}")
            return {'success': False, 'message': str(e)}
    
    def predict_with_ml_system(self, name, X):
        """Make predictions using a specific ML system."""
        try:
            if hasattr(self._model, 'ml_system_model') and self._model.ml_system_model:
                result = self._model.ml_system_model.predict_with_ml_system(name, X)
                
                if result.get('success'):
                    predictions = result.get('predictions', [])
                    self.presenter.main_listbox.set_text(f"ML System '{name}' made {len(predictions)} predictions")
                else:
                    self.presenter.main_listbox.set_text(f"ML System '{name}' prediction failed: {result.get('message')}")
                
                return result
            
            return {'success': False, 'message': 'ML System not available'}
            
        except Exception as e:
            self.app_logger.error(f"Error making ML predictions: {e}")
            return {'success': False, 'message': str(e)}


class RLSystemPresenter:
    """Presenter for the RL system integration."""
    
    def __init__(self, model, view, presenter) -> None:
        self._model = model
        self.main_view = view
        self.presenter = presenter
        self.app_logger = app_logger
        self.app_logger.info("RLSystemPresenter initialized")
    
    def create_rl_agent(self, name, agent_type, state_dim, action_dim, config=None):
        """Create a new RL agent."""
        try:
            if hasattr(self._model, 'rl_system_model') and self._model.rl_system_model:
                agent = self._model.rl_system_model.create_agent(name, agent_type, state_dim, action_dim, config)
                
                if agent:
                    self.presenter.main_listbox.set_text(f"RL Agent '{name}' created with type {agent_type}")
                    return agent
                else:
                    self.presenter.main_listbox.set_text(f"Failed to create RL Agent '{name}'")
                    return None
            
            self.presenter.main_listbox.set_text("RL System not available")
            return None
            
        except Exception as e:
            self.app_logger.error(f"Error creating RL agent: {e}")
            self.presenter.main_listbox.set_text(f"RL Agent error: {e}")
            return None
    
    def list_rl_agents(self):
        """List all created RL agents."""
        try:
            if hasattr(self._model, 'rl_system_model') and self._model.rl_system_model:
                agents = self._model.rl_system_model.list_agents()
                
                if agents:
                    agent_names = list(agents.keys())
                    self.presenter.main_listbox.set_text(f"RL Agents: {', '.join(agent_names)}")
                else:
                    self.presenter.main_listbox.set_text("No RL agents created yet")
                
                return agents
            return {}
            
        except Exception as e:
            self.app_logger.error(f"Error listing RL agents: {e}")
            return {}
    
    def train_rl_agent(self, name, environment_data, config=None):
        """Train a specific RL agent."""
        try:
            if hasattr(self._model, 'rl_system_model') and self._model.rl_system_model:
                result = self._model.rl_system_model.train_agent(name, environment_data, config)
                
                if result.get('success'):
                    self.presenter.main_listbox.set_text(f"RL Agent '{name}' trained successfully")
                else:
                    self.presenter.main_listbox.set_text(f"RL Agent '{name}' training failed: {result.get('message')}")
                
                return result
            
            return {'success': False, 'message': 'RL System not available'}
            
        except Exception as e:
            self.app_logger.error(f"Error training RL agent: {e}")
            return {'success': False, 'message': str(e)}
    
    def evaluate_rl_agent(self, name, environment_data, num_episodes=10):
        """Evaluate a specific RL agent."""
        try:
            if hasattr(self._model, 'rl_system_model') and self._model.rl_system_model:
                result = self._model.rl_system_model.evaluate_agent(name, environment_data, num_episodes)
                
                if result.get('success'):
                    mean_reward = result.get('results', {}).get('mean_reward', 'N/A')
                    self.presenter.main_listbox.set_text(f"RL Agent '{name}' evaluated - Mean reward: {mean_reward}")
                else:
                    self.presenter.main_listbox.set_text(f"RL Agent '{name}' evaluation failed: {result.get('message')}")
                
                return result
            
            return {'success': False, 'message': 'RL System not available'}
            
        except Exception as e:
            self.app_logger.error(f"Error evaluating RL agent: {e}")
            return {'success': False, 'message': str(e)}
    
    def get_available_algorithms(self):
        """Get available RL algorithms."""
        try:
            if hasattr(self._model, 'rl_system_model') and self._model.rl_system_model:
                algorithms = self._model.rl_system_model.get_available_algorithms()
                
                if algorithms:
                    algo_names = list(algorithms.keys())
                    self.presenter.main_listbox.set_text(f"Available RL algorithms: {', '.join(algo_names)}")
                else:
                    self.presenter.main_listbox.set_text("No RL algorithms available")
                
                return algorithms
            return {}
            
        except Exception as e:
            self.app_logger.error(f"Error getting RL algorithms: {e}")
            return {}
