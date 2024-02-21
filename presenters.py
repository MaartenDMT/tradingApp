import os
import pprint
import threading
import traceback
from tkinter import messagebox

from ttkbootstrap import Frame

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

    def run(self) -> None:
        self._view.mainloop()

    def get_exchange(self, test_mode=True):
        exchange = self._model.get_exchange(test_mode=test_mode)
        return exchange

    # Login view -------------------------------------------------------------------

    def on_login_button_clicked(self) -> None:
        username = self.loginview.get_username() or 'test'
        password = self.loginview.get_password() or 't'
        self._model.login_model.set_credentials(username, password)

        if username == 'test' or self._model.login_model.check_credentials():
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
        self.trading_presenter = TradePresenter(self._model, main_view, self)
        self.bot_tab = BotPresenter(self._model, main_view, self)
        self.chart_tab = ChartPresenter(self._model, main_view, self)
        self.exchange_tab = ExchangePresenter(self._model, main_view, self)
        self.ml_tab = MLPresenter(self._model, main_view, self)
        self.rl_tab = RLPresenter(self._model, main_view, self)


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
        stop_loss_percentage = float(trade_tab.stoploss_slider.get())

        # Validate that the stop loss percentage is between 0 and 100
        if not (0 <= stop_loss_percentage <= 100):
            raise ValueError("Stop loss percentage must be between 0 and 100")

        # Calculate the stop loss value based on the percentage
        # Your stop loss calculation logic here
        stop_loss_value = price - (1 + (stop_loss_percentage / 100))

        return stop_loss_value

    def calculate_take_profit(self, trade_tab, price):
        take_profit_percentage = float(trade_tab.takeprofit_slider.get())

        # Validate that the take profit percentage is between 0 and 100
        if not (0 <= take_profit_percentage <= 100):
            raise ValueError(
                "Take profit percentage must be between 0 and 100")

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
        self.presenter.main_listbox.set_text(f"refreshing the data")

    def start_refresh_data_thread(self):
        refresh_thread = threading.Thread(target=self.refresh_data)
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

    def start_bot(self, index: int) -> None:
        started = self._model.bottab_model.start_bot(index)

        if started:
            bot_tab = self.bot_tab_view()
            name = self.get_bot_name(bot_tab, index)
            bot_tab.update_bot_status("Started", index, name)
        self.presenter.main_listbox.set_text(
            f"bot {index}: {name} has been started")

    def stop_bot(self, index: int) -> None:
        stopped = self._model.bottab_model.stop_bot(index)

        if stopped:
            bot_tab = self.bot_tab_view()
            name = self.get_bot_name(bot_tab, index)
            bot_tab.update_bot_status("Stopped", index, name)
        self.presenter.main_listbox.set_text(
            f"bot {index}: {name} has been stopped")

    def create_bot(self) -> None:
        self._model.bottab_model.create_bot()
        bot_tab = self.bot_tab_view()
        name = self.get_bot_name(bot_tab, self.bot_count)
        self.bot_count += 1
        bot_tab.add_bot_to_optionmenu(self.bot_count, name)
        self.presenter.main_listbox.set_text(
            f"bot {self.bot_count}: {name} has been created")

    def threading_createbot(self) -> None:
        t = threading.Thread(target=self.create_bot)
        t.setDaemon(True)
        t.start()

    def destroy_bot(self, index: int) -> None:
        if index < len(self._model.bottab_model.bots):
            destroyed = self._model.bottab_model.destroy_bot(index)

            if destroyed:
                bot_tab = self.bot_tab_view()
                bot_tab.remove_bot_from_optionmenu(index)
            self.presenter.main_listbox.set_text(
                f"bot {index}:has been destroyed")
        else:
            messagebox.showerror("Error", "There is no bot to destroy.")

    def get_data_ml_files(self) -> list:
        files = self._model.bottab_model.get_data_ml_files()
        return files

    def get_auto_bot(self):
        self.bot_tab = self.bot_tab_view()
        exchange = self.get_autobot_exchange()
        symbol = self.bot_tab.exchange_var.get()
        amount = float(self.bot_tab.amount_slider.get())
        stoploss = float(self.bot_tab.loss_slider.get())
        takeprofit = float(self.bot_tab.profit_slider.get())
        time = self.bot_tab.time_var.get()
        file = self.bot_tab.optionmenu_var.get()

        autobot = self._model.bottab_model.get_autobot(
            exchange, symbol, amount, stoploss, takeprofit, file, time)
        self.append_botname_bottab(self.bot_tab, autobot.__str__())
        self.presenter.main_listbox.set_text(
            f"bot {autobot.__str__()}:has been selected")
        return autobot

    def append_botname_bottab(self, bottab, name) -> None:
        bottab.bot_names.append(name)

    def get_bot_name(self, bot_tab, index) -> str:
        return bot_tab.bot_names[index]

    def get_autobot_exchange(self):
        exchange = self.presenter.exchange_tab.select_exchange()
        exchange_tab = self.presenter.exchange_tab.exchange_tab_view()
        l = exchange_tab.text_exchange_var.get()

        if exchange == None:
            exchange_name = "phemex"
            key = os.environ.get('API_KEY_PHE_TEST')
            secret = os.environ.get('API_SECRET_PHE_TEST')
            l = True

        self.presenter.main_listbox.set_text(
            f"exchange {exchange}:has been created")

        return exchange


class ChartPresenter:
    def __init__(self, model, view, presenter) -> None:
        self._model = model
        self.main_view = view
        self.presenter = presenter
    # Implement the methods related to the Chart Tab here

    def chart_tab_view(self) -> Frame:
        chart_tab_view = self.main_view.chart_tab
        return chart_tab_view

    def update_chart(self, stop_event) -> None:
        # Check if the stop event has been set
        if not stop_event.is_set():
            # Check if automatic trading is enabled
            print('AUTO CHART TRADING --> TESTER')
            # Clear the figure
            self.chart_tab.axes.clear()
            # Get the data for the chart
            data = self._model.charttab_model.get_data()
            # Plot the data on the figure
            self.chart_tab.axes.plot(data.index, data.close)
            # Redraw the canvas
            self.chart_tab.canvas.draw()
            # Call the update_chart function again after 5 seconds
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
        exchange = self._model.exchangetab_model.get_exchange(index)
        self.presenter.main_listbox.set_text(
            f"exchange {exchange}:has been selected")
        return exchange


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
        selected_model = self.get_ML_model()
        ML = self._model.get_ML()
        model = ML.train_evaluate_and_save_model_thread(selected_model)
        return model

    def predict_with_model(self) -> int:
        model = self.get_ML_model()
        ML = self._model.get_ML()
        predict = ML.predict(model)
        self.presenter.main_listbox.set_text(f"ML has predicted {predict}")
        return predict

    def get_ml(self):
        ml = self._model.get_ML()
        return ml


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
        pass

    def start_rlmodel(self) -> None:
        self.presenter.main_listbox.set_text(
            f"Starting the DQRL Model")
        self._model.rltab_model.start()
        score = self._model.rltab_model.result
        app_logger.info(score)
        self.presenter.main_listbox.set_text(f"DQRL has scored: {score}")
