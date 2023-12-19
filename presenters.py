import os
import threading
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

    def get_exchange(self):
        exchange = self._model.get_exchange()
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

    def update_balance(self):
        usdt, btc = self._model.get_balance()
        trade_tab = self.trade_tab()
        trade_tab.usdt_label.config(
            text=f"USDT: Total {usdt['total']} | Free {usdt['free']}")
        trade_tab.btc_label.config(
            text=f"BTC: Total {btc['total']} | Free {btc['free']}")
        self.presenter.main_listbox.set_text(
            f"getting the Balance: USDT free {usdt['free']} & BTC free {btc['free']} ")

    def place_trade(self):
        try:
            trade_tab = self.trade_tab()
            # Method to handle the trade button click
            trade_type = trade_tab.type_var.get()
            symbol = trade_tab.symbol
            amount = int(trade_tab.amount_slider.get())
            price = float(trade_tab.price_entry.get()
                          ) if trade_type == 'limit' else None
            stoploss = trade_tab.stoploss_slider.get()
            takeprofit = trade_tab.takeprofit_slider.get()
            side = "buy" if trade_tab.buy_var == True else "sell"
            self._model.place_trade(
                symbol, side, trade_type, amount, price, stoploss, takeprofit)
            self.presenter.main_listbox.set_text(
                f"place trade: {symbol}|{amount}|{price}")
        except Exception as e:
            app_logger.error(f"error with place trade:{e}")

    def update_stoploss(self, new_stoploss):
        trade_tab = self.trade_tab()
        new_stoploss = trade_tab.stoploss_slider.get()
        self._model.update_stoploss(new_stoploss)
        self.presenter.main_listbox.set_text(
            f"updating stoploss: {new_stoploss}")

    def update_takeprofit(self, new_takeprofit):
        trade_tab = self.trade_tab()
        new_takeprofit = trade_tab.takeprofit_slider.get()
        self._model.update_takeprofit(new_takeprofit)
        self.presenter.main_listbox.set_text(
            f"updating takeprofit: {new_takeprofit}")

    # Bid and ask
    def update_bid_ask(self):
        bid, ask = self._model.get_bid_ask()
        trade_tab = self.trade_tab()

        # Insert the new bid and ask data
        trade_tab.bid_label.config(text=f"Bid: {bid}")
        trade_tab.ask_label.config(text=f"Ask: {ask}")

    # Open trades
    def update_open_trades(self):
        trade_tab = self.trade_tab()
        open_trades = self._model.fetch_open_trades(trade_tab.symbol)
        for i in open_trades:
            trade_tab.open_trades_listbox.insert('end', open_trades[i])

    # Update ticker prices
    def update_ticker_price(self):
        last_price = self._model.get_ticker_price()
        trade_tab = self.trade_tab()
        trade_tab.ticker_price_label.config(text=f"Ticker Price: {last_price}")

    def get_real_time_date(self):
        return self._model.get_real_time_data()

    def refresh_data(self):
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

    def get_symbol(self):
        trade_tab = self.trade_tab()
        return trade_tab.symbol


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

        if l:
            exchange.set_sandbox_mode(True)
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
        exchange = self._model.exchangetab_model.set_first_exchange()
        exchange_tab = self.exchange_tab_view()
        value = exchange_tab.text_exchange_var.get()
        exchange.set_sandbox_mode(value)
        exchange_tab.add_exchange_optionmenu(exchange)
        self.presenter.trading_presenter.get_balance()

    def create_exchange(self) -> None:
        exchange_tab = self.exchange_tab_view()
        exchange_name = exchange_tab.exchange_var.get()
        api_key = exchange_tab.api_key_entry.get()
        api_secret = exchange_tab.api_secret_entry.get()
        exchange = self._model.exchangetab_model.create_exchange(
            exchange_name, api_key, api_secret)
        exchange_tab.add_exchange_optionmenu(exchange)
        self.presenter.main_listbox.set_text(
            f"exchange {exchange}: has been created")

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
