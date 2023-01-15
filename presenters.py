import os
from tkinter import messagebox

import ccxt
from ttkbootstrap import Frame

MAX_POSITION_SIZE = 0.01
MIN_STOP_LOSS_LEVEL = 0.10


class Presenter:
    def __init__(self, model, view):
        self._model = model
        self._view = view
        self.get_frames()

        
    def run(self):
        self._view.mainloop()
        
    def get_exchange(self):
        exchange = self._model.get_exchange()
        return exchange
        

    # Login view
    def on_login_button_clicked(self):
        username = self.loginview.get_username()
        password = self.loginview.get_password()
        self._model.login_model.set_credentials(username, password)

        if username=='test' or self._model.login_model.check_credentials():
            self.get_main_view()
        else:
            self.loginview.login_failed()
            
    def on_register_button_clicked(self):
        username = self.loginview.get_username()
        password = self.loginview.get_password()
        
        self._model.login_model.set_credentials(username, password)
        registered = self._model.login_model.register()
        if registered and self._model.login_model.check_credentials():
            self.get_main_view()
        else:
            self.loginview.login_failed()
    
    def get_frames(self):
        self._view.frames = {}
        self.loginview = self._view.loginview(self)
        self._view.show_frame(self.loginview, self)
        
    
    #Main view  ----------------------------------------------------------------
    def get_main_view(self):
        self.loginview.destroy()
        self.main_view = self._view.main_view(self._view)
        self._model.create_tabmodels(self)
        self._view.show_frame(self.main_view, self)
        
        # self.exchange_tab_view = self.main_view.exchange_tab
        # self.bot_tab_view = self.main_view.bot_tab
        # self.chart_tab_view = self.main_view.chart_tab
        # self.ml_tab_view = self.main_view.ml_tab
    
    def trade_tab(self) -> Frame:
        trade_tab_view = self.main_view.trade_tab
        return trade_tab_view
    
    def exchange_tab_view(self)-> Frame:
        exchange_tab_view = self.main_view.exchange_tab
        return exchange_tab_view
    
    def bot_tab_view(self)-> Frame:
        bot_tab_view = self.main_view.bot_tab
        return bot_tab_view
    
    def chart_tab_view(self)-> Frame:
        chart_tab_view = self.main_view.chart_tab
        return chart_tab_view
    
    def ml_tab_view(self)-> Frame:
        ml_tab_view = self.main_view.ml_tab
        return ml_tab_view
      
            
    # Trading tab     ----------------------------------------------------        
    def update_stoploss(self):
        trade_tab_view = self.trade_tab()
        stoploss_slider = trade_tab_view.stoploss_slider.get()
        self._model.tradetab_model.update_stoploss(stoploss_slider)
    
    def update_takeprofit(self):
        trade_tab_view = self.trade_tab()
        takeprofit_slider = trade_tab_view.takeprofit_slider.get()
        self._model.tradetab_model.update_takeprofit(takeprofit_slider)
    
    def place_trade(self):        
        trade_tab_view = self.trade_tab()
        
        # Get the trade type, amount, and price from the GUI
        trade_type = trade_tab_view.type_var.get()
        amount = float( trade_tab_view.amount_entry.get())
        price = float(trade_tab_view.price_entry.get())
        
        # Create a Trading instance with the selected exchange, symbol, side, stop loss, and take profit levels, and the API key

        allowed_position = self._model.tradetab_model._trading.getBalancUSDT * MAX_POSITION_SIZE
        if abs(float(amount)) > allowed_position:
            messagebox.showerror(
                "Error", "Trade size exceeds maximum position size")
            return
        
        bid = self._model.tradetab_model._trading.getBidAsk()[0]
        # Check if the stop loss level is above the minimum level
        if trade_type == "stop" and price < float(bid) * MIN_STOP_LOSS_LEVEL:
            messagebox.showerror(
                "Error", "Stop loss level is below minimum level")
            return
        
        
        # Get the stop loss and take profit levels from the sliders
        stoploss = price - (price * (float(trade_tab_view.stoploss_slider.get())))
        takeprofit = price + (price * (float(trade_tab_view.takeprofit_slider.get())))

        ml_tab = self.ml_tab_view()
        
        file = ml_tab.type_var.get()
        
        self._model.tradetab_model.place_trade(trade_type, amount, price, stoploss, takeprofit, file)
        
        self._model.logger.info(f"Type: {trade_type} {amount} at {price}")
        # Update the trade history list
        # self.trade_tab.update_history()
        
        
        
    # Bot tab  ----------------------------------------------------------------------
    def toggle_auto_trade(self) ->None:
        self.auto_trade = self._model.bottab_model.toggle_auto_trade()
        self.bot_tab = self.bot_tab_view()
        if self.auto_trade:
            # Update the status label
            self.bot_tab.auto_trade_button.config(text="Stop Auto Trade")
        else:
            # Update the status label
            self.bot_tab.auto_trade_button.config(text="Start Auto Trade")
    
    def get_bot(self):
        bot = self._model.tradetab_model.get_trading(self._model.tradetab_model.symbol)
        return bot
    

    def get_data_ml_files(self) -> list:
        files = self._model.bottab_model.get_data_ml_files()
        return files
    
    # def get_auto_bot(self):
    #     exchange = self.get_autobot_exchange()
    #     symbol = 
    #     amount = 
    #     stoploss = 
    #     takeprofit = 
        
    #     autobot = self._model.get_autobot(exchange, symbol, amount,stoploss,takeprofit)
    #     return autobot
    
    def get_autobot_exchange(self):
        exchange_tab =  self.exchange_tab_view()
        exchange_name = exchange_tab.exchange_var.get()
        key = exchange_tab.api_key_entry.get()
        secret = exchange_tab.api_secret_entry.get()
        
        if key or secret or exchange_name == None:
            exchange_name = "phemex"
            key = os.environ.get('API_KEY_PHE_TEST')
            secret = os.environ.get('API_SECRET_PHE_TEST')
        
        exchange = getattr(ccxt, exchange_name)({'apiKey': key,
                        'secret': secret,
                        'rateLimit': 2000,
                        'enableRateLimit': True})
        return exchange
        
    # Chart tab  -------------------------------------------------------------------
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
            self._view.after(60_000, self.update_chart, stop_event)
            
    def toggle_auto_charting(self) -> None:
        self.auto_chart = self._model.charttab_model.toggle_auto_charting()
        self.chart_tab = self.chart_tab_view()
        if self.auto_chart:
            # Clear the figure
            self.chart_tab.axes.clear()
            # Update the status label
            self.chart_tab.start_autochart_button.config(text="Stop Auto Charting")
            
        else:
            # Clear the figure
            self.chart_tab.axes.clear()
            # Update the status label
            self.chart_tab.start_autochart_button.config(text="Start Auto Charting")
    
    
    # ML tab -------------------------------------------------------------------
    def get_ML_model(self) -> str:
        self.ml_tab = self.ml_tab_view()
        selected_algorithm = self.ml_tab.type_var.get()
        self.ml_tab.current_ml_label.config(text=f"{selected_algorithm}")
        self._model.logger.info(f"machine learning model {selected_algorithm} selected")
        
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
        
        return predict
    
    def get_ml(self):
        ml = self._model.get_ML()
        return ml