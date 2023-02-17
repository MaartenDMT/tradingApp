import logging
import os
import threading
import time
from tkinter import BOTH, BOTTOM, LEFT, RIGHT, TOP, messagebox

import ccxt
import pandas as pd
from autobot import AutoBot
from dotenv import load_dotenv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ttkbootstrap import (Button, Entry, Frame, Label, Notebook, OptionMenu,
                          StringVar, Window)

# Set the path to the .env file
dotenv_path = r'.env'
# Load the environment variables from the .env file located two directories above
load_dotenv(dotenv_path)

class Example(Window):
    def __init__(self):
        super().__init__()
        # Create the main window with a tabbed interface
        self.notebook = Notebook(self, width='600')

        self.trade_page = Frame(self.notebook)
        self.history_page = Frame(self.notebook)

        # Create a frame to hold the trade button and exchange dropdown menu
        self.exchange_pages = Frame(self.notebook)

        # Create a frame to hold the figure
        self.chart_frame = Frame(self.notebook)
        # ... GUI elements for the order size and unit are created here ...
        # Create a figure and a canvas to display the chart
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.chart_frame)
        self.canvas.draw()
        self.axes.set_ylabel('Price')
        self.axes.set_xlabel('Time')

        # Add a button for enabling and disabling automatic charting
        self.auto_chart = False
        self.start_autochart_button = Button(
            self.chart_frame, text="Start Auto Charting", command=self.update_chart)

        self.notebook.add(self.trade_page, text="Trade")
        self.notebook.add(self.history_page, text="History")
        self.notebook.add(self.exchange_pages, text="Exchanges")
        self.notebook.add(self.chart_frame, text="Chart")

        self.bot_button = Button(
            self.trade_page, text="create a bot", command=self.get_bot)
        # Create an instance of the exchange using the CCXT library
        self.exchange = getattr(ccxt, "phemex")({'apiKey': os.environ.get('API_KEY_PHE_TEST'),
                        'secret': os.environ.get('API_SECRET_PHE_TEST'),
                        'rateLimit': 2000,
                        'enableRateLimit': True})
        self.exchange.set_sandbox_mode(True)
        # get the balance of the exchange
        self.balance = self.exchange.fetch_balance()
        # Create a dropdown menu for selecting the exchange
        self.exchange_var = StringVar(self.exchange_pages)
        self.exchange_var.set("BTC/USDT")  # default value
        self.exchange_menu = OptionMenu(self.exchange_pages, self.exchange_var,
                                            *self.exchange.load_markets().keys())
        # Create a label and entry box for the Amount to be used 
        self.order_size_label = Label(
            self.trade_page, text="Amount")
        self.order_size_entry = Entry(self.trade_page)
        # Create a label and entry box for the order unit
        self.order_unit_label = Label(
            self.trade_page, text="Order unit")
        self.order_unit_entry = Entry(self.trade_page)
        # Create a trade button
        # self.trade_button = Button(
        #     self.trade_page, text="Trade", command=self.trade)
        # Create a label and entry box for the symbol
        self.symbol_label = Label(self.trade_page, text="Symbol")
        self.symbol_entry = Entry(self.trade_page)
        # Create a label and entry box for the timeframe
        self.timeframe_label = Label(self.trade_page, text="Timeframe")
        self.timeframe_entry = Entry(self.trade_page)
        # Create a button for fetching the chart data
        # self.fetch_data_button = Button(
        #     self.trade_page, text="Fetch Data", command=self.fetch_data)
        # Create a label and entry box for the amount of time to wait between trades
        self.wait_time_label = Label(
            self.trade_page, text="Wait Time (s)")
        self.wait_time_entry = Entry(self.trade_page)
        # Create a start and stop button for the bot
        self.start_bot_button = Button(
            self.trade_page, text="Start Bot", command=self.start_bot)
        self.stop_bot_button = Button(
            self.trade_page, text="Stop Bot", command=self.stop_bot)
        self.bot_status_label = Label(self.trade_page, text="Bot is stopped")
        
        # Add the widgets to the window
        self.notebook.pack(expand=True, fill=BOTH)
        self.start_autochart_button.pack(side=RIGHT)
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        self.start_bot_button.pack()
        self.stop_bot_button.pack()
        self.bot_status_label.pack()
        
        
        self.exchange_menu.pack()
        self.symbol_label.pack()
        self.symbol_entry.pack()
        
        self.timeframe_label.pack()
        self.timeframe_entry.pack()
        
        self.order_size_label.pack()
        self.order_size_entry.pack()
        
        self.order_unit_label.pack()
        self.order_unit_entry.pack()
        
        # Start the charting thread
        # self.chart_thread = threading.Thread(target=self.auto_chart_thread)
        # self.chart_thread.daemon = True
        # self.chart_thread.start()
    
    def start_bot(self):
        """Start the bot."""
        if self.bot is None:
            messagebox.showerror(
                "Error", "Please create a bot using the 'Create a bot' button.")
            return

        if self.bot.is_running:
            messagebox.showerror("Error", "Bot is already running.")
            return

        self.bot.start()

    # This function is called when the user clicks the Start/Stop button for a bot
    def start_stop_bot(self, bot):
        if bot.running:
            bot.stop()
        else:
            bot.start()

    def stop_bot(self):
        """Stop the bot."""
        if self.bot is None:
            messagebox.showerror(
                "Error", "Please create a bot using the 'Create a bot' button.")
            return

        if not self.bot.is_running:
            messagebox.showerror("Error", "Bot is not running.")
            return

        self.bot.stop()

    def get_bot(self):
        """Get an instance of the bot."""
        if self.bot is None:
            self.bot = AutoBot(self)
        else:
            self.bot.destroy()
            self.bot = AutoBot(self)
            
    # This function is called when the user clicks the Destroy button for a bot
    def destroy_bot(self, bot):
        if bot.running:
            bot.stop()
        self.bots.remove(bot)

    def update_chart(self):
        if self.auto_chart:
            self.start_autochart_button.config(text="Stop Auto Charting")
            while self.auto_chart:
                # Fetch data from the exchange
                data = self.exchange.fetch_ohlcv(self.bot.symbol)
                df = pd.DataFrame(data)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
                # Update the chart
                self.axes.clear()
                self.axes.plot(df["timestamp"], df["close"])
                self.canvas.draw()
                time.sleep(5)
        else:
            self.start_autochart_button.config(text="Start Auto Charting")

    def plot_data(self, data, type):
        data = pd.DataFrame(data)
        data = data.sort_values(by='timestamp')
        self.axes.clear()
        self.axes.plot(data['timestamp'], data['close'], label=type)
        self.axes.set_ylabel('Price')
        self.axes.set_xlabel('Time')
        self.canvas.draw()

    def get_bot(self):
        symbol = self.symbol_entry.get()
        order_size = float(self.order_size_entry.get())
        order_unit = float(self.order_unit_entry.get())
        # Get the selected exchange
        exchange = self.selected_exchange.get()
        # Get the selected market
        market = self.selected_market.get()
        # Create a new instance of the bot using the selected exchange and market
        self.bot = AutoBot(exchange, symbol)
        # Enable the start and stop button
        self.start_bot_button.config(state='normal')
        self.stop_bot_button.config(state='normal')
        # Clear the prices list and start charting the new bot's price
        self.prices.clear()
        self.update_chart()
        # Call the update_chart function every second when automatic charting is enabled
        if self.auto_chart:
            self.update_chart()
            self.after(1000, self.update_chart)

    # This function is called when the user selects a different exchange from the dropdown menu
    def change_exchange(self, event):
        # Get the name of the selected exchange
        exchange_name = event.widget.get()
        # Destroy any existing bots
        for bot in self.bots:
            self.destroy_bot(bot)
        self.bots = []
        # Create a new instance of the exchange using the CCXT library
        self.exchange = getattr(ccxt, exchange_name)()
        # Create a frame for the selected exchange
        exchange_frame = Frame(self.exchange_pages)
        # Create a label showing the selected exchange
        exchange_label = Label(exchange_frame, text=exchange_name)
        # Create a start and stop button for the bot
        self.start_stop_bot_button = Button(
            exchange_frame, text="Start", command=lambda: self.start_stop_bot)

    def on_close(self):
        if self.bot:
            self.bot.close()
        super().on_close()

def main():
    # create instance of the app
    app = Example()
    # Run the app loop
    app.mainloop()


if __name__ == "__main__":
    main()
