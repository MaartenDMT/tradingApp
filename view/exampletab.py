from ttkbootstrap import Button, Frame, Label, OptionMenu, StringVar, Notebook, Entry
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import ccxt
from tkinter import BOTH, RIGHT, LEFT, TOP, BOTTOM, messagebox
import threading
import time 
import pandas as pd
import logging
from model.trading import Trading

class Example(Frame):
    def __init__(self):
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
        self.exchange = ccxt.binance()
        # get the balance of the exchange
        self.balance = self.exchange.fetch_balance()
        # Create a dropdown menu for selecting the exchange
        self.exchange_var = StringVar(self.exchange_pages)
        self.exchange_var.set("Binance")  # default value
        self.exchange_menu = OptionMenu(self.exchange_pages, self.exchange_var,
                                            *self.exchange.load_markets().keys())
        # Create a label and entry box for the order size
        self.order_size_label = Label(
            self.trade_page, text="Order size")
        self.order_size_entry = Entry(self.trade_page)
        # Create a label and entry box for the order unit
        self.order_unit_label = Label(
            self.trade_page, text="Order unit")
        self.order_unit_entry = Entry(self.trade_page)
        # Create a trade button
        self.trade_button = Button(
            self.trade_page, text="Trade", command=self.trade)
        # Create a label and entry box for the symbol
        self.symbol_label = Label(self.trade_page, text="Symbol")
        self.symbol_entry = Entry(self.trade_page)
        # Create a label and entry box for the timeframe
        self.timeframe_label = Label(self.trade_page, text="Timeframe")
        self.timeframe_entry = Entry(self.trade_page)
        # Create a button for fetching the chart data
        self.fetch_data_button = Button(
            self.trade_page, text="Fetch Data", command=self.fetch_data)
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

        # Start the charting thread
        self.chart_thread = threading.Thread(target=self.auto_chart_thread)
        self.chart_thread.daemon = True
        self.chart_thread.start()
    
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
            self.bot = Trading(self)
        else:
            self.bot.destroy()
            self.bot = Trading(self)
            
    # This function is called when the user clicks the Destroy button for a bot
    def destroy_bot(self, bot):
        if bot.running:
            bot.stop()
        self.bots.remove(bot)

    def update_chart(self):
        if self.auto_chart:
            self.auto_chart = False
            self.start_autochart_button.config(text="Start Auto Charting")
            self.bot.subscribe_to_market(self.bot.symbol, self.update_chart)
        else:
            self.auto_chart = True
            self.start_autochart_button.config(text="Stop Auto Charting")
            self.bot.unsubscribe_from_market(
                self.bot.symbol, self.update_chart)
            thread = threading.Thread(target=self.chart_bot_price)
            thread.daemon = True
            thread.start()

    def plot_data(self, data, type):
        data = pd.DataFrame(data)
        data = data.sort_values(by='timestamp')
        self.axes.clear()
        self.axes.plot(data['timestamp'], data['close'], label=type)
        self.axes.set_ylabel('Price')
        self.axes.set_xlabel('Time')
        self.canvas.draw()

    def chart_bot_price(self):
        while self.auto_chart:
            try:
                # Get the current price from the bot
                price = self.bot.get_price()
                # Add the price to the list of prices
                self.prices.append(price)
                # Clear the axis
                self.axes.clear()
                # Plot the new list of prices
                self.axes.plot(self.prices)
                # Draw the canvas
                self.canvas.draw()
                # Sleep for 5 seconds
                time.sleep(5)
            except Exception as e:
                logging.error(e)
                break

    def get_bot(self):
        # Get the selected exchange
        exchange = self.selected_exchange.get()
        # Get the selected market
        market = self.selected_market.get()
        # Create a new instance of the bot using the selected exchange and market
        self.bot = Trading(exchange, market)
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
