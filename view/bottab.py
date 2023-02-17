from tkinter import messagebox

from ttkbootstrap import Button, Frame, Label, OptionMenu, Scale, StringVar


class BotTab(Frame):
    def __init__(self, parent, presenter) -> None:
        super().__init__(parent)
        self._parent = parent
        self._presenter = presenter
        self.exchange = self._presenter.get_exchange()
        self.bot_names = []
        
        # ... GUI elements for the trade page go here ...
        self.options_ml_frame = Frame(self)
        self.bot_frame = Frame(self)
        
        #add the bots 
        self.bot_select_var = StringVar(self.bot_frame)
        self.bot_select = OptionMenu(self.bot_frame,self.bot_select_var , *[])
        self.bot_select.config(width=25)
        
        # Machine learning 
        self.optionmenu_var = StringVar(self.options_ml_frame)
        self.optionmenu_var.set('Select a file')
        files = self._presenter.get_data_ml_files()
        self.options_ml_model = OptionMenu(self.options_ml_frame, self.optionmenu_var, *files)
        
        
        index = 0
        # Add the auto Trade pages 
        self.auto_trade_label = Label(self.bot_frame, text='Trading Bot', font=('Arial', 16))
        # # Add a button for enabling and disabling automatic trading
        self.auto_trade_button = Button(
            self.bot_frame, text="Start Auto Trade", command=lambda: self._presenter.toggle_auto_trade(index))
        
        # Create a dropdown menu for selecting the exchange
        list_time = ['1m', '5m', '30m', '1h','3h']
        self.time_label = Label(self.bot_frame, text="Time", font=('Arial', 12))
        self.time_var = StringVar(self.bot_frame)
        self.time_var.set("30m")  # default value
        self.time_menu = OptionMenu(self.bot_frame, self.time_var,
                                            *list_time)
        
        # Create a dropdown menu for selecting the exchange
        self.symbol_label = Label(self.bot_frame, text="Symbol", font=('Arial', 12))
        self.exchange_var = StringVar(self.bot_frame)
        self.exchange_var.set("BTC/USDT")  # default value
        self.exchange_menu = OptionMenu(self.bot_frame, self.exchange_var,
                                            *self.exchange.load_markets().keys())

        self.amount_used = Label(self.bot_frame, text="percentage bot use", font=('Arial', 12))
        self.amount_slider = Scale(self.bot_frame, from_=0, to_=100, orient='horizontal')
        self.amount_slider.config(value=50)
        
        self.profit_used = Label(self.bot_frame, text="profit percentage", font=('Arial', 12))
        self.profit_slider = Scale(self.bot_frame, from_=0, to_=100, orient='horizontal')
        self.profit_slider.config(value=15)
        
        self.loss_used = Label(self.bot_frame, text="loss percentage", font=('Arial', 12))
        self.loss_slider = Scale(self.bot_frame, from_=0, to_=100, orient='horizontal')
        self.loss_slider.config(value=8)
        
        # self.bot_button = Button(
        # self.bot_frame, text="create a bot", command=self._presenter.get_bot)  
        self.start_button = Button(self.bot_frame, text="Start Bot", command=self.start_bot)
        self.start_button.config(width=15)
        
        self.stop_button = Button(self.bot_frame, text="Stop Bot", command=self.stop_bot)
        self.stop_button.config(width=15)
        
        self.create_button = Button(self.bot_frame, text="Create Bot", command=self.create_bot)
        self.create_button.config(width=15)
        
        self.destroy_button = Button(self.bot_frame, text="Destroy Bot", command=self.destroy_bot)
        self.destroy_button.config(width=15)
        
        # AUTO TRADING BOT
        
        self.options_ml_frame.grid(row=0, column=0, padx=5, pady=5)
        self.options_ml_model.grid(row=0, column=0, padx=5, pady=5, sticky='NSEW')
        #-----------------------------------------------------------------------------
        
        self.bot_frame.grid(row=1, column=0, padx=5, pady=5, sticky='NSEW')
        
        self.auto_trade_label.grid(row=2, column=0, padx=5, pady=5)
        
        self.create_button.grid(row=3, column=0, padx=5, pady=5)
        self.start_button.grid(row=4, column=0, padx=5, pady=5)
        self.stop_button.grid(row=5, column=0, padx=5, pady=5)
        self.destroy_button.grid(row=6, column=0, padx=5, pady=5)
        # self.auto_trade_button.grid(row=3, column=0, padx=5, pady=5)
        
        self.amount_used.grid(row=2, column=1, padx=5, pady=5)
        self.amount_slider.grid(row=3, column=1, padx=5, pady=5)
        
        self.profit_used.grid(row=4, column=1, padx=5, pady=5)
        self.profit_slider.grid(row=5, column=1, padx=5, pady=5)
        
        self.loss_used.grid(row=6, column=1, padx=5, pady=5)
        self.loss_slider.grid(row=7, column=1, padx=5, pady=5)   
        # self.bot_button.grid(row=8, column=1, padx=5, pady=5)
        
        #------------------------------------------------------------------------------
        self.symbol_label.grid(row=2, column=2, padx=5, pady=5)
        self.exchange_menu.grid(row=3, column=2, padx=5, pady=5)
        
        self.time_label.grid(row=4, column=2, padx=5, pady=5)
        self.time_menu.grid(row=5, column=2, padx=5, pady=5)
        self.bot_select.grid(row=6, column=2, padx=5, pady=5)
        
    

    def start_bot(self):
        selection = int(self.bot_select_var.get()) -1
        print(selection)
        if selection == None:
            messagebox.showerror("Error", "there is no bot selected!")
        if isinstance(selection, int):
            self._presenter.start_bot(selection)
        else:
            messagebox.showerror("Error", "you need to select a bot to start")
            
    def stop_bot(self):
        selection = int(self.bot_select_var.get()) -1 
        print(selection)
        if selection == None:
            messagebox.showerror("Error", "there is no bot selected!")
        if isinstance(selection, int):
            self._presenter.stop_bot(selection)
        else:
            messagebox.showerror("Error", "there is no bot to stop")

    
    def create_bot(self) -> None:
        self._presenter.threading_createbot()

    def destroy_bot(self) -> None:
        selection = int(self.bot_select_var.get()) - 1
        print(selection)
        if isinstance(selection, int):
            self._presenter.destroy_bot(selection)
        else:
            messagebox.showerror("Error", "there is no bot to destroy")

    def add_bot_to_optionmenu(self, count: int, name)-> None:
        self.bot_select["menu"].add_command(label=f"{name}{count}", command=lambda: self.bot_select_var.set(f"{count}"))

    def remove_bot_from_optionmenu(self, index: int) -> None:
        if index < int(self.bot_select_var.get()):
            self.bot_select["menu"].delete(index)
        else:
            messagebox.showerror("Error", "There is no bot to remove.")

    def update_bot_status(self, status: str, index: int, name) -> None:
        self.bot_select["menu"].entryconfigure(index, label=f"{name}{index + 1} ({status})")