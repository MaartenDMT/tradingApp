


from ttkbootstrap import Button, Entry, Frame, Label, OptionMenu, StringVar, Scale
from ttkbootstrap.constants import *

from util.utils import validate_float


class TradeTab(Frame):
    
    def __init__(self, parent, presenter) -> None:
        super().__init__(parent)
        self._parent = parent
        self._presenter = presenter
        
        # ... GUI elements for the trade page go here ...
        self.type_var = StringVar(self)
        self.type_var.set("limit")
        self.type_menu = OptionMenu(
            self, self.type_var, "market", "limit", "stop")
        
         # Add sliders for setting the stop loss and take profit levels on the trade page
        self.stoploss_slider = Scale(self, from_=0, to=100, orient='horizontal')
        self.takeprofit_slider = Scale(self, from_=0, to=100, orient='horizontal')
        self.stoploss_slider.config(value=8)
        self.takeprofit_slider.config(value=10)
        
        # Add a button for placing the trade
        self.trade_button = Button(
            self, text="Trade", command=self._presenter.place_trade)

        # # Add buttons for modifying the stop loss and take profit levels
        self.stop_loss_button = Button(self, text="Modify Stop Loss",
                                           command=self._presenter.update_stoploss)
        self.take_profit_button = Button(self, text="Modify Take Profit",
                                             command=self._presenter.update_takeprofit)
         # Add text boxes for entering the trade amount and price
        self.amount_label = Label(
            self, text='Amount', font=('Arial', 12))
        self.amount_entry = Entry(self, validate='key', validatecommand=(
            self.register(validate_float), '%d', '%i', '%P', '%S', '%T'))

        self.price_label = Label(
            self, text='Price', font=('Arial', 12))
        self.price_entry = Entry(self, validate='key', validatecommand=(
            self.register(validate_float), '%d', '%i', '%P', '%S', '%T'))
        
    
        # Notebook Trading page --------------------------------------------------------
        
        self.type_menu.grid(row=0, column=0, pady=5)

        self.amount_label.grid(row=1, column=0, pady=5)
        self.price_label.grid(row=1, column=1, pady=5)

        self.amount_entry.grid(row=2, column=0, pady=5)
        self.price_entry.grid(row=2, column=1, pady=5)

        self.trade_button.grid(row=3, column=0, columnspan=2, pady=5)

        self.stop_loss_button.grid(row=4, column=0, pady=5)
        self.take_profit_button.grid(row=4, column=1, pady=5)

        self.stoploss_slider.grid(row=6, column=0, pady=5)
        self.takeprofit_slider.grid(row=6, column=1, pady=5)
        
        