from tkinter import END, BooleanVar, Listbox

import ccxt
from ttkbootstrap import (Button, Checkbutton, Entry, Frame, Label, OptionMenu,
                          StringVar)

# class ExchangeTab(Frame):
#     def __init__(self, parent, presenter) -> None:
#         super().__init__(parent)
#         self._parent = parent
#         self._presenter = presenter
#         # ... GUI elements for the trade page go here ...

#         # Create a dropdown menu for selecting the exchange
#         self.exchange_var = StringVar(self)
#         self.exchange_menu = OptionMenu(
#             self, self.exchange_var, 'phemex',  *ccxt.exchanges, direction='flush')
#         self.api_key_label = Label(self, text='API Key:')
#         self.api_key_entry = Entry(self)
#         self.api_secret_label = Label(
#             self, text='API Secret:')
#         self.api_secret_entry = Entry(self, show='*')

#         self.load_firxt_exchange = Button(
#             self, text='Load', command=self._presenter.exchange_tab.save_first_exchange)
#         self.submit_exchange = Button(
#             self, text='Create', command=self._presenter.exchange_tab.create_exchange)
#         self.remove_exchange = Button(
#             self, text='Remove', command=self._presenter.exchange_tab.remove_exchange)
#         self.history_exchange = Listbox(self)

#         self.text_exchange_var = BooleanVar(self)
#         self.test_exchange = Checkbutton(
#             self, text='Testnet', variable=self.text_exchange_var)

#         # self exchange page -------------------------------------------------

#         self.exchange_menu.grid(row=0, column=0, pady=5)
#         self.load_firxt_exchange.grid(row=0, column=1, pady=5)
#         self.api_key_label.grid(row=1, column=0, pady=5)
#         self.api_key_entry.grid(row=1, column=1, pady=5)
#         self.api_secret_label.grid(row=2, column=0, pady=5)
#         self.api_secret_entry.grid(row=2, column=1, pady=5)
#         self.submit_exchange.grid(row=3, column=0, pady=5)
#         self.remove_exchange.grid(row=3, column=1, pady=5)
#         self.history_exchange.grid(row=6, column=0, pady=5, padx=5)
#         self.test_exchange.grid(row=6, column=1, pady=5, padx=5)

#     def add_exchange_optionmenu(self, exchange):
#         self.history_exchange.insert(END, exchange)

#     def select_exchange(self):
#         exchange = self.history_exchange.curselection()[0]
#         return exchange

#     def remove_exchange_from_optionmenu(self) -> None:
#         index = self.history_exchange.curselection()[0]
#         self.history_exchange.delete(index)
#         return index

#     def update_exchange_status(self, status: str, index: int, name) -> None:
#         ...

class ExchangeTab(Frame):
    def __init__(self, parent, presenter) -> None:
        super().__init__(parent)
        self._presenter = presenter

        self.exchange_var = StringVar(self)
        self.exchange_var.set("Select Exchange")
        self.exchange_menu = OptionMenu(
            self, self.exchange_var, 'phemex', *ccxt.exchanges)

        self.api_key_label = Label(self, text='API Key:')
        self.api_key_entry = Entry(self)
        self.api_secret_label = Label(self, text='API Secret:')
        self.api_secret_entry = Entry(self, show='*')
        self.show_secret_button = Button(
            self, text='Show', command=self.toggle_secret_visibility)

        self.load_first_exchange_button = Button(
            self, text='Load', command=self._presenter.exchange_tab.save_first_exchange)
        self.submit_exchange_button = Button(
            self, text='Create', command=self._presenter.exchange_tab.create_exchange)
        self.remove_exchange_button = Button(
            self, text='Remove', command=self._presenter.exchange_tab.remove_exchange)

        self.history_exchange = Listbox(self)
        self.test_exchange_check = Checkbutton(
            self, text='Testnet', variable=BooleanVar(self))

        # Layout the widgets
        self.layout_widgets()

    def layout_widgets(self):
        self.exchange_menu.grid(row=0, column=0, padx=5, pady=5)
        self.load_first_exchange_button.grid(row=0, column=1, padx=5, pady=5)
        self.api_key_label.grid(row=1, column=0, padx=5, pady=5)
        self.api_key_entry.grid(row=1, column=1, padx=5, pady=5)
        self.api_secret_label.grid(row=2, column=0, padx=5, pady=5)
        self.api_secret_entry.grid(row=2, column=1, padx=5, pady=5)
        self.show_secret_button.grid(row=2, column=2, padx=5, pady=5)
        self.submit_exchange_button.grid(row=3, column=0, padx=5, pady=5)
        self.remove_exchange_button.grid(row=3, column=1, padx=5, pady=5)
        self.history_exchange.grid(
            row=4, column=0, columnspan=2, padx=5, pady=5)
        self.test_exchange_check.grid(
            row=5, column=0, columnspan=2, padx=5, pady=5)

    def toggle_secret_visibility(self):
        current_show = self.api_secret_entry.cget('show')
        new_show = '' if current_show == '*' else '*'
        self.api_secret_entry.config(show=new_show)
        self.show_secret_button.config(
            text='Hide' if new_show == '' else 'Show')

    def add_exchange_optionmenu(self, exchange):
        self.history_exchange.insert(END, exchange)

    def select_exchange(self):
        exchange = self.history_exchange.curselection()[0]
        return exchange

    def remove_exchange_from_optionmenu(self) -> None:
        index = self.history_exchange.curselection()[0]
        self.history_exchange.delete(index)
        return index

    def update_exchange_status(self, status: str, index: int, name) -> None:
        ...
