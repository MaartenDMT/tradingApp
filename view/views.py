
from tkinter import Listbox, messagebox

from ttkbootstrap import (Button, Entry, Frame, Label, Menu, Notebook,
                          StringVar, Style, Window)

from view.bottab import BotTab
from view.charttab import ChartTab
from view.exchangetab import ExchangeTab
from view.mltab import MLTab
from view.tradetab import TradeTab


class WindowView(Window):
    def __init__(self):
        super().__init__()
        
        self.geometry("300x200")
        self.title("Trading App")

        self._style.theme_use(themename='superhero')
        self.loginview = LoginView
        self.main_view = MainView
        
    def show_frame(self, cont, presenter)-> None:
        # Destroy the current frame
        for widget in self.winfo_children():
            widget.destroy()
            
        # Raise the new frame
        frame = cont 
        frame.__init__(self)
        frame.create_ui(presenter)
        frame.grid(row=1, column=0)
        frame.tkraise()

    



class LoginView(Frame):
    def __init__(self, parent):
        super().__init__()
        self._parent = parent

        self._username_var = StringVar()
        self._password_var = StringVar()
        
    def create_ui(self, presenter):
        self._presenter = presenter

        Label(self, text="Username").grid(row=0, column=0, padx=5, pady=5)
        self.username_entry = Entry(self, textvariable=self._username_var).grid(row=0, column=1, padx=5, pady=5)

        Label(self, text="Password").grid(row=1, column=0, padx=5, pady=5)
        self.password_entry = Entry(self, textvariable=self._password_var, show="*").grid(row=1, column=1, padx=5, pady=5)

        self.login = Button(self, text="Login", command=self._presenter.on_login_button_clicked)
        self.login.config(width=10)
         
        self.registering = Button(self, text="Register", command=self._presenter.on_register_button_clicked)
        self.registering.config(width=10)

        self.login.grid(row=2, column=0, pady=5)
        self.registering.grid(row=2, column=1, pady=5)

    def get_username(self) -> str:
        return self._username_var.get()

    def get_password(self) -> str:
        return self._password_var.get()
    
    def login_failed(self) -> None:
        self.username_entry.delete(0, "end")
        self.password_entry.delete(0, "end")
        self.username_entry.insert(0, "Invalid credentials. Try again.")

    def show_error_message(self, message: str):
        messagebox.showerror("Error", message)


class MainView(Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._parent = parent
        
        
        self._parent.title("Main Screen")
        self._parent.geometry("860x600")
        style = Style()
        themes = style.theme_names()
    
        self._parent.menu = Menu(self)
        self._parent.theme_menu = Menu(self._parent.menu, tearoff=0)
        self._parent.menu.add_cascade(label="Themes", menu=self._parent.theme_menu)
        

        # Add a menu item for each theme
        for theme in themes:
            self._parent.theme_menu.add_command(
                label=theme, command=lambda theme=theme: self.changer(theme))

        # Set the menu as the menu bar of the root window
        self._parent.config(menu=self._parent.menu)
        
    def create_ui(self, presenter):
        self._presenter = presenter
        
        Label(self, text="Welcome, to the trading app!").grid(row=0, column=0, padx=5, pady=5)
        
        # Add a list box for displaying the trade history on the history page
        self.history_list = Listbox(self)

        # Create the main window with a tabbed interface
        self.notebook = Notebook(self, width='600')
        self.trade_tab= TradeTab(self.notebook, self._presenter)
        self.exchange_tab = ExchangeTab(self.notebook, self._presenter)
        self.bot_tab = BotTab(self.notebook, self._presenter)
        self.chart_tab = ChartTab(self.notebook, self._presenter)
        self.ml_tab = MLTab(self.notebook, self._presenter)
        
        
        self.notebook.add(self.trade_tab, text="Trade")
        self.notebook.add(self.exchange_tab, text="Exchanges")
        self.notebook.add(self.bot_tab, text="Bot")
        self.notebook.add(self.chart_tab, text="Chart")
        self.notebook.add(self.ml_tab, text="Machine Learning")
        
        
        # Tkinter App main page ----------------------------------------------
        self.notebook.grid(row=1, column=0,padx=5)
        self.history_list.grid(row=2, column=0, padx=5, pady=5, sticky='nsew')
        
    def changer(self, theme) -> None:
        Style().theme_use(theme)
