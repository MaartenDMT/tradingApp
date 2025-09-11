"""
Refactored Views Module

This module provides refactored and enhanced versions of the trading application views
with improved user feedback, consistent styling, and better performance.
"""

import asyncio
import threading
import traceback
from datetime import datetime
from tkinter import Listbox, messagebox
from typing import Any, Callable, Dict, List, Optional

try:
    from ttkbootstrap import (
        END, Button, Entry, Frame, Label, Menu, Notebook, StringVar, Style, 
        Treeview, Window, Progressbar, Separator, ToastNotification
    )
    from ttkbootstrap.constants import *
    HAS_TTKBOOTSTRAP = True
except ImportError:
    # Fallback to tkinter equivalents
    from tkinter import (
        END, Button, Entry, Frame, Label, Menu, StringVar, Tk as Window
    )
    from tkinter.ttk import Notebook, Style, Progressbar, Separator, Treeview
    HAS_TTKBOOTSTRAP = False

# Import tab views
from view.tradetab import OptimizedTradeTab
from view.bottab_optimized import OptimizedBotTab
from view.charttab import OptimizedChartTab
from view.mltab import OptimizedMLTab
from view.rltab import OptimizedRLTab

# Import new system tabs
from view.trading_system_tab import TradingSystemTab
from view.ml_system_tab import MLSystemTab
from view.rl_system_tab import RLSystemTab

# Import advanced features tabs
from view.advanced_rl_system_tab import AdvancedRLSystemTab

# Import view utilities
from view.utils import UIThemeManager, ValidationMixin, StatusIndicator

import util.loggers as loggers

logger_dict = loggers.setup_loggers()
app_logger = logger_dict['app']


class BaseView(Frame):
    """Base class for all views in the trading application."""
    
    def __init__(self, parent):
        """Initialize the base view.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.parent = parent
        self.logger = app_logger
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize the view.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            if not self._initialized:
                self.logger.info(f"Initializing {self.__class__.__name__}")
                self._initialize()
                self._initialized = True
                self.logger.info(f"Successfully initialized {self.__class__.__name__}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.__class__.__name__}: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def _initialize(self):
        """Internal initialization method to be implemented by subclasses."""
        pass
    
    def show_message(self, title: str, message: str, level: str = "info"):
        """Show a message to the user.
        
        Args:
            title: Message title
            message: Message content
            level: Message level ('info', 'warning', 'error')
        """
        try:
            if HAS_TTKBOOTSTRAP:
                # Use ToastNotification for better user experience
                toast = ToastNotification(
                    title=title,
                    message=message,
                    duration=3000,  # 3 seconds
                    alert=level == "error"
                )
                toast.show_toast()
            else:
                # Fallback to messagebox
                if level == "error":
                    messagebox.showerror(title, message)
                elif level == "warning":
                    messagebox.showwarning(title, message)
                else:
                    messagebox.showinfo(title, message)
        except Exception as e:
            self.logger.error(f"Failed to show message: {e}")
            
    def show_progress(self, message: str, progress: float):
        """Show progress to the user.
        
        Args:
            message: Progress message
            progress: Progress value (0.0 to 1.0)
        """
        # This would typically update a progress bar in the UI
        self.logger.info(f"Progress: {message} ({progress:.1%})")


class OptimizedWindowView(Window):
    """
    Enhanced main window with optimized performance and modern UI features.
    """
    
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.geometry("1200x800")
        self.title("Trading Application - Professional Edition")
        self.minsize(800, 600)
        
        # Initialize theme manager
        self.theme_manager = UIThemeManager(self)
        self.theme_manager.set_theme('superhero')
        
        # Performance tracking
        self._start_time = datetime.now()
        self._frame_switches = 0
        self._last_frame_switch = datetime.now()
        
        # View references
        self._loginview_class = OptimizedLoginView
        self._main_view_class = OptimizedMainView
        
        # Status tracking
        self._current_frame = None
        self._cleanup_callbacks = []
        
        # Enhanced window close handling
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Configure window icon and styling
        self._configure_window_appearance()
        
        # Create status bar
        self._create_status_bar()
        
        app_logger.info("OptimizedWindowView initialized successfully")

    def _configure_window_appearance(self):
        """Configure window appearance and styling."""
        try:
            # Set window icon if available
            # self.iconbitmap('path/to/icon.ico')  # Uncomment if you have an icon
            
            # Configure styling
            style = Style()
            style.configure('TButton', padding=6)
            style.configure('TLabel', padding=3)
            
        except Exception as e:
            self.logger.warning(f"Failed to configure window appearance: {e}")

    def _create_status_bar(self):
        """Create a status bar for user feedback."""
        try:
            self.status_frame = Frame(self)
            self.status_frame.pack(side='bottom', fill='x')
            
            self.status_label = Label(
                self.status_frame, 
                text="Ready", 
                anchor='w',
                padding=(10, 5)
            )
            self.status_label.pack(side='left', fill='x', expand=True)
            
            self.progress_bar = Progressbar(
                self.status_frame,
                mode='determinate',
                length=100
            )
            self.progress_bar.pack(side='right', padx=(0, 10))
            
        except Exception as e:
            self.logger.warning(f"Failed to create status bar: {e}")

    def loginview(self, presenter):
        """Create and return a login view instance."""
        login_view = self._loginview_class(self)
        login_view._presenter = presenter
        return login_view

    def mainview(self, presenter):
        """Create and return a main view instance."""
        main_view = self._main_view_class(self)
        main_view._presenter = presenter
        return main_view

    def set_title(self, title: str):
        """Set the window title.
        
        Args:
            title: New window title
        """
        self.title(title)

    def set_geometry(self, geometry: str):
        """Set the window geometry.
        
        Args:
            geometry: Window geometry (e.g., "800x600")
        """
        self.geometry(geometry)

    def show_main_view(self):
        """Show the main application view."""
        try:
            # Clear any existing content
            for widget in self.winfo_children():
                if widget != self.status_frame:
                    widget.destroy()
            
            # Create main view
            main_view = self.mainview(None)  # Presenter will be set later
            main_view.pack(fill='both', expand=True)
            
            self._current_frame = main_view
            self.update_status("Ready")
            
        except Exception as e:
            self.logger.error(f"Failed to show main view: {e}")
            self.show_message("Error", f"Failed to show main view: {str(e)}", "error")

    def show_error(self, message: str):
        """Show an error message to the user.
        
        Args:
            message: Error message
        """
        self.show_message("Error", message, "error")
        self.update_status(f"Error: {message}")

    def show_trade_result(self, message: str):
        """Show a trade result message to the user.
        
        Args:
            message: Trade result message
        """
        self.show_message("Trade Result", message, "info")
        self.update_status(message)

    def show_bot_creation_result(self, message: str):
        """Show a bot creation result message to the user.
        
        Args:
            message: Bot creation result message
        """
        self.show_message("Bot Creation", message, "info")
        self.update_status(message)

    def show_ml_training_result(self, message: str):
        """Show an ML training result message to the user.
        
        Args:
            message: ML training result message
        """
        self.show_message("ML Training", message, "info")
        self.update_status(message)

    def show_rl_training_result(self, message: str):
        """Show an RL training result message to the user.
        
        Args:
            message: RL training result message
        """
        self.show_message("RL Training", message, "info")
        self.update_status(message)

    def update_status(self, message: str):
        """Update the status bar message.
        
        Args:
            message: Status message
        """
        try:
            if hasattr(self, 'status_label'):
                self.status_label.config(text=message)
        except Exception as e:
            self.logger.warning(f"Failed to update status: {e}")

    def update_progress(self, progress: float):
        """Update the progress bar.
        
        Args:
            progress: Progress value (0.0 to 1.0)
        """
        try:
            if hasattr(self, 'progress_bar'):
                self.progress_bar['value'] = progress * 100
        except Exception as e:
            self.logger.warning(f"Failed to update progress: {e}")

    def on_close(self):
        """Handle window close event."""
        try:
            # Confirm exit
            if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
                self.logger.info("Application shutting down")
                self.destroy()
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.destroy()


class OptimizedLoginView(BaseView):
    """Enhanced login view with better user feedback."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self._presenter = None
        self.username_var = StringVar()
        self.password_var = StringVar()
        
    def _initialize(self):
        """Internal initialization method."""
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the login UI."""
        try:
            # Create main frame
            main_frame = Frame(self)
            main_frame.pack(expand=True)
            
            # Title
            title_label = Label(
                main_frame,
                text="Trading Application Login",
                font=('Helvetica', 16, 'bold'),
                padding=(0, 0, 0, 20)
            )
            title_label.pack()
            
            # Username
            username_frame = Frame(main_frame)
            username_frame.pack(pady=5)
            
            Label(username_frame, text="Username:").pack(side='left')
            self.username_entry = Entry(
                username_frame,
                textvariable=self.username_var,
                width=20
            )
            self.username_entry.pack(side='left', padx=(10, 0))
            
            # Password
            password_frame = Frame(main_frame)
            password_frame.pack(pady=5)
            
            Label(password_frame, text="Password:").pack(side='left')
            self.password_entry = Entry(
                password_frame,
                textvariable=self.password_var,
                show="*",
                width=20
            )
            self.password_entry.pack(side='left', padx=(10, 0))
            
            # Login button
            self.login_button = Button(
                main_frame,
                text="Login",
                command=self._on_login,
                padding=(20, 5)
            )
            self.login_button.pack(pady=20)
            
            # Bind Enter key to login
            self.username_entry.bind('<Return>', lambda e: self.password_entry.focus())
            self.password_entry.bind('<Return>', lambda e: self._on_login())
            
            # Focus on username entry
            self.username_entry.focus()
            
        except Exception as e:
            self.logger.error(f"Failed to set up login UI: {e}")
            self.show_message("Error", f"Failed to set up login UI: {str(e)}", "error")

    def _on_login(self):
        """Handle login button click."""
        try:
            username = self.username_var.get()
            password = self.password_var.get()
            
            if not username or not password:
                self.show_message("Error", "Please enter both username and password", "error")
                return
                
            if self._presenter:
                self._presenter.authenticate_user(username, password)
            else:
                self.show_message("Error", "Presenter not set", "error")
                
        except Exception as e:
            self.logger.error(f"Error during login: {e}")
            self.show_message("Error", f"Login failed: {str(e)}", "error")


class OptimizedMainView(BaseView):
    """Enhanced main view with improved organization and user feedback."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self._presenter = None
        self.notebook = None
        self.tabs = {}
        
    def _initialize(self):
        """Internal initialization method."""
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the main UI."""
        try:
            # Create notebook for tabs
            self.notebook = Notebook(self)
            self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Create tabs
            self._create_tabs()
            
            # Set up menu
            self._setup_menu()
            
        except Exception as e:
            self.logger.error(f"Failed to set up main UI: {e}")
            self.show_message("Error", f"Failed to set up main UI: {str(e)}", "error")

    def _create_tabs(self):
        """Create all application tabs."""
        try:
            # Trading tab
            trading_tab = OptimizedTradeTab(self.notebook)
            self.notebook.add(trading_tab, text="Trading")
            self.tabs['trading'] = trading_tab
            
            # Bot tab
            bot_tab = OptimizedBotTab(self.notebook)
            self.notebook.add(bot_tab, text="Bots")
            self.tabs['bots'] = bot_tab
            
            # Chart tab
            chart_tab = OptimizedChartTab(self.notebook)
            self.notebook.add(chart_tab, text="Charts")
            self.tabs['charts'] = chart_tab
            
            # ML tab
            ml_tab = OptimizedMLTab(self.notebook)
            self.notebook.add(ml_tab, text="ML")
            self.tabs['ml'] = ml_tab
            
            # RL tab
            rl_tab = OptimizedRLTab(self.notebook)
            self.notebook.add(rl_tab, text="RL")
            self.tabs['rl'] = rl_tab
            
            # Trading system tab
            trading_system_tab = TradingSystemTab(self.notebook)
            self.notebook.add(trading_system_tab, text="Trading System")
            self.tabs['trading_system'] = trading_system_tab
            
            # ML system tab
            ml_system_tab = MLSystemTab(self.notebook)
            self.notebook.add(ml_system_tab, text="ML System")
            self.tabs['ml_system'] = ml_system_tab
            
            # RL system tab
            rl_system_tab = RLSystemTab(self.notebook)
            self.notebook.add(rl_system_tab, text="RL System")
            self.tabs['rl_system'] = rl_system_tab
            
            # Advanced RL system tab
            advanced_rl_system_tab = AdvancedRLSystemTab(self.notebook)
            self.notebook.add(advanced_rl_system_tab, text="Advanced RL")
            self.tabs['advanced_rl'] = advanced_rl_system_tab
            
        except Exception as e:
            self.logger.error(f"Failed to create tabs: {e}")
            self.show_message("Error", f"Failed to create tabs: {str(e)}", "error")

    def _setup_menu(self):
        """Set up the application menu."""
        try:
            menubar = Menu(self.parent)
            
            # File menu
            file_menu = Menu(menubar, tearoff=0)
            file_menu.add_command(label="Exit", command=self.parent.on_close)
            menubar.add_cascade(label="File", menu=file_menu)
            
            # View menu
            view_menu = Menu(menubar, tearoff=0)
            view_menu.add_command(label="Refresh", command=self._refresh_view)
            menubar.add_cascade(label="View", menu=view_menu)
            
            # Help menu
            help_menu = Menu(menubar, tearoff=0)
            help_menu.add_command(label="About", command=self._show_about)
            menubar.add_cascade(label="Help", menu=help_menu)
            
            self.parent.config(menu=menubar)
            
        except Exception as e:
            self.logger.warning(f"Failed to set up menu: {e}")

    def _refresh_view(self):
        """Refresh the current view."""
        try:
            self.show_message("Refresh", "View refreshed successfully", "info")
            self.parent.update_status("View refreshed")
        except Exception as e:
            self.logger.error(f"Failed to refresh view: {e}")
            self.show_message("Error", f"Failed to refresh view: {str(e)}", "error")

    def _show_about(self):
        """Show about dialog."""
        try:
            messagebox.showinfo(
                "About",
                "Advanced Trading Application\\nVersion 1.0\\n\\nA comprehensive trading application with multiple strategies."
            )
        except Exception as e:
            self.logger.error(f"Failed to show about dialog: {e}")