"""
Enhanced Views Module

Optimized version of the views module with:
- Better integration with optimized presenter and models
- Enhanced UI responsiveness and performance monitoring
- Modern UI patterns and improved user experience
- Async operation support and better error handling
- Professional styling and accessibility features
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
        Treeview, Window, Progressbar, Separator
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

# Import optimized tab views
from view.tradetab_optimized import OptimizedTradeTab
from view.bottab_optimized import OptimizedBotTab
from view.charttab_optimized import OptimizedChartTab
from view.exchangetab_optimized import OptimizedExchangeTab
from view.mltab_optimized import OptimizedMLTab
from view.rltab_optimized import OptimizedRLTab

# Import new system tabs
from view.trading_system_tab import TradingSystemTab
from view.ml_system_tab import MLSystemTab
from view.rl_system_tab import RLSystemTab

# Import view utilities
from view.utils import UIThemeManager, ValidationMixin, StatusIndicator

import util.loggers as loggers

logger_dict = loggers.setup_loggers()
app_logger = logger_dict['app']


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
        self.loginview = OptimizedLoginView
        self.main_view = OptimizedMainView
        
        # Status tracking
        self._current_frame = None
        self._cleanup_callbacks = []
        
        # Enhanced window close handling
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Configure window icon and styling
        self._configure_window_appearance()
        
        app_logger.info("OptimizedWindowView initialized successfully")

    def _configure_window_appearance(self):
        """Configure window appearance and styling."""
        try:
            # Set window icon if available
            # self.iconbitmap("assets/icon.ico")  # Uncomment when icon is available
            
            # Configure window attributes
            self.configure(padx=5, pady=5)
            
            # Center window on screen
            self.center_window()
            
        except Exception as e:
            app_logger.warning(f"Could not configure window appearance: {e}")

    def center_window(self):
        """Center the window on the screen."""
        try:
            # Get screen dimensions
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            
            # Calculate position
            window_width = 1200
            window_height = 800
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            
            # Set position
            self.geometry(f"{window_width}x{window_height}+{x}+{y}")
            
        except Exception as e:
            app_logger.error(f"Error centering window: {e}")

    def show_frame(self, cont, presenter) -> None:
        """Enhanced frame switching with performance monitoring and cleanup."""
        try:
            switch_start = datetime.now()
            
            # Cleanup previous frame
            self._cleanup_current_frame()
            
            # Performance tracking
            self._frame_switches += 1
            self._last_frame_switch = switch_start
            
            # Create and show new frame
            frame = cont
            frame.__init__(self)
            
            # Enhanced UI creation with error handling
            if hasattr(frame, 'create_ui'):
                frame.create_ui(presenter)
            
            # Configure grid
            frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
            frame.tkraise()
            
            # Configure grid weights for responsive design
            self.grid_rowconfigure(0, weight=1)
            self.grid_columnconfigure(0, weight=1)
            
            self._current_frame = frame
            
            # Log performance metrics
            switch_time = (datetime.now() - switch_start).total_seconds()
            app_logger.debug(f"Frame switch completed in {switch_time:.3f}s")
            
        except Exception as e:
            app_logger.error(f"Error showing frame: {e}")
            messagebox.showerror("Error", f"Failed to display view: {str(e)}")

    def _cleanup_current_frame(self):
        """Cleanup current frame resources."""
        try:
            # Cleanup current frame
            for widget in self.winfo_children():
                if hasattr(widget, 'cleanup'):
                    widget.cleanup()
                widget.destroy()
                
            # Execute cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    app_logger.error(f"Cleanup callback error: {e}")
            
            self._cleanup_callbacks.clear()
            
        except Exception as e:
            app_logger.error(f"Error during frame cleanup: {e}")

    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback."""
        self._cleanup_callbacks.append(callback)

    def on_close(self):
        """Enhanced window close handling with cleanup."""
        try:
            app_logger.info("Closing application...")
            
            # Calculate uptime
            uptime = (datetime.now() - self._start_time).total_seconds()
            app_logger.info(f"Application uptime: {uptime:.1f}s, Frame switches: {self._frame_switches}")
            
            # Cleanup resources
            self._cleanup_current_frame()
            
            # Additional cleanup if needed
            if hasattr(self, '_presenter'):
                if hasattr(self._presenter, 'cleanup'):
                    self._presenter.cleanup()
            
            app_logger.info("Application closed successfully")
            
        except Exception as e:
            app_logger.error(f"Error during application close: {e}")
        finally:
            self.destroy()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get window performance metrics."""
        uptime = (datetime.now() - self._start_time).total_seconds()
        return {
            'uptime_seconds': uptime,
            'frame_switches': self._frame_switches,
            'avg_switch_time': uptime / max(self._frame_switches, 1),
            'current_frame': type(self._current_frame).__name__ if self._current_frame else None
        }


class OptimizedLoginView(Frame, ValidationMixin):
    """
    Enhanced login view with improved validation and user experience.
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        self._parent = parent
        self._presenter = None
        
        # Enhanced UI variables
        self._username_var = StringVar()
        self._password_var = StringVar()
        self._remember_var = StringVar()
        
        # UI state
        self._login_attempts = 0
        self._max_attempts = 5
        self._is_logging_in = False
        
        # Status components
        self._status_indicator = None
        self._progress_bar = None
        
        app_logger.info("OptimizedLoginView initialized")

    def create_ui(self, presenter):
        """Create enhanced login UI with modern styling."""
        try:
            self._presenter = presenter
            
            # Configure main frame
            self.configure(padx=20, pady=20)
            
            # Create title section
            self._create_title_section()
            
            # Create input section  
            self._create_input_section()
            
            # Create button section
            self._create_button_section()
            
            # Create status section
            self._create_status_section()
            
            # Configure grid weights for responsiveness
            self._configure_grid_weights()
            
            # Focus on username entry
            self.after(100, lambda: self.username_entry.focus())
            
            app_logger.info("Login UI created successfully")
            
        except Exception as e:
            app_logger.error(f"Error creating login UI: {e}")
            messagebox.showerror("Error", f"Failed to create login interface: {str(e)}")

    def _create_title_section(self):
        """Create title section with branding."""
        title_frame = Frame(self)
        title_frame.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        title_label = Label(
            title_frame, 
            text="Trading Application", 
            font=('Arial', 20, 'bold')
        )
        title_label.grid(row=0, column=0)
        
        subtitle_label = Label(
            title_frame,
            text="Professional Trading Platform",
            font=('Arial', 10)
        )
        subtitle_label.grid(row=1, column=0, pady=(5, 0))

    def _create_input_section(self):
        """Create input fields with validation."""
        # Username section
        Label(
            self, 
            text="Username:", 
            font=('Arial', 12, 'bold')
        ).grid(row=1, column=0, sticky='e', padx=(0, 10), pady=5)
        
        self.username_entry = Entry(
            self, 
            textvariable=self._username_var,
            font=('Arial', 11),
            width=20,
            validate='key',
            validatecommand=(self.register(self._validate_username), '%P')
        )
        self.username_entry.grid(row=1, column=1, pady=5, sticky='ew')
        
        # Password section
        Label(
            self, 
            text="Password:", 
            font=('Arial', 12, 'bold')
        ).grid(row=2, column=0, sticky='e', padx=(0, 10), pady=5)
        
        self.password_entry = Entry(
            self, 
            textvariable=self._password_var, 
            show="*",
            font=('Arial', 11),
            width=20,
            validate='key',
            validatecommand=(self.register(self._validate_password), '%P')
        )
        self.password_entry.grid(row=2, column=1, pady=5, sticky='ew')
        
        # Bind Enter key to login
        self.username_entry.bind('<Return>', lambda e: self._on_login_clicked())
        self.password_entry.bind('<Return>', lambda e: self._on_login_clicked())

    def _create_button_section(self):
        """Create button section with enhanced styling."""
        button_frame = Frame(self)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        self.login_button = Button(
            button_frame, 
            text="Login",
            command=self._on_login_clicked,
            width=12,
            style='success.TButton'
        )
        self.login_button.grid(row=0, column=0, padx=(0, 10))
        
        self.register_button = Button(
            button_frame, 
            text="Register", 
            command=self._on_register_clicked,
            width=12,
            style='info.TButton'
        )
        self.register_button.grid(row=0, column=1, padx=(10, 0))
        
        # Development quick login (can be removed in production)
        dev_button = Button(
            button_frame,
            text="Dev Login",
            command=self._on_dev_login,
            width=12,
            style='warning.TButton'
        )
        dev_button.grid(row=1, column=0, columnspan=2, pady=(10, 0))

    def _create_status_section(self):
        """Create status section with progress indicator."""
        status_frame = Frame(self)
        status_frame.grid(row=4, column=0, columnspan=2, pady=(10, 0), sticky='ew')
        
        # Status indicator
        self._status_indicator = StatusIndicator(status_frame)
        self._status_indicator.grid(row=0, column=0, sticky='ew')
        
        # Progress bar
        self._progress_bar = Progressbar(
            status_frame,
            mode='indeterminate',
            style='success.Horizontal.TProgressbar'
        )
        self._progress_bar.grid(row=1, column=0, sticky='ew', pady=(5, 0))
        self._progress_bar.grid_remove()  # Initially hidden

    def _configure_grid_weights(self):
        """Configure grid weights for responsive design."""
        self.grid_columnconfigure(1, weight=1)
        
        # Configure parent grid
        if hasattr(self._parent, 'grid_rowconfigure'):
            self._parent.grid_rowconfigure(0, weight=1)
            self._parent.grid_columnconfigure(0, weight=1)

    def _validate_username(self, value: str) -> bool:
        """Validate username input."""
        # Allow alphanumeric and basic special characters
        if not value:
            return True
        return len(value) <= 50 and value.replace('_', '').replace('-', '').isalnum()

    def _validate_password(self, value: str) -> bool:
        """Validate password input."""
        # Basic password validation
        return len(value) <= 100

    def _on_login_clicked(self):
        """Handle login button click with enhanced validation."""
        if self._is_logging_in:
            return
            
        try:
            # Validate inputs
            username = self._username_var.get().strip()
            password = self._password_var.get()
            
            if not username or not password:
                self._show_validation_error("Username and password are required")
                return
            
            if len(username) < 2:
                self._show_validation_error("Username must be at least 2 characters")
                return
            
            # Check attempt limit
            if self._login_attempts >= self._max_attempts:
                self._show_validation_error("Too many login attempts. Please restart the application.")
                return
            
            # Show loading state
            self._set_login_state(True, "Authenticating...")
            
            # Increment attempt counter
            self._login_attempts += 1
            
            # Call presenter
            if self._presenter:
                self._presenter.on_login_button_clicked()
            
        except Exception as e:
            app_logger.error(f"Login error: {e}")
            self._show_validation_error(f"Login error: {str(e)}")
        finally:
            # Reset login state after a delay
            self.after(2000, lambda: self._set_login_state(False))

    def _on_register_clicked(self):
        """Handle register button click."""
        if self._is_logging_in:
            return
            
        try:
            # Validate inputs
            username = self._username_var.get().strip()
            password = self._password_var.get()
            
            if not username or not password:
                self._show_validation_error("Username and password are required")
                return
            
            if len(username) < 3:
                self._show_validation_error("Username must be at least 3 characters")
                return
                
            if len(password) < 4:
                self._show_validation_error("Password must be at least 4 characters")
                return
            
            # Show loading state
            self._set_login_state(True, "Creating account...")
            
            # Call presenter
            if self._presenter:
                self._presenter.on_register_button_clicked()
            
        except Exception as e:
            app_logger.error(f"Registration error: {e}")
            self._show_validation_error(f"Registration error: {str(e)}")
        finally:
            # Reset login state after a delay
            self.after(2000, lambda: self._set_login_state(False))

    def _on_dev_login(self):
        """Quick development login."""
        self._username_var.set('test')
        self._password_var.set('t')
        self._on_login_clicked()

    def _set_login_state(self, is_logging_in: bool, message: str = ""):
        """Set login state and update UI accordingly."""
        try:
            self._is_logging_in = is_logging_in
            
            # Update button states
            state = 'disabled' if is_logging_in else 'normal'
            self.login_button.configure(state=state)
            self.register_button.configure(state=state)
            
            # Update progress bar
            if is_logging_in:
                self._progress_bar.grid()
                self._progress_bar.start(10)
                if message:
                    self._status_indicator.set_status(message, 'info')
            else:
                self._progress_bar.stop()
                self._progress_bar.grid_remove()
                self._status_indicator.clear_status()
                
        except Exception as e:
            app_logger.error(f"Error setting login state: {e}")

    def _show_validation_error(self, message: str):
        """Show validation error to user."""
        try:
            self._status_indicator.set_status(message, 'error')
            
            # Also show in entry field temporarily
            if hasattr(self, 'username_entry'):
                original_text = self._username_var.get()
                self._username_var.set(message)
                self.after(2000, lambda: self._username_var.set(original_text))
                
        except Exception as e:
            app_logger.error(f"Error showing validation error: {e}")

    def get_username(self) -> str:
        """Get username value."""
        return self._username_var.get().strip()

    def get_password(self) -> str:
        """Get password value."""
        return self._password_var.get()

    def login_failed(self) -> None:
        """Handle login failure."""
        try:
            self._set_login_state(False)
            self._show_validation_error("Invalid credentials. Please try again.")
            
            # Clear password field
            self._password_var.set('')
            
            # Focus on password entry
            self.password_entry.focus()
            
        except Exception as e:
            app_logger.error(f"Error handling login failure: {e}")

    def show_error_message(self, message: str):
        """Show error message dialog."""
        try:
            messagebox.showerror("Error", message)
            self._show_validation_error(message)
        except Exception as e:
            app_logger.error(f"Error showing error message: {e}")


class OptimizedMainView(Frame):
    """
    Enhanced main view with optimized tab management and modern UI.
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        self._parent = parent
        self._presenter = None
        
        # Configure main window
        self._parent.title("Trading Application - Main Dashboard")
        self._parent.geometry("1400x900")
        
        # Initialize theme system
        self._initialize_theme_system()
        
        # Tab management
        self._tab_instances = {}
        self._tab_loading_status = {}
        
        # Performance tracking
        self._tab_switches = 0
        self._last_tab_switch = datetime.now()
        
        app_logger.info("OptimizedMainView initialized")

    def _initialize_theme_system(self):
        """Initialize theme system with menu."""
        try:
            style = Style()
            themes = style.theme_names()
            
            # Create menu bar
            self._parent.menu = Menu(self)
            
            # Theme menu
            self._parent.theme_menu = Menu(self._parent.menu, tearoff=0)
            self._parent.menu.add_cascade(label="Themes", menu=self._parent.theme_menu)
            
            # Add theme options
            for theme in themes:
                self._parent.theme_menu.add_command(
                    label=theme.title(), 
                    command=lambda t=theme: self._change_theme(t)
                )
            
            # View menu
            view_menu = Menu(self._parent.menu, tearoff=0)
            self._parent.menu.add_cascade(label="View", menu=view_menu)
            view_menu.add_command(label="Refresh All Tabs", command=self._refresh_all_tabs)
            view_menu.add_command(label="Performance Metrics", command=self._show_performance_metrics)
            
            # Help menu
            help_menu = Menu(self._parent.menu, tearoff=0)
            self._parent.menu.add_cascade(label="Help", menu=help_menu)
            help_menu.add_command(label="About", command=self._show_about)
            help_menu.add_command(label="Keyboard Shortcuts", command=self._show_shortcuts)
            
            # Set menu
            self._parent.config(menu=self._parent.menu)
            
        except Exception as e:
            app_logger.error(f"Error initializing theme system: {e}")

    def create_ui(self, presenter):
        """Create enhanced main UI with optimized tab management."""
        try:
            self._presenter = presenter
            
            # Configure main frame
            self.grid_rowconfigure(1, weight=1)
            self.grid_columnconfigure(0, weight=1)
            
            # Create header section
            self._create_header_section()
            
            # Create main content area with notebook
            self._create_main_content_area()
            
            # Create status bar
            self._create_status_bar()
            
            app_logger.info("Main UI created successfully")
            
        except Exception as e:
            app_logger.error(f"Error creating main UI: {e}")
            messagebox.showerror("Error", f"Failed to create main interface: {str(e)}")

    def _create_header_section(self):
        """Create header section with welcome message and quick actions."""
        header_frame = Frame(self, style='info.TFrame')
        header_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Welcome message
        welcome_label = Label(
            header_frame,
            text="Welcome to the Professional Trading Platform!",
            font=('Arial', 14, 'bold'),
            style='info.TLabel'
        )
        welcome_label.grid(row=0, column=0, padx=10, pady=5)
        
        # Quick actions frame
        actions_frame = Frame(header_frame)
        actions_frame.grid(row=0, column=2, padx=10, pady=5)
        
        # Quick action buttons
        Button(
            actions_frame,
            text="Refresh All",
            command=self._refresh_all_tabs,
            style='secondary.TButton'
        ).grid(row=0, column=0, padx=2)
        
        Button(
            actions_frame,
            text="Settings",
            command=self._show_settings,
            style='secondary.TButton'
        ).grid(row=0, column=1, padx=2)

    def _create_main_content_area(self):
        """Create main content area with enhanced notebook."""
        # Main content frame
        content_frame = Frame(self)
        content_frame.grid(row=1, column=0, sticky='nsew', padx=5)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        
        # Create enhanced notebook
        self.notebook = Notebook(content_frame, style='TNotebook')
        
        # Initialize all tabs
        self._initialize_tabs()
        
        # Bind tab change event
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)
        
        # Grid notebook
        self.notebook.grid(row=0, column=0, sticky='nsew')

    def _initialize_tabs(self):
        """Initialize all tabs with lazy loading capability."""
        try:
            # Core trading tabs
            self._add_tab('trade', "📈 Trading", OptimizedTradeTab)
            self._add_tab('exchange', "🏪 Exchanges", OptimizedExchangeTab)
            self._add_tab('bot', "🤖 Trading Bots", OptimizedBotTab)
            self._add_tab('chart', "📊 Charts", OptimizedChartTab)
            
            # ML/AI tabs
            self._add_tab('ml', "🧠 Machine Learning", OptimizedMLTab)
            self._add_tab('rl', "🎯 Reinforcement Learning", OptimizedRLTab)
            
            # Advanced system tabs (new)
            self._add_tab('trading_system', "⚡ Trading System", TradingSystemTab)
            self._add_tab('ml_system', "🔬 ML System", MLSystemTab)
            self._add_tab('rl_system', "🎮 RL System", RLSystemTab)
            
        except Exception as e:
            app_logger.error(f"Error initializing tabs: {e}")

    def _add_tab(self, tab_id: str, title: str, tab_class):
        """Add a tab with lazy loading."""
        try:
            # Create placeholder frame
            placeholder_frame = Frame(self.notebook)
            
            # Add to notebook
            self.notebook.add(placeholder_frame, text=title)
            
            # Store tab info
            self._tab_instances[tab_id] = {
                'class': tab_class,
                'frame': placeholder_frame,
                'instance': None,
                'loaded': False,
                'title': title
            }
            
            app_logger.debug(f"Tab {tab_id} added with lazy loading")
            
        except Exception as e:
            app_logger.error(f"Error adding tab {tab_id}: {e}")

    def _load_tab(self, tab_id: str):
        """Load a tab on demand."""
        try:
            if tab_id not in self._tab_instances:
                return False
                
            tab_info = self._tab_instances[tab_id]
            
            if tab_info['loaded']:
                return True
                
            # Show loading indicator
            self._show_tab_loading(tab_id, True)
            
            # Create tab instance
            tab_class = tab_info['class']
            placeholder_frame = tab_info['frame']
            
            # Clear placeholder
            for widget in placeholder_frame.winfo_children():
                widget.destroy()
            
            # Create actual tab
            tab_instance = tab_class(placeholder_frame, self._presenter)
            tab_instance.grid(row=0, column=0, sticky='nsew')
            
            # Configure grid weights
            placeholder_frame.grid_rowconfigure(0, weight=1)
            placeholder_frame.grid_columnconfigure(0, weight=1)
            
            # Update tab info
            tab_info['instance'] = tab_instance
            tab_info['loaded'] = True
            
            # Hide loading indicator
            self._show_tab_loading(tab_id, False)
            
            app_logger.info(f"Tab {tab_id} loaded successfully")
            return True
            
        except Exception as e:
            app_logger.error(f"Error loading tab {tab_id}: {e}")
            self._show_tab_loading(tab_id, False)
            return False

    def _show_tab_loading(self, tab_id: str, is_loading: bool):
        """Show/hide loading indicator for tab."""
        try:
            if tab_id not in self._tab_instances:
                return
                
            tab_info = self._tab_instances[tab_id]
            original_title = tab_info['title']
            
            if is_loading:
                # Update tab title to show loading
                tab_index = self._get_tab_index(tab_id)
                if tab_index is not None:
                    self.notebook.tab(tab_index, text=f"⏳ {original_title}")
                    
                # Show loading in frame
                placeholder_frame = tab_info['frame']
                loading_label = Label(
                    placeholder_frame,
                    text="Loading...",
                    font=('Arial', 12)
                )
                loading_label.grid(row=0, column=0)
                
            else:
                # Restore original title
                tab_index = self._get_tab_index(tab_id)
                if tab_index is not None:
                    self.notebook.tab(tab_index, text=original_title)
                    
        except Exception as e:
            app_logger.error(f"Error updating tab loading state: {e}")

    def _get_tab_index(self, tab_id: str) -> Optional[int]:
        """Get the index of a tab in the notebook."""
        try:
            if tab_id not in self._tab_instances:
                return None
                
            tab_frame = self._tab_instances[tab_id]['frame']
            
            for i, tab in enumerate(self.notebook.tabs()):
                if self.notebook.nametowidget(tab) == tab_frame:
                    return i
                    
            return None
            
        except Exception as e:
            app_logger.error(f"Error getting tab index for {tab_id}: {e}")
            return None

    def _on_tab_changed(self, event):
        """Handle tab change events with lazy loading."""
        try:
            # Get selected tab
            selected_tab = self.notebook.nametowidget(self.notebook.select())
            
            # Find corresponding tab_id
            for tab_id, tab_info in self._tab_instances.items():
                if tab_info['frame'] == selected_tab:
                    # Load tab if not loaded
                    if not tab_info['loaded']:
                        self._load_tab(tab_id)
                    break
                    
            # Update performance metrics
            self._tab_switches += 1
            self._last_tab_switch = datetime.now()
            
        except Exception as e:
            app_logger.error(f"Error handling tab change: {e}")

    def _create_status_bar(self):
        """Create status bar with system information."""
        status_frame = Frame(self, relief='sunken', style='secondary.TFrame')
        status_frame.grid(row=2, column=0, sticky='ew', padx=2, pady=2)
        status_frame.grid_columnconfigure(1, weight=1)
        
        # Status label
        self.status_label = Label(
            status_frame,
            text="Ready",
            font=('Arial', 9),
            style='secondary.TLabel'
        )
        self.status_label.grid(row=0, column=0, padx=5, pady=2)
        
        # Performance info
        self.performance_label = Label(
            status_frame,
            text="",
            font=('Arial', 9),
            style='secondary.TLabel'
        )
        self.performance_label.grid(row=0, column=2, padx=5, pady=2)
        
        # Update performance info periodically
        self._update_performance_info()

    def _update_performance_info(self):
        """Update performance information in status bar."""
        try:
            # Calculate basic metrics
            current_time = datetime.now()
            
            # Update performance label
            self.performance_label.configure(
                text=f"Tabs: {len(self._tab_instances)} | Switches: {self._tab_switches}"
            )
            
            # Schedule next update
            self.after(5000, self._update_performance_info)
            
        except Exception as e:
            app_logger.error(f"Error updating performance info: {e}")

    def _change_theme(self, theme: str):
        """Change application theme."""
        try:
            Style().theme_use(theme)
            self.update_status(f"Theme changed to {theme}")
        except Exception as e:
            app_logger.error(f"Error changing theme: {e}")
            messagebox.showerror("Theme Error", f"Failed to change theme: {str(e)}")

    def _refresh_all_tabs(self):
        """Refresh all loaded tabs."""
        try:
            refreshed_count = 0
            for tab_id, tab_info in self._tab_instances.items():
                if tab_info['loaded'] and tab_info['instance']:
                    if hasattr(tab_info['instance'], 'refresh'):
                        tab_info['instance'].refresh()
                        refreshed_count += 1
                        
            self.update_status(f"Refreshed {refreshed_count} tabs")
            
        except Exception as e:
            app_logger.error(f"Error refreshing tabs: {e}")

    def _show_performance_metrics(self):
        """Show performance metrics dialog."""
        try:
            if hasattr(self._presenter, 'get_performance_metrics'):
                metrics = self._presenter.get_performance_metrics()
                
                # Create metrics display
                metrics_text = "Performance Metrics:\n\n"
                for key, value in metrics.items():
                    metrics_text += f"{key}: {value}\n"
                    
                messagebox.showinfo("Performance Metrics", metrics_text)
            else:
                messagebox.showinfo("Performance Metrics", "Metrics not available")
                
        except Exception as e:
            app_logger.error(f"Error showing performance metrics: {e}")

    def _show_settings(self):
        """Show settings dialog."""
        messagebox.showinfo("Settings", "Settings dialog not implemented yet")

    def _show_about(self):
        """Show about dialog."""
        about_text = """Trading Application v2.0
        
Professional Trading Platform
Built with Python & tkinter

Features:
- Advanced Trading Systems
- Machine Learning Integration  
- Reinforcement Learning
- Real-time Market Data
- Automated Trading Bots

© 2024 Trading Application"""
        
        messagebox.showinfo("About", about_text)

    def _show_shortcuts(self):
        """Show keyboard shortcuts."""
        shortcuts_text = """Keyboard Shortcuts:

F5 - Refresh All Tabs
Ctrl+1-9 - Switch to Tab 1-9
Ctrl+R - Refresh Current Tab
Ctrl+T - New Trade
Ctrl+Q - Quit Application

Tab Navigation:
Ctrl+Tab - Next Tab
Ctrl+Shift+Tab - Previous Tab"""
        
        messagebox.showinfo("Keyboard Shortcuts", shortcuts_text)

    def update_status(self, message: str):
        """Update status bar message."""
        try:
            if hasattr(self, 'status_label'):
                self.status_label.configure(text=f"{datetime.now().strftime('%H:%M:%S')} - {message}")
        except Exception as e:
            app_logger.error(f"Error updating status: {e}")

    def list_box(self, text):
        """Enhanced list box method with history management."""
        try:
            # Find current tab's list box if available
            current_tab = self.notebook.nametowidget(self.notebook.select())
            
            # Look for history list in current tab
            if hasattr(current_tab, 'history_list'):
                history_list = current_tab.history_list
                
                # Manage list size
                if history_list.size() >= 20:
                    history_list.delete(0)
                
                # Add new entry with timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted_text = f"[{timestamp}] {text}"
                history_list.insert(END, formatted_text)
                
                # Auto-scroll to bottom
                history_list.see(END)
            else:
                # Log to application logger as fallback
                app_logger.info(f"UI Message: {text}")
                
        except Exception as e:
            app_logger.error(f"Error updating list box: {e}")

    def cleanup(self):
        """Cleanup main view resources."""
        try:
            # Cleanup all tab instances
            for tab_info in self._tab_instances.values():
                if tab_info['loaded'] and tab_info['instance']:
                    if hasattr(tab_info['instance'], 'cleanup'):
                        tab_info['instance'].cleanup()
                        
            app_logger.info("MainView cleanup completed")
            
        except Exception as e:
            app_logger.error(f"Error during MainView cleanup: {e}")


# Backwards compatibility aliases
WindowView = OptimizedWindowView
LoginView = OptimizedLoginView
MainView = OptimizedMainView
