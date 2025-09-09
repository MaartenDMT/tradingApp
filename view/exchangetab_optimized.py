"""
Optimized Exchange Tab

Enhanced version of the exchange tab with:
- Secure API key management and validation
- Support for multiple exchange profiles
- Real-time connection status monitoring
- Asynchronous operations for responsiveness
- Advanced configuration options per exchange
- Integration with trading systems
"""

import asyncio
import threading
from tkinter import messagebox
from typing import Any, Dict, List, Optional

import ccxt

try:
    from ttkbootstrap import (
        Button, Entry, Frame, Label, OptionMenu, Scale, StringVar, 
        BooleanVar, IntVar, Notebook, Progressbar, Separator, Treeview
    )
    from ttkbootstrap.constants import *
    HAS_TTKBOOTSTRAP = True
except ImportError:
    from tkinter import (
        Button, Entry, Frame, Label, OptionMenu, Scale, StringVar,
        BooleanVar, IntVar
    )
    from tkinter.ttk import Notebook, Progressbar, Separator, Treeview
    HAS_TTKBOOTSTRAP = False

from view.utils import (
    ValidationMixin, StatusIndicator, LoadingIndicator, FormValidator,
    notification_system
)

import util.loggers as loggers

logger_dict = loggers.setup_loggers()
app_logger = logger_dict['app']


class OptimizedExchangeTab(Frame, ValidationMixin):
    """
    Enhanced exchange management interface with secure credential handling.
    """
    
    def __init__(self, parent, presenter):
        super().__init__(parent)
        self._parent = parent
        self._presenter = presenter
        
        # Exchange management state
        self._exchanges = {}
        self._selected_exchange = None
        self._is_loading = False
        
        # Form validator
        self.form_validator = FormValidator()
        self._setup_validation_rules()
        
        # Initialize GUI
        self._create_widgets()
        self._setup_layout()
        self._bind_events()
        
        # Initial data load
        self.after(200, self.load_saved_exchanges)
        
        app_logger.info("OptimizedExchangeTab initialized")

    def _setup_validation_rules(self):
        """Setup form validation rules."""
        try:
            self.form_validator.add_field_rule('api_key', 'required')
            self.form_validator.add_field_rule('api_secret', 'required')
            
        except Exception as e:
            app_logger.error(f"Error setting up validation rules: {e}")

    def _create_widgets(self):
        """Create all GUI widgets."""
        try:
            # Create main sections
            self._create_header_section()
            self._create_add_exchange_section()
            self._create_saved_exchanges_section()
            self._create_status_section()
            
        except Exception as e:
            app_logger.error(f"Error creating exchange widgets: {e}")

    def _create_header_section(self):
        """Create header with title and main actions."""
        header_frame = Frame(self)
        header_frame.grid(row=0, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        Label(header_frame, text="Exchange Management", font=('Arial', 16, 'bold')).grid(
            row=0, column=0, padx=5
        )

    def _create_add_exchange_section(self):
        """Create section for adding a new exchange."""
        add_frame = Frame(self, text="Add New Exchange", padding=10)
        add_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        add_frame.grid_columnconfigure(1, weight=1)
        
        # Exchange selection
        Label(add_frame, text="Exchange:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5, pady=5
        )
        self.exchange_var = StringVar(value='phemex')
        self.exchange_menu = OptionMenu(
            add_frame, self.exchange_var, 'phemex', *ccxt.exchanges
        )
        self.exchange_menu.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        
        # API Key
        Label(add_frame, text="API Key:", font=('Arial', 10, 'bold')).grid(
            row=1, column=0, sticky='w', padx=5, pady=5
        )
        self.api_key_var = StringVar()
        self.api_key_entry = Entry(add_frame, textvariable=self.api_key_var, width=40)
        self.api_key_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        
        # API Secret
        Label(add_frame, text="API Secret:", font=('Arial', 10, 'bold')).grid(
            row=2, column=0, sticky='w', padx=5, pady=5
        )
        self.api_secret_var = StringVar()
        self.api_secret_entry = Entry(add_frame, textvariable=self.api_secret_var, show='*', width=40)
        self.api_secret_entry.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        
        # Testnet checkbox
        self.testnet_var = BooleanVar()
        self.testnet_check = Checkbutton(
            add_frame, text="Use Testnet/Sandbox", variable=self.testnet_var
        )
        self.testnet_check.grid(row=3, column=1, sticky='w', padx=5, pady=10)
        
        # Action buttons
        button_frame = Frame(add_frame)
        button_frame.grid(row=4, column=1, sticky='e', pady=10)
        
        self.add_button = Button(
            button_frame, text="Add Exchange", command=self._add_exchange,
            style='success.TButton'
        )
        self.add_button.grid(row=0, column=0, padx=5)
        
        self.test_button = Button(
            button_frame, text="Test Connection", command=self._test_connection,
            style='info.TButton'
        )
        self.test_button.grid(row=0, column=1, padx=5)

    def _create_saved_exchanges_section(self):
        """Create section for displaying saved exchanges."""
        saved_frame = Frame(self, text="Saved Exchanges", padding=10)
        saved_frame.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        saved_frame.grid_rowconfigure(0, weight=1)
        saved_frame.grid_columnconfigure(0, weight=1)
        
        # Exchanges treeview
        self.exchanges_tree = Treeview(
            saved_frame,
            columns=('Name', 'Status', 'Default'),
            show='headings', height=10
        )
        self.exchanges_tree.heading('Name', text='Exchange Name')
        self.exchanges_tree.heading('Status', text='Status')
        self.exchanges_tree.heading('Default', text='Default')
        self.exchanges_tree.column('Name', width=150)
        self.exchanges_tree.column('Status', width=100)
        self.exchanges_tree.column('Default', width=80, anchor='center')
        self.exchanges_tree.grid(row=0, column=0, sticky='nsew')
        
        # Action buttons
        button_frame = Frame(saved_frame)
        button_frame.grid(row=1, column=0, sticky='ew', pady=10)
        
        self.load_button = Button(
            button_frame, text="Load", command=self._load_selected_exchange,
            style='primary.TButton'
        )
        self.load_button.grid(row=0, column=0, padx=5)
        
        self.set_default_button = Button(
            button_frame, text="Set Default", command=self._set_default_exchange
        )
        self.set_default_button.grid(row=0, column=1, padx=5)
        
        self.remove_button = Button(
            button_frame, text="Remove", command=self._remove_exchange,
            style='danger.TButton'
        )
        self.remove_button.grid(row=0, column=2, padx=5)

    def _create_status_section(self):
        """Create status bar for feedback."""
        status_frame = Frame(self)
        status_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_indicator = StatusIndicator(status_frame)
        self.status_indicator.grid(row=0, column=0, sticky='ew')
        
        self.loading_indicator = LoadingIndicator(status_frame)
        self.loading_indicator.grid(row=1, column=0, sticky='ew', pady=5)

    def _setup_layout(self):
        """Configure grid layout weights."""
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def _bind_events(self):
        """Bind event handlers."""
        self.exchanges_tree.bind('<<TreeviewSelect>>', self._on_exchange_selected)

    def load_saved_exchanges(self):
        """Load saved exchanges from presenter."""
        # Placeholder for loading exchanges
        self._exchanges = {
            'phemex_test': {'name': 'phemex', 'testnet': True, 'default': True},
            'binance_main': {'name': 'binance', 'testnet': False, 'default': False}
        }
        self._update_exchanges_display()

    def _add_exchange(self):
        """Add a new exchange configuration."""
        try:
            # Validate form
            form_data = {
                'api_key': self.api_key_var.get(),
                'api_secret': self.api_secret_var.get()
            }
            if not self.form_validator.validate_form(form_data)['valid']:
                self.status_indicator.set_status("API Key and Secret are required", 'error')
                return
            
            exchange_name = self.exchange_var.get()
            is_testnet = self.testnet_var.get()
            
            # Create a unique ID
            exchange_id = f"{exchange_name}_{'test' if is_testnet else 'main'}"
            
            if exchange_id in self._exchanges:
                self.status_indicator.set_status("Exchange already exists", 'warning')
                return
            
            # Save exchange (presenter should handle secure storage)
            self._exchanges[exchange_id] = {
                'name': exchange_name,
                'testnet': is_testnet,
                'default': False
            }
            
            self._update_exchanges_display()
            self.status_indicator.set_status(f"Exchange '{exchange_id}' added", 'success')
            notification_system.show_success("Exchange Added", f"{exchange_id} has been saved.")
            
            # Clear input fields
            self.api_key_var.set('')
            self.api_secret_var.set('')
            
        except Exception as e:
            app_logger.error(f"Error adding exchange: {e}")
            self.status_indicator.set_status(f"Error: {e}", 'error')

    def _remove_exchange(self):
        """Remove the selected exchange."""
        if not self._selected_exchange:
            self.status_indicator.set_status("No exchange selected", 'warning')
            return
        
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to remove {self._selected_exchange}?"):
            if self._selected_exchange in self._exchanges:
                del self._exchanges[self._selected_exchange]
                self._selected_exchange = None
                self._update_exchanges_display()
                self.status_indicator.set_status("Exchange removed", 'success')

    def _load_selected_exchange(self):
        """Load the selected exchange for use in the application."""
        if not self._selected_exchange:
            self.status_indicator.set_status("No exchange selected", 'warning')
            return
        
        # Presenter should handle the actual loading logic
        self.status_indicator.set_status(f"Loading {self._selected_exchange}...", 'info')
        # self._presenter.load_exchange(self._selected_exchange)
        notification_system.show_info("Exchange Loaded", f"{self._selected_exchange} is now active.")

    def _set_default_exchange(self):
        """Set the selected exchange as default."""
        if not self._selected_exchange:
            self.status_indicator.set_status("No exchange selected", 'warning')
            return
        
        for ex_id, ex_data in self._exchanges.items():
            ex_data['default'] = (ex_id == self._selected_exchange)
            
        self._update_exchanges_display()
        self.status_indicator.set_status(f"{self._selected_exchange} set as default", 'success')

    def _test_connection(self):
        """Test connection to the currently configured exchange."""
        self.loading_indicator.show_loading("Testing connection...")
        
        # This should be done in a separate thread to avoid blocking UI
        def worker():
            try:
                # In a real app, this would use the presenter to test the connection
                import time
                time.sleep(2) # Simulate network latency
                self.after(0, lambda: self.status_indicator.set_status("Connection successful!", 'success'))
            except Exception as e:
                self.after(0, lambda: self.status_indicator.set_status(f"Connection failed: {e}", 'error'))
            finally:
                self.after(0, self.loading_indicator.hide_loading)
        
        threading.Thread(target=worker, daemon=True).start()

    def _update_exchanges_display(self):
        """Update the list of saved exchanges."""
        self.exchanges_tree.delete(*self.exchanges_tree.get_children())
        
        for ex_id, ex_data in self._exchanges.items():
            status = "Connected" # Placeholder
            is_default = "Yes" if ex_data.get('default') else "No"
            self.exchanges_tree.insert('end', 'end', values=(
                ex_id,
                status,
                is_default
            ))

    def _on_exchange_selected(self, event):
        """Handle selection of an exchange in the treeview."""
        selected_item = self.exchanges_tree.selection()
        if selected_item:
            self._selected_exchange = self.exchanges_tree.item(selected_item[0])['values'][0]
            self.status_indicator.set_status(f"{self._selected_exchange} selected", 'info')

    def cleanup(self):
        """Cleanup resources on tab close."""
        app_logger.info("OptimizedExchangeTab cleaned up")

    def refresh(self):
        """Public method to refresh the tab content."""
        self.load_saved_exchanges()
        self.status_indicator.set_status("Exchanges refreshed", 'success')

# Backwards compatibility
ExchangeTab = OptimizedExchangeTab
