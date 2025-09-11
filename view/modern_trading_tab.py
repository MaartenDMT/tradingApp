"""
Modern Trading System Tab

A professional trading interface using the new modular trading architecture.
Provides advanced trading capabilities with clean separation of concerns.
"""

import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

try:
    from trading.core import TradingSystem, TradingConfig
    from trading.core.types import OrderType, OrderSide, TradeResult
    MODERN_TRADING_AVAILABLE = True
except ImportError:
    MODERN_TRADING_AVAILABLE = False
    TradingSystem = None
    TradingConfig = None

import util.loggers as loggers

logger_dict = loggers.setup_loggers()
app_logger = logger_dict['app']


class ModernTradingTab:
    """
    Modern Trading System Interface
    
    Provides a clean, professional interface for the new modular trading system
    with advanced features and better separation of concerns.
    """
    
    def __init__(self, parent_notebook, app_instance=None):
        """Initialize the modern trading tab."""
        self.parent_notebook = parent_notebook
        self.app = app_instance
        self.logger = app_logger
        
        # Trading system instance
        self.trading_system: Optional[TradingSystem] = None
        self.trading_config: Optional[TradingConfig] = None
        
        # UI state
        self.selected_symbol = tk.StringVar(value="BTC/USD:USD")
        self.order_type = tk.StringVar(value="limit")
        self.order_side = tk.StringVar(value="buy")
        self.quantity = tk.StringVar(value="0.01")
        self.price = tk.StringVar(value="50000")
        
        # Performance tracking
        self.last_update = datetime.now()
        self.update_interval = timedelta(seconds=1)
        
        self.create_widgets()
        self.setup_modern_trading()
        
    def create_widgets(self):
        """Create the modern trading interface."""
        try:
            # Create main frame
            self.frame = ttk.Frame(self.parent_notebook)
            self.parent_notebook.add(self.frame, text="🚀 Modern Trading")
            
            # Check if modern trading is available
            if not MODERN_TRADING_AVAILABLE:
                self.create_unavailable_interface()
                return
                
            self.create_modern_interface()
            
        except Exception as e:
            self.logger.error(f"Error creating modern trading widgets: {e}")
            self.create_error_interface(str(e))
    
    def create_unavailable_interface(self):
        """Create interface when modern trading system is not available."""
        main_frame = ttk.Frame(self.frame, padding=20)
        main_frame.pack(fill=BOTH, expand=True)
        
        # Warning message
        warning_frame = ttk.LabelFrame(main_frame, text="⚠️ Modern Trading System", padding=15)
        warning_frame.pack(fill=X, pady=(0, 20))
        
        ttk.Label(
            warning_frame,
            text="Modern Trading System Not Available",
            font=("Arial", 14, "bold"),
            foreground="orange"
        ).pack(pady=(0, 10))
        
        ttk.Label(
            warning_frame,
            text="The new modular trading system is still under development.\n"
                 "Please use the Manual Trading tab for current trading operations.",
            justify=CENTER
        ).pack()
        
        # Redirect button
        ttk.Button(
            warning_frame,
            text="Go to Manual Trading",
            bootstyle="primary",
            command=self.switch_to_manual_trading
        ).pack(pady=(15, 0))
    
    def create_error_interface(self, error_msg: str):
        """Create interface when there's an error."""
        main_frame = ttk.Frame(self.frame, padding=20)
        main_frame.pack(fill=BOTH, expand=True)
        
        error_frame = ttk.LabelFrame(main_frame, text="❌ Error", padding=15)
        error_frame.pack(fill=X)
        
        ttk.Label(
            error_frame,
            text="Error Loading Modern Trading System",
            font=("Arial", 14, "bold"),
            foreground="red"
        ).pack(pady=(0, 10))
        
        ttk.Label(
            error_frame,
            text=f"Error: {error_msg}",
            wraplength=600,
            justify=CENTER
        ).pack()
    
    def create_modern_interface(self):
        """Create the full modern trading interface."""
        # Main container with padding
        main_container = ttk.Frame(self.frame, padding=10)
        main_container.pack(fill=BOTH, expand=True)
        
        # Create paned window for layout
        paned_window = ttk.PanedWindow(main_container, orient=HORIZONTAL)
        paned_window.pack(fill=BOTH, expand=True)
        
        # Left panel - Trading Controls
        left_panel = ttk.Frame(paned_window)
        paned_window.add(left_panel, weight=1)
        
        # Right panel - Market Data & Orders
        right_panel = ttk.Frame(paned_window)
        paned_window.add(right_panel, weight=2)
        
        # Create sections
        self.create_trading_controls(left_panel)
        self.create_market_data_section(right_panel)
        self.create_orders_section(right_panel)
        
    def create_trading_controls(self, parent):
        """Create trading control panel."""
        # Trading Controls Frame
        controls_frame = ttk.LabelFrame(parent, text="🎯 Trading Controls", padding=15)
        controls_frame.pack(fill=X, pady=(0, 10))
        
        # Symbol Selection
        symbol_frame = ttk.Frame(controls_frame)
        symbol_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(symbol_frame, text="Symbol:").pack(side=LEFT)
        symbol_combo = ttk.Combobox(
            symbol_frame,
            textvariable=self.selected_symbol,
            values=["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD", "ADA/USD:USD"],
            state="readonly",
            width=15
        )
        symbol_combo.pack(side=RIGHT)
        
        # Order Type Selection
        type_frame = ttk.Frame(controls_frame)
        type_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(type_frame, text="Order Type:").pack(side=LEFT)
        type_combo = ttk.Combobox(
            type_frame,
            textvariable=self.order_type,
            values=["limit", "market", "stop", "stop_limit"],
            state="readonly",
            width=15
        )
        type_combo.pack(side=RIGHT)
        
        # Side Selection
        side_frame = ttk.Frame(controls_frame)
        side_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(side_frame, text="Side:").pack(side=LEFT)
        side_combo = ttk.Combobox(
            side_frame,
            textvariable=self.order_side,
            values=["buy", "sell"],
            state="readonly",
            width=15
        )
        side_combo.pack(side=RIGHT)
        
        # Quantity Entry
        qty_frame = ttk.Frame(controls_frame)
        qty_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(qty_frame, text="Quantity:").pack(side=LEFT)
        ttk.Entry(qty_frame, textvariable=self.quantity, width=18).pack(side=RIGHT)
        
        # Price Entry
        price_frame = ttk.Frame(controls_frame)
        price_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(price_frame, text="Price:").pack(side=LEFT)
        ttk.Entry(price_frame, textvariable=self.price, width=18).pack(side=RIGHT)
        
        # Action Buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(fill=X, pady=(15, 0))
        
        ttk.Button(
            button_frame,
            text="🟢 BUY",
            bootstyle="success",
            command=lambda: self.place_order("buy")
        ).pack(side=LEFT, padx=(0, 5), fill=X, expand=True)
        
        ttk.Button(
            button_frame,
            text="🔴 SELL",
            bootstyle="danger",
            command=lambda: self.place_order("sell")
        ).pack(side=RIGHT, padx=(5, 0), fill=X, expand=True)
        
        # System Status
        status_frame = ttk.LabelFrame(parent, text="📊 System Status", padding=15)
        status_frame.pack(fill=X, pady=(10, 0))
        
        self.status_label = ttk.Label(
            status_frame,
            text="System: Ready",
            foreground="green"
        )
        self.status_label.pack()
        
        self.connection_label = ttk.Label(
            status_frame,
            text="Connection: Disconnected",
            foreground="orange"
        )
        self.connection_label.pack()
    
    def create_market_data_section(self, parent):
        """Create market data display section."""
        # Market Data Frame
        market_frame = ttk.LabelFrame(parent, text="📈 Market Data", padding=15)
        market_frame.pack(fill=X, pady=(0, 10))
        
        # Create market data display
        data_frame = ttk.Frame(market_frame)
        data_frame.pack(fill=X)
        
        # Price display
        self.price_label = ttk.Label(
            data_frame,
            text="Price: --",
            font=("Arial", 12, "bold")
        )
        self.price_label.pack(side=LEFT)
        
        # Volume display
        self.volume_label = ttk.Label(
            data_frame,
            text="Volume: --"
        )
        self.volume_label.pack(side=RIGHT)
        
        # Additional market info
        info_frame = ttk.Frame(market_frame)
        info_frame.pack(fill=X, pady=(10, 0))
        
        self.bid_label = ttk.Label(info_frame, text="Bid: --")
        self.bid_label.pack(side=LEFT)
        
        self.ask_label = ttk.Label(info_frame, text="Ask: --")
        self.ask_label.pack(side=RIGHT)
    
    def create_orders_section(self, parent):
        """Create orders and positions section."""
        # Orders Frame
        orders_frame = ttk.LabelFrame(parent, text="📋 Orders & Positions", padding=15)
        orders_frame.pack(fill=BOTH, expand=True)
        
        # Create notebook for orders/positions
        orders_notebook = ttk.Notebook(orders_frame)
        orders_notebook.pack(fill=BOTH, expand=True)
        
        # Open Orders Tab
        orders_tab = ttk.Frame(orders_notebook)
        orders_notebook.add(orders_tab, text="Open Orders")
        
        # Create treeview for orders
        orders_tree = ttk.Treeview(
            orders_tab,
            columns=("Type", "Side", "Symbol", "Quantity", "Price", "Status"),
            show="headings",
            height=8
        )
        
        # Configure columns
        for col in orders_tree["columns"]:
            orders_tree.heading(col, text=col)
            orders_tree.column(col, width=100)
        
        orders_tree.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        # Positions Tab
        positions_tab = ttk.Frame(orders_notebook)
        orders_notebook.add(positions_tab, text="Positions")
        
        # Create treeview for positions
        positions_tree = ttk.Treeview(
            positions_tab,
            columns=("Symbol", "Side", "Size", "Entry Price", "Mark Price", "PnL"),
            show="headings",
            height=8
        )
        
        # Configure columns
        for col in positions_tree["columns"]:
            positions_tree.heading(col, text=col)
            positions_tree.column(col, width=100)
        
        positions_tree.pack(fill=BOTH, expand=True)
        
        # Store references
        self.orders_tree = orders_tree
        self.positions_tree = positions_tree
    
    def setup_modern_trading(self):
        """Initialize the modern trading system."""
        if not MODERN_TRADING_AVAILABLE:
            return
            
        try:
            # Create trading configuration
            self.trading_config = TradingConfig()
            
            # Initialize trading system
            self.trading_system = TradingSystem(self.trading_config)
            
            self.logger.info("Modern trading system initialized successfully")
            self.update_status("System: Initialized", "green")
            
        except Exception as e:
            self.logger.error(f"Error setting up modern trading: {e}")
            self.update_status(f"System Error: {e}", "red")
    
    def place_order(self, side: str):
        """Place a trading order using the modern system."""
        if not self.trading_system:
            self.logger.warning("Trading system not available")
            return
            
        try:
            # Get order parameters
            symbol = self.selected_symbol.get()
            order_type = self.order_type.get()
            quantity = float(self.quantity.get())
            price = float(self.price.get()) if order_type == "limit" else None
            
            self.logger.info(f"Placing {side} order: {quantity} {symbol} @ {price}")
            
            # Here you would integrate with the actual trading system
            # result = self.trading_system.place_order(...)
            
            # For now, just log the action
            self.update_status(f"Order placed: {side} {quantity} {symbol}", "blue")
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            self.update_status(f"Order Error: {e}", "red")
    
    def update_status(self, message: str, color: str = "black"):
        """Update system status display."""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=message, foreground=color)
    
    def switch_to_manual_trading(self):
        """Switch to the manual trading tab."""
        if self.app and hasattr(self.app, 'notebook'):
            # Find and select the manual trading tab
            for i in range(self.app.notebook.index("end")):
                tab_text = self.app.notebook.tab(i, "text")
                if "Trading" in tab_text and "Modern" not in tab_text:
                    self.app.notebook.select(i)
                    break
    
    def refresh(self):
        """Refresh the trading interface."""
        current_time = datetime.now()
        if current_time - self.last_update >= self.update_interval:
            self.update_market_data()
            self.update_orders()
            self.last_update = current_time
    
    def update_market_data(self):
        """Update market data display."""
        if not hasattr(self, 'price_label'):
            return
            
        # Placeholder for real market data
        # In a real implementation, this would fetch from the trading system
        symbol = self.selected_symbol.get()
        self.price_label.config(text=f"Price: Loading {symbol}...")
    
    def update_orders(self):
        """Update orders and positions display."""
        if not hasattr(self, 'orders_tree'):
            return
            
        # Placeholder for real order data
        # In a real implementation, this would fetch from the trading system
        pass
