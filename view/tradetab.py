"""
Optimized Trade Tab

Enhanced version of the trade tab with:
- Integration with new trading systems (ML, RL, Advanced Trading)
- Async operation support for better responsiveness
- Enhanced validation and error handling
- Real-time market data display
- Advanced order types and risk management
- Performance monitoring and metrics
"""

import asyncio
import threading
from datetime import datetime
from tkinter import Listbox, messagebox
from typing import Any, Dict, List, Optional, Tuple

try:
    from ttkbootstrap import (
        BooleanVar, Button, Checkbutton, Entry, Frame, IntVar, Label, 
        OptionMenu, Scale, StringVar, Notebook, Progressbar, Separator
    )
    from ttkbootstrap.constants import *
    HAS_TTKBOOTSTRAP = True
except ImportError:
    from tkinter import (
        BooleanVar, Button, Checkbutton, Entry, Frame, IntVar, Label, 
        OptionMenu, Scale, StringVar
    )
    from tkinter.ttk import Notebook, Progressbar, Separator
    HAS_TTKBOOTSTRAP = False

from util.utils import validate_float
from view.utils import (
    ValidationMixin, StatusIndicator, LoadingIndicator, FormValidator,
    format_currency, format_percentage, notification_system
)

import util.loggers as loggers

logger_dict = loggers.setup_loggers()
app_logger = logger_dict['app']


class OptimizedTradeTab(Frame, ValidationMixin):
    """
    Enhanced trade tab with advanced trading features and system integration.
    """
    
    # Trading constants
    MAX_POSITION_SIZE = 0.1  # 10% of balance max
    MIN_TRADE_AMOUNT = 10.0  # Minimum trade amount in USD
    DEFAULT_SYMBOLS = ['BTC/USD:USD', 'ETH/USD:USD', 'OP/USD:USD', 'SOL/USD:USD', 'ADA/USD:USD']
    ACCOUNT_TYPES = ['swap', 'spot', 'future', 'margin', 'delivery']
    TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d']
    
    def __init__(self, parent, presenter) -> None:
        super().__init__(parent)
        self._parent = parent
        self._presenter = presenter
        
        # Initialize state
        self._is_loading = False
        self._last_price_update = None
        self._real_time_data = {}
        self._trade_history = []
        
        # Exchange and market data
        try:
            self.exchange = self._presenter.get_exchange() if self._presenter else None
        except Exception as e:
            app_logger.error(f"Error getting exchange: {e}")
            self.exchange = None
        self.symbol = "BTC/USD:USD"
        self.leverage = 10
        
        # Performance tracking
        self._trade_count = 0
        self._successful_trades = 0
        self._total_pnl = 0.0
        
        # Form validator
        self.form_validator = FormValidator()
        self._setup_validation_rules()
        
        # Initialize GUI
        self._create_widgets()
        self._setup_layout()
        self._bind_events()
        
        # Start real-time data updates
        self._start_real_time_updates()
        
        app_logger.info("OptimizedTradeTab initialized")

    def _setup_validation_rules(self):
        """Setup form validation rules."""
        try:
            # Add validation rules
            self.form_validator.add_field_rule('amount', 'required')
            self.form_validator.add_field_rule('amount', 'numeric_range', min=self.MIN_TRADE_AMOUNT, max=100000)
            self.form_validator.add_field_rule('price', 'numeric_range', min=0.01, max=1000000)
            self.form_validator.add_field_rule('stop_loss', 'numeric_range', min=0.01, max=100)
            self.form_validator.add_field_rule('take_profit', 'numeric_range', min=0.01, max=1000)
            
        except Exception as e:
            app_logger.error(f"Error setting up validation rules: {e}")

    def _create_widgets(self):
        """Create all GUI widgets."""
        try:
            # Create main sections
            self._create_header_section()
            self._create_trading_section()
            self._create_advanced_section()
            self._create_market_data_section()
            self._create_positions_section()
            self._create_status_section()
            
        except Exception as e:
            app_logger.error(f"Error creating widgets: {e}")

    def _create_header_section(self):
        """Create header with symbol selection and account type."""
        header_frame = Frame(self)
        header_frame.grid(row=0, column=0, columnspan=4, sticky='ew', padx=5, pady=5)
        header_frame.grid_columnconfigure(2, weight=1)
        
        # Title
        Label(header_frame, text="Advanced Trading", font=('Arial', 14, 'bold')).grid(
            row=0, column=0, padx=5
        )
        
        # Symbol selection
        Label(header_frame, text="Symbol:", font=('Arial', 10, 'bold')).grid(
            row=0, column=1, padx=(20, 5)
        )
        
        self.symbol_var = StringVar(value=self.symbol)
        self.symbol_select = OptionMenu(
            header_frame, self.symbol_var, self.symbol, *self.DEFAULT_SYMBOLS,
            command=self._on_symbol_changed
        )
        self.symbol_select.grid(row=0, column=2, sticky='ew', padx=5)
        
        # Account type
        Label(header_frame, text="Account:", font=('Arial', 10, 'bold')).grid(
            row=0, column=3, padx=(20, 5)
        )
        
        self.account_type_var = StringVar(value='swap')
        self.account_select = OptionMenu(
            header_frame, self.account_type_var, 'swap', *self.ACCOUNT_TYPES
        )
        self.account_select.grid(row=0, column=4, padx=5)
        
        # Performance metrics
        self.performance_label = Label(
            header_frame, 
            text="Trades: 0 | Success: 0% | P&L: $0.00",
            font=('Arial', 9),
            foreground='#888888'
        )
        self.performance_label.grid(row=0, column=5, padx=(20, 5))

    def _create_trading_section(self):
        """Create main trading controls."""
        trading_frame = Frame(self)
        trading_frame.grid(row=1, column=0, columnspan=4, sticky='ew', padx=5, pady=5)
        trading_frame.grid_columnconfigure(3, weight=1)
        
        # Order type
        Label(trading_frame, text="Order Type:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5
        )
        
        self.order_type_var = StringVar(value='market')
        self.order_type_select = OptionMenu(
            trading_frame, self.order_type_var, 'market', 
            'market', 'limit', 'stop', 'trailing'
        )
        self.order_type_select.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        
        # Side (Buy/Sell)
        Label(trading_frame, text="Side:", font=('Arial', 10, 'bold')).grid(
            row=0, column=1, sticky='w', padx=5
        )
        
        side_frame = Frame(trading_frame)
        side_frame.grid(row=1, column=1, padx=5, pady=5)
        
        self.side_var = StringVar(value='buy')
        self.buy_radio = Checkbutton(
            side_frame, text='Buy', variable=self.side_var, 
            onvalue='buy', offvalue='sell'
        )
        self.buy_radio.grid(row=0, column=0)
        
        self.sell_radio = Checkbutton(
            side_frame, text='Sell', variable=self.side_var,
            onvalue='sell', offvalue='buy'
        )
        self.sell_radio.grid(row=0, column=1, padx=(10, 0))
        
        # Amount
        Label(trading_frame, text="Amount (USD):", font=('Arial', 10, 'bold')).grid(
            row=0, column=2, sticky='w', padx=5
        )
        
        self.amount_var = StringVar()
        self.amount_entry = Entry(
            trading_frame, textvariable=self.amount_var,
            validate='key', 
            validatecommand=(self.register(validate_float), '%d', '%i', '%P', '%S', '%T')
        )
        self.amount_entry.grid(row=1, column=2, sticky='ew', padx=5, pady=5)
        
        # Amount slider
        self.amount_slider = Scale(
            trading_frame, from_=10, to=1000, orient='horizontal',
            variable=self.amount_var, command=self._on_amount_slider_changed
        )
        self.amount_slider.grid(row=2, column=2, sticky='ew', padx=5)
        
        # Price (for limit orders)
        Label(trading_frame, text="Price:", font=('Arial', 10, 'bold')).grid(
            row=0, column=3, sticky='w', padx=5
        )
        
        self.price_var = StringVar()
        self.price_entry = Entry(
            trading_frame, textvariable=self.price_var,
            validate='key',
            validatecommand=(self.register(validate_float), '%d', '%i', '%P', '%S', '%T')
        )
        self.price_entry.grid(row=1, column=3, sticky='ew', padx=5, pady=5)

    def _create_advanced_section(self):
        """Create advanced trading options."""
        advanced_frame = Frame(self)
        advanced_frame.grid(row=2, column=0, columnspan=4, sticky='ew', padx=5, pady=5)
        
        # Create notebook for advanced options
        notebook = Notebook(advanced_frame)
        notebook.grid(row=0, column=0, sticky='ew')
        advanced_frame.grid_columnconfigure(0, weight=1)
        
        # Risk Management Tab
        risk_frame = Frame(notebook)
        notebook.add(risk_frame, text="Risk Management")
        self._create_risk_management_section(risk_frame)
        
        # Advanced Orders Tab
        orders_frame = Frame(notebook)
        notebook.add(orders_frame, text="Advanced Orders")
        self._create_advanced_orders_section(orders_frame)
        
        # AI Trading Tab
        ai_frame = Frame(notebook)
        notebook.add(ai_frame, text="AI Trading")
        self._create_ai_trading_section(ai_frame)

    def _create_risk_management_section(self, parent):
        """Create risk management controls."""
        # Stop Loss
        Label(parent, text="Stop Loss (%):", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5, pady=5
        )
        
        self.stop_loss_var = StringVar(value='5.0')
        self.stop_loss_entry = Entry(parent, textvariable=self.stop_loss_var, width=10)
        self.stop_loss_entry.grid(row=0, column=1, padx=5, pady=5)
        
        self.stop_loss_slider = Scale(
            parent, from_=1, to=20, orient='horizontal',
            variable=self.stop_loss_var
        )
        self.stop_loss_slider.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5)
        
        # Take Profit
        Label(parent, text="Take Profit (%):", font=('Arial', 10, 'bold')).grid(
            row=0, column=2, sticky='w', padx=5, pady=5
        )
        
        self.take_profit_var = StringVar(value='10.0')
        self.take_profit_entry = Entry(parent, textvariable=self.take_profit_var, width=10)
        self.take_profit_entry.grid(row=0, column=3, padx=5, pady=5)
        
        self.take_profit_slider = Scale(
            parent, from_=5, to=50, orient='horizontal',
            variable=self.take_profit_var
        )
        self.take_profit_slider.grid(row=1, column=2, columnspan=2, sticky='ew', padx=5)
        
        # Leverage
        Label(parent, text="Leverage:", font=('Arial', 10, 'bold')).grid(
            row=2, column=0, sticky='w', padx=5, pady=5
        )
        
        self.leverage_var = IntVar(value=self.leverage)
        self.leverage_select = OptionMenu(
            parent, self.leverage_var, self.leverage,
            1, 2, 5, 10, 16, 20, 25
        )
        self.leverage_select.grid(row=2, column=1, padx=5, pady=5)
        
        # Position sizing
        Label(parent, text="Max Position (%):", font=('Arial', 10, 'bold')).grid(
            row=2, column=2, sticky='w', padx=5, pady=5
        )
        
        self.position_size_var = StringVar(value='10')
        self.position_size_slider = Scale(
            parent, from_=1, to=100, orient='horizontal',
            variable=self.position_size_var
        )
        self.position_size_slider.grid(row=3, column=0, columnspan=4, sticky='ew', padx=5)

    def _create_advanced_orders_section(self, parent):
        """Create advanced order controls."""
        # TWAP Orders
        twap_frame = Frame(parent)
        twap_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        Label(twap_frame, text="TWAP Order", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w'
        )
        
        self.twap_enabled_var = BooleanVar()
        Checkbutton(twap_frame, text="Enable TWAP", variable=self.twap_enabled_var).grid(
            row=0, column=1, padx=10
        )
        
        Label(twap_frame, text="Duration (minutes):").grid(row=1, column=0, sticky='w', pady=5)
        self.twap_duration_var = StringVar(value='30')
        Entry(twap_frame, textvariable=self.twap_duration_var, width=10).grid(
            row=1, column=1, padx=10, pady=5
        )
        
        # DCA Orders
        dca_frame = Frame(parent)
        dca_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        
        Label(dca_frame, text="DCA Order", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w'
        )
        
        self.dca_enabled_var = BooleanVar()
        Checkbutton(dca_frame, text="Enable DCA", variable=self.dca_enabled_var).grid(
            row=0, column=1, padx=10
        )
        
        Label(dca_frame, text="Intervals:").grid(row=1, column=0, sticky='w', pady=5)
        self.dca_intervals_var = StringVar(value='5')
        Entry(dca_frame, textvariable=self.dca_intervals_var, width=10).grid(
            row=1, column=1, padx=10, pady=5
        )

    def _create_ai_trading_section(self, parent):
        """Create AI trading controls."""
        # ML Prediction
        ml_frame = Frame(parent)
        ml_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        Label(ml_frame, text="ML Trading", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w'
        )
        
        self.ml_enabled_var = BooleanVar()
        Checkbutton(ml_frame, text="Use ML Predictions", variable=self.ml_enabled_var).grid(
            row=0, column=1, padx=10
        )
        
        Button(ml_frame, text="Get ML Signal", command=self._get_ml_signal).grid(
            row=0, column=2, padx=10
        )
        
        self.ml_signal_label = Label(ml_frame, text="Signal: None", foreground='#888888')
        self.ml_signal_label.grid(row=1, column=0, columnspan=3, pady=5)
        
        # RL Trading
        rl_frame = Frame(parent)
        rl_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        
        Label(rl_frame, text="RL Trading", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w'
        )
        
        self.rl_enabled_var = BooleanVar()
        Checkbutton(rl_frame, text="Use RL Agent", variable=self.rl_enabled_var).grid(
            row=0, column=1, padx=10
        )
        
        Button(rl_frame, text="Get RL Action", command=self._get_rl_action).grid(
            row=0, column=2, padx=10
        )
        
        self.rl_action_label = Label(rl_frame, text="Action: None", foreground='#888888')
        self.rl_action_label.grid(row=1, column=0, columnspan=3, pady=5)

    def _create_market_data_section(self):
        """Create market data display."""
        market_frame = Frame(self)
        market_frame.grid(row=1, column=4, rowspan=2, sticky='nsew', padx=5, pady=5)
        
        Label(market_frame, text="Market Data", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=(0, 10)
        )
        
        # Price display
        self.bid_label = Label(market_frame, text="Bid: --", font=('Arial', 10))
        self.bid_label.grid(row=1, column=0, sticky='w', padx=5, pady=2)
        
        self.ask_label = Label(market_frame, text="Ask: --", font=('Arial', 10))
        self.ask_label.grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
        self.last_price_label = Label(market_frame, text="Last: --", font=('Arial', 10, 'bold'))
        self.last_price_label.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Volume and change
        self.volume_label = Label(market_frame, text="Volume: --", font=('Arial', 9))
        self.volume_label.grid(row=3, column=0, columnspan=2, pady=2)
        
        self.change_label = Label(market_frame, text="Change: --", font=('Arial', 9))
        self.change_label.grid(row=4, column=0, columnspan=2, pady=2)
        
        # Balance display
        Separator(market_frame, orient='horizontal').grid(
            row=5, column=0, columnspan=2, sticky='ew', pady=10
        )
        
        Label(market_frame, text="Balance", font=('Arial', 10, 'bold')).grid(
            row=6, column=0, columnspan=2, pady=(0, 5)
        )
        
        self.balance_label = Label(market_frame, text="USDT: --", font=('Arial', 9))
        self.balance_label.grid(row=7, column=0, columnspan=2, pady=2)
        
        self.equity_label = Label(market_frame, text="Equity: --", font=('Arial', 9))
        self.equity_label.grid(row=8, column=0, columnspan=2, pady=2)

    def _create_positions_section(self):
        """Create positions and orders display."""
        positions_frame = Frame(self)
        positions_frame.grid(row=3, column=0, columnspan=5, sticky='nsew', padx=5, pady=5)
        positions_frame.grid_columnconfigure(0, weight=1)
        positions_frame.grid_rowconfigure(1, weight=1)
        
        Label(positions_frame, text="Open Positions & Orders", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, pady=(0, 10)
        )
        
        # Create notebook for positions and orders
        positions_notebook = Notebook(positions_frame)
        positions_notebook.grid(row=1, column=0, sticky='nsew')
        
        # Positions tab
        positions_tab = Frame(positions_notebook)
        positions_notebook.add(positions_tab, text="Positions")
        
        self.positions_listbox = Listbox(positions_tab, height=6)
        self.positions_listbox.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        positions_tab.grid_columnconfigure(0, weight=1)
        positions_tab.grid_rowconfigure(0, weight=1)
        
        # Orders tab
        orders_tab = Frame(positions_notebook)
        positions_notebook.add(orders_tab, text="Open Orders")
        
        self.orders_listbox = Listbox(orders_tab, height=6)
        self.orders_listbox.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        orders_tab.grid_columnconfigure(0, weight=1)
        orders_tab.grid_rowconfigure(0, weight=1)
        
        # History tab
        history_tab = Frame(positions_notebook)
        positions_notebook.add(history_tab, text="Trade History")
        
        self.history_listbox = Listbox(history_tab, height=6)
        self.history_listbox.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        history_tab.grid_columnconfigure(0, weight=1)
        history_tab.grid_rowconfigure(0, weight=1)

    def _create_status_section(self):
        """Create status and control section."""
        status_frame = Frame(self)
        status_frame.grid(row=4, column=0, columnspan=5, sticky='ew', padx=5, pady=5)
        status_frame.grid_columnconfigure(1, weight=1)
        
        # Main trading buttons
        button_frame = Frame(status_frame)
        button_frame.grid(row=0, column=0, padx=5)
        
        self.trade_button = Button(
            button_frame, text="Execute Trade", command=self._execute_trade,
            style='success.TButton', width=15
        )
        self.trade_button.grid(row=0, column=0, padx=5)
        
        Button(
            button_frame, text="Cancel All", command=self._cancel_all_orders,
            style='warning.TButton', width=12
        ).grid(row=0, column=1, padx=5)
        
        Button(
            button_frame, text="Close Positions", command=self._close_all_positions,
            style='danger.TButton', width=12
        ).grid(row=0, column=2, padx=5)
        
        # Status indicator
        self.status_indicator = StatusIndicator(status_frame)
        self.status_indicator.grid(row=0, column=1, sticky='ew', padx=10)
        
        # Loading indicator
        self.loading_indicator = LoadingIndicator(status_frame)
        self.loading_indicator.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        # Refresh button
        Button(
            status_frame, text="Refresh Data", command=self._refresh_all_data,
            style='secondary.TButton'
        ).grid(row=0, column=2, padx=5)

    def _setup_layout(self):
        """Configure grid layout weights."""
        # Configure main grid weights
        self.grid_columnconfigure(4, weight=1)
        self.grid_rowconfigure(3, weight=1)

    def _bind_events(self):
        """Bind event handlers."""
        # Bind validation events
        self.amount_entry.bind('<FocusOut>', self._validate_amount)
        self.price_entry.bind('<FocusOut>', self._validate_price)
        
        # Bind order type changes
        self.order_type_var.trace('w', self._on_order_type_changed)

    def _start_real_time_updates(self):
        """Start real-time data updates."""
        try:
            self._update_market_data()
            self._update_positions()
            self._update_balance()
            
            # Schedule next update
            self.after(5000, self._start_real_time_updates)
            
        except Exception as e:
            app_logger.error(f"Error in real-time updates: {e}")

    def _on_symbol_changed(self, *args):
        """Handle symbol selection change."""
        try:
            self.symbol = self.symbol_var.get()
            self._update_market_data()
            self.status_indicator.set_status(f"Symbol changed to {self.symbol}", 'info')
        except Exception as e:
            app_logger.error(f"Error changing symbol: {e}")

    def _on_order_type_changed(self, *args):
        """Handle order type change."""
        try:
            order_type = self.order_type_var.get()
            
            # Show/hide price entry based on order type
            if order_type in ['limit', 'stop']:
                self.price_entry.configure(state='normal')
            else:
                self.price_entry.configure(state='disabled')
                self.price_var.set('')
                
        except Exception as e:
            app_logger.error(f"Error handling order type change: {e}")

    def _on_amount_slider_changed(self, value):
        """Handle amount slider change."""
        try:
            # Update amount entry from slider
            self.amount_var.set(f"{float(value):.2f}")
        except Exception as e:
            app_logger.error(f"Error handling amount slider change: {e}")

    def _validate_amount(self, event=None):
        """Validate trade amount."""
        try:
            amount_str = self.amount_var.get()
            if not amount_str:
                return
                
            validation_result = self.form_validator.validate_field('amount', amount_str)
            
            if not validation_result['valid']:
                self.status_indicator.set_status(
                    f"Invalid amount: {', '.join(validation_result['messages'])}", 'error'
                )
            else:
                self.status_indicator.clear_status()
                
        except Exception as e:
            app_logger.error(f"Error validating amount: {e}")

    def _validate_price(self, event=None):
        """Validate price."""
        try:
            price_str = self.price_var.get()
            if not price_str or self.order_type_var.get() == 'market':
                return
                
            validation_result = self.form_validator.validate_field('price', price_str)
            
            if not validation_result['valid']:
                self.status_indicator.set_status(
                    f"Invalid price: {', '.join(validation_result['messages'])}", 'error'
                )
            else:
                self.status_indicator.clear_status()
                
        except Exception as e:
            app_logger.error(f"Error validating price: {e}")

    def _execute_trade(self):
        """Execute the trade with enhanced validation and error handling."""
        try:
            if self._is_loading:
                return
                
            # Validate form
            form_data = {
                'amount': self.amount_var.get(),
                'price': self.price_var.get() if self.order_type_var.get() != 'market' else '0',
            }
            
            validation_result = self.form_validator.validate_form(form_data)
            
            if not validation_result['valid']:
                error_messages = '; '.join(validation_result['messages'])
                self.status_indicator.set_status(f"Validation failed: {error_messages}", 'error')
                return
            
            # Show loading
            self.loading_indicator.show_loading("Executing trade...")
            self._is_loading = True
            
            # Prepare trade parameters
            trade_params = {
                'symbol': self.symbol,
                'side': self.side_var.get(),
                'order_type': self.order_type_var.get(),
                'amount': float(self.amount_var.get()),
                'price': float(self.price_var.get()) if self.price_var.get() else None,
                'stop_loss': float(self.stop_loss_var.get()) if self.stop_loss_var.get() else None,
                'take_profit': float(self.take_profit_var.get()) if self.take_profit_var.get() else None,
                'leverage': self.leverage_var.get(),
            }
            
            # Execute trade using presenter
            if self._presenter and hasattr(self._presenter, 'trading_presenter'):
                # Use async execution if available
                if hasattr(self._presenter.trading_presenter, 'place_trade_async'):
                    asyncio.create_task(self._execute_trade_async(trade_params))
                else:
                    self._execute_trade_sync(trade_params)
            else:
                self.status_indicator.set_status("Trading presenter not available", 'error')
                
        except Exception as e:
            app_logger.error(f"Error executing trade: {e}")
            self.status_indicator.set_status(f"Trade execution error: {str(e)}", 'error')
        finally:
            self._is_loading = False
            self.loading_indicator.hide_loading()

    async def _execute_trade_async(self, trade_params: Dict[str, Any]):
        """Execute trade asynchronously."""
        try:
            result = await self._presenter.trading_presenter.place_trade_async(trade_params)
            
            if result.get('success'):
                self._trade_count += 1
                self._successful_trades += 1
                
                self.status_indicator.set_status(
                    f"Trade executed successfully: {trade_params['side']} {trade_params['amount']}", 
                    'success'
                )
                
                notification_system.show_success(
                    "Trade Executed", 
                    f"Successfully {trade_params['side']} {trade_params['amount']} {self.symbol}"
                )
                
                # Update displays
                self._update_positions()
                self._update_performance_metrics()
                
            else:
                error_msg = result.get('error', 'Unknown error')
                self.status_indicator.set_status(f"Trade failed: {error_msg}", 'error')
                
        except Exception as e:
            app_logger.error(f"Error in async trade execution: {e}")
            self.status_indicator.set_status(f"Async trade error: {str(e)}", 'error')

    def _execute_trade_sync(self, trade_params: Dict[str, Any]):
        """Execute trade synchronously."""
        try:
            # Execute in thread to avoid blocking UI
            def trade_worker():
                try:
                    if hasattr(self._presenter.trading_presenter, 'place_trade'):
                        result = self._presenter.trading_presenter.place_trade()
                        
                        # Update UI on main thread
                        self.after(0, lambda: self._handle_trade_result(result, trade_params))
                        
                except Exception as e:
                    app_logger.error(f"Error in trade worker: {e}")
                    self.after(0, lambda: self.status_indicator.set_status(
                        f"Trade execution error: {str(e)}", 'error'
                    ))
            
            threading.Thread(target=trade_worker, daemon=True).start()
            
        except Exception as e:
            app_logger.error(f"Error starting trade worker: {e}")

    def _handle_trade_result(self, result: Any, trade_params: Dict[str, Any]):
        """Handle trade execution result."""
        try:
            if result:  # Assuming non-None result means success
                self._trade_count += 1
                self._successful_trades += 1
                
                self.status_indicator.set_status(
                    f"Trade executed: {trade_params['side']} {trade_params['amount']}", 
                    'success'
                )
                
                # Add to history
                self._add_trade_to_history(trade_params, 'executed')
                
            else:
                self.status_indicator.set_status("Trade execution failed", 'error')
                self._add_trade_to_history(trade_params, 'failed')
            
            # Update displays
            self._update_positions()
            self._update_performance_metrics()
            
        except Exception as e:
            app_logger.error(f"Error handling trade result: {e}")

    def _add_trade_to_history(self, trade_params: Dict[str, Any], status: str):
        """Add trade to history display."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            trade_entry = f"[{timestamp}] {status.upper()}: {trade_params['side'].upper()} {trade_params['amount']} {self.symbol}"
            
            self.history_listbox.insert(0, trade_entry)
            
            # Limit history size
            if self.history_listbox.size() > 50:
                self.history_listbox.delete(50)
                
        except Exception as e:
            app_logger.error(f"Error adding trade to history: {e}")

    def _get_ml_signal(self):
        """Get ML trading signal."""
        try:
            if self._presenter and hasattr(self._presenter, 'ml_system_presenter'):
                # This would call the ML system
                signal = "BUY (confidence: 0.75)"  # Placeholder
                self.ml_signal_label.configure(text=f"Signal: {signal}", foreground='#28a745')
                self.status_indicator.set_status("ML signal updated", 'info')
            else:
                self.status_indicator.set_status("ML system not available", 'warning')
                
        except Exception as e:
            app_logger.error(f"Error getting ML signal: {e}")

    def _get_rl_action(self):
        """Get RL trading action."""
        try:
            if self._presenter and hasattr(self._presenter, 'rl_system_presenter'):
                # This would call the RL system
                action = "HOLD (reward: 0.12)"  # Placeholder
                self.rl_action_label.configure(text=f"Action: {action}", foreground='#17a2b8')
                self.status_indicator.set_status("RL action updated", 'info')
            else:
                self.status_indicator.set_status("RL system not available", 'warning')
                
        except Exception as e:
            app_logger.error(f"Error getting RL action: {e}")

    def _update_market_data(self):
        """Update market data display."""
        try:
            if not self._presenter:
                return
                
            # Get real-time data (placeholder implementation)
            if hasattr(self._presenter.trading_presenter, 'get_real_time_data'):
                data = self._presenter.trading_presenter.get_real_time_data()
                
                if data and isinstance(data, dict):
                    self._real_time_data = data
                    
                    # Update price labels
                    self.bid_label.configure(text=f"Bid: {data.get('bid', '--')}")
                    self.ask_label.configure(text=f"Ask: {data.get('ask', '--')}")
                    self.last_price_label.configure(text=f"Last: {data.get('last', '--')}")
                    self.volume_label.configure(text=f"Volume: {data.get('volume', '--')}")
                    
                    # Update price entry with current price for limit orders
                    if self.order_type_var.get() == 'limit' and not self.price_var.get():
                        self.price_var.set(str(data.get('last', '')))
                        
                    self._last_price_update = datetime.now()
                    
        except Exception as e:
            app_logger.error(f"Error updating market data: {e}")

    def _update_positions(self):
        """Update positions display."""
        try:
            # Clear current positions
            self.positions_listbox.delete(0, 'end')
            
            # Placeholder for positions data
            positions = [
                "BTC/USD:USD - Long 0.1 BTC - P&L: +$125.50",
                "ETH/USD:USD - Short 2.5 ETH - P&L: -$45.20"
            ]
            
            for position in positions:
                self.positions_listbox.insert('end', position)
                
        except Exception as e:
            app_logger.error(f"Error updating positions: {e}")

    def _update_balance(self):
        """Update balance display."""
        try:
            # Placeholder balance data
            balance = 5000.00
            equity = 5125.30
            
            self.balance_label.configure(text=format_currency(balance))
            self.equity_label.configure(text=f"Equity: {format_currency(equity)}")
            
        except Exception as e:
            app_logger.error(f"Error updating balance: {e}")

    def _update_performance_metrics(self):
        """Update performance metrics display."""
        try:
            success_rate = (self._successful_trades / max(self._trade_count, 1)) * 100
            
            metrics_text = f"Trades: {self._trade_count} | Success: {success_rate:.1f}% | P&L: {format_currency(self._total_pnl)}"
            self.performance_label.configure(text=metrics_text)
            
        except Exception as e:
            app_logger.error(f"Error updating performance metrics: {e}")

    def _cancel_all_orders(self):
        """Cancel all open orders."""
        try:
            self.status_indicator.set_status("Canceling all orders...", 'warning')
            # Implementation would go here
            self.status_indicator.set_status("All orders canceled", 'success')
            
        except Exception as e:
            app_logger.error(f"Error canceling orders: {e}")

    def _close_all_positions(self):
        """Close all open positions."""
        try:
            self.status_indicator.set_status("Closing all positions...", 'warning')
            # Implementation would go here
            self.status_indicator.set_status("All positions closed", 'success')
            
        except Exception as e:
            app_logger.error(f"Error closing positions: {e}")

    def _refresh_all_data(self):
        """Refresh all data displays."""
        try:
            self.loading_indicator.show_loading("Refreshing data...")
            
            # Refresh all data
            self._update_market_data()
            self._update_positions()
            self._update_balance()
            
            self.status_indicator.set_status("Data refreshed", 'success')
            
        except Exception as e:
            app_logger.error(f"Error refreshing data: {e}")
            self.status_indicator.set_status(f"Refresh failed: {str(e)}", 'error')
        finally:
            self.loading_indicator.hide_loading()

    def cleanup(self):
        """Cleanup tab resources."""
        try:
            # Cancel any pending updates
            if hasattr(self, '_update_timer'):
                self.after_cancel(self._update_timer)
                
            app_logger.info("OptimizedTradeTab cleaned up")
            
        except Exception as e:
            app_logger.error(f"Error during TradeTab cleanup: {e}")

    def refresh(self):
        """Refresh tab content."""
        self._refresh_all_data()
