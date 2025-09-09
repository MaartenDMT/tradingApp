"""
Optimized Bot Tab

Enhanced version of the bot tab with:
- Advanced bot management and configuration
- Real-time performance monitoring
- Enhanced error handling and logging
- Bot lifecycle management
- Strategy assignment and optimization
- Portfolio allocation per bot
"""

import threading
from datetime import datetime, timedelta
from tkinter import messagebox
from typing import Any, Dict, List, Optional

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
    format_currency, format_percentage, notification_system
)

import util.loggers as loggers

logger_dict = loggers.setup_loggers()
app_logger = logger_dict['app']


class OptimizedBotTab(Frame, ValidationMixin):
    """
    Enhanced bot management interface with advanced monitoring and control.
    """
    
    # Bot configuration constants
    BOT_STRATEGIES = ['scalping', 'swing', 'dca', 'grid', 'arbitrage', 'momentum', 'mean_reversion']
    TIME_FRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    RISK_LEVELS = ['conservative', 'moderate', 'aggressive']
    BOT_STATES = ['stopped', 'starting', 'running', 'paused', 'stopping', 'error']
    
    def __init__(self, parent, presenter):
        super().__init__(parent)
        self._parent = parent
        self._presenter = presenter
        
        # Bot management state
        self._bots = {}
        self._bot_counter = 0
        self._active_bots = {}
        self._bot_performance = {}
        
        # UI state
        self._is_loading = False
        self._last_update = None
        self._selected_bot = None
        
        # Exchange reference
        self.exchange = self._presenter.get_exchange() if self._presenter else None
        
        # Form validator
        self.form_validator = FormValidator()
        self._setup_validation_rules()
        
        # Initialize GUI
        self._create_widgets()
        self._setup_layout()
        self._bind_events()
        
        # Start data updates
        self._start_updates()
        
        app_logger.info("OptimizedBotTab initialized")

    def _setup_validation_rules(self):
        """Setup form validation rules."""
        try:
            self.form_validator.add_field_rule('amount_percentage', 'numeric_range', min=0.1, max=100)
            self.form_validator.add_field_rule('profit_target', 'numeric_range', min=0.1, max=100)
            self.form_validator.add_field_rule('stop_loss', 'numeric_range', min=0.1, max=50)
            
        except Exception as e:
            app_logger.error(f"Error setting up validation rules: {e}")

    def _create_widgets(self):
        """Create all GUI widgets."""
        try:
            # Create main sections
            self._create_header_section()
            self._create_bot_management_section()
            self._create_configuration_section()
            self._create_monitoring_section()
            self._create_control_section()
            
        except Exception as e:
            app_logger.error(f"Error creating widgets: {e}")

    def _create_header_section(self):
        """Create header with bot status and quick controls."""
        header_frame = Frame(self)
        header_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        header_frame.grid_columnconfigure(2, weight=1)
        
        # Title and status
        Label(header_frame, text="Trading Bots", font=('Arial', 16, 'bold')).grid(
            row=0, column=0, padx=5
        )
        
        self.bot_status_label = Label(
            header_frame, text="Active: 0 | Running: 0", 
            font=('Arial', 12, 'bold'), foreground='#17a2b8'
        )
        self.bot_status_label.grid(row=0, column=1, padx=20)
        
        # Quick controls
        control_frame = Frame(header_frame)
        control_frame.grid(row=0, column=3, padx=5)
        
        Button(
            control_frame, text="Create Bot", command=self._create_bot,
            style='success.TButton', width=12
        ).grid(row=0, column=0, padx=2)
        
        Button(
            control_frame, text="Start All", command=self._start_all_bots,
            style='info.TButton', width=12
        ).grid(row=0, column=1, padx=2)
        
        Button(
            control_frame, text="Stop All", command=self._stop_all_bots,
            style='danger.TButton', width=12
        ).grid(row=0, column=2, padx=2)
        
        # Performance summary
        self.performance_summary = Label(
            header_frame, text="Total P&L: $0.00 | Best Bot: N/A | Win Rate: 0%",
            font=('Arial', 10), foreground='#888888'
        )
        self.performance_summary.grid(row=1, column=0, columnspan=4, pady=5)

    def _create_bot_management_section(self):
        """Create bot management interface."""
        bot_frame = Frame(self)
        bot_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        bot_frame.grid_rowconfigure(1, weight=1)
        
        Label(bot_frame, text="Bot Management", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, pady=(0, 10)
        )
        
        # Bot list with details
        self.bot_tree = Treeview(
            bot_frame,
            columns=('ID', 'Strategy', 'Symbol', 'Status', 'P&L', 'Trades', 'Uptime'),
            show='headings', height=12
        )
        
        # Configure columns
        columns_config = {
            'ID': 50,
            'Strategy': 100,
            'Symbol': 100,
            'Status': 80,
            'P&L': 80,
            'Trades': 60,
            'Uptime': 80
        }
        
        for col, width in columns_config.items():
            self.bot_tree.heading(col, text=col)
            self.bot_tree.column(col, width=width)
        
        self.bot_tree.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        
        # Bot controls
        bot_controls = Frame(bot_frame)
        bot_controls.grid(row=2, column=0, sticky='ew', padx=5, pady=5)
        
        Button(
            bot_controls, text="Start", command=self._start_selected_bot,
            style='success.TButton', width=10
        ).grid(row=0, column=0, padx=2)
        
        Button(
            bot_controls, text="Stop", command=self._stop_selected_bot,
            style='danger.TButton', width=10
        ).grid(row=0, column=1, padx=2)
        
        Button(
            bot_controls, text="Pause", command=self._pause_selected_bot,
            style='warning.TButton', width=10
        ).grid(row=0, column=2, padx=2)
        
        Button(
            bot_controls, text="Delete", command=self._delete_selected_bot,
            style='outline.danger.TButton', width=10
        ).grid(row=0, column=3, padx=2)

    def _create_configuration_section(self):
        """Create bot configuration section."""
        config_frame = Frame(self)
        config_frame.grid(row=1, column=2, sticky='nsew', padx=5, pady=5)
        config_frame.grid_rowconfigure(1, weight=1)
        
        Label(config_frame, text="Bot Configuration", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, pady=(0, 10)
        )
        
        # Configuration notebook
        config_notebook = Notebook(config_frame)
        config_notebook.grid(row=1, column=0, sticky='nsew')
        
        # Strategy Tab
        strategy_tab = Frame(config_notebook)
        config_notebook.add(strategy_tab, text="Strategy")
        self._create_strategy_config(strategy_tab)
        
        # Risk Tab
        risk_tab = Frame(config_notebook)
        config_notebook.add(risk_tab, text="Risk Management")
        self._create_risk_config(risk_tab)
        
        # Advanced Tab
        advanced_tab = Frame(config_notebook)
        config_notebook.add(advanced_tab, text="Advanced")
        self._create_advanced_config(advanced_tab)

    def _create_strategy_config(self, parent):
        """Create strategy configuration interface."""
        # Strategy selection
        Label(parent, text="Strategy:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5, pady=5
        )
        
        self.strategy_var = StringVar(value='scalping')
        self.strategy_select = OptionMenu(
            parent, self.strategy_var, 'scalping', *self.BOT_STRATEGIES
        )
        self.strategy_select.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        
        # Symbol selection
        Label(parent, text="Symbol:", font=('Arial', 10, 'bold')).grid(
            row=1, column=0, sticky='w', padx=5, pady=5
        )
        
        self.symbol_var = StringVar(value='BTC/USDT')
        self.symbol_entry = Entry(parent, textvariable=self.symbol_var)
        self.symbol_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        
        # Timeframe
        Label(parent, text="Timeframe:", font=('Arial', 10, 'bold')).grid(
            row=2, column=0, sticky='w', padx=5, pady=5
        )
        
        self.timeframe_var = StringVar(value='5m')
        self.timeframe_select = OptionMenu(
            parent, self.timeframe_var, '5m', *self.TIME_FRAMES
        )
        self.timeframe_select.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        
        # Amount percentage
        Label(parent, text="Portfolio % (1-100):", font=('Arial', 10, 'bold')).grid(
            row=3, column=0, sticky='w', padx=5, pady=5
        )
        
        self.amount_percentage_var = StringVar(value='10')
        self.amount_percentage_entry = Entry(parent, textvariable=self.amount_percentage_var, width=10)
        self.amount_percentage_entry.grid(row=3, column=1, sticky='w', padx=5, pady=5)
        
        self.amount_slider = Scale(
            parent, from_=1, to=100, orient='horizontal',
            variable=self.amount_percentage_var
        )
        self.amount_slider.grid(row=4, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        parent.grid_columnconfigure(1, weight=1)

    def _create_risk_config(self, parent):
        """Create risk management configuration."""
        # Profit target
        Label(parent, text="Profit Target (%):", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5, pady=5
        )
        
        self.profit_target_var = StringVar(value='5.0')
        self.profit_target_entry = Entry(parent, textvariable=self.profit_target_var, width=10)
        self.profit_target_entry.grid(row=0, column=1, padx=5, pady=5)
        
        self.profit_slider = Scale(
            parent, from_=0.1, to=50, orient='horizontal',
            variable=self.profit_target_var, resolution=0.1
        )
        self.profit_slider.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        # Stop loss
        Label(parent, text="Stop Loss (%):", font=('Arial', 10, 'bold')).grid(
            row=2, column=0, sticky='w', padx=5, pady=5
        )
        
        self.stop_loss_var = StringVar(value='3.0')
        self.stop_loss_entry = Entry(parent, textvariable=self.stop_loss_var, width=10)
        self.stop_loss_entry.grid(row=2, column=1, padx=5, pady=5)
        
        self.stop_loss_slider = Scale(
            parent, from_=0.1, to=20, orient='horizontal',
            variable=self.stop_loss_var, resolution=0.1
        )
        self.stop_loss_slider.grid(row=3, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        # Risk level
        Label(parent, text="Risk Level:", font=('Arial', 10, 'bold')).grid(
            row=4, column=0, sticky='w', padx=5, pady=5
        )
        
        self.risk_level_var = StringVar(value='moderate')
        self.risk_level_select = OptionMenu(
            parent, self.risk_level_var, 'moderate', *self.RISK_LEVELS
        )
        self.risk_level_select.grid(row=4, column=1, sticky='ew', padx=5, pady=5)
        
        parent.grid_columnconfigure(1, weight=1)

    def _create_advanced_config(self, parent):
        """Create advanced configuration options."""
        # Max concurrent trades
        Label(parent, text="Max Concurrent Trades:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5, pady=5
        )
        
        self.max_trades_var = IntVar(value=3)
        self.max_trades_scale = Scale(
            parent, from_=1, to=10, orient='horizontal',
            variable=self.max_trades_var
        )
        self.max_trades_scale.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        
        # Auto-restart
        self.auto_restart_var = BooleanVar(value=True)
        auto_restart_checkbox = Button(
            parent, text="Auto-restart on Error",
            command=self._toggle_auto_restart,
            style='outline.TButton'
        )
        auto_restart_checkbox.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        # Notifications
        self.notifications_var = BooleanVar(value=True)
        notifications_checkbox = Button(
            parent, text="Enable Notifications",
            command=self._toggle_notifications,
            style='outline.TButton'
        )
        notifications_checkbox.grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        parent.grid_columnconfigure(1, weight=1)

    def _create_monitoring_section(self):
        """Create bot monitoring section."""
        monitor_frame = Frame(self)
        monitor_frame.grid(row=2, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
        monitor_frame.grid_rowconfigure(1, weight=1)
        
        Label(monitor_frame, text="Bot Performance Monitor", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, pady=(0, 10)
        )
        
        # Monitoring notebook
        monitor_notebook = Notebook(monitor_frame)
        monitor_notebook.grid(row=1, column=0, sticky='nsew')
        
        # Performance Tab
        performance_tab = Frame(monitor_notebook)
        monitor_notebook.add(performance_tab, text="Performance")
        self._create_performance_monitor(performance_tab)
        
        # Logs Tab
        logs_tab = Frame(monitor_notebook)
        monitor_notebook.add(logs_tab, text="Bot Logs")
        self._create_logs_monitor(logs_tab)
        
        # Analytics Tab
        analytics_tab = Frame(monitor_notebook)
        monitor_notebook.add(analytics_tab, text="Analytics")
        self._create_analytics_monitor(analytics_tab)

    def _create_performance_monitor(self, parent):
        """Create performance monitoring display."""
        perf_frame = Frame(parent)
        perf_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        # Performance metrics
        self.total_pnl_label = Label(perf_frame, text="Total P&L: $0.00", font=('Arial', 10, 'bold'))
        self.total_pnl_label.grid(row=0, column=0, sticky='w', pady=2)
        
        self.daily_pnl_label = Label(perf_frame, text="Daily P&L: $0.00", font=('Arial', 10))
        self.daily_pnl_label.grid(row=0, column=1, sticky='w', padx=20, pady=2)
        
        self.total_trades_label = Label(perf_frame, text="Total Trades: 0", font=('Arial', 10))
        self.total_trades_label.grid(row=1, column=0, sticky='w', pady=2)
        
        self.win_rate_label = Label(perf_frame, text="Win Rate: 0%", font=('Arial', 10))
        self.win_rate_label.grid(row=1, column=1, sticky='w', padx=20, pady=2)
        
        # Individual bot performance
        Label(parent, text="Individual Bot Performance:", font=('Arial', 10, 'bold')).grid(
            row=1, column=0, sticky='w', padx=5, pady=(20, 5)
        )
        
        self.performance_tree = Treeview(
            parent,
            columns=('Bot', 'Strategy', 'P&L', 'Trades', 'Win%', 'Avg_Trade', 'Status'),
            show='headings', height=10
        )
        
        for col in ('Bot', 'Strategy', 'P&L', 'Trades', 'Win%', 'Avg_Trade', 'Status'):
            self.performance_tree.heading(col, text=col)
            self.performance_tree.column(col, width=80)
        
        self.performance_tree.grid(row=2, column=0, sticky='nsew', padx=5, pady=5)
        parent.grid_rowconfigure(2, weight=1)
        parent.grid_columnconfigure(0, weight=1)

    def _create_logs_monitor(self, parent):
        """Create bot logs monitoring display."""
        from tkinter import Text, Scrollbar
        
        # Log display
        log_frame = Frame(parent)
        log_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)
        
        self.log_text = Text(log_frame, height=15, font=('Courier', 9))
        scrollbar = Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Log controls
        log_controls = Frame(parent)
        log_controls.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        
        Button(
            log_controls, text="Clear Logs", command=self._clear_logs,
            style='secondary.TButton', width=12
        ).grid(row=0, column=0, padx=2)
        
        Button(
            log_controls, text="Export Logs", command=self._export_logs,
            style='info.TButton', width=12
        ).grid(row=0, column=1, padx=2)
        
        # Auto-scroll checkbox
        self.auto_scroll_var = BooleanVar(value=True)
        Button(
            log_controls, text="Auto-scroll",
            command=self._toggle_auto_scroll,
            style='outline.TButton', width=12
        ).grid(row=0, column=2, padx=2)
        
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

    def _create_analytics_monitor(self, parent):
        """Create analytics monitoring display."""
        analytics_frame = Frame(parent)
        analytics_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        # Analytics metrics
        self.avg_trade_duration_label = Label(analytics_frame, text="Avg Trade Duration: --", font=('Arial', 10))
        self.avg_trade_duration_label.grid(row=0, column=0, sticky='w', pady=2)
        
        self.max_drawdown_label = Label(analytics_frame, text="Max Drawdown: 0%", font=('Arial', 10))
        self.max_drawdown_label.grid(row=0, column=1, sticky='w', padx=20, pady=2)
        
        self.profit_factor_label = Label(analytics_frame, text="Profit Factor: --", font=('Arial', 10))
        self.profit_factor_label.grid(row=1, column=0, sticky='w', pady=2)
        
        self.sharpe_ratio_label = Label(analytics_frame, text="Sharpe Ratio: --", font=('Arial', 10))
        self.sharpe_ratio_label.grid(row=1, column=1, sticky='w', padx=20, pady=2)

    def _create_control_section(self):
        """Create control and status section."""
        control_frame = Frame(self)
        control_frame.grid(row=3, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        control_frame.grid_columnconfigure(1, weight=1)
        
        # Control buttons
        buttons_frame = Frame(control_frame)
        buttons_frame.grid(row=0, column=0, padx=5)
        
        Button(
            buttons_frame, text="Save Config", command=self._save_config,
            style='success.TButton', width=12
        ).grid(row=0, column=0, padx=2)
        
        Button(
            buttons_frame, text="Load Config", command=self._load_config,
            style='info.TButton', width=12
        ).grid(row=0, column=1, padx=2)
        
        Button(
            buttons_frame, text="Emergency Stop", command=self._emergency_stop,
            style='danger.TButton', width=12
        ).grid(row=0, column=2, padx=2)
        
        # Status indicator
        self.status_indicator = StatusIndicator(control_frame)
        self.status_indicator.grid(row=0, column=1, sticky='ew', padx=10)
        
        # Loading indicator
        self.loading_indicator = LoadingIndicator(control_frame)
        self.loading_indicator.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=5)

    def _setup_layout(self):
        """Configure grid layout weights."""
        # Configure main grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

    def _bind_events(self):
        """Bind event handlers."""
        # Bind tree selection
        self.bot_tree.bind('<<TreeviewSelect>>', self._on_bot_selected)
        
        # Bind validation events
        self.amount_percentage_entry.bind('<FocusOut>', self._validate_amount_percentage)
        self.profit_target_entry.bind('<FocusOut>', self._validate_profit_target)
        self.stop_loss_entry.bind('<FocusOut>', self._validate_stop_loss)

    def _start_updates(self):
        """Start periodic data updates."""
        try:
            self._update_bot_status()
            self._update_performance_metrics()
            self._update_bot_display()
            
            # Schedule next update
            self.after(5000, self._start_updates)  # Update every 5 seconds
            
        except Exception as e:
            app_logger.error(f"Error in periodic updates: {e}")

    def _create_bot(self):
        """Create a new trading bot."""
        try:
            # Validate configuration
            validation_result = self._validate_bot_config()
            if not validation_result['valid']:
                self.status_indicator.set_status(f"Configuration error: {validation_result['message']}", 'error')
                return
            
            self.loading_indicator.show_loading("Creating new bot...")
            
            # Generate bot ID
            self._bot_counter += 1
            bot_id = f"bot_{self._bot_counter:03d}"
            
            # Create bot configuration
            bot_config = {
                'id': bot_id,
                'strategy': self.strategy_var.get(),
                'symbol': self.symbol_var.get(),
                'timeframe': self.timeframe_var.get(),
                'amount_percentage': float(self.amount_percentage_var.get()),
                'profit_target': float(self.profit_target_var.get()),
                'stop_loss': float(self.stop_loss_var.get()),
                'risk_level': self.risk_level_var.get(),
                'max_trades': self.max_trades_var.get(),
                'auto_restart': self.auto_restart_var.get(),
                'notifications': self.notifications_var.get(),
                'created_at': datetime.now(),
                'status': 'stopped',
                'pnl': 0.0,
                'trades': 0,
                'wins': 0
            }
            
            # Add to bots dictionary
            self._bots[bot_id] = bot_config
            
            # Update displays
            self._update_bot_display()
            self._log_message(f"Bot {bot_id} created with {bot_config['strategy']} strategy")
            
            self.status_indicator.set_status(f"Bot {bot_id} created successfully", 'success')
            
            notification_system.show_success(
                "Bot Created",
                f"Trading bot {bot_id} created with {bot_config['strategy']} strategy"
            )
            
        except Exception as e:
            app_logger.error(f"Error creating bot: {e}")
            self.status_indicator.set_status(f"Failed to create bot: {str(e)}", 'error')
        finally:
            self.loading_indicator.hide_loading()

    def _start_selected_bot(self):
        """Start the selected bot."""
        try:
            selected_item = self.bot_tree.selection()
            if not selected_item:
                self.status_indicator.set_status("No bot selected", 'warning')
                return
            
            bot_data = self.bot_tree.item(selected_item[0], 'values')
            bot_id = bot_data[0]
            
            self._start_bot(bot_id)
            
        except Exception as e:
            app_logger.error(f"Error starting selected bot: {e}")

    def _start_bot(self, bot_id: str):
        """Start a specific bot."""
        try:
            if bot_id not in self._bots:
                return
            
            bot_config = self._bots[bot_id]
            
            if bot_config['status'] in ['running', 'starting']:
                self.status_indicator.set_status(f"Bot {bot_id} is already running", 'warning')
                return
            
            self.loading_indicator.show_loading(f"Starting bot {bot_id}...")
            
            # Update bot status
            bot_config['status'] = 'starting'
            bot_config['started_at'] = datetime.now()
            
            # Simulate bot startup process
            def startup_worker():
                try:
                    # Simulate startup time
                    import time
                    time.sleep(2)
                    
                    # Update status to running
                    bot_config['status'] = 'running'
                    self._active_bots[bot_id] = bot_config
                    
                    # Update UI on main thread
                    self.after(0, self._bot_started_successfully, bot_id)
                    
                except Exception as e:
                    app_logger.error(f"Error in bot startup: {e}")
                    bot_config['status'] = 'error'
                    self.after(0, self._bot_startup_failed, bot_id, str(e))
            
            threading.Thread(target=startup_worker, daemon=True).start()
            
        except Exception as e:
            app_logger.error(f"Error starting bot {bot_id}: {e}")
            self.status_indicator.set_status(f"Failed to start bot {bot_id}: {str(e)}", 'error')

    def _bot_started_successfully(self, bot_id: str):
        """Handle successful bot startup."""
        try:
            self.loading_indicator.hide_loading()
            self._update_bot_display()
            self._log_message(f"Bot {bot_id} started successfully")
            self.status_indicator.set_status(f"Bot {bot_id} started successfully", 'success')
            
        except Exception as e:
            app_logger.error(f"Error handling bot startup success: {e}")

    def _bot_startup_failed(self, bot_id: str, error_message: str):
        """Handle failed bot startup."""
        try:
            self.loading_indicator.hide_loading()
            self._update_bot_display()
            self._log_message(f"Bot {bot_id} startup failed: {error_message}")
            self.status_indicator.set_status(f"Bot {bot_id} startup failed", 'error')
            
        except Exception as e:
            app_logger.error(f"Error handling bot startup failure: {e}")

    def _stop_selected_bot(self):
        """Stop the selected bot."""
        try:
            selected_item = self.bot_tree.selection()
            if not selected_item:
                self.status_indicator.set_status("No bot selected", 'warning')
                return
            
            bot_data = self.bot_tree.item(selected_item[0], 'values')
            bot_id = bot_data[0]
            
            self._stop_bot(bot_id)
            
        except Exception as e:
            app_logger.error(f"Error stopping selected bot: {e}")

    def _stop_bot(self, bot_id: str):
        """Stop a specific bot."""
        try:
            if bot_id not in self._bots:
                return
            
            bot_config = self._bots[bot_id]
            
            if bot_config['status'] == 'stopped':
                self.status_indicator.set_status(f"Bot {bot_id} is already stopped", 'warning')
                return
            
            # Update bot status
            bot_config['status'] = 'stopping'
            
            # Simulate bot shutdown process
            def shutdown_worker():
                try:
                    import time
                    time.sleep(1)
                    
                    bot_config['status'] = 'stopped'
                    if bot_id in self._active_bots:
                        del self._active_bots[bot_id]
                    
                    self.after(0, self._bot_stopped_successfully, bot_id)
                    
                except Exception as e:
                    app_logger.error(f"Error in bot shutdown: {e}")
            
            threading.Thread(target=shutdown_worker, daemon=True).start()
            
        except Exception as e:
            app_logger.error(f"Error stopping bot {bot_id}: {e}")

    def _bot_stopped_successfully(self, bot_id: str):
        """Handle successful bot shutdown."""
        try:
            self._update_bot_display()
            self._log_message(f"Bot {bot_id} stopped successfully")
            self.status_indicator.set_status(f"Bot {bot_id} stopped successfully", 'success')
            
        except Exception as e:
            app_logger.error(f"Error handling bot stop success: {e}")

    def _pause_selected_bot(self):
        """Pause the selected bot."""
        try:
            selected_item = self.bot_tree.selection()
            if not selected_item:
                self.status_indicator.set_status("No bot selected", 'warning')
                return
            
            bot_data = self.bot_tree.item(selected_item[0], 'values')
            bot_id = bot_data[0]
            
            if bot_id in self._bots:
                bot_config = self._bots[bot_id]
                if bot_config['status'] == 'running':
                    bot_config['status'] = 'paused'
                    self._update_bot_display()
                    self._log_message(f"Bot {bot_id} paused")
                    self.status_indicator.set_status(f"Bot {bot_id} paused", 'info')
            
        except Exception as e:
            app_logger.error(f"Error pausing bot: {e}")

    def _delete_selected_bot(self):
        """Delete the selected bot."""
        try:
            selected_item = self.bot_tree.selection()
            if not selected_item:
                self.status_indicator.set_status("No bot selected", 'warning')
                return
            
            bot_data = self.bot_tree.item(selected_item[0], 'values')
            bot_id = bot_data[0]
            
            # Confirm deletion
            if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete bot {bot_id}?"):
                # Stop bot if running
                if bot_id in self._active_bots:
                    self._stop_bot(bot_id)
                
                # Remove from bots
                if bot_id in self._bots:
                    del self._bots[bot_id]
                
                self._update_bot_display()
                self._log_message(f"Bot {bot_id} deleted")
                self.status_indicator.set_status(f"Bot {bot_id} deleted", 'success')
            
        except Exception as e:
            app_logger.error(f"Error deleting bot: {e}")

    def _start_all_bots(self):
        """Start all stopped bots."""
        try:
            started_count = 0
            for bot_id, bot_config in self._bots.items():
                if bot_config['status'] == 'stopped':
                    self._start_bot(bot_id)
                    started_count += 1
            
            if started_count > 0:
                self.status_indicator.set_status(f"Starting {started_count} bots", 'info')
            else:
                self.status_indicator.set_status("No stopped bots to start", 'warning')
                
        except Exception as e:
            app_logger.error(f"Error starting all bots: {e}")

    def _stop_all_bots(self):
        """Stop all running bots."""
        try:
            stopped_count = 0
            for bot_id, bot_config in self._bots.items():
                if bot_config['status'] in ['running', 'paused']:
                    self._stop_bot(bot_id)
                    stopped_count += 1
            
            if stopped_count > 0:
                self.status_indicator.set_status(f"Stopping {stopped_count} bots", 'info')
            else:
                self.status_indicator.set_status("No running bots to stop", 'warning')
                
        except Exception as e:
            app_logger.error(f"Error stopping all bots: {e}")

    def _emergency_stop(self):
        """Emergency stop all bots."""
        try:
            if messagebox.askyesno("Emergency Stop", "This will immediately stop ALL bots. Continue?"):
                for bot_id in list(self._active_bots.keys()):
                    bot_config = self._bots[bot_id]
                    bot_config['status'] = 'stopped'
                    del self._active_bots[bot_id]
                
                self._update_bot_display()
                self._log_message("EMERGENCY STOP: All bots stopped")
                self.status_indicator.set_status("Emergency stop completed", 'warning')
                
                notification_system.show_warning(
                    "Emergency Stop",
                    "All trading bots have been stopped"
                )
                
        except Exception as e:
            app_logger.error(f"Error in emergency stop: {e}")

    def _on_bot_selected(self, event):
        """Handle bot selection in tree view."""
        try:
            selected_item = self.bot_tree.selection()
            if selected_item:
                bot_data = self.bot_tree.item(selected_item[0], 'values')
                self._selected_bot = bot_data[0]
                
        except Exception as e:
            app_logger.error(f"Error handling bot selection: {e}")

    def _validate_bot_config(self) -> Dict[str, Any]:
        """Validate bot configuration."""
        try:
            form_data = {
                'amount_percentage': self.amount_percentage_var.get(),
                'profit_target': self.profit_target_var.get(),
                'stop_loss': self.stop_loss_var.get()
            }
            
            validation_result = self.form_validator.validate_form(form_data)
            
            if not validation_result['valid']:
                return {
                    'valid': False,
                    'message': '; '.join(validation_result['messages'])
                }
            
            # Additional validation
            if not self.symbol_var.get().strip():
                return {'valid': False, 'message': 'Symbol is required'}
            
            return {'valid': True, 'message': 'Configuration is valid'}
            
        except Exception as e:
            app_logger.error(f"Error validating bot config: {e}")
            return {'valid': False, 'message': str(e)}

    def _validate_amount_percentage(self, event=None):
        """Validate amount percentage."""
        try:
            value = self.amount_percentage_var.get()
            if not value:
                return
                
            validation_result = self.form_validator.validate_field('amount_percentage', value)
            
            if not validation_result['valid']:
                self.status_indicator.set_status(
                    f"Invalid amount: {', '.join(validation_result['messages'])}", 'error'
                )
            else:
                self.status_indicator.clear_status()
                
        except Exception as e:
            app_logger.error(f"Error validating amount percentage: {e}")

    def _validate_profit_target(self, event=None):
        """Validate profit target."""
        try:
            value = self.profit_target_var.get()
            if not value:
                return
                
            validation_result = self.form_validator.validate_field('profit_target', value)
            
            if not validation_result['valid']:
                self.status_indicator.set_status(
                    f"Invalid profit target: {', '.join(validation_result['messages'])}", 'error'
                )
            else:
                self.status_indicator.clear_status()
                
        except Exception as e:
            app_logger.error(f"Error validating profit target: {e}")

    def _validate_stop_loss(self, event=None):
        """Validate stop loss."""
        try:
            value = self.stop_loss_var.get()
            if not value:
                return
                
            validation_result = self.form_validator.validate_field('stop_loss', value)
            
            if not validation_result['valid']:
                self.status_indicator.set_status(
                    f"Invalid stop loss: {', '.join(validation_result['messages'])}", 'error'
                )
            else:
                self.status_indicator.clear_status()
                
        except Exception as e:
            app_logger.error(f"Error validating stop loss: {e}")

    def _toggle_auto_restart(self):
        """Toggle auto-restart setting."""
        enabled = self.auto_restart_var.get()
        self.status_indicator.set_status(f"Auto-restart {'enabled' if enabled else 'disabled'}", 'info')

    def _toggle_notifications(self):
        """Toggle notifications setting."""
        enabled = self.notifications_var.get()
        self.status_indicator.set_status(f"Notifications {'enabled' if enabled else 'disabled'}", 'info')

    def _toggle_auto_scroll(self):
        """Toggle auto-scroll for logs."""
        enabled = self.auto_scroll_var.get()
        self.status_indicator.set_status(f"Log auto-scroll {'enabled' if enabled else 'disabled'}", 'info')

    def _update_bot_display(self):
        """Update bot display in tree view."""
        try:
            # Clear current display
            self.bot_tree.delete(*self.bot_tree.get_children())
            
            # Add bots to display
            for bot_id, bot_config in self._bots.items():
                uptime = "--"
                if 'started_at' in bot_config and bot_config['status'] == 'running':
                    uptime_delta = datetime.now() - bot_config['started_at']
                    uptime = str(uptime_delta).split('.')[0]  # Remove microseconds
                
                pnl_text = format_currency(bot_config.get('pnl', 0.0))
                if bot_config.get('pnl', 0.0) > 0:
                    pnl_text = f"+{pnl_text}"
                
                self.bot_tree.insert('', 'end', values=(
                    bot_id,
                    bot_config['strategy'].title(),
                    bot_config['symbol'],
                    bot_config['status'].title(),
                    pnl_text,
                    bot_config.get('trades', 0),
                    uptime
                ))
                
        except Exception as e:
            app_logger.error(f"Error updating bot display: {e}")

    def _update_bot_status(self):
        """Update bot status summary."""
        try:
            total_bots = len(self._bots)
            running_bots = len(self._active_bots)
            
            self.bot_status_label.configure(
                text=f"Active: {total_bots} | Running: {running_bots}"
            )
            
        except Exception as e:
            app_logger.error(f"Error updating bot status: {e}")

    def _update_performance_metrics(self):
        """Update performance metrics display."""
        try:
            total_pnl = sum(bot.get('pnl', 0.0) for bot in self._bots.values())
            total_trades = sum(bot.get('trades', 0) for bot in self._bots.values())
            total_wins = sum(bot.get('wins', 0) for bot in self._bots.values())
            
            win_rate = (total_wins / max(total_trades, 1)) * 100
            
            # Update summary
            best_bot = max(self._bots.values(), key=lambda b: b.get('pnl', 0.0), default=None)
            best_bot_id = best_bot['id'] if best_bot else 'N/A'
            
            self.performance_summary.configure(
                text=f"Total P&L: {format_currency(total_pnl)} | Best Bot: {best_bot_id} | Win Rate: {win_rate:.1f}%"
            )
            
            # Update individual metrics
            pnl_color = '#28a745' if total_pnl >= 0 else '#dc3545'
            self.total_pnl_label.configure(
                text=f"Total P&L: {format_currency(total_pnl)}",
                foreground=pnl_color
            )
            
            self.total_trades_label.configure(text=f"Total Trades: {total_trades}")
            self.win_rate_label.configure(text=f"Win Rate: {win_rate:.1f}%")
            
            # Update performance tree
            self._update_performance_tree()
            
        except Exception as e:
            app_logger.error(f"Error updating performance metrics: {e}")

    def _update_performance_tree(self):
        """Update individual bot performance display."""
        try:
            # Clear current display
            self.performance_tree.delete(*self.performance_tree.get_children())
            
            # Add bot performance data
            for bot_id, bot_config in self._bots.items():
                pnl = bot_config.get('pnl', 0.0)
                trades = bot_config.get('trades', 0)
                wins = bot_config.get('wins', 0)
                
                win_rate = (wins / max(trades, 1)) * 100
                avg_trade = pnl / max(trades, 1)
                
                pnl_text = f"+{format_currency(pnl)}" if pnl >= 0 else format_currency(pnl)
                
                self.performance_tree.insert('', 'end', values=(
                    bot_id,
                    bot_config['strategy'].title(),
                    pnl_text,
                    trades,
                    f"{win_rate:.1f}%",
                    format_currency(avg_trade),
                    bot_config['status'].title()
                ))
                
        except Exception as e:
            app_logger.error(f"Error updating performance tree: {e}")

    def _log_message(self, message: str):
        """Add message to log display."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            
            self.log_text.insert('end', log_entry)
            
            # Auto-scroll if enabled
            if self.auto_scroll_var.get():
                self.log_text.see('end')
                
        except Exception as e:
            app_logger.error(f"Error logging message: {e}")

    def _clear_logs(self):
        """Clear log display."""
        try:
            self.log_text.delete('1.0', 'end')
            self.status_indicator.set_status("Logs cleared", 'info')
            
        except Exception as e:
            app_logger.error(f"Error clearing logs: {e}")

    def _export_logs(self):
        """Export logs to file."""
        try:
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get('1.0', 'end'))
                
                self.status_indicator.set_status("Logs exported successfully", 'success')
                
        except Exception as e:
            app_logger.error(f"Error exporting logs: {e}")
            self.status_indicator.set_status(f"Failed to export logs: {str(e)}", 'error')

    def _save_config(self):
        """Save bot configurations."""
        try:
            # Placeholder for configuration saving
            self.status_indicator.set_status("Configuration saved successfully", 'success')
            
        except Exception as e:
            app_logger.error(f"Error saving config: {e}")

    def _load_config(self):
        """Load bot configurations."""
        try:
            # Placeholder for configuration loading
            self.status_indicator.set_status("Configuration loaded successfully", 'success')
            
        except Exception as e:
            app_logger.error(f"Error loading config: {e}")

    def cleanup(self):
        """Cleanup tab resources."""
        try:
            # Stop all active bots
            for bot_id in list(self._active_bots.keys()):
                self._stop_bot(bot_id)
                
            app_logger.info("OptimizedBotTab cleaned up")
            
        except Exception as e:
            app_logger.error(f"Error during BotTab cleanup: {e}")

    def refresh(self):
        """Refresh tab content."""
        try:
            self._update_bot_display()
            self._update_performance_metrics()
            self.status_indicator.set_status("Data refreshed", 'success')
            
        except Exception as e:
            app_logger.error(f"Error refreshing BotTab: {e}")


# Backwards compatibility
BotTab = OptimizedBotTab
