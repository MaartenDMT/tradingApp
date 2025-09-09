"""
Trading System Tab

Advanced trading system interface for:
- Strategy management and configuration
- Signal generation and analysis
- Portfolio optimization
- Risk management controls
- Performance analytics
- System monitoring and alerts
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


class TradingSystemTab(Frame, ValidationMixin):
    """
    Advanced trading system interface with strategy management and analytics.
    """
    
    # Trading system constants
    STRATEGIES = ['momentum', 'mean_reversion', 'arbitrage', 'pairs_trading', 'grid_trading']
    TIME_FRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    RISK_LEVELS = ['conservative', 'moderate', 'aggressive']
    
    def __init__(self, parent, presenter):
        super().__init__(parent)
        self._parent = parent
        self._presenter = presenter
        
        # System state
        self._system_running = False
        self._active_strategies = {}
        self._performance_data = {}
        self._signals_history = []
        
        # UI state
        self._is_loading = False
        self._last_update = None
        
        # Form validator
        self.form_validator = FormValidator()
        self._setup_validation_rules()
        
        # Initialize GUI
        self._create_widgets()
        self._setup_layout()
        self._bind_events()
        
        # Start data updates
        self._start_updates()
        
        app_logger.info("TradingSystemTab initialized")

    def _setup_validation_rules(self):
        """Setup form validation rules."""
        try:
            self.form_validator.add_field_rule('max_position', 'numeric_range', min=1, max=100)
            self.form_validator.add_field_rule('stop_loss', 'numeric_range', min=0.1, max=50)
            self.form_validator.add_field_rule('take_profit', 'numeric_range', min=0.1, max=200)
            
        except Exception as e:
            app_logger.error(f"Error setting up validation rules: {e}")

    def _create_widgets(self):
        """Create all GUI widgets."""
        try:
            # Create main sections
            self._create_header_section()
            self._create_strategy_section()
            self._create_signals_section()
            self._create_performance_section()
            self._create_control_section()
            
        except Exception as e:
            app_logger.error(f"Error creating widgets: {e}")

    def _create_header_section(self):
        """Create header with system status and controls."""
        header_frame = Frame(self)
        header_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        header_frame.grid_columnconfigure(2, weight=1)
        
        # Title and status
        Label(header_frame, text="Trading System", font=('Arial', 16, 'bold')).grid(
            row=0, column=0, padx=5
        )
        
        self.system_status_label = Label(
            header_frame, text="System: STOPPED", 
            font=('Arial', 12, 'bold'), foreground='#dc3545'
        )
        self.system_status_label.grid(row=0, column=1, padx=20)
        
        # System controls
        control_frame = Frame(header_frame)
        control_frame.grid(row=0, column=3, padx=5)
        
        self.start_button = Button(
            control_frame, text="Start System", command=self._start_system,
            style='success.TButton', width=12
        )
        self.start_button.grid(row=0, column=0, padx=2)
        
        self.stop_button = Button(
            control_frame, text="Stop System", command=self._stop_system,
            style='danger.TButton', width=12, state='disabled'
        )
        self.stop_button.grid(row=0, column=1, padx=2)
        
        # Performance summary
        self.performance_summary = Label(
            header_frame, text="P&L: $0.00 | Win Rate: 0% | Sharpe: 0.0",
            font=('Arial', 10), foreground='#888888'
        )
        self.performance_summary.grid(row=1, column=0, columnspan=4, pady=5)

    def _create_strategy_section(self):
        """Create strategy configuration section."""
        strategy_frame = Frame(self)
        strategy_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        strategy_frame.grid_rowconfigure(1, weight=1)
        
        Label(strategy_frame, text="Strategy Configuration", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, pady=(0, 10)
        )
        
        # Create notebook for strategy sections
        strategy_notebook = Notebook(strategy_frame)
        strategy_notebook.grid(row=1, column=0, sticky='nsew')
        
        # Strategy Selection Tab
        selection_tab = Frame(strategy_notebook)
        strategy_notebook.add(selection_tab, text="Strategy Selection")
        self._create_strategy_selection(selection_tab)
        
        # Parameters Tab
        params_tab = Frame(strategy_notebook)
        strategy_notebook.add(params_tab, text="Parameters")
        self._create_strategy_parameters(params_tab)
        
        # Risk Management Tab
        risk_tab = Frame(strategy_notebook)
        strategy_notebook.add(risk_tab, text="Risk Management")
        self._create_risk_management(risk_tab)

    def _create_strategy_selection(self, parent):
        """Create strategy selection interface."""
        # Active strategies
        Label(parent, text="Active Strategies:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5, pady=5
        )
        
        self.strategy_vars = {}
        for i, strategy in enumerate(self.STRATEGIES):
            var = BooleanVar()
            self.strategy_vars[strategy] = var
            
            checkbox = Button(
                parent, text=strategy.replace('_', ' ').title(),
                command=lambda s=strategy: self._toggle_strategy(s),
                style='outline.TButton', width=15
            )
            checkbox.grid(row=1 + i//3, column=i%3, padx=5, pady=2, sticky='ew')
        
        # Strategy priority
        Label(parent, text="Priority Order:", font=('Arial', 10, 'bold')).grid(
            row=4, column=0, sticky='w', padx=5, pady=(20, 5)
        )
        
        self.priority_listbox = Treeview(parent, columns=('Strategy', 'Status'), show='headings', height=6)
        self.priority_listbox.heading('Strategy', text='Strategy')
        self.priority_listbox.heading('Status', text='Status')
        self.priority_listbox.grid(row=5, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

    def _create_strategy_parameters(self, parent):
        """Create strategy parameters interface."""
        # Common parameters
        params_frame = Frame(parent)
        params_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        Label(params_frame, text="Lookback Period:", font=('Arial', 10)).grid(
            row=0, column=0, sticky='w', padx=5, pady=5
        )
        
        self.lookback_var = IntVar(value=20)
        self.lookback_scale = Scale(
            params_frame, from_=5, to=200, orient='horizontal',
            variable=self.lookback_var
        )
        self.lookback_scale.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        
        # Time frame
        Label(params_frame, text="Time Frame:", font=('Arial', 10)).grid(
            row=1, column=0, sticky='w', padx=5, pady=5
        )
        
        self.timeframe_var = StringVar(value='1h')
        self.timeframe_select = OptionMenu(
            params_frame, self.timeframe_var, '1h', *self.TIME_FRAMES
        )
        self.timeframe_select.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        
        # Signal threshold
        Label(params_frame, text="Signal Threshold:", font=('Arial', 10)).grid(
            row=2, column=0, sticky='w', padx=5, pady=5
        )
        
        self.threshold_var = StringVar(value='0.75')
        self.threshold_scale = Scale(
            params_frame, from_=0.1, to=1.0, orient='horizontal',
            variable=self.threshold_var
        )
        self.threshold_scale.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        
        params_frame.grid_columnconfigure(1, weight=1)

    def _create_risk_management(self, parent):
        """Create risk management controls."""
        risk_frame = Frame(parent)
        risk_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        # Maximum position size
        Label(risk_frame, text="Max Position Size (%):", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky='w', padx=5, pady=5
        )
        
        self.max_position_var = StringVar(value='10')
        self.max_position_entry = Entry(risk_frame, textvariable=self.max_position_var, width=10)
        self.max_position_entry.grid(row=0, column=1, padx=5, pady=5)
        
        self.max_position_scale = Scale(
            risk_frame, from_=1, to=50, orient='horizontal',
            variable=self.max_position_var
        )
        self.max_position_scale.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5)
        
        # Stop loss
        Label(risk_frame, text="Stop Loss (%):", font=('Arial', 10, 'bold')).grid(
            row=2, column=0, sticky='w', padx=5, pady=5
        )
        
        self.stop_loss_var = StringVar(value='5')
        self.stop_loss_entry = Entry(risk_frame, textvariable=self.stop_loss_var, width=10)
        self.stop_loss_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # Take profit
        Label(risk_frame, text="Take Profit (%):", font=('Arial', 10, 'bold')).grid(
            row=3, column=0, sticky='w', padx=5, pady=5
        )
        
        self.take_profit_var = StringVar(value='15')
        self.take_profit_entry = Entry(risk_frame, textvariable=self.take_profit_var, width=10)
        self.take_profit_entry.grid(row=3, column=1, padx=5, pady=5)
        
        # Risk level
        Label(risk_frame, text="Risk Level:", font=('Arial', 10, 'bold')).grid(
            row=4, column=0, sticky='w', padx=5, pady=5
        )
        
        self.risk_level_var = StringVar(value='moderate')
        self.risk_level_select = OptionMenu(
            risk_frame, self.risk_level_var, 'moderate', *self.RISK_LEVELS
        )
        self.risk_level_select.grid(row=4, column=1, padx=5, pady=5)
        
        risk_frame.grid_columnconfigure(1, weight=1)

    def _create_signals_section(self):
        """Create signals monitoring section."""
        signals_frame = Frame(self)
        signals_frame.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        signals_frame.grid_rowconfigure(1, weight=1)
        
        Label(signals_frame, text="Trading Signals", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, pady=(0, 10)
        )
        
        # Signals display
        self.signals_tree = Treeview(
            signals_frame, 
            columns=('Time', 'Symbol', 'Signal', 'Confidence', 'Status'),
            show='headings', height=12
        )
        
        # Configure columns
        for col in ('Time', 'Symbol', 'Signal', 'Confidence', 'Status'):
            self.signals_tree.heading(col, text=col)
            self.signals_tree.column(col, width=80 if col != 'Symbol' else 120)
        
        self.signals_tree.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        
        # Signal controls
        signal_controls = Frame(signals_frame)
        signal_controls.grid(row=2, column=0, sticky='ew', padx=5, pady=5)
        
        Button(
            signal_controls, text="Generate Signals", command=self._generate_signals,
            style='info.TButton', width=15
        ).grid(row=0, column=0, padx=5)
        
        Button(
            signal_controls, text="Clear History", command=self._clear_signals,
            style='secondary.TButton', width=15
        ).grid(row=0, column=1, padx=5)

    def _create_performance_section(self):
        """Create performance analytics section."""
        perf_frame = Frame(self)
        perf_frame.grid(row=1, column=2, sticky='nsew', padx=5, pady=5)
        perf_frame.grid_rowconfigure(2, weight=1)
        
        Label(perf_frame, text="Performance Analytics", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, pady=(0, 10)
        )
        
        # Performance metrics
        metrics_frame = Frame(perf_frame)
        metrics_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        
        # Key metrics
        self.total_pnl_label = Label(metrics_frame, text="Total P&L: $0.00", font=('Arial', 10, 'bold'))
        self.total_pnl_label.grid(row=0, column=0, sticky='w', pady=2)
        
        self.win_rate_label = Label(metrics_frame, text="Win Rate: 0%", font=('Arial', 10))
        self.win_rate_label.grid(row=1, column=0, sticky='w', pady=2)
        
        self.sharpe_ratio_label = Label(metrics_frame, text="Sharpe Ratio: 0.0", font=('Arial', 10))
        self.sharpe_ratio_label.grid(row=2, column=0, sticky='w', pady=2)
        
        self.max_drawdown_label = Label(metrics_frame, text="Max Drawdown: 0%", font=('Arial', 10))
        self.max_drawdown_label.grid(row=3, column=0, sticky='w', pady=2)
        
        # Strategy performance
        Separator(perf_frame, orient='horizontal').grid(row=3, column=0, sticky='ew', pady=10)
        
        Label(perf_frame, text="Strategy Performance", font=('Arial', 10, 'bold')).grid(
            row=4, column=0, sticky='w', pady=(0, 5)
        )
        
        self.strategy_perf_tree = Treeview(
            perf_frame,
            columns=('Strategy', 'Trades', 'Win%', 'P&L'),
            show='headings', height=8
        )
        
        for col in ('Strategy', 'Trades', 'Win%', 'P&L'):
            self.strategy_perf_tree.heading(col, text=col)
            self.strategy_perf_tree.column(col, width=80)
        
        self.strategy_perf_tree.grid(row=5, column=0, sticky='nsew', padx=5, pady=5)

    def _create_control_section(self):
        """Create system control and status section."""
        control_frame = Frame(self)
        control_frame.grid(row=2, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        control_frame.grid_columnconfigure(1, weight=1)
        
        # Control buttons
        buttons_frame = Frame(control_frame)
        buttons_frame.grid(row=0, column=0, padx=5)
        
        Button(
            buttons_frame, text="Save Config", command=self._save_configuration,
            style='success.TButton', width=12
        ).grid(row=0, column=0, padx=2)
        
        Button(
            buttons_frame, text="Load Config", command=self._load_configuration,
            style='info.TButton', width=12
        ).grid(row=0, column=1, padx=2)
        
        Button(
            buttons_frame, text="Reset System", command=self._reset_system,
            style='warning.TButton', width=12
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

    def _bind_events(self):
        """Bind event handlers."""
        # Bind validation events
        self.max_position_entry.bind('<FocusOut>', self._validate_max_position)
        self.stop_loss_entry.bind('<FocusOut>', self._validate_stop_loss)
        self.take_profit_entry.bind('<FocusOut>', self._validate_take_profit)

    def _start_updates(self):
        """Start periodic data updates."""
        try:
            self._update_performance_data()
            self._update_signals_display()
            
            # Schedule next update
            self.after(10000, self._start_updates)  # Update every 10 seconds
            
        except Exception as e:
            app_logger.error(f"Error in periodic updates: {e}")

    def _toggle_strategy(self, strategy: str):
        """Toggle strategy activation."""
        try:
            if strategy in self._active_strategies:
                # Deactivate strategy
                del self._active_strategies[strategy]
                self.status_indicator.set_status(f"Deactivated {strategy}", 'warning')
            else:
                # Activate strategy
                self._active_strategies[strategy] = {
                    'status': 'active',
                    'trades': 0,
                    'pnl': 0.0,
                    'activated_at': datetime.now()
                }
                self.status_indicator.set_status(f"Activated {strategy}", 'success')
            
            self._update_strategy_display()
            
        except Exception as e:
            app_logger.error(f"Error toggling strategy {strategy}: {e}")

    def _start_system(self):
        """Start the trading system."""
        try:
            if not self._active_strategies:
                self.status_indicator.set_status("No strategies selected", 'error')
                return
            
            self.loading_indicator.show_loading("Starting trading system...")
            
            # Validate configuration
            validation_result = self._validate_system_config()
            if not validation_result['valid']:
                self.status_indicator.set_status(f"Configuration error: {validation_result['message']}", 'error')
                return
            
            # Start system (placeholder for actual implementation)
            self._system_running = True
            
            # Update UI
            self.start_button.configure(state='disabled')
            self.stop_button.configure(state='normal')
            self.system_status_label.configure(
                text="System: RUNNING", 
                foreground='#28a745'
            )
            
            self.status_indicator.set_status("Trading system started successfully", 'success')
            
            notification_system.show_success(
                "Trading System",
                "Trading system started successfully"
            )
            
        except Exception as e:
            app_logger.error(f"Error starting trading system: {e}")
            self.status_indicator.set_status(f"Failed to start system: {str(e)}", 'error')
        finally:
            self.loading_indicator.hide_loading()

    def _stop_system(self):
        """Stop the trading system."""
        try:
            self.loading_indicator.show_loading("Stopping trading system...")
            
            # Stop system (placeholder for actual implementation)
            self._system_running = False
            
            # Update UI
            self.start_button.configure(state='normal')
            self.stop_button.configure(state='disabled')
            self.system_status_label.configure(
                text="System: STOPPED",
                foreground='#dc3545'
            )
            
            self.status_indicator.set_status("Trading system stopped", 'warning')
            
        except Exception as e:
            app_logger.error(f"Error stopping trading system: {e}")
        finally:
            self.loading_indicator.hide_loading()

    def _validate_system_config(self) -> Dict[str, Any]:
        """Validate system configuration."""
        try:
            # Check required fields
            form_data = {
                'max_position': self.max_position_var.get(),
                'stop_loss': self.stop_loss_var.get(),
                'take_profit': self.take_profit_var.get()
            }
            
            validation_result = self.form_validator.validate_form(form_data)
            
            if not validation_result['valid']:
                return {
                    'valid': False,
                    'message': '; '.join(validation_result['messages'])
                }
            
            # Check if strategies are selected
            if not self._active_strategies:
                return {
                    'valid': False,
                    'message': 'At least one strategy must be selected'
                }
            
            return {'valid': True, 'message': 'Configuration is valid'}
            
        except Exception as e:
            app_logger.error(f"Error validating system config: {e}")
            return {'valid': False, 'message': str(e)}

    def _validate_max_position(self, event=None):
        """Validate maximum position size."""
        try:
            value = self.max_position_var.get()
            if not value:
                return
                
            validation_result = self.form_validator.validate_field('max_position', value)
            
            if not validation_result['valid']:
                self.status_indicator.set_status(
                    f"Invalid max position: {', '.join(validation_result['messages'])}", 'error'
                )
            else:
                self.status_indicator.clear_status()
                
        except Exception as e:
            app_logger.error(f"Error validating max position: {e}")

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

    def _validate_take_profit(self, event=None):
        """Validate take profit."""
        try:
            value = self.take_profit_var.get()
            if not value:
                return
                
            validation_result = self.form_validator.validate_field('take_profit', value)
            
            if not validation_result['valid']:
                self.status_indicator.set_status(
                    f"Invalid take profit: {', '.join(validation_result['messages'])}", 'error'
                )
            else:
                self.status_indicator.clear_status()
                
        except Exception as e:
            app_logger.error(f"Error validating take profit: {e}")

    def _generate_signals(self):
        """Generate trading signals."""
        try:
            self.loading_indicator.show_loading("Generating signals...")
            
            # Placeholder signal generation
            signals = [
                {
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'symbol': 'BTC/USD:USD',
                    'signal': 'BUY',
                    'confidence': '0.85',
                    'status': 'NEW'
                },
                {
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'symbol': 'ETH/USD:USD',
                    'signal': 'SELL',
                    'confidence': '0.72',
                    'status': 'NEW'
                }
            ]
            
            # Add to signals history
            for signal in signals:
                self.signals_tree.insert('', 0, values=(
                    signal['time'], signal['symbol'], signal['signal'],
                    signal['confidence'], signal['status']
                ))
            
            # Limit signal history
            children = self.signals_tree.get_children()
            if len(children) > 100:
                for item in children[100:]:
                    self.signals_tree.delete(item)
            
            self.status_indicator.set_status(f"Generated {len(signals)} new signals", 'success')
            
        except Exception as e:
            app_logger.error(f"Error generating signals: {e}")
            self.status_indicator.set_status(f"Signal generation failed: {str(e)}", 'error')
        finally:
            self.loading_indicator.hide_loading()

    def _clear_signals(self):
        """Clear signals history."""
        try:
            self.signals_tree.delete(*self.signals_tree.get_children())
            self.status_indicator.set_status("Signals history cleared", 'info')
            
        except Exception as e:
            app_logger.error(f"Error clearing signals: {e}")

    def _update_strategy_display(self):
        """Update strategy priority display."""
        try:
            # Clear current display
            self.priority_listbox.delete(*self.priority_listbox.get_children())
            
            # Add active strategies
            for strategy, data in self._active_strategies.items():
                self.priority_listbox.insert('', 'end', values=(
                    strategy.replace('_', ' ').title(),
                    data['status'].upper()
                ))
                
        except Exception as e:
            app_logger.error(f"Error updating strategy display: {e}")

    def _update_performance_data(self):
        """Update performance metrics."""
        try:
            # Placeholder performance data
            total_pnl = 1250.75
            win_rate = 65.5
            sharpe_ratio = 1.85
            max_drawdown = 8.2
            
            # Update labels
            pnl_color = '#28a745' if total_pnl >= 0 else '#dc3545'
            self.total_pnl_label.configure(
                text=f"Total P&L: {format_currency(total_pnl)}",
                foreground=pnl_color
            )
            self.win_rate_label.configure(text=f"Win Rate: {win_rate:.1f}%")
            self.sharpe_ratio_label.configure(text=f"Sharpe Ratio: {sharpe_ratio:.2f}")
            self.max_drawdown_label.configure(text=f"Max Drawdown: {max_drawdown:.1f}%")
            
            # Update summary
            self.performance_summary.configure(
                text=f"P&L: {format_currency(total_pnl)} | Win Rate: {win_rate:.1f}% | Sharpe: {sharpe_ratio:.2f}"
            )
            
            # Update strategy performance
            self._update_strategy_performance()
            
        except Exception as e:
            app_logger.error(f"Error updating performance data: {e}")

    def _update_strategy_performance(self):
        """Update individual strategy performance."""
        try:
            # Clear current data
            self.strategy_perf_tree.delete(*self.strategy_perf_tree.get_children())
            
            # Placeholder strategy performance data
            strategy_data = [
                {'name': 'Momentum', 'trades': 45, 'win_rate': 68.9, 'pnl': 850.25},
                {'name': 'Mean Reversion', 'trades': 32, 'win_rate': 62.5, 'pnl': 400.50},
                {'name': 'Arbitrage', 'trades': 78, 'win_rate': 89.7, 'pnl': 125.75}
            ]
            
            for data in strategy_data:
                pnl_text = f"+{format_currency(data['pnl'])}" if data['pnl'] >= 0 else format_currency(data['pnl'])
                self.strategy_perf_tree.insert('', 'end', values=(
                    data['name'],
                    data['trades'],
                    f"{data['win_rate']:.1f}%",
                    pnl_text
                ))
                
        except Exception as e:
            app_logger.error(f"Error updating strategy performance: {e}")

    def _update_signals_display(self):
        """Update signals display with latest data."""
        try:
            # This would update signals from the actual trading system
            self._last_update = datetime.now()
            
        except Exception as e:
            app_logger.error(f"Error updating signals display: {e}")

    def _save_configuration(self):
        """Save current system configuration."""
        try:
            self.loading_indicator.show_loading("Saving configuration...")
            
            config = {
                'active_strategies': list(self._active_strategies.keys()),
                'lookback_period': self.lookback_var.get(),
                'timeframe': self.timeframe_var.get(),
                'signal_threshold': float(self.threshold_var.get()),
                'max_position': float(self.max_position_var.get()),
                'stop_loss': float(self.stop_loss_var.get()),
                'take_profit': float(self.take_profit_var.get()),
                'risk_level': self.risk_level_var.get()
            }
            
            # Placeholder for saving configuration
            app_logger.info(f"Saving configuration: {config}")
            
            self.status_indicator.set_status("Configuration saved successfully", 'success')
            
        except Exception as e:
            app_logger.error(f"Error saving configuration: {e}")
            self.status_indicator.set_status(f"Failed to save configuration: {str(e)}", 'error')
        finally:
            self.loading_indicator.hide_loading()

    def _load_configuration(self):
        """Load system configuration."""
        try:
            self.loading_indicator.show_loading("Loading configuration...")
            
            # Placeholder for loading configuration
            self.status_indicator.set_status("Configuration loaded successfully", 'success')
            
        except Exception as e:
            app_logger.error(f"Error loading configuration: {e}")
            self.status_indicator.set_status(f"Failed to load configuration: {str(e)}", 'error')
        finally:
            self.loading_indicator.hide_loading()

    def _reset_system(self):
        """Reset trading system to default state."""
        try:
            if messagebox.askyesno("Reset System", "Are you sure you want to reset the trading system?"):
                self.loading_indicator.show_loading("Resetting system...")
                
                # Stop system if running
                if self._system_running:
                    self._stop_system()
                
                # Reset state
                self._active_strategies.clear()
                self._performance_data.clear()
                self._signals_history.clear()
                
                # Reset UI
                self._update_strategy_display()
                self.signals_tree.delete(*self.signals_tree.get_children())
                self.strategy_perf_tree.delete(*self.strategy_perf_tree.get_children())
                
                self.status_indicator.set_status("System reset successfully", 'success')
                
        except Exception as e:
            app_logger.error(f"Error resetting system: {e}")
            self.status_indicator.set_status(f"Reset failed: {str(e)}", 'error')
        finally:
            self.loading_indicator.hide_loading()

    def cleanup(self):
        """Cleanup tab resources."""
        try:
            # Stop system if running
            if self._system_running:
                self._stop_system()
                
            app_logger.info("TradingSystemTab cleaned up")
            
        except Exception as e:
            app_logger.error(f"Error during TradingSystemTab cleanup: {e}")

    def refresh(self):
        """Refresh tab content."""
        try:
            self._update_performance_data()
            self._update_signals_display()
            self.status_indicator.set_status("Data refreshed", 'success')
            
        except Exception as e:
            app_logger.error(f"Error refreshing TradingSystemTab: {e}")
