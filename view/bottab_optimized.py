"""
Advanced Bot Tab

Enhanced bot management system with:
- Integration with ML and RL trained models
- Advanced bot configuration and management
- Real-time performance monitoring
- Multi-strategy bot deployment
- Portfolio allocation and risk management
"""

import threading
import time
from datetime import datetime, timedelta
from tkinter import messagebox, filedialog
from typing import Any, Dict, List, Optional

try:
    from ttkbootstrap import (
        Button, Entry, Frame, Label, OptionMenu, Scale, StringVar, 
        BooleanVar, IntVar, DoubleVar, Notebook, Progressbar, Separator, Treeview,
        Combobox
    )
    from ttkbootstrap.constants import *
    HAS_TTKBOOTSTRAP = True
except ImportError:
    from tkinter import (
        Button, Entry, Frame, Label, OptionMenu, Scale, StringVar,
        BooleanVar, IntVar, DoubleVar
    )
    from tkinter.ttk import Notebook, Progressbar, Separator, Treeview, Combobox
    HAS_TTKBOOTSTRAP = False

# Import the new bot system
from model.bot_system import BotManager, BotType, BotStatus, RiskLevel, bot_manager

from view.utils import (
    ValidationMixin, StatusIndicator, LoadingIndicator, FormValidator,
    format_currency, format_percentage, notification_system
)

import util.loggers as loggers

logger_dict = loggers.setup_loggers()
app_logger = logger_dict['app']


class AdvancedBotTab(Frame, ValidationMixin):
    """
    Advanced bot management interface with ML/RL integration.
    """
    
    def __init__(self, parent, presenter):
        super().__init__(parent)
        self._parent = parent
        self._presenter = presenter
        
        # Bot system integration
        self.bot_manager = bot_manager
        
        # UI state
        self._is_loading = False
        self._last_update = None
        self._selected_bot_id = None
        self._update_thread = None
        self._stop_updates = threading.Event()
        
        # Form variables
        self.bot_name_var = StringVar()
        self.bot_type_var = StringVar(value=BotType.ML_BASED.value)
        self.symbol_var = StringVar(value="BTC/USD:USD")
        self.risk_level_var = StringVar(value=RiskLevel.MODERATE.value)
        self.ml_model_path_var = StringVar()
        self.rl_model_path_var = StringVar()
        self.min_confidence_var = DoubleVar(value=0.6)
        self.sleep_interval_var = IntVar(value=60)
        
        # Form validator
        self.form_validator = FormValidator()
        self._setup_validation_rules()
        
        # Initialize GUI
        self._create_widgets()
        self._setup_layout()
        self._bind_events()
        
        # Start data updates
        self._start_updates()
        
        app_logger.info("AdvancedBotTab initialized")

    def _setup_validation_rules(self):
        """Setup form validation rules."""
        try:
            self.form_validator.add_field_rule('min_confidence', 'numeric_range', min=0.1, max=1.0)
            self.form_validator.add_field_rule('sleep_interval', 'numeric_range', min=10, max=3600)
            
        except Exception as e:
            app_logger.error(f"Error setting up validation rules: {e}")

    def _create_widgets(self):
        """Create all GUI widgets."""
        try:
            # Create main sections
            self._create_header_section()
            self._create_bot_creation_section()
            self._create_bot_management_section()
            self._create_performance_section()
            self._create_model_management_section()
            
        except Exception as e:
            app_logger.error(f"Error creating widgets: {e}")

    def _create_header_section(self):
        """Create header with system status."""
        header_frame = Frame(self)
        header_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        header_frame.grid_columnconfigure(2, weight=1)
        
        # Title
        Label(header_frame, text="🤖 Advanced Trading Bot System", 
              font=('Arial', 16, 'bold')).grid(row=0, column=0, padx=5)
        
        # System status
        self.system_status_label = Label(
            header_frame, text="Total: 0 | Running: 0 | Profit: $0.00", 
            font=('Arial', 11, 'bold'), foreground='#17a2b8'
        )
        self.system_status_label.grid(row=0, column=1, padx=20)
        
        # Quick controls
        control_frame = Frame(header_frame)
        control_frame.grid(row=0, column=3, padx=5)
        
        Button(
            control_frame, text="⚡ Start All", command=self._start_all_bots,
            style='success.TButton', width=12
        ).grid(row=0, column=0, padx=2)
        
        Button(
            control_frame, text="⏹ Stop All", command=self._stop_all_bots,
            style='danger.TButton', width=12
        ).grid(row=0, column=1, padx=2)

    def _create_bot_creation_section(self):
        """Create bot creation and configuration section."""
        creation_frame = Frame(self)
        creation_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        creation_frame.grid_columnconfigure(1, weight=1)
        
        Label(creation_frame, text="🚀 Create New Bot", 
              font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Bot configuration form
        config_notebook = Notebook(creation_frame)
        config_notebook.grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)
        
        # Basic configuration tab
        basic_frame = Frame(config_notebook)
        config_notebook.add(basic_frame, text="Basic Config")
        
        # Bot name
        Label(basic_frame, text="Bot Name:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        Entry(basic_frame, textvariable=self.bot_name_var, width=20).grid(
            row=0, column=1, sticky='ew', padx=5, pady=2)
        
        # Bot type
        Label(basic_frame, text="Bot Type:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        bot_type_combo = Combobox(basic_frame, textvariable=self.bot_type_var, width=18)
        bot_type_combo['values'] = [bt.value for bt in BotType]
        bot_type_combo.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        
        # Symbol
        Label(basic_frame, text="Symbol:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        symbol_combo = Combobox(basic_frame, textvariable=self.symbol_var, width=18)
        symbol_combo['values'] = ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD", "DOGE/USD:USD"]
        symbol_combo.grid(row=2, column=1, sticky='ew', padx=5, pady=2)
        
        # Risk level
        Label(basic_frame, text="Risk Level:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        risk_combo = Combobox(basic_frame, textvariable=self.risk_level_var, width=18)
        risk_combo['values'] = [rl.value for rl in RiskLevel]
        risk_combo.grid(row=3, column=1, sticky='ew', padx=5, pady=2)
        
        basic_frame.grid_columnconfigure(1, weight=1)
        
        # AI Models configuration tab
        models_frame = Frame(config_notebook)
        config_notebook.add(models_frame, text="AI Models")
        
        # ML Model
        Label(models_frame, text="ML Model Path:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ml_path_frame = Frame(models_frame)
        ml_path_frame.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        ml_path_frame.grid_columnconfigure(0, weight=1)
        
        Entry(ml_path_frame, textvariable=self.ml_model_path_var).grid(
            row=0, column=0, sticky='ew', padx=(0, 5))
        Button(ml_path_frame, text="📁", command=self._browse_ml_model, width=3).grid(
            row=0, column=1)
        
        # RL Model
        Label(models_frame, text="RL Model Path:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        rl_path_frame = Frame(models_frame)
        rl_path_frame.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        rl_path_frame.grid_columnconfigure(0, weight=1)
        
        Entry(rl_path_frame, textvariable=self.rl_model_path_var).grid(
            row=0, column=0, sticky='ew', padx=(0, 5))
        Button(rl_path_frame, text="📁", command=self._browse_rl_model, width=3).grid(
            row=0, column=1)
        
        models_frame.grid_columnconfigure(1, weight=1)
        
        # Advanced configuration tab
        advanced_frame = Frame(config_notebook)
        config_notebook.add(advanced_frame, text="Advanced")
        
        # Min confidence
        Label(advanced_frame, text="Min Confidence:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        confidence_frame = Frame(advanced_frame)
        confidence_frame.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        
        Scale(confidence_frame, from_=0.1, to=1.0, orient='horizontal',
              variable=self.min_confidence_var, length=150).grid(row=0, column=0)
        Label(confidence_frame, textvariable=self.min_confidence_var).grid(row=0, column=1, padx=5)
        
        # Sleep interval
        Label(advanced_frame, text="Update Interval (s):").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        interval_frame = Frame(advanced_frame)
        interval_frame.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        
        Scale(interval_frame, from_=10, to=300, orient='horizontal',
              variable=self.sleep_interval_var, length=150).grid(row=0, column=0)
        Label(interval_frame, textvariable=self.sleep_interval_var).grid(row=0, column=1, padx=5)
        
        # Create bot button
        Button(
            creation_frame, text="🤖 Create Bot", command=self._create_bot,
            style='success.TButton', width=20
        ).grid(row=2, column=0, columnspan=2, pady=10)

    def _create_bot_management_section(self):
        """Create bot management interface."""
        management_frame = Frame(self)
        management_frame.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        management_frame.grid_rowconfigure(1, weight=1)
        
        Label(management_frame, text="🛠 Bot Management", 
              font=('Arial', 12, 'bold')).grid(row=0, column=0, pady=(0, 10))
        
        # Bot list
        self.bot_tree = Treeview(
            management_frame,
            columns=('ID', 'Name', 'Type', 'Symbol', 'Status', 'Trades', 'Profit'),
            show='headings', height=10
        )
        
        # Configure columns
        columns_config = {
            'ID': 80,
            'Name': 120,
            'Type': 100,
            'Symbol': 100,
            'Status': 80,
            'Trades': 60,
            'Profit': 80
        }
        
        for col, width in columns_config.items():
            self.bot_tree.heading(col, text=col)
            self.bot_tree.column(col, width=width)
        
        self.bot_tree.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        
        # Bot controls
        controls_frame = Frame(management_frame)
        controls_frame.grid(row=2, column=0, sticky='ew', padx=5, pady=5)
        
        Button(controls_frame, text="▶️ Start", command=self._start_selected_bot,
               style='success.TButton', width=8).grid(row=0, column=0, padx=2)
        
        Button(controls_frame, text="⏸ Pause", command=self._pause_selected_bot,
               style='warning.TButton', width=8).grid(row=0, column=1, padx=2)
        
        Button(controls_frame, text="⏹ Stop", command=self._stop_selected_bot,
               style='danger.TButton', width=8).grid(row=0, column=2, padx=2)
        
        Button(controls_frame, text="🗑 Delete", command=self._delete_selected_bot,
               style='danger.TButton', width=8).grid(row=0, column=3, padx=2)

    def _create_performance_section(self):
        """Create performance monitoring section."""
        perf_frame = Frame(self)
        perf_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        Label(perf_frame, text="📊 Performance Overview", 
              font=('Arial', 12, 'bold')).grid(row=0, column=0, pady=(0, 10))
        
        # Performance metrics
        metrics_frame = Frame(perf_frame)
        metrics_frame.grid(row=1, column=0, sticky='ew', padx=5)
        metrics_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Total profit
        profit_frame = Frame(metrics_frame, relief='ridge', borderwidth=1)
        profit_frame.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        Label(profit_frame, text="Total Profit", font=('Arial', 10, 'bold')).pack()
        self.total_profit_label = Label(profit_frame, text="$0.00", 
                                       font=('Arial', 12), foreground='#28a745')
        self.total_profit_label.pack()
        
        # Total trades
        trades_frame = Frame(metrics_frame, relief='ridge', borderwidth=1)
        trades_frame.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        Label(trades_frame, text="Total Trades", font=('Arial', 10, 'bold')).pack()
        self.total_trades_label = Label(trades_frame, text="0", font=('Arial', 12))
        self.total_trades_label.pack()
        
        # Active bots
        active_frame = Frame(metrics_frame, relief='ridge', borderwidth=1)
        active_frame.grid(row=0, column=2, padx=5, pady=5, sticky='ew')
        Label(active_frame, text="Active Bots", font=('Arial', 10, 'bold')).pack()
        self.active_bots_label = Label(active_frame, text="0", font=('Arial', 12))
        self.active_bots_label.pack()
        
        # System uptime
        uptime_frame = Frame(metrics_frame, relief='ridge', borderwidth=1)
        uptime_frame.grid(row=0, column=3, padx=5, pady=5, sticky='ew')
        Label(uptime_frame, text="System Uptime", font=('Arial', 10, 'bold')).pack()
        self.uptime_label = Label(uptime_frame, text="0h 0m", font=('Arial', 12))
        self.uptime_label.pack()

    def _create_model_management_section(self):
        """Create AI model management section."""
        model_frame = Frame(self)
        model_frame.grid(row=3, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        Label(model_frame, text="🧠 AI Model Management", 
              font=('Arial', 12, 'bold')).grid(row=0, column=0, pady=(0, 10))
        
        # Model loading controls
        load_frame = Frame(model_frame)
        load_frame.grid(row=1, column=0, sticky='ew', padx=5)
        
        Button(load_frame, text="📥 Load ML Models", command=self._load_ml_models,
               style='info.TButton', width=15).grid(row=0, column=0, padx=5)
        
        Button(load_frame, text="📥 Load RL Models", command=self._load_rl_models,
               style='info.TButton', width=15).grid(row=0, column=1, padx=5)
        
        Button(load_frame, text="🔄 Reload All", command=self._reload_all_models,
               style='secondary.TButton', width=15).grid(row=0, column=2, padx=5)

    def _setup_layout(self):
        """Setup the layout configuration."""
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def _bind_events(self):
        """Bind event handlers."""
        try:
            self.bot_tree.bind('<<TreeviewSelect>>', self._on_bot_select)
            
        except Exception as e:
            app_logger.error(f"Error binding events: {e}")

    def _start_updates(self):
        """Start the update thread."""
        try:
            self._stop_updates.clear()
            self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self._update_thread.start()
            
        except Exception as e:
            app_logger.error(f"Error starting updates: {e}")

    def _update_loop(self):
        """Main update loop for refreshing bot data."""
        while not self._stop_updates.is_set():
            try:
                self._update_bot_list()
                self._update_performance_metrics()
                self._update_system_status()
                
                # Update every 2 seconds
                time.sleep(2)
                
            except Exception as e:
                app_logger.error(f"Error in update loop: {e}")
                time.sleep(5)

    def _browse_ml_model(self):
        """Browse for ML model file."""
        try:
            filename = filedialog.askopenfilename(
                title="Select ML Model",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if filename:
                self.ml_model_path_var.set(filename)
                
        except Exception as e:
            app_logger.error(f"Error browsing ML model: {e}")

    def _browse_rl_model(self):
        """Browse for RL model file."""
        try:
            filename = filedialog.askopenfilename(
                title="Select RL Model",
                filetypes=[("Model files", "*.pth *.h5 *.pkl"), ("All files", "*.*")]
            )
            if filename:
                self.rl_model_path_var.set(filename)
                
        except Exception as e:
            app_logger.error(f"Error browsing RL model: {e}")

    def _create_bot(self):
        """Create a new trading bot."""
        try:
            # Validate form
            if not self.bot_name_var.get().strip():
                messagebox.showerror("Error", "Bot name is required")
                return
            
            # Get exchange from presenter (safely)
            exchange = None
            try:
                if hasattr(self._presenter, 'get_exchange'):
                    exchange = self._presenter.get_exchange()
            except Exception as e:
                app_logger.warning(f"Could not get exchange: {e}")
            
            # Prepare bot configuration
            config = {
                'risk_level': RiskLevel(self.risk_level_var.get()),
                'min_confidence': self.min_confidence_var.get(),
                'sleep_interval': self.sleep_interval_var.get(),
                'ml_model_path': self.ml_model_path_var.get() if self.ml_model_path_var.get() else None,
                'rl_model_path': self.rl_model_path_var.get() if self.rl_model_path_var.get() else None
            }
            
            # Create bot
            bot_id = self.bot_manager.create_bot(
                name=self.bot_name_var.get().strip(),
                bot_type=BotType(self.bot_type_var.get()),
                symbol=self.symbol_var.get(),
                exchange=exchange,
                config=config
            )
            
            if bot_id:
                # Load models if specified
                bot = self.bot_manager.get_bot(bot_id)
                if bot:
                    if config['ml_model_path']:
                        bot.load_ml_model(config['ml_model_path'])
                    if config['rl_model_path']:
                        bot.load_rl_model(config['rl_model_path'])
                
                messagebox.showinfo("Success", f"Bot '{self.bot_name_var.get()}' created successfully!")
                self._clear_form()
                
            else:
                messagebox.showerror("Error", "Failed to create bot")
                
        except Exception as e:
            app_logger.error(f"Error creating bot: {e}")
            messagebox.showerror("Error", f"Failed to create bot: {str(e)}")

    def _clear_form(self):
        """Clear the bot creation form."""
        self.bot_name_var.set("")
        self.ml_model_path_var.set("")
        self.rl_model_path_var.set("")
        self.min_confidence_var.set(0.6)
        self.sleep_interval_var.set(60)

    def _on_bot_select(self, event):
        """Handle bot selection in the tree."""
        try:
            selection = self.bot_tree.selection()
            if selection:
                item = self.bot_tree.item(selection[0])
                self._selected_bot_id = item['values'][0]  # Bot ID is first column
                
        except Exception as e:
            app_logger.error(f"Error handling bot selection: {e}")

    def _start_selected_bot(self):
        """Start the selected bot."""
        if self._selected_bot_id:
            try:
                success = self.bot_manager.start_bot(self._selected_bot_id)
                if success:
                    messagebox.showinfo("Success", "Bot started successfully")
                else:
                    messagebox.showerror("Error", "Failed to start bot")
            except Exception as e:
                app_logger.error(f"Error starting bot: {e}")
                messagebox.showerror("Error", f"Failed to start bot: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Please select a bot first")

    def _pause_selected_bot(self):
        """Pause the selected bot."""
        if self._selected_bot_id:
            try:
                success = self.bot_manager.pause_bot(self._selected_bot_id)
                if success:
                    messagebox.showinfo("Success", "Bot paused successfully")
                else:
                    messagebox.showerror("Error", "Failed to pause bot")
            except Exception as e:
                app_logger.error(f"Error pausing bot: {e}")
                messagebox.showerror("Error", f"Failed to pause bot: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Please select a bot first")

    def _stop_selected_bot(self):
        """Stop the selected bot."""
        if self._selected_bot_id:
            try:
                success = self.bot_manager.stop_bot(self._selected_bot_id)
                if success:
                    messagebox.showinfo("Success", "Bot stopped successfully")
                else:
                    messagebox.showerror("Error", "Failed to stop bot")
            except Exception as e:
                app_logger.error(f"Error stopping bot: {e}")
                messagebox.showerror("Error", f"Failed to stop bot: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Please select a bot first")

    def _delete_selected_bot(self):
        """Delete the selected bot."""
        if self._selected_bot_id:
            try:
                result = messagebox.askyesno("Confirm", "Are you sure you want to delete this bot?")
                if result:
                    success = self.bot_manager.delete_bot(self._selected_bot_id)
                    if success:
                        messagebox.showinfo("Success", "Bot deleted successfully")
                        self._selected_bot_id = None
                    else:
                        messagebox.showerror("Error", "Failed to delete bot")
            except Exception as e:
                app_logger.error(f"Error deleting bot: {e}")
                messagebox.showerror("Error", f"Failed to delete bot: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Please select a bot first")

    def _start_all_bots(self):
        """Start all bots."""
        try:
            bots = self.bot_manager.list_bots()
            started_count = 0
            
            for bot_info in bots:
                if bot_info['status'] != BotStatus.RUNNING.value:
                    success = self.bot_manager.start_bot(bot_info['bot_id'])
                    if success:
                        started_count += 1
            
            messagebox.showinfo("Success", f"Started {started_count} bots")
            
        except Exception as e:
            app_logger.error(f"Error starting all bots: {e}")
            messagebox.showerror("Error", f"Failed to start bots: {str(e)}")

    def _stop_all_bots(self):
        """Stop all bots."""
        try:
            result = messagebox.askyesno("Confirm", "Are you sure you want to stop all bots?")
            if result:
                self.bot_manager.stop_all_bots()
                messagebox.showinfo("Success", "All bots stopped")
                
        except Exception as e:
            app_logger.error(f"Error stopping all bots: {e}")
            messagebox.showerror("Error", f"Failed to stop bots: {str(e)}")

    def _update_bot_list(self):
        """Update the bot list display."""
        try:
            # Clear current items
            for item in self.bot_tree.get_children():
                self.bot_tree.delete(item)
            
            # Add current bots
            bots = self.bot_manager.list_bots()
            for bot_info in bots:
                self.bot_tree.insert('', 'end', values=(
                    bot_info['bot_id'][:8],  # Shortened ID
                    bot_info['name'],
                    bot_info.get('bot_type', 'Unknown'),
                    bot_info.get('symbol', 'N/A'),
                    bot_info['status'],
                    bot_info['trades_executed'],
                    format_currency(bot_info['total_profit'])
                ))
                
        except Exception as e:
            app_logger.error(f"Error updating bot list: {e}")

    def _update_performance_metrics(self):
        """Update performance metrics display."""
        try:
            perf = self.bot_manager.get_system_performance()
            
            self.total_profit_label.config(text=format_currency(perf['total_profit']))
            self.total_trades_label.config(text=str(perf['total_trades']))
            self.active_bots_label.config(text=f"{perf['running_bots']}/{perf['total_bots']}")
            
            # Update profit color
            if perf['total_profit'] > 0:
                self.total_profit_label.config(foreground='#28a745')  # Green
            elif perf['total_profit'] < 0:
                self.total_profit_label.config(foreground='#dc3545')  # Red
            else:
                self.total_profit_label.config(foreground='#6c757d')  # Gray
                
        except Exception as e:
            app_logger.error(f"Error updating performance metrics: {e}")

    def _update_system_status(self):
        """Update system status display."""
        try:
            perf = self.bot_manager.get_system_performance()
            
            status_text = f"Total: {perf['total_bots']} | Running: {perf['running_bots']} | Profit: {format_currency(perf['total_profit'])}"
            self.system_status_label.config(text=status_text)
            
        except Exception as e:
            app_logger.error(f"Error updating system status: {e}")

    def _load_ml_models(self):
        """Load ML models for bots."""
        try:
            # This would implement bulk ML model loading
            messagebox.showinfo("Info", "ML model loading feature coming soon")
            
        except Exception as e:
            app_logger.error(f"Error loading ML models: {e}")
            messagebox.showerror("Error", f"Failed to load ML models: {str(e)}")

    def _load_rl_models(self):
        """Load RL models for bots."""
        try:
            # This would implement bulk RL model loading
            messagebox.showinfo("Info", "RL model loading feature coming soon")
            
        except Exception as e:
            app_logger.error(f"Error loading RL models: {e}")
            messagebox.showerror("Error", f"Failed to load RL models: {str(e)}")

    def _reload_all_models(self):
        """Reload all models for all bots."""
        try:
            messagebox.showinfo("Info", "Model reloading feature coming soon")
            
        except Exception as e:
            app_logger.error(f"Error reloading models: {e}")
            messagebox.showerror("Error", f"Failed to reload models: {str(e)}")

    def cleanup(self):
        """Cleanup resources when tab is destroyed."""
        try:
            self._stop_updates.set()
            if self._update_thread and self._update_thread.is_alive():
                self._update_thread.join(timeout=2)
            
            app_logger.info("AdvancedBotTab cleaned up")
            
        except Exception as e:
            app_logger.error(f"Error during cleanup: {e}")


# Alias for compatibility
OptimizedBotTab = AdvancedBotTab
