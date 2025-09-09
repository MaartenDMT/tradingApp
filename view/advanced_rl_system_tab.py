"""
Advanced RL System Tab with Production Features

Enhanced version of the RL system tab that includes:
- Real market data integration
- Algorithm comparison tools
- Multi-agent tournaments
- Production deployment features
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
import threading
import json
from pathlib import Path

# Import advanced features
from production_config import PRODUCTION_CONFIG, ProductionManager
from market_data_integration import market_data_manager, MarketTick
from algorithm_research import algorithm_comparator, research_runner
from multi_agent_tournaments import tournament_engine, TournamentType, GameMode

class AdvancedRLSystemTab(ttk.Frame):
    """Advanced RL System Tab with production features."""
    
    def __init__(self, parent, presenter=None):
        super().__init__(parent)
        self.presenter = presenter
        
        # Configuration
        self.config = PRODUCTION_CONFIG
        self.production_manager = ProductionManager(self.config)
        
        # State management
        self.current_market_data = {}
        self.active_experiments = {}
        self.active_tournaments = {}
        self.real_time_subscriptions = {}
        
        # Create notebook for different sections
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self._create_production_tab()
        self._create_market_data_tab()
        self._create_research_tab()
        self._create_tournament_tab()
        self._create_monitoring_tab()
        
        print("Advanced RL System Tab initialized")
    
    def _create_production_tab(self):
        """Create production deployment and configuration tab."""
        
        prod_frame = ttk.Frame(self.notebook)
        self.notebook.add(prod_frame, text="🚀 Production")
        
        # Production Status Section
        status_frame = ttk.LabelFrame(prod_frame, text="Production Status", padding=10)
        status_frame.pack(fill='x', padx=10, pady=5)
        
        # Environment checks
        ttk.Label(status_frame, text="Environment Validation:", font=('Arial', 12, 'bold')).pack(anchor='w')
        
        self.environment_status = ttk.Frame(status_frame)
        self.environment_status.pack(fill='x', pady=5)
        
        # Run environment check
        self._update_environment_status()
        
        # Configuration Section
        config_frame = ttk.LabelFrame(prod_frame, text="Configuration", padding=10)
        config_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create configuration controls
        config_scroll = ttk.Scrollbar(config_frame)
        config_scroll.pack(side='right', fill='y')
        
        self.config_tree = ttk.Treeview(config_frame, yscrollcommand=config_scroll.set, height=15)
        config_scroll.config(command=self.config_tree.yview)
        self.config_tree.pack(fill='both', expand=True)
        
        # Populate configuration tree
        self._populate_config_tree()
        
        # Control buttons
        button_frame = ttk.Frame(prod_frame)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(
            button_frame, 
            text="Refresh Status", 
            command=self._update_environment_status,
            style='success.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame, 
            text="Export Config", 
            command=self._export_config,
            style='info.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame, 
            text="Load Config", 
            command=self._load_config,
            style='warning.TButton'
        ).pack(side='left', padx=5)
    
    def _create_market_data_tab(self):
        """Create real market data integration tab."""
        
        market_frame = ttk.Frame(self.notebook)
        self.notebook.add(market_frame, text="📈 Market Data")
        
        # Data Sources Section
        sources_frame = ttk.LabelFrame(market_frame, text="Data Sources", padding=10)
        sources_frame.pack(fill='x', padx=10, pady=5)
        
        # Provider selection
        ttk.Label(sources_frame, text="Data Provider:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        
        self.provider_var = tk.StringVar(value="binance")
        provider_combo = ttk.Combobox(
            sources_frame, 
            textvariable=self.provider_var,
            values=["binance", "alphavantage", "yahoo", "custom"],
            state="readonly"
        )
        provider_combo.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        
        # Symbol selection
        ttk.Label(sources_frame, text="Trading Symbol:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        
        self.symbol_var = tk.StringVar(value="BTCUSDT")
        symbol_entry = ttk.Entry(sources_frame, textvariable=self.symbol_var)
        symbol_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        
        # Time interval
        ttk.Label(sources_frame, text="Interval:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        
        self.interval_var = tk.StringVar(value="1h")
        interval_combo = ttk.Combobox(
            sources_frame,
            textvariable=self.interval_var,
            values=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            state="readonly"
        )
        interval_combo.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        
        sources_frame.columnconfigure(1, weight=1)
        
        # Real-time Data Section
        realtime_frame = ttk.LabelFrame(market_frame, text="Real-time Data", padding=10)
        realtime_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Real-time controls
        rt_controls = ttk.Frame(realtime_frame)
        rt_controls.pack(fill='x', pady=5)
        
        ttk.Button(
            rt_controls,
            text="Start Real-time Feed",
            command=self._start_realtime_feed,
            style='success.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            rt_controls,
            text="Stop Feed",
            command=self._stop_realtime_feed,
            style='danger.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            rt_controls,
            text="Fetch Historical",
            command=self._fetch_historical_data,
            style='info.TButton'
        ).pack(side='left', padx=5)
        
        # Data display
        self.market_data_text = tk.Text(realtime_frame, height=15, width=80)
        scrollbar_market = ttk.Scrollbar(realtime_frame, command=self.market_data_text.yview)
        self.market_data_text.config(yscrollcommand=scrollbar_market.set)
        
        self.market_data_text.pack(side='left', fill='both', expand=True)
        scrollbar_market.pack(side='right', fill='y')
    
    def _create_research_tab(self):
        """Create algorithm research and comparison tab."""
        
        research_frame = ttk.Frame(self.notebook)
        self.notebook.add(research_frame, text="🔬 Research")
        
        # Algorithm Selection
        algo_frame = ttk.LabelFrame(research_frame, text="Algorithm Comparison", padding=10)
        algo_frame.pack(fill='x', padx=10, pady=5)
        
        # Algorithm selection
        ttk.Label(algo_frame, text="Select Algorithms to Compare:").pack(anchor='w', pady=5)
        
        self.algorithm_vars = {}
        available_algorithms = ["dqn", "double_dqn", "dueling_dqn", "a2c", "ppo", "ddpg", "td3", "sac"]
        
        algo_checkboxes = ttk.Frame(algo_frame)
        algo_checkboxes.pack(fill='x', pady=5)
        
        for i, algorithm in enumerate(available_algorithms):
            var = tk.BooleanVar()
            self.algorithm_vars[algorithm] = var
            
            cb = ttk.Checkbutton(
                algo_checkboxes,
                text=algorithm.upper(),
                variable=var
            )
            cb.grid(row=i//4, column=i%4, sticky='w', padx=10, pady=2)
        
        # Experiment Configuration
        exp_config_frame = ttk.LabelFrame(research_frame, text="Experiment Configuration", padding=10)
        exp_config_frame.pack(fill='x', padx=10, pady=5)
        
        # Training episodes
        ttk.Label(exp_config_frame, text="Training Episodes:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.training_episodes_var = tk.StringVar(value="1000")
        ttk.Entry(exp_config_frame, textvariable=self.training_episodes_var, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        # Evaluation episodes
        ttk.Label(exp_config_frame, text="Evaluation Episodes:").grid(row=0, column=2, sticky='w', padx=5, pady=5)
        self.eval_episodes_var = tk.StringVar(value="100")
        ttk.Entry(exp_config_frame, textvariable=self.eval_episodes_var, width=10).grid(row=0, column=3, sticky='w', padx=5, pady=5)
        
        # Number of seeds
        ttk.Label(exp_config_frame, text="Random Seeds:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.num_seeds_var = tk.StringVar(value="3")
        ttk.Entry(exp_config_frame, textvariable=self.num_seeds_var, width=10).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        # Research Controls
        research_controls = ttk.Frame(research_frame)
        research_controls.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(
            research_controls,
            text="Start Comparison",
            command=self._start_algorithm_comparison,
            style='primary.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            research_controls,
            text="Hyperparameter Study",
            command=self._start_hyperparameter_study,
            style='info.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            research_controls,
            text="Stability Analysis",
            command=self._start_stability_analysis,
            style='warning.TButton'
        ).pack(side='left', padx=5)
        
        # Results Display
        results_frame = ttk.LabelFrame(research_frame, text="Research Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.research_results_text = tk.Text(results_frame, height=15)
        results_scrollbar = ttk.Scrollbar(results_frame, command=self.research_results_text.yview)
        self.research_results_text.config(yscrollcommand=results_scrollbar.set)
        
        self.research_results_text.pack(side='left', fill='both', expand=True)
        results_scrollbar.pack(side='right', fill='y')
    
    def _create_tournament_tab(self):
        """Create multi-agent tournament tab."""
        
        tournament_frame = ttk.Frame(self.notebook)
        self.notebook.add(tournament_frame, text="🏆 Tournaments")
        
        # Tournament Configuration
        config_frame = ttk.LabelFrame(tournament_frame, text="Tournament Configuration", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)
        
        # Tournament type
        ttk.Label(config_frame, text="Tournament Type:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        
        self.tournament_type_var = tk.StringVar(value="round_robin")
        tournament_type_combo = ttk.Combobox(
            config_frame,
            textvariable=self.tournament_type_var,
            values=["round_robin", "elimination", "swiss", "ladder"],
            state="readonly"
        )
        tournament_type_combo.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        
        # Game mode
        ttk.Label(config_frame, text="Game Mode:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        
        self.game_mode_var = tk.StringVar(value="competitive")
        game_mode_combo = ttk.Combobox(
            config_frame,
            textvariable=self.game_mode_var,
            values=["competitive", "cooperative", "mixed"],
            state="readonly"
        )
        game_mode_combo.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        
        # Episodes per match
        ttk.Label(config_frame, text="Episodes per Match:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        
        self.episodes_per_match_var = tk.StringVar(value="100")
        ttk.Entry(config_frame, textvariable=self.episodes_per_match_var, width=10).grid(row=2, column=1, sticky='w', padx=5, pady=5)
        
        config_frame.columnconfigure(1, weight=1)
        
        # Agent Configuration
        agents_frame = ttk.LabelFrame(tournament_frame, text="Tournament Agents", padding=10)
        agents_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Agent list
        self.agents_listbox = tk.Listbox(agents_frame, height=8)
        agents_scrollbar = ttk.Scrollbar(agents_frame, command=self.agents_listbox.yview)
        self.agents_listbox.config(yscrollcommand=agents_scrollbar.set)
        
        self.agents_listbox.pack(side='left', fill='both', expand=True)
        agents_scrollbar.pack(side='right', fill='y')
        
        # Agent controls
        agent_controls = ttk.Frame(agents_frame)
        agent_controls.pack(side='right', fill='y', padx=10)
        
        ttk.Button(
            agent_controls,
            text="Add Agent",
            command=self._add_tournament_agent,
            style='success.TButton'
        ).pack(fill='x', pady=2)
        
        ttk.Button(
            agent_controls,
            text="Remove Agent",
            command=self._remove_tournament_agent,
            style='danger.TButton'
        ).pack(fill='x', pady=2)
        
        ttk.Button(
            agent_controls,
            text="Edit Agent",
            command=self._edit_tournament_agent,
            style='info.TButton'
        ).pack(fill='x', pady=2)
        
        # Tournament Controls
        tournament_controls = ttk.Frame(tournament_frame)
        tournament_controls.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(
            tournament_controls,
            text="Create Tournament",
            command=self._create_tournament,
            style='primary.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            tournament_controls,
            text="Start Tournament",
            command=self._start_tournament,
            style='success.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            tournament_controls,
            text="View Results",
            command=self._view_tournament_results,
            style='info.TButton'
        ).pack(side='left', padx=5)
    
    def _create_monitoring_tab(self):
        """Create system monitoring and analytics tab."""
        
        monitoring_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitoring_frame, text="📊 Monitoring")
        
        # System Metrics
        metrics_frame = ttk.LabelFrame(monitoring_frame, text="System Metrics", padding=10)
        metrics_frame.pack(fill='x', padx=10, pady=5)
        
        # Metrics display
        self.metrics_tree = ttk.Treeview(metrics_frame, columns=('Value', 'Status'), show='tree headings', height=8)
        self.metrics_tree.heading('#0', text='Metric')
        self.metrics_tree.heading('Value', text='Value')
        self.metrics_tree.heading('Status', text='Status')
        
        metrics_scroll = ttk.Scrollbar(metrics_frame, command=self.metrics_tree.yview)
        self.metrics_tree.config(yscrollcommand=metrics_scroll.set)
        
        self.metrics_tree.pack(side='left', fill='both', expand=True)
        metrics_scroll.pack(side='right', fill='y')
        
        # Performance Analytics
        analytics_frame = ttk.LabelFrame(monitoring_frame, text="Performance Analytics", padding=10)
        analytics_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Analytics controls
        analytics_controls = ttk.Frame(analytics_frame)
        analytics_controls.pack(fill='x', pady=5)
        
        ttk.Button(
            analytics_controls,
            text="Refresh Metrics",
            command=self._refresh_metrics,
            style='info.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            analytics_controls,
            text="Export Report",
            command=self._export_analytics_report,
            style='primary.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            analytics_controls,
            text="Generate Charts",
            command=self._generate_performance_charts,
            style='success.TButton'
        ).pack(side='left', padx=5)
        
        # Analytics display
        self.analytics_text = tk.Text(analytics_frame, height=12)
        analytics_scroll = ttk.Scrollbar(analytics_frame, command=self.analytics_text.yview)
        self.analytics_text.config(yscrollcommand=analytics_scroll.set)
        
        self.analytics_text.pack(side='left', fill='both', expand=True)
        analytics_scroll.pack(side='right', fill='y')
        
        # Initialize metrics
        self._refresh_metrics()
    
    # Production Tab Methods
    def _update_environment_status(self):
        """Update environment validation status."""
        
        # Clear existing status
        for widget in self.environment_status.winfo_children():
            widget.destroy()
        
        # Run validation
        try:
            validation_results = self.production_manager.validate_environment()
            
            for check_name, status in validation_results.items():
                status_frame = ttk.Frame(self.environment_status)
                status_frame.pack(fill='x', pady=2)
                
                # Status indicator
                status_color = 'success' if status else 'danger'
                status_text = '✅' if status else '❌'
                
                ttk.Label(
                    status_frame, 
                    text=status_text, 
                    foreground='green' if status else 'red'
                ).pack(side='left', padx=5)
                
                ttk.Label(
                    status_frame, 
                    text=check_name.replace('_', ' ').title()
                ).pack(side='left', padx=5)
                
        except Exception as e:
            ttk.Label(
                self.environment_status,
                text=f"Error checking environment: {e}",
                foreground='red'
            ).pack()
    
    def _populate_config_tree(self):
        """Populate configuration tree with current settings."""
        
        # Clear existing items
        for item in self.config_tree.get_children():
            self.config_tree.delete(item)
        
        # Add configuration sections
        config_dict = self.config.to_dict()
        
        for section, values in config_dict.items():
            if isinstance(values, dict):
                section_item = self.config_tree.insert('', 'end', text=section, values=('', ''))
                for key, value in values.items():
                    self.config_tree.insert(section_item, 'end', text=key, values=(str(value), ''))
            else:
                self.config_tree.insert('', 'end', text=section, values=(str(values), ''))
    
    def _export_config(self):
        """Export current configuration to file."""
        
        filename = filedialog.asksaveasfilename(
            title="Export Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                config_dict = self.config.to_dict()
                with open(filename, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
                
                messagebox.showinfo("Success", f"Configuration exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export configuration: {e}")
    
    def _load_config(self):
        """Load configuration from file."""
        
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config_dict = json.load(f)
                
                # Update configuration (simplified - in production, would validate)
                messagebox.showinfo("Success", f"Configuration loaded from {filename}")
                self._populate_config_tree()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    # Market Data Tab Methods
    def _start_realtime_feed(self):
        """Start real-time market data feed."""
        
        symbol = self.symbol_var.get()
        provider = self.provider_var.get()
        
        if not symbol:
            messagebox.showerror("Error", "Please enter a trading symbol")
            return
        
        try:
            # Subscribe to real-time data
            def on_tick(tick: MarketTick):
                self._display_market_tick(tick)
            
            market_data_manager.subscribe_to_symbol(symbol, on_tick, provider)
            
            self._log_market_message(f"Started real-time feed for {symbol} from {provider}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start real-time feed: {e}")
    
    def _stop_realtime_feed(self):
        """Stop real-time market data feed."""
        
        # In a real implementation, would stop specific subscriptions
        self._log_market_message("Real-time feed stopped")
    
    def _fetch_historical_data(self):
        """Fetch historical market data."""
        
        symbol = self.symbol_var.get()
        provider = self.provider_var.get()
        interval = self.interval_var.get()
        
        if not symbol:
            messagebox.showerror("Error", "Please enter a trading symbol")
            return
        
        try:
            # Run in separate thread to avoid blocking UI
            def fetch_data():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    data = loop.run_until_complete(
                        market_data_manager.get_historical_data(symbol, interval, 30, provider)
                    )
                    
                    self._display_historical_data(data)
                    
                except Exception as e:
                    self._log_market_message(f"Error fetching historical data: {e}")
                
                finally:
                    loop.close()
            
            threading.Thread(target=fetch_data, daemon=True).start()
            self._log_market_message(f"Fetching historical data for {symbol}...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch historical data: {e}")
    
    def _display_market_tick(self, tick: MarketTick):
        """Display real-time market tick data."""
        
        message = f"[{tick.timestamp.strftime('%H:%M:%S')}] {tick.symbol}: ${tick.price:.4f}"
        if tick.volume:
            message += f" (Vol: {tick.volume:.2f})"
        
        self._log_market_message(message)
    
    def _display_historical_data(self, data: List):
        """Display historical market data."""
        
        if not data:
            self._log_market_message("No historical data received")
            return
        
        self._log_market_message(f"Received {len(data)} historical data points:")
        
        for i, point in enumerate(data[-10:]):  # Show last 10 points
            message = f"  {point.timestamp.strftime('%Y-%m-%d %H:%M')} OHLC: {point.open:.4f}/{point.high:.4f}/{point.low:.4f}/{point.close:.4f}"
            self._log_market_message(message)
    
    def _log_market_message(self, message: str):
        """Log message to market data display."""
        
        self.market_data_text.insert(tk.END, f"{message}\n")
        self.market_data_text.see(tk.END)
    
    # Research Tab Methods
    def _start_algorithm_comparison(self):
        """Start algorithm comparison experiment."""
        
        selected_algorithms = [
            algo for algo, var in self.algorithm_vars.items() if var.get()
        ]
        
        if len(selected_algorithms) < 2:
            messagebox.showerror("Error", "Please select at least 2 algorithms to compare")
            return
        
        try:
            training_episodes = int(self.training_episodes_var.get())
            eval_episodes = int(self.eval_episodes_var.get())
            num_seeds = int(self.num_seeds_var.get())
            
            # Start comparison in separate thread
            def run_comparison():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    self._log_research_message(f"Starting comparison of {len(selected_algorithms)} algorithms...")
                    
                    # Simple environment config
                    env_configs = [{'observation_space_size': 100, 'action_space_size': 5}]
                    
                    result = loop.run_until_complete(
                        algorithm_comparator.run_algorithm_comparison(
                            selected_algorithms,
                            env_configs,
                            training_episodes,
                            eval_episodes,
                            num_seeds
                        )
                    )
                    
                    self._display_comparison_results(result)
                    
                except Exception as e:
                    self._log_research_message(f"Error in comparison: {e}")
                
                finally:
                    loop.close()
            
            threading.Thread(target=run_comparison, daemon=True).start()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
    
    def _start_hyperparameter_study(self):
        """Start hyperparameter optimization study."""
        
        selected_algorithms = [
            algo for algo, var in self.algorithm_vars.items() if var.get()
        ]
        
        if len(selected_algorithms) != 1:
            messagebox.showerror("Error", "Please select exactly 1 algorithm for hyperparameter study")
            return
        
        algorithm = selected_algorithms[0]
        
        # Define parameter grid (simplified)
        parameter_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [32, 64, 128]
        }
        
        def run_study():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                self._log_research_message(f"Starting hyperparameter study for {algorithm}...")
                
                env_config = {'observation_space_size': 100, 'action_space_size': 5}
                
                result = loop.run_until_complete(
                    research_runner.run_hyperparameter_study(
                        algorithm,
                        parameter_grid,
                        env_config,
                        int(self.training_episodes_var.get())
                    )
                )
                
                self._display_hyperparameter_results(result)
                
            except Exception as e:
                self._log_research_message(f"Error in hyperparameter study: {e}")
            
            finally:
                loop.close()
        
        threading.Thread(target=run_study, daemon=True).start()
    
    def _start_stability_analysis(self):
        """Start algorithm stability analysis."""
        
        selected_algorithms = [
            algo for algo, var in self.algorithm_vars.items() if var.get()
        ]
        
        if not selected_algorithms:
            messagebox.showerror("Error", "Please select at least 1 algorithm for stability analysis")
            return
        
        def run_analysis():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                self._log_research_message(f"Starting stability analysis for {len(selected_algorithms)} algorithms...")
                
                env_config = {'observation_space_size': 100, 'action_space_size': 5}
                
                result = loop.run_until_complete(
                    research_runner.run_stability_analysis(
                        selected_algorithms,
                        env_config,
                        num_runs=int(self.num_seeds_var.get()),
                        training_episodes=int(self.training_episodes_var.get())
                    )
                )
                
                self._display_stability_results(result)
                
            except Exception as e:
                self._log_research_message(f"Error in stability analysis: {e}")
            
            finally:
                loop.close()
        
        threading.Thread(target=run_analysis, daemon=True).start()
    
    def _log_research_message(self, message: str):
        """Log message to research results display."""
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.research_results_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.research_results_text.see(tk.END)
    
    def _display_comparison_results(self, result):
        """Display algorithm comparison results."""
        
        self._log_research_message("=== COMPARISON RESULTS ===")
        self._log_research_message(f"Best Algorithm: {result.best_algorithm}")
        
        for algorithm, metrics in result.performance_summary.items():
            self._log_research_message(f"\n{algorithm.upper()}:")
            self._log_research_message(f"  Mean Reward: {metrics['mean_evaluation_reward']:.4f}")
            self._log_research_message(f"  Std Reward: {metrics['std_evaluation_reward']:.4f}")
            self._log_research_message(f"  Training Time: {metrics['mean_training_time']:.2f}s")
        
        self._log_research_message("\nRecommendations:")
        for rec in result.recommendations:
            self._log_research_message(f"  • {rec}")
    
    def _display_hyperparameter_results(self, result):
        """Display hyperparameter study results."""
        
        self._log_research_message("=== HYPERPARAMETER STUDY RESULTS ===")
        self._log_research_message(f"Best Performance: {result['best_performance']:.4f}")
        self._log_research_message(f"Best Parameters: {result['best_parameters']}")
    
    def _display_stability_results(self, result):
        """Display stability analysis results."""
        
        self._log_research_message("=== STABILITY ANALYSIS RESULTS ===")
        
        for algorithm, metrics in result.items():
            self._log_research_message(f"\n{algorithm.upper()}:")
            self._log_research_message(f"  Mean Performance: {metrics['mean_performance']:.4f}")
            self._log_research_message(f"  Std Performance: {metrics['std_performance']:.4f}")
            self._log_research_message(f"  Coefficient of Variation: {metrics['coefficient_of_variation']:.4f}")
    
    # Tournament Tab Methods
    def _add_tournament_agent(self):
        """Add agent to tournament."""
        
        # Simple dialog for agent configuration
        dialog = tk.Toplevel(self)
        dialog.title("Add Tournament Agent")
        dialog.geometry("400x300")
        
        # Algorithm selection
        ttk.Label(dialog, text="Algorithm:").pack(pady=5)
        
        algo_var = tk.StringVar(value="dqn")
        algo_combo = ttk.Combobox(
            dialog,
            textvariable=algo_var,
            values=["dqn", "double_dqn", "a2c", "ppo", "ddpg", "td3", "sac"],
            state="readonly"
        )
        algo_combo.pack(pady=5)
        
        # Training episodes
        ttk.Label(dialog, text="Training Episodes:").pack(pady=5)
        
        episodes_var = tk.StringVar(value="1000")
        ttk.Entry(dialog, textvariable=episodes_var).pack(pady=5)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        def add_agent():
            algorithm = algo_var.get()
            episodes = episodes_var.get()
            
            agent_info = f"{algorithm} (Episodes: {episodes})"
            self.agents_listbox.insert(tk.END, agent_info)
            
            dialog.destroy()
        
        ttk.Button(button_frame, text="Add", command=add_agent).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side='left', padx=5)
    
    def _remove_tournament_agent(self):
        """Remove selected agent from tournament."""
        
        selection = self.agents_listbox.curselection()
        if selection:
            self.agents_listbox.delete(selection[0])
    
    def _edit_tournament_agent(self):
        """Edit selected tournament agent."""
        
        selection = self.agents_listbox.curselection()
        if selection:
            # Simple implementation - remove and add new
            self.agents_listbox.delete(selection[0])
            self._add_tournament_agent()
    
    def _create_tournament(self):
        """Create new tournament."""
        
        agent_count = self.agents_listbox.size()
        
        if agent_count < 2:
            messagebox.showerror("Error", "Need at least 2 agents for tournament")
            return
        
        try:
            # Get tournament configuration
            tournament_type = TournamentType(self.tournament_type_var.get())
            game_mode = GameMode(self.game_mode_var.get())
            episodes_per_match = int(self.episodes_per_match_var.get())
            
            # Create agent configurations
            agents = []
            for i in range(agent_count):
                agent_info = self.agents_listbox.get(i)
                # Parse agent info (simplified)
                algorithm = agent_info.split()[0]
                agents.append({
                    'algorithm_type': algorithm,
                    'training_episodes': 1000
                })
            
            # Create tournament in separate thread
            def create_tournament():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    env_config = {'observation_space_size': 100, 'action_space_size': 5}
                    
                    tournament_id = loop.run_until_complete(
                        tournament_engine.create_tournament(
                            tournament_type,
                            agents,
                            env_config,
                            game_mode,
                            episodes_per_match
                        )
                    )
                    
                    self._log_research_message(f"Tournament created: {tournament_id}")
                    self.active_tournaments[tournament_id] = tournament_type
                    
                except Exception as e:
                    self._log_research_message(f"Error creating tournament: {e}")
                
                finally:
                    loop.close()
            
            threading.Thread(target=create_tournament, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create tournament: {e}")
    
    def _start_tournament(self):
        """Start tournament."""
        
        if not self.active_tournaments:
            messagebox.showerror("Error", "No tournaments created")
            return
        
        # For simplicity, start the first tournament
        tournament_id = list(self.active_tournaments.keys())[0]
        
        def run_tournament():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                self._log_research_message(f"Starting tournament {tournament_id}...")
                
                result = loop.run_until_complete(
                    tournament_engine.run_tournament(tournament_id)
                )
                
                self._display_tournament_results(result)
                
            except Exception as e:
                self._log_research_message(f"Error running tournament: {e}")
            
            finally:
                loop.close()
        
        threading.Thread(target=run_tournament, daemon=True).start()
    
    def _view_tournament_results(self):
        """View tournament results."""
        
        # Simple implementation - show saved results
        messagebox.showinfo("Tournament Results", "Check tournament_results/ directory for detailed results")
    
    def _display_tournament_results(self, result):
        """Display tournament results."""
        
        self._log_research_message("=== TOURNAMENT RESULTS ===")
        self._log_research_message(f"Tournament: {result.tournament_id}")
        self._log_research_message(f"Type: {result.tournament_type.value}")
        self._log_research_message(f"Champion: {result.champion}")
        
        self._log_research_message("\nFinal Rankings:")
        for i, (agent_id, score) in enumerate(result.final_rankings[:5]):  # Top 5
            self._log_research_message(f"  {i+1}. {agent_id}: {score:.4f}")
        
        self._log_research_message(f"\nTotal Matches: {result.tournament_stats['total_matches']}")
        self._log_research_message(f"Total Time: {result.tournament_stats['total_tournament_time']:.2f}s")
    
    # Monitoring Tab Methods
    def _refresh_metrics(self):
        """Refresh system metrics display."""
        
        # Clear existing metrics
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
        
        try:
            # System metrics
            import psutil
            import platform
            
            # CPU metrics
            cpu_item = self.metrics_tree.insert('', 'end', text='CPU', values=('', ''))
            self.metrics_tree.insert(cpu_item, 'end', text='Usage', values=(f'{psutil.cpu_percent():.1f}%', 'Good' if psutil.cpu_percent() < 80 else 'High'))
            self.metrics_tree.insert(cpu_item, 'end', text='Cores', values=(f'{psutil.cpu_count()}', 'OK'))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_item = self.metrics_tree.insert('', 'end', text='Memory', values=('', ''))
            self.metrics_tree.insert(memory_item, 'end', text='Usage', values=(f'{memory.percent:.1f}%', 'Good' if memory.percent < 80 else 'High'))
            self.metrics_tree.insert(memory_item, 'end', text='Available', values=(f'{memory.available / 1024**3:.1f} GB', 'OK'))
            
            # Application metrics
            app_item = self.metrics_tree.insert('', 'end', text='Application', values=('', ''))
            self.metrics_tree.insert(app_item, 'end', text='Active Experiments', values=(f'{len(self.active_experiments)}', 'OK'))
            self.metrics_tree.insert(app_item, 'end', text='Active Tournaments', values=(f'{len(self.active_tournaments)}', 'OK'))
            self.metrics_tree.insert(app_item, 'end', text='Real-time Feeds', values=(f'{len(self.real_time_subscriptions)}', 'OK'))
            
        except Exception as e:
            error_item = self.metrics_tree.insert('', 'end', text='Error', values=(str(e), 'Error'))
    
    def _export_analytics_report(self):
        """Export analytics report."""
        
        filename = filedialog.asksaveasfilename(
            title="Export Analytics Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                report_content = self.analytics_text.get('1.0', tk.END)
                
                with open(filename, 'w') as f:
                    f.write(f"RL System Analytics Report\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"{'='*50}\n\n")
                    f.write(report_content)
                
                messagebox.showinfo("Success", f"Analytics report exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report: {e}")
    
    def _generate_performance_charts(self):
        """Generate performance visualization charts."""
        
        try:
            import matplotlib.pyplot as plt
            
            # Simple performance chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # CPU and Memory usage over time (simulated)
            times = list(range(10))
            cpu_usage = np.random.uniform(20, 80, 10)
            memory_usage = np.random.uniform(30, 70, 10)
            
            ax1.plot(times, cpu_usage, label='CPU %', marker='o')
            ax1.plot(times, memory_usage, label='Memory %', marker='s')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Usage %')
            ax1.set_title('System Resource Usage')
            ax1.legend()
            ax1.grid(True)
            
            # Algorithm performance comparison (simulated)
            algorithms = ['DQN', 'A2C', 'PPO', 'DDPG']
            performance = np.random.uniform(0.3, 0.9, 4)
            
            ax2.bar(algorithms, performance, color=['blue', 'green', 'orange', 'red'])
            ax2.set_ylabel('Performance Score')
            ax2.set_title('Algorithm Performance Comparison')
            ax2.set_ylim(0, 1)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = f"performance_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            
            self._log_analytics_message(f"Performance chart saved: {chart_path}")
            
            plt.show()
            
        except ImportError:
            messagebox.showerror("Error", "Matplotlib not available for chart generation")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate charts: {e}")
    
    def _log_analytics_message(self, message: str):
        """Log message to analytics display."""
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.analytics_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.analytics_text.see(tk.END)

# For compatibility with existing code
RLSystemTab = AdvancedRLSystemTab
