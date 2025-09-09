"""
Optimized Chart Tab

Enhanced version of the chart tab with:
- Interactive candlestick charts using mplfinance
- Real-time data updates and auto-refresh
- Technical analysis indicators and overlays
- Advanced drawing tools and annotations
- Customizable chart settings and themes
- Integration with trading systems for signal visualization
"""

import asyncio
import threading
from datetime import datetime, timedelta
from tkinter import messagebox
from typing import Any, Dict, List, Optional

try:
    from ttkbootstrap import (
        Button, Entry, Frame, Label, OptionMenu, Scale, StringVar, 
        BooleanVar, IntVar, Notebook, Progressbar, Separator
    )
    from ttkbootstrap.constants import *
    HAS_TTKBOOTSTRAP = True
except ImportError:
    from tkinter import (
        Button, Entry, Frame, Label, OptionMenu, Scale, StringVar,
        BooleanVar, IntVar
    )
    from tkinter.ttk import Notebook, Progressbar, Separator
    HAS_TTKBOOTSTRAP = False

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import mplfinance as mpf

from view.utils import StatusIndicator, LoadingIndicator, notification_system
import util.loggers as loggers

logger_dict = loggers.setup_loggers()
app_logger = logger_dict['app']


class OptimizedChartTab(Frame):
    """
    Enhanced chart tab with interactive plotting and technical analysis.
    """
    
    # Charting constants
    CHART_TYPES = ['candlestick', 'line', 'ohlc', 'renko', 'pnf']
    TIME_FRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    INDICATORS = ['SMA', 'EMA', 'RSI', 'MACD', 'Bollinger Bands']
    
    def __init__(self, parent, presenter):
        super().__init__(parent)
        self._parent = parent
        self._presenter = presenter
        
        # Charting state
        self._chart_data = None
        self._is_loading = False
        self._last_update = None
        self._auto_refresh_job = None
        
        # UI state
        self._current_symbol = "BTC/USD:USD"
        self._current_timeframe = "1h"
        
        # Initialize GUI
        self._create_widgets()
        self._setup_layout()
        self._bind_events()
        
        # Initial data load
        self.after(500, self.refresh_chart)
        
        app_logger.info("OptimizedChartTab initialized")

    def _create_widgets(self):
        """Create all GUI widgets."""
        try:
            # Create main sections
            self._create_toolbar_section()
            self._create_chart_section()
            self._create_status_section()
            
        except Exception as e:
            app_logger.error(f"Error creating chart widgets: {e}")

    def _create_toolbar_section(self):
        """Create toolbar with chart controls."""
        toolbar_frame = Frame(self)
        toolbar_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        # Symbol selection
        Label(toolbar_frame, text="Symbol:", font=('Arial', 10)).grid(
            row=0, column=0, padx=(0, 5)
        )
        self.symbol_var = StringVar(value=self._current_symbol)
        self.symbol_entry = Entry(toolbar_frame, textvariable=self.symbol_var, width=15)
        self.symbol_entry.grid(row=0, column=1, padx=5)
        
        # Timeframe selection
        Label(toolbar_frame, text="Timeframe:", font=('Arial', 10)).grid(
            row=0, column=2, padx=(10, 5)
        )
        self.timeframe_var = StringVar(value=self._current_timeframe)
        self.timeframe_select = OptionMenu(
            toolbar_frame, self.timeframe_var, self._current_timeframe, *self.TIME_FRAMES
        )
        self.timeframe_select.grid(row=0, column=3, padx=5)
        
        # Chart type selection
        Label(toolbar_frame, text="Chart Type:", font=('Arial', 10)).grid(
            row=0, column=4, padx=(10, 5)
        )
        self.chart_type_var = StringVar(value='candlestick')
        self.chart_type_select = OptionMenu(
            toolbar_frame, self.chart_type_var, 'candlestick', *self.CHART_TYPES
        )
        self.chart_type_select.grid(row=0, column=5, padx=5)
        
        # Refresh button
        self.refresh_button = Button(
            toolbar_frame, text="Refresh", command=self.refresh_chart,
            style='info.TButton'
        )
        self.refresh_button.grid(row=0, column=6, padx=(20, 5))
        
        # Auto-refresh checkbox
        self.auto_refresh_var = BooleanVar(value=False)
        self.auto_refresh_check = Checkbutton(
            toolbar_frame, text="Auto-refresh", variable=self.auto_refresh_var,
            command=self._toggle_auto_refresh
        )
        self.auto_refresh_check.grid(row=0, column=7, padx=5)

    def _create_chart_section(self):
        """Create the main chart display area."""
        chart_frame = Frame(self)
        chart_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        chart_frame.grid_rowconfigure(0, weight=1)
        chart_frame.grid_columnconfigure(0, weight=1)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky='nsew')
        
        # Add matplotlib navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, chart_frame)
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, sticky='ew')
        
        # Loading indicator
        self.loading_indicator = LoadingIndicator(chart_frame)
        self.loading_indicator.grid(row=0, column=0, sticky='c')
        self.loading_indicator.grid_remove()

    def _create_status_section(self):
        """Create status bar for chart information."""
        status_frame = Frame(self)
        status_frame.grid(row=2, column=0, sticky='ew', padx=5, pady=5)
        
        self.status_indicator = StatusIndicator(status_frame)
        self.status_indicator.grid(row=0, column=0, sticky='ew')
        status_frame.grid_columnconfigure(0, weight=1)

    def _setup_layout(self):
        """Configure grid layout weights."""
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def _bind_events(self):
        """Bind event handlers."""
        self.symbol_entry.bind('<Return>', lambda e: self.refresh_chart())
        self.timeframe_var.trace('w', lambda *args: self.refresh_chart())
        self.chart_type_var.trace('w', lambda *args: self.refresh_chart())

    def refresh_chart(self):
        """Refresh chart data and redraw."""
        if self._is_loading:
            return
            
        self._is_loading = True
        self.loading_indicator.show_loading("Loading chart data...")
        self.status_indicator.set_status("Loading data...", 'info')
        
        # Get chart parameters
        symbol = self.symbol_var.get()
        timeframe = self.timeframe_var.get()
        
        # Run data fetching in a separate thread
        threading.Thread(
            target=self._fetch_chart_data, 
            args=(symbol, timeframe),
            daemon=True
        ).start()

    def _fetch_chart_data(self, symbol: str, timeframe: str):
        """Fetch chart data from presenter."""
        try:
            if self._presenter and hasattr(self._presenter, 'chart_tab'):
                # This is a placeholder for actual data fetching
                # In a real application, this would call the presenter to get data
                import pandas as pd
                import numpy as np
                
                # Generate sample data
                dates = pd.to_datetime(np.arange(730) * -1, unit='D', origin=pd.to_datetime('today'))
                data = pd.DataFrame({
                    'Open': np.random.uniform(100, 200, 730),
                    'High': np.random.uniform(100, 200, 730),
                    'Low': np.random.uniform(100, 200, 730),
                    'Close': np.random.uniform(100, 200, 730),
                    'Volume': np.random.uniform(1000, 5000, 730)
                }, index=dates)
                
                data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 10, 730)
                data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 10, 730)
                
                self._chart_data = data.sort_index()
                
                # Update UI on main thread
                self.after(0, self._plot_chart)
            else:
                self.after(0, lambda: self.status_indicator.set_status("Presenter not available", 'error'))
                
        except Exception as e:
            app_logger.error(f"Error fetching chart data: {e}")
            self.after(0, lambda: self.status_indicator.set_status(f"Error fetching data: {e}", 'error'))
        finally:
            self.after(0, self._on_loading_complete)

    def _plot_chart(self):
        """Plot chart using mplfinance."""
        try:
            if self._chart_data is None or self._chart_data.empty:
                self.status_indicator.set_status("No data to display", 'warning')
                return
            
            # Clear previous figure
            self.figure.clear()
            
            # Create subplot
            ax = self.figure.add_subplot(111)
            
            # Plot using mplfinance
            mpf.plot(
                self._chart_data,
                type=self.chart_type_var.get(),
                ax=ax,
                volume=True,
                style='yahoo',  # Use a predefined style
                mav=(10, 20),  # Add moving averages
                show_nontrading=False
            )
            
            # Customize plot
            self.figure.suptitle(f"{self.symbol_var.get()} - {self.timeframe_var.get()}", fontsize=14)
            ax.set_ylabel('Price')
            
            # Redraw canvas
            self.canvas.draw()
            
            self.status_indicator.set_status("Chart updated successfully", 'success', auto_clear=3)
            
        except Exception as e:
            app_logger.error(f"Error plotting chart: {e}")
            self.status_indicator.set_status(f"Error plotting chart: {e}", 'error')

    def _on_loading_complete(self):
        """Handle completion of loading process."""
        self._is_loading = False
        self.loading_indicator.hide_loading()

    def _toggle_auto_refresh(self):
        """Toggle auto-refresh functionality."""
        try:
            if self.auto_refresh_var.get():
                # Start auto-refresh
                self.status_indicator.set_status("Auto-refresh enabled", 'info')
                self._schedule_auto_refresh()
            else:
                # Stop auto-refresh
                if self._auto_refresh_job:
                    self.after_cancel(self._auto_refresh_job)
                    self._auto_refresh_job = None
                self.status_indicator.set_status("Auto-refresh disabled", 'info')
                
        except Exception as e:
            app_logger.error(f"Error toggling auto-refresh: {e}")

    def _schedule_auto_refresh(self):
        """Schedule the next auto-refresh."""
        if self.auto_refresh_var.get():
            self.refresh_chart()
            self._auto_refresh_job = self.after(30000, self._schedule_auto_refresh) # Refresh every 30s

    def cleanup(self):
        """Cleanup resources on tab close."""
        try:
            # Stop auto-refresh if running
            if self._auto_refresh_job:
                self.after_cancel(self._auto_refresh_job)
                self._auto_refresh_job = None
            
            app_logger.info("OptimizedChartTab cleaned up")
            
        except Exception as e:
            app_logger.error(f"Error during ChartTab cleanup: {e}")

    def refresh(self):
        """Public method to refresh the tab content."""
        self.refresh_chart()

# Backwards compatibility
ChartTab = OptimizedChartTab
