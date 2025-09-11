"""
Refactored Presenters Module

This module provides refactored and enhanced versions of the trading application presenters
with improved organization, error handling, and performance optimizations.
"""

import asyncio
import os
import threading
import traceback
import weakref
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from queue import Queue, Empty
from tkinter import messagebox
from typing import Any, Dict, List, Optional, Callable, Union

try:
    from ttkbootstrap import Frame
    HAS_TTKBOOTSTRAP = True
except Exception:
    from tkinter import Frame
    HAS_TTKBOOTSTRAP = False

import util.loggers as loggers
from util.config_manager import get_config
from model.base_model import BaseModel

# Constants for performance tuning
MAX_POSITION_SIZE = 0.01
MIN_STOP_LOSS_LEVEL = 0.10
UI_UPDATE_INTERVAL = 100  # milliseconds
CACHE_TTL = 300  # seconds
MAX_CONCURRENT_OPERATIONS = 5

logger_dict = loggers.setup_loggers()
app_logger = logger_dict['app']

# Performance and UI state management
@dataclass
class UIMetrics:
    """UI performance metrics tracking."""
    operations_count: int = 0
    avg_response_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    last_update: Optional[datetime] = None
    slow_operations: List[Dict] = None
    
    def __post_init__(self):
        if self.slow_operations is None:
            self.slow_operations = []

# Global UI metrics
ui_metrics = UIMetrics()

def ui_performance_monitor(func):
    """Decorator to monitor UI operation performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            ui_metrics.operations_count += 1
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update average response time
            ui_metrics.avg_response_time = (
                (ui_metrics.avg_response_time * (ui_metrics.operations_count - 1) + execution_time) 
                / ui_metrics.operations_count
            )
            
            # Track slow operations
            if execution_time > 1.0:
                ui_metrics.slow_operations.append({
                    'function': func.__name__,
                    'duration': execution_time,
                    'timestamp': datetime.now()
                })
                # Keep only recent slow operations
                ui_metrics.slow_operations = ui_metrics.slow_operations[-50:]
                
            ui_metrics.last_update = datetime.now()
            return result
            
        except Exception as e:
            ui_metrics.error_count += 1
            app_logger.error(f"UI operation error in {func.__name__}: {e}")
            raise e
    return wrapper


class BasePresenter(ABC):
    """Base class for all presenters in the trading application."""
    
    def __init__(self, model, view):
        """Initialize the base presenter.
        
        Args:
            model: The model instance
            view: The view instance
        """
        self.model = model
        self.view = view
        self.config = get_config()
        self.logger = self._setup_logger()
        self._initialized = False
        self._lock = threading.RLock()
        
    def _setup_logger(self) -> Any:
        """Set up logger for this presenter."""
        return app_logger
    
    def initialize(self) -> bool:
        """Initialize the presenter.
        
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
    
    @abstractmethod
    def _initialize(self):
        """Internal initialization method to be implemented by subclasses."""
        pass
    
    def handle_exception(self, e: Exception, context: str = "") -> None:
        """Handle an exception with proper logging and user feedback.
        
        Args:
            e: The exception that occurred
            context: Additional context about where the exception occurred
        """
        message = f"Exception in {self.__class__.__name__}"
        if context:
            message += f" ({context})"
        message += f": {str(e)}"
        
        self.logger.error(message)
        self.logger.debug(traceback.format_exc())
        
        # Show error message to user
        try:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        except Exception:
            # If we can't show a messagebox, just log it
            self.logger.warning("Failed to show error message to user")


class OptimizedPresenter(BasePresenter):
    """Main presenter class that coordinates all UI functionality."""
    
    def __init__(self, model, view):
        """Initialize the optimized presenter."""
        super().__init__(model, view)
        self.executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_OPERATIONS)
        self._cache = weakref.WeakValueDictionary()
        self._cache_timestamps = {}
        
    def _initialize(self):
        """Internal initialization method."""
        # No special initialization needed for the main presenter
        pass
        
    def get_main_view(self):
        """Get the main application view."""
        try:
            # Set up the main view
            self.view.set_title("Advanced Trading Application")
            self.view.set_geometry(self.config.window_size)
            
            # Initialize all tabs
            self._initialize_tabs()
            
            return self.view
        except Exception as e:
            self.handle_exception(e, "main view initialization")
            return None
            
    def _initialize_tabs(self):
        """Initialize all application tabs."""
        try:
            # This would initialize all the different tabs in the application
            # For now, we'll just log that we're doing this
            self.logger.info("Initializing application tabs")
        except Exception as e:
            self.handle_exception(e, "tab initialization")
            
    def run(self):
        """Run the application."""
        try:
            self.logger.info("Starting application")
            self.view.mainloop()
        except Exception as e:
            self.handle_exception(e, "application run")
            
    def shutdown(self):
        """Shutdown the application gracefully."""
        try:
            self.logger.info("Shutting down application")
            self.executor.shutdown(wait=True)
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Individual presenter classes for different parts of the application
class LoginPresenter(BasePresenter):
    """Presenter for the login functionality."""
    
    def _initialize(self):
        """Internal initialization method."""
        # No special initialization needed
        pass
        
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate a user.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            bool: True if authentication was successful, False otherwise
        """
        try:
            self.model.login_model.set_credentials(username, password)
            result = self.model.login_model.authenticate()
            
            if result:
                self.logger.info("User authenticated successfully")
                # Update the view to show the main application
                self.view.show_main_view()
                return True
            else:
                self.logger.warning("Authentication failed")
                # Show error message in the view
                self.view.show_error("Invalid username or password")
                return False
        except Exception as e:
            self.handle_exception(e, "user authentication")
            return False


class TradingPresenter(BasePresenter):
    """Presenter for manual trading functionality."""
    
    def _initialize(self):
        """Internal initialization method."""
        # No special initialization needed
        pass
        
    def execute_trade(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> bool:
        """Execute a trade.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            side: Trade side ('buy' or 'sell')
            amount: Trade amount
            price: Optional limit price
            
        Returns:
            bool: True if trade was executed successfully, False otherwise
        """
        try:
            result = self.model.trading_model.execute_trade(symbol, side, amount, price)
            
            if result:
                self.logger.info(f"Trade executed: {side} {amount} {symbol}")
                # Update the view to show the trade result
                self.view.show_trade_result(f"Successfully executed {side} order for {amount} {symbol}")
                return True
            else:
                self.logger.error(f"Failed to execute trade: {side} {amount} {symbol}")
                # Show error message in the view
                self.view.show_error("Failed to execute trade")
                return False
        except Exception as e:
            self.handle_exception(e, "trade execution")
            return False


class BotPresenter(BasePresenter):
    """Presenter for bot management functionality."""
    
    def _initialize(self):
        """Internal initialization method."""
        # No special initialization needed
        pass
        
    def create_bot(self, name: str, strategy: str, parameters: Dict[str, Any]) -> bool:
        """Create a new trading bot.
        
        Args:
            name: Bot name
            strategy: Trading strategy
            parameters: Strategy parameters
            
        Returns:
            bool: True if bot was created successfully, False otherwise
        """
        try:
            result = self.model.bot_model.create_bot(name, strategy, parameters)
            
            if result:
                self.logger.info(f"Bot created: {name}")
                # Update the view to show the bot creation result
                self.view.show_bot_creation_result(f"Successfully created bot: {name}")
                return True
            else:
                self.logger.error(f"Failed to create bot: {name}")
                # Show error message in the view
                self.view.show_error("Failed to create bot")
                return False
        except Exception as e:
            self.handle_exception(e, "bot creation")
            return False


class MLPresenter(BasePresenter):
    """Presenter for machine learning functionality."""
    
    def _initialize(self):
        """Internal initialization method."""
        # No special initialization needed
        pass
        
    def train_model(self, data: Any, target: str) -> bool:
        """Train a machine learning model.
        
        Args:
            data: Training data
            target: Target column name
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            result = self.model.ml_model.train_model(data, target)
            
            if result:
                self.logger.info("ML model trained successfully")
                # Update the view to show the training result
                self.view.show_ml_training_result("ML model trained successfully")
                return True
            else:
                self.logger.error("Failed to train ML model")
                # Show error message in the view
                self.view.show_error("Failed to train ML model")
                return False
        except Exception as e:
            self.handle_exception(e, "ML model training")
            return False


class RLPresenter(BasePresenter):
    """Presenter for reinforcement learning functionality."""
    
    def _initialize(self):
        """Internal initialization method."""
        # No special initialization needed
        pass
        
    def train_agent(self, environment: str, episodes: int) -> bool:
        """Train a reinforcement learning agent.
        
        Args:
            environment: Environment name
            episodes: Number of training episodes
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            result = self.model.rl_model.train_agent(environment, episodes)
            
            if result:
                self.logger.info("RL agent trained successfully")
                # Update the view to show the training result
                self.view.show_rl_training_result("RL agent trained successfully")
                return True
            else:
                self.logger.error("Failed to train RL agent")
                # Show error message in the view
                self.view.show_error("Failed to train RL agent")
                return False
        except Exception as e:
            self.handle_exception(e, "RL agent training")
            return False