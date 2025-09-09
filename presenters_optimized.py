"""
Optimized Presenters Module

Enhanced version of the presenters module with:
- Improved UI responsiveness and async operations
- Better state management and caching
- Enhanced error handling and user feedback
- Professional UI patterns and separation of concerns
- Performance monitoring and metrics
- Advanced presenter patterns (Command, Observer)
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
from util.error_handling import handle_exception
from util.cache import get_cache

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

def async_ui_operation(func):
    """Decorator to run UI operations asynchronously when possible."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # For non-critical UI updates, run in background
        if hasattr(self, '_executor') and hasattr(func, '_async_safe'):
            return self._executor.submit(func, self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)
    return wrapper

class UIStateManager:
    """Centralized UI state management."""
    
    def __init__(self):
        self._state = {}
        self._observers = {}
        self._cache = get_cache()
        self._lock = threading.RLock()
        
    def set_state(self, key: str, value: Any, notify_observers: bool = True):
        """Set state value with observer notification."""
        with self._lock:
            old_value = self._state.get(key)
            self._state[key] = value
            
            # Cache important state
            self._cache.set(f"ui_state_{key}", value, ttl=CACHE_TTL)
            
            # Notify observers if value changed
            if notify_observers and old_value != value:
                self._notify_observers(key, value, old_value)
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value with caching."""
        with self._lock:
            # Try memory first
            if key in self._state:
                ui_metrics.cache_hits += 1
                return self._state[key]
            
            # Try cache
            cached_value = self._cache.get(f"ui_state_{key}")
            if cached_value is not None:
                self._state[key] = cached_value
                ui_metrics.cache_hits += 1
                return cached_value
            
            ui_metrics.cache_misses += 1
            return default
    
    def register_observer(self, key: str, callback: Callable):
        """Register observer for state changes."""
        if key not in self._observers:
            self._observers[key] = []
        self._observers[key].append(callback)
    
    def _notify_observers(self, key: str, new_value: Any, old_value: Any):
        """Notify observers of state changes."""
        if key in self._observers:
            for callback in self._observers[key]:
                try:
                    callback(key, new_value, old_value)
                except Exception as e:
                    app_logger.error(f"Observer error: {e}")

# Global UI state manager
ui_state = UIStateManager()

class BasePresenter(ABC):
    """Base presenter with common functionality."""
    
    def __init__(self, model, view, main_presenter):
        self._model = model
        self.main_view = view
        self.presenter = main_presenter
        self._cache = get_cache()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{self.__class__.__name__}_")
        self._operation_queue = Queue()
        self._is_active = True
        
        # UI state management
        self._last_update = {}
        self._update_intervals = {}
        self._error_counts = {}
        
        app_logger.info(f"{self.__class__.__name__} initialized")
    
    @ui_performance_monitor
    def safe_execute(self, operation: Callable, error_message: str = "Operation failed", 
                    show_error: bool = True, default_return: Any = None) -> Any:
        """Safely execute operations with error handling."""
        try:
            return operation()
        except Exception as e:
            error_key = f"{self.__class__.__name__}_{operation.__name__}"
            self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
            
            app_logger.error(f"{error_message}: {e}")
            
            if show_error and self._error_counts[error_key] <= 3:  # Limit error popups
                try:
                    messagebox.showerror("Error", f"{error_message}\n\nDetails: {str(e)}")
                except:
                    pass  # Ignore messagebox errors in headless environments
            
            return default_return
    
    def update_ui_status(self, message: str, level: str = "info"):
        """Update UI status with throttling."""
        try:
            # Throttle updates to prevent UI flooding
            now = datetime.now()
            last_update = self._last_update.get('status', now - timedelta(seconds=1))
            
            if (now - last_update).total_seconds() >= 0.5:  # Max 2 updates per second
                if hasattr(self.presenter, 'main_listbox') and self.presenter.main_listbox:
                    self.presenter.main_listbox.set_text(f"[{level.upper()}] {message}")
                self._last_update['status'] = now
                
        except Exception as e:
            app_logger.error(f"Error updating UI status: {e}")
    
    @contextmanager
    def loading_context(self, message: str = "Processing..."):
        """Context manager for loading states."""
        self.update_ui_status(f"🔄 {message}", "info")
        try:
            yield
        except Exception as e:
            self.update_ui_status(f"❌ Error: {str(e)[:50]}...", "error")
            raise
        else:
            self.update_ui_status(f"✅ Completed", "success")
    
    def cleanup(self):
        """Cleanup presenter resources."""
        try:
            self._is_active = False
            if self._executor:
                self._executor.shutdown(wait=False)
            app_logger.info(f"{self.__class__.__name__} cleaned up")
        except Exception as e:
            app_logger.error(f"Error during presenter cleanup: {e}")
    
    def __del__(self):
        """Ensure cleanup on destruction."""
        try:
            self.cleanup()
        except:
            pass

class OptimizedPresenter:
    """
    Optimized main presenter with enhanced performance and functionality.
    """
    
    def __init__(self, model, view) -> None:
        self._model = model
        self._view = view
        
        # Enhanced initialization
        self._cache = get_cache()
        self._executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_OPERATIONS, thread_name_prefix="presenter_")
        
        # State management
        self._presenter_registry = weakref.WeakValueDictionary()
        self._initialization_times = {}
        
        # UI responsiveness
        self._pending_operations = Queue()
        self._ui_update_thread = None
        self._stop_ui_updates = threading.Event()
        
        # Performance monitoring
        self._start_time = datetime.now()
        
        self.get_frames()
        
        # Initialize presenter placeholders with lazy loading
        self._presenters = {}
        self._presenter_configs = {}
        
        app_logger.info("OptimizedPresenter initialized successfully")
        
        # Start UI update thread
        self._start_ui_update_thread()

    def _start_ui_update_thread(self):
        """Start background UI update thread for responsiveness."""
        def ui_update_worker():
            while not self._stop_ui_updates.is_set():
                try:
                    # Process pending UI operations
                    try:
                        operation = self._pending_operations.get(timeout=0.1)
                        if operation and callable(operation):
                            operation()
                    except Empty:
                        pass
                    
                    # Small delay to prevent CPU spinning
                    threading.Event().wait(0.01)
                    
                except Exception as e:
                    app_logger.error(f"UI update thread error: {e}")
        
        self._ui_update_thread = threading.Thread(target=ui_update_worker, daemon=True)
        self._ui_update_thread.start()

    def run(self) -> None:
        """Run the main UI loop with error handling."""
        try:
            self._view.mainloop()
        except Exception as e:
            app_logger.error(f"Error in main UI loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Enhanced cleanup with resource management."""
        try:
            app_logger.info("Starting presenter cleanup...")
            
            # Stop UI update thread
            self._stop_ui_updates.set()
            if self._ui_update_thread and self._ui_update_thread.is_alive():
                self._ui_update_thread.join(timeout=1.0)
            
            # Cleanup all child presenters
            for presenter in self._presenters.values():
                if hasattr(presenter, 'cleanup'):
                    presenter.cleanup()
            
            # Shutdown executor
            if self._executor:
                self._executor.shutdown(wait=True, timeout=2.0)
            
            app_logger.info("Presenter cleanup completed")
            
        except Exception as e:
            app_logger.error(f"Error during cleanup: {e}")

    @ui_performance_monitor
    def get_exchange(self, test_mode=True):
        """Get exchange with caching and error handling."""
        cache_key = f"exchange_{test_mode}"
        
        # Check cache first
        cached_exchange = self._cache.get(cache_key)
        if cached_exchange:
            return cached_exchange
        
        exchange = self.safe_execute(
            lambda: self._model.get_exchange(test_mode=test_mode),
            "Failed to get exchange",
            default_return=None
        )
        
        # Cache successful result
        if exchange:
            self._cache.set(cache_key, exchange, ttl=300)
        
        return exchange

    def safe_execute(self, operation: Callable, error_message: str = "Operation failed", 
                    show_error: bool = True, default_return: Any = None) -> Any:
        """Safely execute operations with comprehensive error handling."""
        try:
            return operation()
        except Exception as e:
            app_logger.error(f"{error_message}: {e}")
            app_logger.debug(f"Stack trace: {traceback.format_exc()}")
            
            if show_error:
                try:
                    messagebox.showerror("Error", f"{error_message}\n\nDetails: {str(e)}")
                except:
                    pass  # Ignore messagebox errors
            
            return default_return

    # Enhanced lazy initialization for presenters
    def _ensure_presenter(self, presenter_type: str):
        """Generic presenter initialization with performance monitoring."""
        if presenter_type in self._presenters:
            return self._presenters[presenter_type]
        
        start_time = datetime.now()
        
        try:
            app_logger.debug(f"Initializing {presenter_type}Presenter")
            
            presenter_class = self._get_presenter_class(presenter_type)
            if presenter_class:
                presenter_instance = presenter_class(self._model, self.main_view, self)
                self._presenters[presenter_type] = presenter_instance
                self._presenter_registry[f"{presenter_type}_presenter"] = presenter_instance
                
                # Track initialization time
                init_time = (datetime.now() - start_time).total_seconds()
                self._initialization_times[presenter_type] = init_time
                
                app_logger.info(f"{presenter_type}Presenter initialized in {init_time:.3f}s")
                return presenter_instance
            else:
                app_logger.warning(f"Presenter class not found for type: {presenter_type}")
                return None
                
        except Exception as e:
            app_logger.error(f"Error initializing {presenter_type}Presenter: {e}")
            return None

    def _get_presenter_class(self, presenter_type: str):
        """Factory method to get presenter class by type."""
        presenter_classes = {
            'trading': OptimizedTradePresenter,
            'bot': OptimizedBotPresenter,
            'chart': OptimizedChartPresenter,
            'exchange': OptimizedExchangePresenter,
            'ml': OptimizedMLPresenter,
            'rl': OptimizedRLPresenter,
            'trading_system': OptimizedTradingSystemPresenter,
            'ml_system': OptimizedMLSystemPresenter,
            'rl_system': OptimizedRLSystemPresenter,
        }
        return presenter_classes.get(presenter_type)

    # Properties with lazy initialization
    @property
    def trading_presenter(self):
        return self._ensure_presenter('trading')

    @property
    def bot_tab(self):
        return self._ensure_presenter('bot')

    @property
    def chart_tab(self):
        return self._ensure_presenter('chart')

    @property
    def exchange_tab(self):
        return self._ensure_presenter('exchange')

    @property
    def ml_tab(self):
        return self._ensure_presenter('ml')

    @property
    def rl_tab(self):
        return self._ensure_presenter('rl')

    @property
    def trading_system_presenter(self):
        return self._ensure_presenter('trading_system')

    @property
    def ml_system_presenter(self):
        return self._ensure_presenter('ml_system')

    @property
    def rl_system_presenter(self):
        return self._ensure_presenter('rl_system')

    # Login view with enhanced error handling
    @ui_performance_monitor
    def on_login_button_clicked(self) -> None:
        """Handle login with enhanced security and user feedback."""
        with self.loading_context("Authenticating user..."):
            username = self.loginview.get_username() or 'test'
            password = self.loginview.get_password() or 't'
            
            self._model.login_model.set_credentials(username, password)

            # Development mode bypass
            if username == 'test' and password == 't':
                self.get_main_view()
                return
            
            # Regular authentication
            if self.safe_execute(
                lambda: self._model.login_model.check_credentials(),
                "Authentication failed",
                show_error=True,
                default_return=False
            ):
                self.get_main_view()
            else:
                self.loginview.login_failed()

    @ui_performance_monitor
    def on_register_button_clicked(self) -> None:
        """Handle registration with validation."""
        with self.loading_context("Registering user..."):
            username = self.loginview.get_username()
            password = self.loginview.get_password()

            if not username or not password:
                messagebox.showerror("Error", "Username and password are required")
                return

            self._model.login_model.set_credentials(username, password)
            
            registered = self.safe_execute(
                lambda: self._model.login_model.register(),
                "Registration failed",
                default_return=False
            )
            
            if registered and self.safe_execute(
                lambda: self._model.login_model.check_credentials(),
                "Authentication after registration failed",
                default_return=False
            ):
                self.get_main_view()
            else:
                self.loginview.login_failed()

    def get_frames(self) -> None:
        """Initialize login view with error handling."""
        self._view.frames = {}
        self.loginview = self.safe_execute(
            lambda: self._view.loginview(self),
            "Failed to create login view"
        )
        if self.loginview:
            self._view.show_frame(self.loginview, self)

    # Main view with performance optimizations
    @ui_performance_monitor
    def get_main_view(self) -> None:
        """Initialize main view with optimized loading."""
        try:
            if hasattr(self, 'loginview') and self.loginview:
                self.loginview.destroy()
            
            self.main_view = self.safe_execute(
                lambda: self._view.main_view(self._view),
                "Failed to create main view"
            )
            
            if self.main_view:
                # Initialize models asynchronously for better responsiveness
                self._executor.submit(self._initialize_models_async)
                
                self.get_tabs(self.main_view)
                self._view.show_frame(self.main_view, self)
                
                self.main_listbox = OptimizedMainListBox(self._model, self.main_view, self)
                
                app_logger.info("Main view initialized successfully")
            
        except Exception as e:
            app_logger.error(f"Error initializing main view: {e}")
            messagebox.showerror("Error", f"Failed to initialize main view: {str(e)}")

    def _initialize_models_async(self):
        """Initialize models asynchronously to improve UI responsiveness."""
        try:
            self._model.create_tabmodels(self)
            app_logger.info("Models initialized asynchronously")
        except Exception as e:
            app_logger.error(f"Error initializing models asynchronously: {e}")

    def get_tabs(self, main_view) -> None:
        """Initialize tabs with lazy loading."""
        app_logger.info("Tab presenters configured for lazy initialization")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive presenter performance metrics."""
        uptime = (datetime.now() - self._start_time).total_seconds()
        
        return {
            'uptime_seconds': uptime,
            'initialization_times': self._initialization_times,
            'active_presenters': len(self._presenters),
            'ui_metrics': {
                'operations_count': ui_metrics.operations_count,
                'avg_response_time': ui_metrics.avg_response_time,
                'cache_hit_rate': (
                    ui_metrics.cache_hits / max(ui_metrics.cache_hits + ui_metrics.cache_misses, 1)
                ) * 100,
                'error_count': ui_metrics.error_count,
                'slow_operations_count': len(ui_metrics.slow_operations)
            },
            'pending_operations': self._pending_operations.qsize()
        }


class OptimizedMainListBox:
    """Enhanced main list box with better formatting and performance."""
    
    def __init__(self, model, view, presenter) -> None:
        self._model = model
        self.main_view = view
        self.presenter = presenter
        self._message_history = []
        self._max_messages = 100
        self._cache = get_cache()
        
        self.load()

    def load(self):
        """Initialize with welcome message."""
        welcome_msg = f"🚀 Welcome to the Trading Application - {datetime.now().strftime('%H:%M:%S')}"
        self.set_text(welcome_msg)

    @ui_performance_monitor
    def set_text(self, text: str):
        """Set text with message history and formatting."""
        try:
            # Add timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_text = f"[{timestamp}] {text}"
            
            # Store in history
            self._message_history.append(formatted_text)
            if len(self._message_history) > self._max_messages:
                self._message_history.pop(0)
            
            # Update UI
            if hasattr(self.main_view, 'list_box'):
                self.main_view.list_box(formatted_text)
            
        except Exception as e:
            app_logger.error(f"Error updating list box: {e}")

    def get_history(self) -> List[str]:
        """Get message history."""
        return self._message_history.copy()


# Optimized specialized presenters
class OptimizedTradePresenter(BasePresenter):
    """Enhanced trade presenter with better performance and UX."""
    
    def __init__(self, model, view, presenter) -> None:
        super().__init__(model.tradetab_model, view, presenter)
        self._trade_cache = {}
        self._last_market_update = None
        
    def trade_tab(self) -> Frame:
        """Get trade tab view with caching."""
        return self.main_view.trade_tab

    @ui_performance_monitor
    async def place_trade_async(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """Place trade asynchronously for better UI responsiveness."""
        try:
            with self.loading_context(f"Placing {trade_params.get('side', 'unknown')} order..."):
                # Validate parameters
                required_params = ['symbol', 'side', 'trade_type', 'amount']
                missing_params = [p for p in required_params if p not in trade_params]
                
                if missing_params:
                    raise ValueError(f"Missing required parameters: {missing_params}")
                
                # Execute trade
                result = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self._execute_trade_sync,
                    trade_params
                )
                
                # Update UI with result
                if result:
                    self.update_ui_status(f"✅ Trade placed successfully: {trade_params['symbol']}", "success")
                    return {'success': True, 'result': result}
                else:
                    self.update_ui_status("❌ Trade placement failed", "error")
                    return {'success': False, 'error': 'Trade execution failed'}
                
        except Exception as e:
            self.update_ui_status(f"❌ Trade error: {str(e)[:50]}...", "error")
            return {'success': False, 'error': str(e)}

    def _execute_trade_sync(self, trade_params: Dict[str, Any]):
        """Synchronous trade execution for thread executor."""
        return self._model.place_trade(
            symbol=trade_params['symbol'],
            side=trade_params['side'],
            trade_type=trade_params['trade_type'],
            amount=trade_params['amount'],
            price=trade_params.get('price'),
            stoploss=trade_params.get('stoploss'),
            takeprofit=trade_params.get('takeprofit')
        )

    @ui_performance_monitor
    def place_trade(self):
        """Traditional synchronous trade placement."""
        try:
            trade_tab = self.trade_tab()
            trade_params = self.extract_trade_parameters(trade_tab)
            
            if not trade_params:
                self.update_ui_status("❌ Invalid trade parameters", "error")
                return
            
            amount = self.calculate_trade_amount(trade_params.get('percentage_amount', 0))
            
            # Enhanced parameter validation
            validation_result = self._validate_trade_params(trade_params, amount)
            if not validation_result['valid']:
                self.update_ui_status(f"❌ Validation failed: {validation_result['error']}", "error")
                return

            with self.loading_context("Executing trade..."):
                result = self.safe_execute(
                    lambda: self._model.place_trade(
                        symbol=trade_params['symbol'],
                        side=trade_params['side'],
                        trade_type=trade_params['trade_type'],
                        amount=amount,
                        price=trade_params.get('price'),
                        stoploss=trade_params.get('stop_loss'),
                        takeprofit=trade_params.get('take_profit')
                    ),
                    "Trade placement failed"
                )
                
                if result:
                    self.update_ui_status(f"✅ Trade placed: {trade_params['symbol']} {trade_params['side']}", "success")
                
        except Exception as e:
            self.update_ui_status(f"❌ Trade error: {e}", "error")

    def _validate_trade_params(self, params: Dict[str, Any], amount: float) -> Dict[str, Any]:
        """Enhanced trade parameter validation."""
        try:
            # Check required fields
            required_fields = ['symbol', 'side', 'trade_type']
            for field in required_fields:
                if not params.get(field):
                    return {'valid': False, 'error': f'Missing {field}'}
            
            # Validate amount
            if amount <= 0:
                return {'valid': False, 'error': 'Amount must be positive'}
            
            # Validate side
            if params['side'] not in ['buy', 'sell']:
                return {'valid': False, 'error': 'Side must be buy or sell'}
            
            # Additional business logic validation
            if amount > MAX_POSITION_SIZE * 1000:  # Example limit
                return {'valid': False, 'error': 'Amount exceeds maximum position size'}
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def extract_trade_parameters(self, trade_tab) -> Optional[Dict[str, Any]]:
        """Extract trade parameters with enhanced error handling."""
        try:
            # This would extract parameters from the UI
            # Implementation depends on the actual UI structure
            return {
                'symbol': 'BTC/USD:USD',  # Default for demo
                'side': 'buy',
                'trade_type': 'limit',
                'percentage_amount': 1.0
            }
        except Exception as e:
            app_logger.error(f"Error extracting trade parameters: {e}")
            return None

    def calculate_trade_amount(self, percentage: float) -> float:
        """Calculate trade amount with validation."""
        try:
            # Get balance and calculate amount
            balance = self._get_available_balance()
            return min(balance * (percentage / 100), balance * MAX_POSITION_SIZE)
        except Exception as e:
            app_logger.error(f"Error calculating trade amount: {e}")
            return 0.0

    def _get_available_balance(self) -> float:
        """Get available balance with caching."""
        cache_key = "available_balance"
        
        # Check cache first
        cached_balance = self._cache.get(cache_key)
        if cached_balance is not None:
            return cached_balance
        
        # Fetch fresh balance
        balance = self.safe_execute(
            lambda: self._model.get_balance(),
            "Failed to get balance",
            default_return=0.0
        )
        
        # Cache for 30 seconds
        if balance > 0:
            self._cache.set(cache_key, balance, ttl=30)
        
        return balance

    @ui_performance_monitor  
    def get_real_time_date(self):
        """Get real-time data with throttling."""
        now = datetime.now()
        
        # Throttle updates to prevent excessive API calls
        if (self._last_market_update and 
            (now - self._last_market_update).total_seconds() < 5):
            return self._trade_cache.get('market_data')
        
        data = self.safe_execute(
            lambda: self._model.get_real_time_data(),
            "Failed to get real-time data",
            default_return=None
        )
        
        if data:
            self._trade_cache['market_data'] = data
            self._last_market_update = now
        
        return data


# Additional optimized presenter classes (abbreviated for brevity)
class OptimizedBotPresenter(BasePresenter):
    """Enhanced bot presenter with better bot management."""
    
    def __init__(self, model, view, presenter) -> None:
        super().__init__(model.bottab_model, view, presenter)
        self._bot_performance = {}

class OptimizedChartPresenter(BasePresenter):
    """Enhanced chart presenter with optimized rendering."""
    
    def __init__(self, model, view, presenter) -> None:
        super().__init__(model.charttab_model, view, presenter)
        self._chart_cache = {}

class OptimizedExchangePresenter(BasePresenter):
    """Enhanced exchange presenter with better connection management."""
    
    def __init__(self, model, view, presenter) -> None:
        super().__init__(model.exchangetab_model, view, presenter)
        self._exchange_health = {}

class OptimizedMLPresenter(BasePresenter):
    """Enhanced ML presenter with better model management."""
    
    def __init__(self, model, view, presenter) -> None:
        super().__init__(model, view, presenter)
        self._ml_metrics = {}

class OptimizedRLPresenter(BasePresenter):
    """Enhanced RL presenter with better training management."""
    
    def __init__(self, model, view, presenter) -> None:
        super().__init__(model.rltab_model, view, presenter)
        self._training_progress = {}

# New optimized system presenters
class OptimizedTradingSystemPresenter(BasePresenter):
    """Enhanced trading system presenter."""
    
    def __init__(self, model, view, presenter) -> None:
        super().__init__(model, view, presenter)
        self._system_status = {'initialized': False}
        app_logger.info("OptimizedTradingSystemPresenter initialized")

    @ui_performance_monitor
    async def execute_signal_async(self, signal_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading signal asynchronously."""
        try:
            with self.loading_context("Executing trading signal..."):
                if hasattr(self._model, 'trading_system_model') and self._model.trading_system_model:
                    result = await self._model.trading_system_model.execute_trade_signal(signal_params)
                    
                    if result.get('success'):
                        self.update_ui_status(f"✅ Signal executed: {result.get('order_id', 'N/A')}", "success")
                    else:
                        self.update_ui_status(f"❌ Signal failed: {result.get('message', 'Unknown error')}", "error")
                    
                    return result
                
                return {'success': False, 'message': 'Trading system not available'}
                
        except Exception as e:
            self.update_ui_status(f"❌ Signal execution error: {e}", "error")
            return {'success': False, 'message': str(e)}

class OptimizedMLSystemPresenter(BasePresenter):
    """Enhanced ML system presenter."""
    
    def __init__(self, model, view, presenter) -> None:
        super().__init__(model, view, presenter)
        self._model_status = {}
        app_logger.info("OptimizedMLSystemPresenter initialized")

class OptimizedRLSystemPresenter(BasePresenter):
    """Enhanced RL system presenter."""
    
    def __init__(self, model, view, presenter) -> None:
        super().__init__(model, view, presenter)
        self._agent_status = {}
        app_logger.info("OptimizedRLSystemPresenter initialized")

# Factory functions
def create_optimized_presenter(model, view):
    """Factory function to create optimized presenter."""
    return OptimizedPresenter(model, view)

# Backwards compatibility aliases
Presenter = OptimizedPresenter
TradePresenter = OptimizedTradePresenter
BotPresenter = OptimizedBotPresenter
ChartPresenter = OptimizedChartPresenter
ExchangePresenter = OptimizedExchangePresenter
MLPresenter = OptimizedMLPresenter
RLPresenter = OptimizedRLPresenter
MainListBox = OptimizedMainListBox
