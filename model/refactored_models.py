"""
Refactored Models Module

This module provides refactored and enhanced versions of the trading application models
with improved organization, error handling, and performance optimizations.
"""

import asyncio
import os
import threading
import traceback
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from time import sleep
from typing import Any, Dict, List, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import ccxt
import pandas as pd
from dotenv import load_dotenv

import util.loggers as loggers
from util.config_manager import get_config
from model.base_model import BaseModel

# Import legacy components with error handling
try:
    from model.features import Tradex_indicator
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    Tradex_indicator = None

try:
    from model.machinelearning.autobot import AutoBot
    from model.machinelearning.machinelearning import MachineLearning
    LEGACY_ML_AVAILABLE = True
except ImportError:
    LEGACY_ML_AVAILABLE = False
    AutoBot = None
    MachineLearning = None

try:
    from model.manualtrading.trading import Trading
    MANUAL_TRADING_AVAILABLE = True
except ImportError:
    MANUAL_TRADING_AVAILABLE = False
    Trading = None

try:
    from model.reinforcement.agents.agent_manager import StablebaselineModel
    LEGACY_RL_AVAILABLE = True
except ImportError:
    LEGACY_RL_AVAILABLE = False
    StablebaselineModel = None

# Import new optimized systems with error handling
try:
    from trading.core import TradingSystem, TradingConfig
    TRADING_SYSTEM_AVAILABLE = True
except ImportError:
    TRADING_SYSTEM_AVAILABLE = False

try:
    from model.ml_system.core import MLSystem
    from model.ml_system.config.ml_config import MLConfig
    ML_SYSTEM_AVAILABLE = True
except ImportError:
    ML_SYSTEM_AVAILABLE = False

try:
    from model.rl_system.integration import RLSystemManager
    RL_SYSTEM_AVAILABLE = True
except ImportError:
    RL_SYSTEM_AVAILABLE = False

# Load configuration and environment
config = get_config()
dotenv_path = r'.env'
load_dotenv(dotenv_path)

logger = loggers.setup_loggers()

# Performance monitoring and metrics
@dataclass
class ModelMetrics:
    """Metrics tracking for model performance."""
    creation_count: int = 0
    initialization_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    last_error: Optional[str] = None


class OptimizedModels(BaseModel):
    """Main models class that coordinates all trading functionality."""
    
    def __init__(self):
        """Initialize the optimized models."""
        super().__init__("OptimizedModels")
        self.metrics = ModelMetrics()
        self._lock = threading.RLock()
        self._cache = weakref.WeakValueDictionary()
        
        # Initialize subsystems
        self.login_model = None
        self.trading_model = None
        self.bot_model = None
        self.ml_model = None
        self.rl_model = None
        self.features_model = None
        
    def _initialize(self):
        """Internal initialization method."""
        start_time = datetime.now()
        
        try:
            # Initialize login model
            self.login_model = LoginModel()
            self.login_model.initialize()
            
            # Initialize trading model
            if MANUAL_TRADING_AVAILABLE:
                self.trading_model = TradingModel()
                self.trading_model.initialize()
            
            # Initialize bot model
            self.bot_model = BotModel()
            self.bot_model.initialize()
            
            # Initialize ML model
            if ML_SYSTEM_AVAILABLE:
                self.ml_model = MLModel()
                self.ml_model.initialize()
            
            # Initialize RL model
            if RL_SYSTEM_AVAILABLE:
                self.rl_model = RLModel()
                self.rl_model.initialize()
            
            # Initialize features model
            if FEATURES_AVAILABLE:
                self.features_model = FeaturesModel()
                self.features_model.initialize()
                
            # Calculate initialization time
            end_time = datetime.now()
            self.metrics.initialization_time = (end_time - start_time).total_seconds()
            
        except Exception as e:
            self.handle_exception(e, "model initialization")
            raise
    
    def get_exchange(self, exchange_name: str, is_test: bool = False) -> Optional[ccxt.Exchange]:
        """Get an exchange instance.
        
        Args:
            exchange_name: Name of the exchange (e.g., 'binance', 'phemex')
            is_test: Whether to use testnet
            
        Returns:
            Exchange instance or None if failed
        """
        with self._lock:
            cache_key = f"exchange_{exchange_name}_{is_test}"
            if cache_key in self._cache:
                self.metrics.cache_hits += 1
                return self._cache[cache_key]
            
            self.metrics.cache_misses += 1
            
            try:
                # Get API keys from config
                api_key = config.get_api_key(exchange_name, 'api', is_test)
                secret_key = config.get_api_key(exchange_name, 'secret', is_test)
                password = config.get_api_key(exchange_name, 'password', is_test)
                
                if not api_key or not secret_key:
                    self.logger.warning(f"API keys not found for {exchange_name}")
                    return None
                
                # Create exchange instance
                exchange_class = getattr(ccxt, exchange_name.lower())
                exchange_options = {
                    'apiKey': api_key,
                    'secret': secret_key,
                }
                
                if password:
                    exchange_options['password'] = password
                    
                if is_test:
                    exchange_options['enableRateLimit'] = True
                    # Add testnet URLs if available
                    if exchange_name.lower() == 'binance':
                        exchange_options['urls'] = {
                            'api': {
                                'public': 'https://testnet.binance.vision/api/',
                                'private': 'https://testnet.binance.vision/api/',
                                'v3': 'https://testnet.binance.vision/api/v3/',
                                'v1': 'https://testnet.binance.vision/api/v1/'
                            }
                        }
                
                exchange = exchange_class(exchange_options)
                
                # Test the connection
                exchange.load_markets()
                
                # Cache the exchange instance
                self._cache[cache_key] = exchange
                
                return exchange
                
            except Exception as e:
                self.handle_exception(e, f"exchange creation for {exchange_name}")
                return None


# Individual model classes
class LoginModel(BaseModel):
    """Model for handling user authentication."""
    
    def __init__(self):
        super().__init__("LoginModel")
        self._username = None
        self._password = None
        self._authenticated = False
        
    def _initialize(self):
        """Internal initialization method."""
        # No special initialization needed
        pass
        
    def set_credentials(self, username: str, password: str):
        """Set user credentials.
        
        Args:
            username: Username
            password: Password
        """
        self._username = username
        self._password = password
        
    def authenticate(self) -> bool:
        """Authenticate the user.
        
        Returns:
            bool: True if authentication was successful, False otherwise
        """
        try:
            # In a real application, this would check against a database
            # For now, we'll use the default test credentials
            if self._username == 'test' and self._password == 't':
                self._authenticated = True
                self.logger.info("User authenticated successfully")
                return True
            else:
                self._authenticated = False
                self.logger.warning("Authentication failed")
                return False
        except Exception as e:
            self.handle_exception(e, "authentication")
            return False
            
    def is_authenticated(self) -> bool:
        """Check if the user is authenticated.
        
        Returns:
            bool: True if authenticated, False otherwise
        """
        return self._authenticated


class TradingModel(BaseModel):
    """Model for manual trading functionality."""
    
    def __init__(self):
        super().__init__("TradingModel")
        self.trading_instance = None
        
    def _initialize(self):
        """Internal initialization method."""
        if MANUAL_TRADING_AVAILABLE and Trading:
            try:
                self.trading_instance = Trading()
                self.logger.info("Trading instance initialized")
            except Exception as e:
                self.handle_exception(e, "trading instance initialization")
                self.trading_instance = None
        else:
            self.logger.warning("Manual trading functionality not available")
            
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
        if not self.trading_instance:
            self.logger.error("Trading instance not available")
            return False
            
        try:
            if price:
                result = self.trading_instance.limit_trade(symbol, side, amount, price)
            else:
                result = self.trading_instance.market_trade(symbol, side, amount)
                
            if result:
                self.logger.info(f"Trade executed: {side} {amount} {symbol}{' at ' + str(price) if price else ''}")
                return True
            else:
                self.logger.error(f"Failed to execute trade: {side} {amount} {symbol}")
                return False
        except Exception as e:
            self.handle_exception(e, "trade execution")
            return False


class BotModel(BaseModel):
    """Model for bot management functionality."""
    
    def __init__(self):
        super().__init__("BotModel")
        self.bot_system = None
        
    def _initialize(self):
        """Internal initialization method."""
        try:
            from model.bot_system import BotSystem
            self.bot_system = BotSystem()
            self.logger.info("Bot system initialized")
        except ImportError:
            self.logger.warning("Bot system not available")
            self.bot_system = None
        except Exception as e:
            self.handle_exception(e, "bot system initialization")
            self.bot_system = None
            
    def create_bot(self, name: str, strategy: str, parameters: Dict[str, Any]) -> bool:
        """Create a new trading bot.
        
        Args:
            name: Bot name
            strategy: Trading strategy
            parameters: Strategy parameters
            
        Returns:
            bool: True if bot was created successfully, False otherwise
        """
        if not self.bot_system:
            self.logger.error("Bot system not available")
            return False
            
        try:
            result = self.bot_system.create_bot(name, strategy, parameters)
            if result:
                self.logger.info(f"Bot created: {name}")
                return True
            else:
                self.logger.error(f"Failed to create bot: {name}")
                return False
        except Exception as e:
            self.handle_exception(e, "bot creation")
            return False


class MLModel(BaseModel):
    """Model for machine learning functionality."""
    
    def __init__(self):
        super().__init__("MLModel")
        self.ml_system = None
        
    def _initialize(self):
        """Internal initialization method."""
        if ML_SYSTEM_AVAILABLE:
            try:
                self.ml_system = MLSystem()
                self.logger.info("ML system initialized")
            except Exception as e:
                self.handle_exception(e, "ML system initialization")
                self.ml_system = None
        else:
            self.logger.warning("ML system not available")
            
    def train_model(self, data: pd.DataFrame, target: str) -> bool:
        """Train a machine learning model.
        
        Args:
            data: Training data
            target: Target column name
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        if not self.ml_system:
            self.logger.error("ML system not available")
            return False
            
        try:
            result = self.ml_system.train(data, target)
            if result:
                self.logger.info("ML model trained successfully")
                return True
            else:
                self.logger.error("Failed to train ML model")
                return False
        except Exception as e:
            self.handle_exception(e, "ML model training")
            return False


class RLModel(BaseModel):
    """Model for reinforcement learning functionality."""
    
    def __init__(self):
        super().__init__("RLModel")
        self.rl_system = None
        
    def _initialize(self):
        """Internal initialization method."""
        if RL_SYSTEM_AVAILABLE:
            try:
                self.rl_system = RLSystemManager()
                self.logger.info("RL system initialized")
            except Exception as e:
                self.handle_exception(e, "RL system initialization")
                self.rl_system = None
        else:
            self.logger.warning("RL system not available")
            
    def train_agent(self, environment: str, episodes: int) -> bool:
        """Train a reinforcement learning agent.
        
        Args:
            environment: Environment name
            episodes: Number of training episodes
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        if not self.rl_system:
            self.logger.error("RL system not available")
            return False
            
        try:
            result = self.rl_system.train(environment, episodes)
            if result:
                self.logger.info("RL agent trained successfully")
                return True
            else:
                self.logger.error("Failed to train RL agent")
                return False
        except Exception as e:
            self.handle_exception(e, "RL agent training")
            return False


class FeaturesModel(BaseModel):
    """Model for technical analysis features."""
    
    def __init__(self):
        super().__init__("FeaturesModel")
        self.indicator = None
        
    def _initialize(self):
        """Internal initialization method."""
        if FEATURES_AVAILABLE and Tradex_indicator:
            try:
                self.indicator = Tradex_indicator()
                self.logger.info("Features indicator initialized")
            except Exception as e:
                self.handle_exception(e, "features indicator initialization")
                self.indicator = None
        else:
            self.logger.warning("Features functionality not available")
            
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators.
        
        Args:
            data: Input data
            
        Returns:
            DataFrame with calculated indicators
        """
        if not self.indicator:
            self.logger.error("Features indicator not available")
            return data
            
        try:
            result = self.indicator.calculate(data)
            self.logger.info("Indicators calculated successfully")
            return result
        except Exception as e:
            self.handle_exception(e, "indicator calculation")
            return data