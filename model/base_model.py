"""
Base Model Module

This module provides a base model class that all other models should inherit from.
It includes common functionality like logging, error handling, and configuration management.
"""

import logging
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from util.config_manager import get_config
from util.loggers import setup_loggers

class BaseModel(ABC):
    """Base class for all models in the trading application."""
    
    def __init__(self, name: str):
        """Initialize the base model.
        
        Args:
            name: Name of the model for logging purposes
        """
        self.name = name
        self.config = get_config()
        self.logger = self._setup_logger()
        self._initialized = False
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for this model."""
        loggers = setup_loggers()
        # Use the model name as the logger name, or fall back to 'model'
        return loggers.get(self.name.lower(), loggers.get('model', logging.getLogger(__name__)))
    
    def initialize(self) -> bool:
        """Initialize the model. This method should be called after creation.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            if not self._initialized:
                self.logger.info(f"Initializing {self.name} model")
                self._initialize()
                self._initialized = True
                self.logger.info(f"Successfully initialized {self.name} model")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.name} model: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    @abstractmethod
    def _initialize(self):
        """Internal initialization method to be implemented by subclasses."""
        pass
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key (e.g., 'trading_mode', 'portfolio_balance')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        # Try to get from model-specific config first
        if hasattr(self.config, key):
            return getattr(self.config, key)
        
        # Try to get from config dict
        config_dict = self.config.to_dict()
        return config_dict.get(key, default)
    
    def handle_exception(self, e: Exception, context: str = "") -> None:
        """Handle an exception with proper logging.
        
        Args:
            e: The exception that occurred
            context: Additional context about where the exception occurred
        """
        message = f"Exception in {self.name}"
        if context:
            message += f" ({context})"
        message += f": {str(e)}"
        
        self.logger.error(message)
        self.logger.debug(traceback.format_exc())
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(name={self.name})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return f"{self.__class__.__module__}.{self.__class__.__name__}(name={self.name})"