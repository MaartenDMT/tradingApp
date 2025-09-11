"""
Standardized Logging Module

This module provides a standardized logging setup for the trading application
with consistent formatting, configurable levels, and improved performance.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional
from util.config_manager import get_config

# Global logger instances
_loggers: Dict[str, logging.Logger] = {}
_config = get_config()

# Default logging configuration
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 5

# Standard logger names
LOGGER_NAMES = [
    'app',          # Main application logger
    'model',        # Model layer logger
    'view',         # View layer logger
    'presenter',    # Presenter layer logger
    'autobot',      # Autobot system logger
    'tradex',       # Trading execution logger
    'rl_trading',   # RL trading logger
    'rl',           # Reinforcement learning logger
    'env',          # Environment logger
    'agent',        # Agent logger
    'manual',       # Manual trading logger
    'ml',           # Machine learning logger
    'bot',          # Bot system logger
    'exchange',     # Exchange integration logger
    'data',         # Data handling logger
    'security',     # Security-related logger
    'performance',  # Performance monitoring logger
]

def setup_loggers(log_level: Optional[int] = None, 
                  log_format: Optional[str] = None,
                  date_format: Optional[str] = None) -> Dict[str, logging.Logger]:
    """Set up all application loggers with standardized configuration.
    
    Args:
        log_level: Logging level (defaults to INFO or config value)
        log_format: Log message format (defaults to standard format)
        date_format: Date format (defaults to standard format)
        
    Returns:
        Dictionary of configured logger instances
    """
    global _loggers
    
    # Use provided values or defaults
    if log_level is None:
        log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), DEFAULT_LOG_LEVEL)
    
    if log_format is None:
        log_format = DEFAULT_LOG_FORMAT
        
    if date_format is None:
        date_format = DEFAULT_DATE_FORMAT
    
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(_config.data_paths.get('log_file', 'data/logs/app.log'))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Set up each logger
    for logger_name in LOGGER_NAMES:
        if logger_name not in _loggers:
            logger = _setup_individual_logger(
                logger_name, 
                log_level, 
                log_format, 
                date_format
            )
            _loggers[logger_name] = logger
    
    return _loggers

def _setup_individual_logger(logger_name: str,
                            log_level: int,
                            log_format: str,
                            date_format: str) -> logging.Logger:
    """Set up an individual logger with standardized configuration.
    
    Args:
        logger_name: Name of the logger
        log_level: Logging level
        log_format: Log message format
        date_format: Date format
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    
    # Skip if already configured
    if logger.handlers:
        return logger
    
    # Set logger level
    logger.setLevel(log_level)
    
    # Determine log file path
    log_file = _get_log_file_path(logger_name)
    
    # Create file handler with rotation
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=DEFAULT_MAX_BYTES,
            backupCount=DEFAULT_BACKUP_COUNT
        )
        
        # Set formatter
        formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
    except Exception as e:
        # Fallback to console logging if file logging fails
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(log_format, datefmt=date_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.warning(f"Failed to set up file logging for {logger_name}: {e}. Using console logging.")
    
    return logger

def _get_log_file_path(logger_name: str) -> str:
    """Get the log file path for a logger.
    
    Args:
        logger_name: Name of the logger
        
    Returns:
        Log file path
    """
    # Try to get from config first
    log_file_key = f"{logger_name}_log_file"
    if hasattr(_config, log_file_key):
        return getattr(_config, log_file_key)
    
    # Try to get from data paths in config
    data_paths = _config.data_paths
    if f"{logger_name}_log" in data_paths:
        return data_paths[f"{logger_name}_log"]
    
    # Default path
    return f"data/logs/{logger_name}.log"

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance by name.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    global _loggers
    
    if name not in _loggers:
        # If logger not yet set up, create it with default settings
        _loggers[name] = _setup_individual_logger(
            name,
            DEFAULT_LOG_LEVEL,
            DEFAULT_LOG_FORMAT,
            DEFAULT_DATE_FORMAT
        )
    
    return _loggers[name]

def set_log_level(logger_name: str, level: int):
    """Set the logging level for a specific logger.
    
    Args:
        logger_name: Name of the logger
        level: Logging level
    """
    logger = get_logger(logger_name)
    logger.setLevel(level)
    
    # Also set level for all handlers
    for handler in logger.handlers:
        handler.setLevel(level)

def add_console_handler(logger_name: str):
    """Add a console handler to a logger for development/debugging.
    
    Args:
        logger_name: Name of the logger
    """
    logger = get_logger(logger_name)
    
    # Check if console handler already exists
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            return  # Console handler already exists
    
    # Add console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Initialize loggers on module import
setup_loggers()