"""
Error handling utilities for the Trading Application.

This module provides consistent error handling patterns across the application.
"""

import traceback
import logging

def handle_exception(logger: logging.Logger, operation: str, exception: Exception, 
                    rethrow: bool = False, default_return = None):
    """
    Handle an exception with consistent logging and optional rethrow.
    
    Args:
        logger (logging.Logger): The logger to use for logging
        operation (str): Description of the operation that failed
        exception (Exception): The exception that occurred
        rethrow (bool): Whether to rethrow the exception
        default_return: What to return if not rethrowing
        
    Returns:
        The default_return value if not rethrowing
        
    Raises:
        The original exception if rethrow is True
    """
    logger.error(f"Error during {operation}: {exception}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    if rethrow:
        raise exception
    else:
        return default_return

def safe_execute(logger: logging.Logger, operation: str, func, *args, **kwargs):
    """
    Safely execute a function with error handling.
    
    Args:
        logger (logging.Logger): The logger to use for logging
        operation (str): Description of the operation being executed
        func (callable): The function to execute
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function if successful, None otherwise
    """
    try:
        logger.debug(f"Executing {operation}")
        result = func(*args, **kwargs)
        logger.debug(f"Successfully executed {operation}")
        return result
    except Exception as e:
        return handle_exception(logger, operation, e, rethrow=False)

class SafeExecutor:
    """
    A context manager for safe execution of code blocks.
    """
    
    def __init__(self, logger: logging.Logger, operation: str, 
                 rethrow: bool = False, default_return = None):
        self.logger = logger
        self.operation = operation
        self.rethrow = rethrow
        self.default_return = default_return
    
    def __enter__(self):
        self.logger.debug(f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            handle_exception(self.logger, self.operation, exc_value, 
                           self.rethrow, self.default_return)
            return not self.rethrow  # Suppress exception if not rethrowing
        self.logger.debug(f"Successfully completed {self.operation}")
        return True