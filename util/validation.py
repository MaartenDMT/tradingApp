"""
Input validation utilities for the Trading Application.

This module provides functions for validating and sanitizing user inputs.
"""

import re
from typing import Any, Optional, Union

def validate_string(value: Any, min_length: int = 1, max_length: int = 255, 
                   pattern: Optional[str] = None, allow_empty: bool = False) -> bool:
    """
    Validate a string input.
    
    Args:
        value (Any): The value to validate
        min_length (int): Minimum length of the string
        max_length (int): Maximum length of the string
        pattern (Optional[str]): Regular expression pattern to match
        allow_empty (bool): Whether to allow empty strings
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if value is None
    if value is None:
        return allow_empty
    
    # Convert to string if not already
    if not isinstance(value, str):
        value = str(value)
    
    # Check for empty string
    if len(value) == 0:
        return allow_empty
    
    # Check length constraints
    if len(value) < min_length or len(value) > max_length:
        return False
    
    # Check pattern if provided
    if pattern and not re.match(pattern, value):
        return False
    
    return True

def validate_number(value: Any, min_value: Optional[Union[int, float]] = None, 
                   max_value: Optional[Union[int, float]] = None, 
                   allow_float: bool = True) -> bool:
    """
    Validate a numeric input.
    
    Args:
        value (Any): The value to validate
        min_value (Optional[Union[int, float]]): Minimum allowed value
        max_value (Optional[Union[int, float]]): Maximum allowed value
        allow_float (bool): Whether to allow floating point numbers
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Try to convert to number
    try:
        if allow_float:
            num_value = float(value)
        else:
            # Check if it's an integer
            if isinstance(value, float) and not value.is_integer():
                return False
            num_value = int(float(value))  # Convert via float to handle string representations
    except (ValueError, TypeError):
        return False
    
    # Check range constraints
    if min_value is not None and num_value < min_value:
        return False
    
    if max_value is not None and num_value > max_value:
        return False
    
    return True

def validate_percentage(value: Any) -> bool:
    """
    Validate a percentage value (0-100).
    
    Args:
        value (Any): The value to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    return validate_number(value, min_value=0, max_value=100, allow_float=True)

def validate_leverage(value: Any) -> bool:
    """
    Validate a leverage value (1-100).
    
    Args:
        value (Any): The value to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    return validate_number(value, min_value=1, max_value=100, allow_float=False)

def validate_symbol(value: Any) -> bool:
    """
    Validate a trading symbol (e.g., BTC/USD:USD).
    
    Args:
        value (Any): The value to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not validate_string(value, min_length=3, max_length=50):
        return False
    
    # Basic symbol pattern: LETTERS/LETTERS or LETTERS/LETTERS:LETTERS
    pattern = r'^[A-Z0-9]+/[A-Z0-9]+(:[A-Z0-9]+)?$'
    return bool(re.match(pattern, str(value)))

def sanitize_string(value: Any, max_length: int = 255) -> str:
    """
    Sanitize a string input by removing potentially dangerous characters.
    
    Args:
        value (Any): The value to sanitize
        max_length (int): Maximum length of the result
        
    Returns:
        str: The sanitized string
    """
    if value is None:
        return ""
    
    # Convert to string
    if not isinstance(value, str):
        value = str(value)
    
    # Remove potentially dangerous characters
    # Allow alphanumeric, spaces, and common punctuation
    sanitized = re.sub(r'[^\w\s\-_/.:]', '', value)
    
    # Limit length
    return sanitized[:max_length]

def validate_and_sanitize_symbol(value: Any) -> Optional[str]:
    """
    Validate and sanitize a trading symbol.
    
    Args:
        value (Any): The value to validate and sanitize
        
    Returns:
        Optional[str]: The sanitized symbol if valid, None otherwise
    """
    # First sanitize
    sanitized = sanitize_string(value, max_length=50)
    
    # Then validate
    if validate_symbol(sanitized):
        return sanitized
    else:
        return None

class InputValidator:
    """
    A class for validating multiple inputs at once.
    """
    
    def __init__(self):
        self.errors = []
    
    def add_error(self, field: str, message: str):
        """Add an error message."""
        self.errors.append(f"{field}: {message}")
    
    def validate_field(self, field_name: str, value: Any, validation_func, 
                      error_message: str, *args, **kwargs) -> bool:
        """
        Validate a single field.
        
        Args:
            field_name (str): Name of the field
            value (Any): Value to validate
            validation_func (callable): Function to use for validation
            error_message (str): Error message if validation fails
            *args, **kwargs: Additional arguments for validation_func
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not validation_func(value, *args, **kwargs):
            self.add_error(field_name, error_message)
            return False
        return True
    
    def is_valid(self) -> bool:
        """Check if all validations passed."""
        return len(self.errors) == 0
    
    def get_errors(self) -> list:
        """Get list of error messages."""
        return self.errors.copy()