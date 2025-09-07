"""
Secure configuration loader for the Trading Application.

This module provides functions for loading configuration settings while
protecting sensitive information like API keys and passwords.
"""

import os
from configparser import ConfigParser
from util.secure_credentials import mask_sensitive_data

def load_secure_config(config_file: str = 'config.ini') -> dict:
    """
    Load configuration from a file, masking sensitive values.
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary with masked sensitive values
    """
    config = ConfigParser()
    config.read(config_file)
    
    # Convert to dictionary format
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for key, value in config.items(section):
            # Mask sensitive values
            if 'key' in key.lower() or 'secret' in key.lower() or 'password' in key.lower():
                config_dict[section][key] = mask_sensitive_data(value)
            else:
                config_dict[section][key] = value
                
    return config_dict

def get_secure_config_value(section: str, key: str, default=None) -> str:
    """
    Get a configuration value, masking it if it's sensitive.
    
    Args:
        section (str): Configuration section name
        key (str): Configuration key name
        default: Default value if key is not found
        
    Returns:
        str: Configuration value (masked if sensitive)
    """
    config = ConfigParser()
    config.read('config.ini')
    
    try:
        value = config.get(section, key)
        # Mask sensitive values
        if 'key' in key.lower() or 'secret' in key.lower() or 'password' in key.lower():
            return mask_sensitive_data(value)
        return value
    except:
        return default

def validate_config() -> list:
    """
    Validate the configuration and return a list of issues.
    
    Returns:
        list: List of configuration issues
    """
    issues = []
    config = ConfigParser()
    config.read('config.ini')
    
    # Check for required sections
    required_sections = ['Application', 'WebSocket', 'Data', 'Params', 'Env', 'Tradingenv']
    for section in required_sections:
        if not config.has_section(section):
            issues.append(f"Missing required section: {section}")
    
    # Check for sensitive data in config file (should be in environment variables)
    sensitive_keys = ['api_key', 'api_secret', 'password']
    for section in config.sections():
        for key in config.options(section):
            value = config.get(section, key)
            for sensitive in sensitive_keys:
                if sensitive in key.lower() and value.strip():
                    issues.append(f"Sensitive data found in config file: [{section}] {key}. "
                                "This should be stored in environment variables instead.")
    
    return issues