"""
Configuration Management Module

This module provides a centralized configuration management system
that consolidates settings from multiple sources:
- config.ini (application settings)
- .env (secrets and environment variables)
- production_config.py (production settings)
- command line arguments
"""

import os
import configparser
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import json

@dataclass
class AppConfig:
    """Main application configuration class."""
    
    # Application settings
    window_size: str = "800x640"
    
    # WebSocket settings
    websocket_retry_delay: int = 10
    websocket_timeout: int = 8
    
    # Data settings
    timeframes: list = field(default_factory=lambda: ["1m", "30m", "1h", "2h", "4h", "1d"])
    symbol: str = "BTCUSDT"
    percentage: int = 15
    
    # Path settings
    data_paths: Dict[str, str] = field(default_factory=dict)
    
    # Model parameters
    gamma: float = 0.95
    learning_rate: float = 0.01
    batch_size: int = 32
    episodes: int = 400
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    min_acc: float = 0.55
    test_episodes: int = 25
    env_actions: int = 5
    hidden_units: int = 24
    dropout: float = 0.25
    loss: str = "mse"
    
    # Environment settings
    patience: int = 8
    shifts: int = -3
    trading_mode: str = "future"
    
    # Trading environment settings
    portfolio_balance: float = 10000.0
    position_size: float = 0.01
    
    # API keys (from environment)
    api_keys: Dict[str, str] = field(default_factory=dict)
    
    # Database settings (from environment)
    database_config: Dict[str, str] = field(default_factory=dict)
    
    # Production settings
    production_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_files(cls, config_path: str = "config.ini", env_path: str = ".env") -> 'AppConfig':
        """Create configuration from config files and environment variables."""
        config = cls()
        
        # Load from config.ini
        if os.path.exists(config_path):
            config_parser = configparser.ConfigParser()
            config_parser.read(config_path)
            
            # Application settings
            if 'Application' in config_parser:
                config.window_size = config_parser['Application'].get('window_size', config.window_size)
            
            # WebSocket settings
            if 'WebSocket' in config_parser:
                config.websocket_retry_delay = config_parser['WebSocket'].getint('retry_delay', config.websocket_retry_delay)
                config.websocket_timeout = config_parser['WebSocket'].getint('timeout', config.websocket_timeout)
            
            # Data settings
            if 'Data' in config_parser:
                timeframes_str = config_parser['Data'].get('timeframes', '')
                if timeframes_str:
                    config.timeframes = [t.strip() for t in timeframes_str.split(',')]
                config.symbol = config_parser['Data'].get('symbol', config.symbol)
                config.percentage = config_parser['Data'].getint('percentage', config.percentage)
            
            # Path settings
            if 'Path' in config_parser:
                config.data_paths = dict(config_parser['Path'])
            
            # Params settings
            if 'Params' in config_parser:
                params = config_parser['Params']
                config.gamma = params.getfloat('gamma', config.gamma)
                config.learning_rate = params.getfloat('learning_rate', config.learning_rate)
                config.batch_size = params.getint('batch_size', config.batch_size)
                config.episodes = params.getint('episodes', config.episodes)
                config.epsilon_min = params.getfloat('epsilon_min', config.epsilon_min)
                config.epsilon_decay = params.getfloat('epsilon_decay', config.epsilon_decay)
                config.min_acc = params.getfloat('min_acc', config.min_acc)
                config.test_episodes = params.getint('test_episodes', config.test_episodes)
                config.env_actions = params.getint('env_actions', config.env_actions)
                config.hidden_units = params.getint('hidden_units', config.hidden_units)
                config.dropout = params.getfloat('dropout', config.dropout)
                config.loss = params.get('loss', config.loss)
            
            # Env settings
            if 'Env' in config_parser:
                env_section = config_parser['Env']
                config.patience = env_section.getint('patience', config.patience)
                config.shifts = env_section.getint('shifts', config.shifts)
                config.trading_mode = env_section.get('trading', config.trading_mode)
            
            # Tradingenv settings
            if 'Tradingenv' in config_parser:
                trading_section = config_parser['Tradingenv']
                config.portfolio_balance = trading_section.getfloat('portfolio_balance', config.portfolio_balance)
                config.position_size = trading_section.getfloat('position_size', config.position_size)
        
        # Load API keys and database settings from environment/.env
        config.api_keys = {
            'BINANCE_API_KEY': os.getenv('API_KEY_BINANCE', ''),
            'BINANCE_SECRET_KEY': os.getenv('API_SECRET_BINANCE', ''),
            'PHEMEX_API_KEY': os.getenv('API_KEY_PHEMEX', ''),
            'PHEMEX_SECRET_KEY': os.getenv('API_SECRET_PHEMEX', ''),
            'COINBASE_API_KEY': os.getenv('API_KEY_COINBASE', ''),
            'COINBASE_SECRET_KEY': os.getenv('API_SECRET_COINBASE', ''),
            'KUCOIN_API_KEY': os.getenv('API_KEY_KUCOIN', ''),
            'KUCOIN_SECRET_KEY': os.getenv('API_SECRET_KUCOIN', ''),
            'KUCOIN_PASSWORD': os.getenv('PASSWORD_KUCOIN', ''),
            'GATE_API_KEY': os.getenv('API_KEY_GATE', ''),
            'GATE_SECRET_KEY': os.getenv('API_SECRET_GATE', ''),
            'BINANCE_TEST_API_KEY': os.getenv('API_KEY_BINANCE_TEST', ''),
            'BINANCE_TEST_SECRET_KEY': os.getenv('API_SECRET_BINANCE_TEST', ''),
            'PHEMEX_TEST_API_KEY': os.getenv('API_KEY_PHEMEX_TEST', ''),
            'PHEMEX_TEST_SECRET_KEY': os.getenv('API_SECRET_PHEMEX_TEST', ''),
            'KUCOIN_TEST_API_KEY': os.getenv('API_KEY_KUCOIN_TEST', ''),
            'KUCOIN_TEST_SECRET_KEY': os.getenv('API_SECRET_KUCOIN_TEST', ''),
            'KUCOIN_TEST_PASSWORD': os.getenv('PASSWORD_KUCOIN_TEST', ''),
        }
        
        config.database_config = {
            'PGUSER': os.getenv('PGUSER', ''),
            'PGPASSWORD': os.getenv('PGPASSWORD', ''),
            'PGHOST': os.getenv('PGHOST', ''),
            'PGPORT': os.getenv('PGPORT', ''),
            'PGDATABASE': os.getenv('PGDATABASE', ''),
        }
        
        # Load production config if available
        try:
            from production_config import PRODUCTION_CONFIG
            config.production_config = PRODUCTION_CONFIG.to_dict()
        except ImportError:
            config.production_config = {}
        
        return config
    
    def get_api_key(self, exchange: str, key_type: str = 'api', is_test: bool = False) -> Optional[str]:
        """Get API key for a specific exchange."""
        key_name = f"{exchange.upper()}_{'TEST_' if is_test else ''}{key_type.upper()}_KEY"
        return self.api_keys.get(key_name)
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        db_config = self.database_config
        if all(db_config.values()):
            return f"postgresql://{db_config['PGUSER']}:{db_config['PGPASSWORD']}@{db_config['PGHOST']}:{db_config['PGPORT']}/{db_config['PGDATABASE']}"
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'window_size': self.window_size,
            'websocket_retry_delay': self.websocket_retry_delay,
            'websocket_timeout': self.websocket_timeout,
            'timeframes': self.timeframes,
            'symbol': self.symbol,
            'percentage': self.percentage,
            'data_paths': self.data_paths,
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'episodes': self.episodes,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'min_acc': self.min_acc,
            'test_episodes': self.test_episodes,
            'env_actions': self.env_actions,
            'hidden_units': self.hidden_units,
            'dropout': self.dropout,
            'loss': self.loss,
            'patience': self.patience,
            'shifts': self.shifts,
            'trading_mode': self.trading_mode,
            'portfolio_balance': self.portfolio_balance,
            'position_size': self.position_size,
            'database_config': self.database_config,  # Note: Not including API keys for security
            'production_config': self.production_config,
        }
    
    def save_to_file(self, filepath: str):
        """Save configuration to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"AppConfig(window_size={self.window_size}, symbol={self.symbol}, timeframes={self.timeframes})"

# Global configuration instance
app_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Get the global application configuration instance."""
    global app_config
    if app_config is None:
        app_config = AppConfig.from_files()
    return app_config

def reload_config() -> AppConfig:
    """Reload the global application configuration."""
    global app_config
    app_config = AppConfig.from_files()
    return app_config