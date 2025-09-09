"""
Trading System Configuration

Comprehensive configuration management for the trading system with
validation, default values, and environment-specific settings.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal

from .types import OrderType, TimeInForce, RiskLevel


@dataclass
class ExchangeConfig:
    """Exchange-specific configuration."""
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    sandbox: bool = True
    rate_limit: bool = True
    enable_rate_limit: bool = True
    timeout: int = 30000
    default_type: str = "spot"  # spot, margin, future, swap
    options: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Load API credentials from environment if not provided."""
        if not self.api_key:
            self.api_key = os.getenv(f'API_KEY_{self.name.upper()}{"_TEST" if self.sandbox else ""}')
        if not self.api_secret:
            self.api_secret = os.getenv(f'API_SECRET_{self.name.upper()}{"_TEST" if self.sandbox else ""}')


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_risk_per_trade: float = 0.02  # 2% of capital per trade
    max_portfolio_risk: float = 0.1   # 10% of capital total
    max_positions: int = 10
    max_correlation: float = 0.7      # Maximum correlation between positions
    stop_loss_enabled: bool = True
    take_profit_enabled: bool = True
    trailing_stop_enabled: bool = False
    default_stop_loss_pct: float = 0.05  # 5%
    default_take_profit_pct: float = 0.10  # 10%
    risk_free_rate: float = 0.02      # 2% annual risk-free rate
    var_confidence_level: float = 0.95 # 95% VaR confidence
    max_drawdown_threshold: float = 0.2  # 20% max drawdown before stopping
    kelly_criterion_enabled: bool = False
    position_sizing_method: str = "fixed_fractional"  # fixed_fractional, kelly, volatility_adjusted
    
    def validate(self) -> bool:
        """Validate risk configuration values."""
        if not (0 < self.max_risk_per_trade <= 0.1):
            raise ValueError("max_risk_per_trade must be between 0 and 10%")
        if not (0 < self.max_portfolio_risk <= 0.5):
            raise ValueError("max_portfolio_risk must be between 0 and 50%")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")
        return True


@dataclass
class TradingConfig:
    """Main trading system configuration."""
    
    # Exchange Configuration
    exchange: ExchangeConfig = field(default_factory=lambda: ExchangeConfig("phemex"))
    
    # Risk Management
    risk: RiskConfig = field(default_factory=RiskConfig)
    
    # Trading Parameters
    default_symbol: str = "BTC/USD:USD"
    default_timeframe: str = "1h"
    default_order_type: OrderType = OrderType.LIMIT
    default_time_in_force: TimeInForce = TimeInForce.GTC
    
    # Portfolio Settings
    initial_capital: Union[float, Decimal] = Decimal('10000')
    base_currency: str = "USD"
    quote_currencies: List[str] = field(default_factory=lambda: ["USD", "USDT", "BUSD"])
    
    # Execution Settings
    slippage_tolerance: float = 0.001  # 0.1% slippage tolerance
    min_order_value: Union[float, Decimal] = Decimal('10')
    max_order_retries: int = 3
    order_timeout: int = 30  # seconds
    
    # Data Settings
    enable_real_time_data: bool = True
    data_buffer_size: int = 1000
    price_precision: int = 8
    amount_precision: int = 6
    
    # Logging and Monitoring
    log_level: str = "INFO"
    log_trades: bool = True
    log_signals: bool = True
    enable_performance_tracking: bool = True
    save_trade_history: bool = True
    
    # Strategy Settings
    max_active_strategies: int = 5
    strategy_rebalance_frequency: str = "1h"  # How often to rebalance strategy allocations
    
    # Advanced Settings
    enable_partial_fills: bool = True
    enable_advanced_orders: bool = True
    enable_portfolio_margin: bool = False
    margin_ratio: float = 0.1  # 10x leverage maximum
    
    # Performance Settings
    use_async_execution: bool = True
    max_concurrent_orders: int = 20
    cache_market_data: bool = True
    cache_duration: int = 60  # seconds
    
    # Development and Testing
    paper_trading: bool = True
    backtest_mode: bool = False
    enable_debug_mode: bool = False
    save_debug_data: bool = False
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self.validate()
        self.setup_environment()
    
    def validate(self) -> bool:
        """Validate all configuration values."""
        # Validate risk settings
        self.risk.validate()
        
        # Validate trading parameters
        if self.slippage_tolerance < 0 or self.slippage_tolerance > 0.1:
            raise ValueError("slippage_tolerance must be between 0 and 10%")
        
        if float(self.min_order_value) <= 0:
            raise ValueError("min_order_value must be positive")
        
        if float(self.initial_capital) <= 0:
            raise ValueError("initial_capital must be positive")
        
        if self.max_order_retries <= 0:
            raise ValueError("max_order_retries must be positive")
        
        # Validate precision settings
        if self.price_precision < 0 or self.price_precision > 18:
            raise ValueError("price_precision must be between 0 and 18")
        
        if self.amount_precision < 0 or self.amount_precision > 18:
            raise ValueError("amount_precision must be between 0 and 18")
        
        return True
    
    def setup_environment(self):
        """Set up environment-specific settings."""
        # Override settings based on environment
        env = os.getenv('TRADING_ENV', 'development')
        
        if env == 'production':
            self.exchange.sandbox = False
            self.paper_trading = False
            self.enable_debug_mode = False
            self.log_level = "WARNING"
        elif env == 'testing':
            self.exchange.sandbox = True
            self.paper_trading = True
            self.save_trade_history = False
            self.enable_debug_mode = True
        else:  # development
            self.exchange.sandbox = True
            self.paper_trading = True
            self.enable_debug_mode = True
            self.log_level = "DEBUG"
    
    def get_symbol_config(self, symbol: str) -> Dict[str, Any]:
        """Get symbol-specific configuration."""
        # This could be extended to load symbol-specific settings
        # from a database or configuration file
        return {
            'symbol': symbol,
            'min_order_size': self.min_order_value,
            'price_precision': self.price_precision,
            'amount_precision': self.amount_precision,
            'slippage_tolerance': self.slippage_tolerance
        }
    
    def get_risk_config_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """Get risk configuration specific to a symbol."""
        base_config = {
            'max_risk_per_trade': self.risk.max_risk_per_trade,
            'stop_loss_enabled': self.risk.stop_loss_enabled,
            'take_profit_enabled': self.risk.take_profit_enabled,
            'default_stop_loss_pct': self.risk.default_stop_loss_pct,
            'default_take_profit_pct': self.risk.default_take_profit_pct
        }
        
        # Symbol-specific overrides could be added here
        # For example, higher risk for more volatile assets
        volatility_multipliers = {
            'BTC/USD:USD': 1.0,
            'ETH/USD:USD': 1.2,
            'ADA/USD:USD': 1.5,
            'DOGE/USD:USD': 2.0
        }
        
        multiplier = volatility_multipliers.get(symbol, 1.0)
        base_config['default_stop_loss_pct'] *= multiplier
        
        return base_config
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> 'TradingConfig':
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                if key == 'exchange' and isinstance(value, dict):
                    # Update exchange config
                    for ex_key, ex_value in value.items():
                        if hasattr(self.exchange, ex_key):
                            setattr(self.exchange, ex_key, ex_value)
                elif key == 'risk' and isinstance(value, dict):
                    # Update risk config
                    for risk_key, risk_value in value.items():
                        if hasattr(self.risk, risk_key):
                            setattr(self.risk, risk_key, risk_value)
                else:
                    setattr(self, key, value)
        
        # Re-validate after updates
        self.validate()
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'exchange': {
                'name': self.exchange.name,
                'sandbox': self.exchange.sandbox,
                'rate_limit': self.exchange.rate_limit,
                'timeout': self.exchange.timeout,
                'default_type': self.exchange.default_type
            },
            'risk': {
                'max_risk_per_trade': self.risk.max_risk_per_trade,
                'max_portfolio_risk': self.risk.max_portfolio_risk,
                'max_positions': self.risk.max_positions,
                'stop_loss_enabled': self.risk.stop_loss_enabled,
                'take_profit_enabled': self.risk.take_profit_enabled
            },
            'default_symbol': self.default_symbol,
            'default_timeframe': self.default_timeframe,
            'initial_capital': float(self.initial_capital),
            'base_currency': self.base_currency,
            'paper_trading': self.paper_trading,
            'enable_real_time_data': self.enable_real_time_data,
            'log_level': self.log_level
        }


def create_default_config() -> TradingConfig:
    """Create a default trading configuration."""
    return TradingConfig()


def create_production_config() -> TradingConfig:
    """Create a production trading configuration."""
    config = TradingConfig()
    config.exchange.sandbox = False
    config.paper_trading = False
    config.enable_debug_mode = False
    config.log_level = "INFO"
    config.risk.max_risk_per_trade = 0.01  # More conservative in production
    return config


def create_backtest_config() -> TradingConfig:
    """Create a configuration for backtesting."""
    config = TradingConfig()
    config.backtest_mode = True
    config.paper_trading = True
    config.enable_real_time_data = False
    config.save_trade_history = True
    config.enable_performance_tracking = True
    return config
