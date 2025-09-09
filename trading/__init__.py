"""
Optimized Trading System

A comprehensive, modular trading system with clear separation of concerns,
professional architecture, and extensible design patterns.

This system incorporates best practices from the ML and RL systems while
providing a clean, maintainable structure for trading operations.
"""

from .core import TradingSystem, TradingConfig, TradeResult
from .execution import OrderExecutor, OrderManager, TradeExecutor
from .risk_management import RiskManager, PositionSizer, RiskAssessment
from .strategies import BaseStrategy, TechnicalStrategy, SentimentStrategy
from .market_data import MarketDataProvider, RealTimeDataHandler, DataValidator
from .portfolio import Portfolio, PortfolioManager, PerformanceTracker
from .utils import TradingUtils, PriceCalculator, ValidationUtils

__version__ = "1.0.0"

__all__ = [
    # Core components
    'TradingSystem',
    'TradingConfig', 
    'TradeResult',
    
    # Execution
    'OrderExecutor',
    'OrderManager',
    'TradeExecutor',
    
    # Risk Management
    'RiskManager',
    'PositionSizer',
    'RiskAssessment',
    
    # Strategies
    'BaseStrategy',
    'TechnicalStrategy',
    'SentimentStrategy',
    
    # Market Data
    'MarketDataProvider',
    'RealTimeDataHandler',
    'DataValidator',
    
    # Portfolio Management
    'Portfolio',
    'PortfolioManager',
    'PerformanceTracker',
    
    # Utilities
    'TradingUtils',
    'PriceCalculator',
    'ValidationUtils',
]
