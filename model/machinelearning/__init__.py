"""
Machine Learning Module for Trading Application

This module provides comprehensive machine learning functionality including:
- Optimized model training and prediction
- Async operations for better performance
- Model caching and persistence
- Performance monitoring and evaluation
- Automated trading with ML predictions
"""

from typing import Any, Dict, List, Optional

from .autobot import AutoBot
# Core ML components
from .machinelearning import MachineLearning
from .ml_models import get_model, get_model_parameters
from .ml_util import (classifier, column_1d, future_score_clas,
                      future_score_reg, regression, spot_score_clas,
                      spot_score_reg)

# Export commonly used functions and classes
__all__ = [
    # Core ML components
    'MachineLearning',
    'AutoBot',
    # Module management
    'MLModuleManager',
    'get_ml_manager',
    'create_optimized_ml_pipeline',
    # Configuration
    'DEFAULT_ML_CONFIG',
    'OPTIMIZATIONS_AVAILABLE',
    # ML model functions
    'get_model',
    'get_model_parameters',
    # ML utilities
    'classifier',
    'column_1d',
    'future_score_clas',
    'future_score_reg',
    'regression',
    'spot_score_clas',
    'spot_score_reg'
]

# Import optimized utilities
try:
    from ..util.cache import HybridCache, get_global_cache
    from ..util.ml_optimization import MLConfig, OptimizedMLPipeline
    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False

# Version information
__version__ = "2.0.0"
__author__ = "Trading App Team"

# Module configuration
DEFAULT_ML_CONFIG = {
    "cache_enabled": True,
    "async_enabled": True,
    "parallel_processing": True,
    "model_persistence": True,
    "performance_monitoring": True
}

# Global cache for ML models and predictions
_ml_cache: Optional[HybridCache] = None
_ml_models_cache: Dict[str, Any] = {}


async def initialize_ml_module(config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize the ML module with optimizations."""
    global _ml_cache

    if config is None:
        config = DEFAULT_ML_CONFIG

    if config.get("cache_enabled", True) and OPTIMIZATIONS_AVAILABLE:
        try:
            _ml_cache = await get_global_cache()
        except Exception:
            _ml_cache = None


async def get_ml_cache() -> Optional[HybridCache]:
    """Get the ML cache instance."""
    global _ml_cache
    if _ml_cache is None and OPTIMIZATIONS_AVAILABLE:
        await initialize_ml_module()
    return _ml_cache


def get_available_models() -> Dict[str, List[str]]:
    """Get list of available ML models."""
    return {
        "classifiers": classifier,
        "regressors": regression,
        "column_1d_models": column_1d
    }


def create_optimized_ml_pipeline(config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """Create an optimized ML pipeline if available."""
    if not OPTIMIZATIONS_AVAILABLE:
        return None

    try:
        ml_config = MLConfig(**(config or {}))
        return OptimizedMLPipeline(config=ml_config)
    except Exception:
        return None


class MLModuleManager:
    """Manager for ML module operations and optimizations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or DEFAULT_ML_CONFIG
        self._initialized = False
        self._ml_instances: Dict[str, MachineLearning] = {}
        self._autobot_instances: Dict[str, AutoBot] = {}

    async def initialize(self) -> None:
        """Initialize the ML module manager."""
        if self._initialized:
            return

        await initialize_ml_module(self.config)
        self._initialized = True

    def get_ml_instance(self, exchange, symbol: str) -> MachineLearning:
        """Get or create a MachineLearning instance."""
        key = f"{exchange}_{symbol}"
        if key not in self._ml_instances:
            self._ml_instances[key] = MachineLearning(exchange, symbol)
        return self._ml_instances[key]

    def get_autobot_instance(
        self,
        exchange,
        symbol: str,
        amount: float,
        stop_loss: float,
        take_profit: float,
        model,
        timeframe: str,
        ml,
        trade_x,
        df
    ) -> AutoBot:
        """Get or create an AutoBot instance."""
        key = f"{exchange}_{symbol}_{timeframe}"
        if key not in self._autobot_instances:
            self._autobot_instances[key] = AutoBot(
                exchange, symbol, amount, stop_loss, take_profit,
                model, timeframe, ml, trade_x, df
            )
        return self._autobot_instances[key]

    async def cleanup(self) -> None:
        """Cleanup ML module resources."""
        # Clean up instances
        for autobot in self._autobot_instances.values():
            try:
                if hasattr(autobot, 'cleanup'):
                    await autobot.cleanup()
            except Exception:
                pass

        self._ml_instances.clear()
        self._autobot_instances.clear()
        self._initialized = False


# Global manager instance
_ml_manager: Optional[MLModuleManager] = None


def get_ml_manager(config: Optional[Dict[str, Any]] = None) -> MLModuleManager:
    """Get or create the global ML manager."""
    global _ml_manager
    if _ml_manager is None:
        _ml_manager = MLModuleManager(config)
    return _ml_manager
