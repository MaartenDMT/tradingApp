"""
ML System __init__.py

Main module initialization for the ML system.
Provides easy access to core classes and functions.
"""

# Algorithms
from .algorithms.registry import AlgorithmRegistry
from .config.hyperparameter_config import HyperparameterConfig

# Configuration
from .config.ml_config import MLConfig
from .core.base_estimator import BaseMLEstimator

# Core ML System
from .core.ml_system import MLSystem
from .core.pipeline_manager import PipelineManager

# Training components
from .training.hyperparameter_optimizer import HyperparameterOptimizer
from .training.model_evaluator import ModelEvaluator
from .training.model_persistence import ModelPersistence
from .utils.data_validator import DataValidator

# Utilities
from .utils.feature_engineering import FeatureEngineer
from .utils.performance_tracker import PerformanceTracker

# Version information
__version__ = "1.0.0"
__author__ = "Trading App Development Team"

# Module metadata
__all__ = [
    # Core classes
    'MLSystem',
    'BaseMLEstimator',
    'PipelineManager',

    # Configuration
    'MLConfig',
    'HyperparameterConfig',

    # Algorithms
    'AlgorithmRegistry',

    # Training
    'HyperparameterOptimizer',
    'ModelEvaluator',
    'ModelPersistence',

    # Utilities
    'FeatureEngineer',
    'DataValidator',
    'PerformanceTracker'
]

# Quick access factory functions
def create_trading_regressor(**kwargs):
    """
    Create a pre-configured ML system for trading regression tasks.

    Args:
        **kwargs: Additional configuration parameters

    Returns:
        MLSystem: Configured ML system for regression
    """
    config = MLConfig.for_trading_regression()

    # Update config with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return MLSystem(config)


def create_trading_classifier(**kwargs):
    """
    Create a pre-configured ML system for trading classification tasks.

    Args:
        **kwargs: Additional configuration parameters

    Returns:
        MLSystem: Configured ML system for classification
    """
    config = MLConfig.for_trading_classification()

    # Update config with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return MLSystem(config)


def get_available_algorithms(target_type='regression'):
    """
    Get list of available algorithms for a given target type.

    Args:
        target_type (str): 'regression' or 'classification'

    Returns:
        list: Available algorithm names
    """
    registry = AlgorithmRegistry()
    return registry.get_available_algorithms(target_type)


# Add factory functions to __all__
__all__.extend([
    'create_trading_regressor',
    'create_trading_classifier',
    'get_available_algorithms'
])

# Module documentation
__doc__ = """
ML System Package

A comprehensive machine learning system designed for trading applications.
Provides modular, extensible components for:

- Model training and evaluation
- Hyperparameter optimization
- Feature engineering
- Model persistence and versioning
- Performance tracking

Quick Start:
    >>> from model.ml_system import create_trading_regressor
    >>> ml_system = create_trading_regressor(algorithm='random_forest')
    >>> results = ml_system.train(X_train, y_train)
    >>> predictions = ml_system.predict(X_test)

For more advanced usage, use the individual components:
    >>> from model.ml_system import MLSystem, MLConfig
    >>> config = MLConfig(algorithm='gradient_boosting', hyperparameter_optimization=True)
    >>> ml_system = MLSystem(config)
"""


__version__ = "1.0.0"
__author__ = "Trading App ML Team"

__all__ = [
    'MLSystem',
    'BaseMLEstimator',
    'PipelineManager',
    'MLConfig',
    'HyperparameterConfig'
]
