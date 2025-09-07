"""
Visualization utilities and initialization for RL Trading System.

This module provides the main interface for all visualization components
including training progress, performance analysis, trading behavior, and curiosity analysis.
"""

from .curiosity_analysis import CuriosityVisualizer, quick_plot_curiosity
from .performance_analysis import PerformanceAnalyzer
from .trading_behavior import (
                               TradingBehaviorVisualizer,
                               quick_plot_actions,
                               quick_plot_trading_patterns,
)
from .training_plots import (
                               TrainingVisualizer,
                               quick_compare_algorithms,
                               quick_plot_training,
)

__all__ = [
    # Training visualization
    'TrainingVisualizer',
    'quick_plot_training',
    'quick_compare_algorithms',

    # Performance analysis
    'PerformanceAnalyzer',

    # Trading behavior
    'TradingBehaviorVisualizer',
    'quick_plot_actions',
    'quick_plot_trading_patterns',

    # Curiosity analysis
    'CuriosityVisualizer',
    'quick_plot_curiosity',
]

# Version info
__version__ = "1.0.0"
__author__ = "RL Trading System"
__description__ = "Comprehensive visualization suite for reinforcement learning trading analysis"
