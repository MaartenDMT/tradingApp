"""
Exploration algorithms for enhanced RL learning.

This module provides exploration enhancement techniques including:
- Intrinsic Curiosity Module (ICM) for curiosity-driven exploration
- Other exploration methods and utilities

Key Components:
- ICMModule: Complete intrinsic curiosity implementation
- CuriosityDrivenAgent: Wrapper for integrating ICM with any RL agent
- Factory functions for easy agent creation
"""

from .icm import (
                  CuriosityDrivenAgent,
                  FeatureExtractor,
                  ForwardModel,
                  ICMModule,
                  InverseModel,
                  create_curiosity_driven_agent,
)

__all__ = [
    # Core ICM components
    'ICMModule',
    'CuriosityDrivenAgent',
    'FeatureExtractor',
    'InverseModel',
    'ForwardModel',

    # Factory functions
    'create_curiosity_driven_agent',
]

# Version info
__version__ = "1.0.0"
__author__ = "RL Trading System"
__description__ = "Advanced exploration techniques for reinforcement learning"
