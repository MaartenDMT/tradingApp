"""
RL System Examples Package.

This package contains comprehensive examples and usage guides for the
reinforcement learning trading system.
"""

from .comprehensive_examples import (
                                     cleanup_experiment_directories,
                                     example_1_basic_usage,
                                     example_2_algorithm_comparison,
                                     example_3_custom_configuration,
                                     example_4_save_load_agents,
                                     example_5_quick_experiment,
                                     example_6_comprehensive_experiment,
                                     example_7_testing_and_benchmarking,
                                     generate_sample_data,
)

__all__ = [
    'generate_sample_data',
    'example_1_basic_usage',
    'example_2_algorithm_comparison',
    'example_3_custom_configuration',
    'example_4_save_load_agents',
    'example_5_quick_experiment',
    'example_6_comprehensive_experiment',
    'example_7_testing_and_benchmarking',
    'cleanup_experiment_directories'
]

__version__ = "1.0.0"
