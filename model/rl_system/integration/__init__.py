"""
Integration System for RL Trading Application.

This module provides advanced integration capabilities including:
- Algorithm factory with professional optimizations
- Seamless algorithm switching
- Comprehensive training management
- Performance monitoring and visualization
"""

from .algorithm_factory import (AdvancedAlgorithmFactory, AgentWrapper,
                                AlgorithmRegistry, AlgorithmSpec,
                                AlgorithmType, EnsembleAgent,
                                algorithm_factory, create_agent,
                                get_algorithm_recommendations,
                                list_available_algorithms)
from .algorithm_switching import (AlgorithmSwitcher, PerformanceMonitor,
                                  StateTransferManager, SwitchCondition,
                                  SwitchEvent, SwitchTrigger,
                                  create_performance_based_strategy,
                                  create_switcher)
from .integration_manager import (EnvironmentSpec, IntegrationManager,
                                  TrainingConfig, integration_manager,
                                  quick_train, setup_training_environment)
from .rl_system import (RLSystemManager, compare_algorithms, create_rl_system,
                        quick_experiment)

__all__ = [
    # Algorithm Factory
    'AlgorithmType',
    'AlgorithmSpec',
    'AgentWrapper',
    'EnsembleAgent',
    'AdvancedAlgorithmFactory',
    'AlgorithmRegistry',
    'algorithm_factory',
    'create_agent',
    'get_algorithm_recommendations',
    'list_available_algorithms',

    # Algorithm Switching
    'SwitchTrigger',
    'SwitchCondition',
    'SwitchEvent',
    'StateTransferManager',
    'PerformanceMonitor',
    'AlgorithmSwitcher',
    'create_switcher',
    'create_performance_based_strategy',

    # Integration Manager
    'TrainingConfig',
    'EnvironmentSpec',
    'IntegrationManager',
    'integration_manager',
    'setup_training_environment',
    'quick_train',

    # Legacy RL System
    'RLSystemManager',
    'compare_algorithms',
    'create_rl_system',
    'quick_experiment',
]
