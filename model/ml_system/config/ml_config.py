"""
ML Configuration Management

Provides comprehensive configuration classes for machine learning operations
including algorithm selection, hyperparameters, and training settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class MLConfig:
    """
    Main configuration class for ML system operations.

    Contains all settings needed for training, evaluation, and deployment
    of machine learning models with sensible defaults based on scikit-learn
    best practices.
    """

    # Algorithm Configuration
    algorithm: str = 'random_forest'
    target_type: str = 'regression'  # 'regression' or 'classification'
    random_state: int = 42

    # Training Configuration
    hyperparameter_optimization: bool = False
    optimization_method: str = 'random_search'  # 'grid_search', 'random_search', 'bayesian'
    optimization_iterations: int = 50
    cross_validation_folds: int = 5
    scoring_metric: str = 'neg_mean_squared_error'  # Will be set based on target_type

    # Data Configuration
    scaling_enabled: bool = True
    feature_engineering_enabled: bool = True
    feature_selection_enabled: bool = False
    feature_selection_k: int = 10

    # Model Persistence
    model_save_dir: Path = field(default_factory=lambda: Path("data/ML/models"))
    model_versioning: bool = True
    auto_save: bool = True

    # Performance and Monitoring
    early_stopping: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    performance_tracking: bool = True

    # Advanced Settings
    n_jobs: int = -1  # Use all available cores
    memory_efficient: bool = True
    cache_size: int = 200  # MB for SVM models
    max_iter: int = 1000

    # Preprocessing Settings
    handle_missing_values: bool = True
    missing_value_strategy: str = 'mean'  # 'mean', 'median', 'most_frequent'
    outlier_detection: bool = False
    outlier_threshold: float = 3.0

    def __post_init__(self):
        """Post-initialization to set dependent defaults."""
        # Set appropriate scoring metric based on target type
        if hasattr(self, '_scoring_metric_set'):
            return

        if self.target_type == 'classification':
            if self.scoring_metric == 'neg_mean_squared_error':
                self.scoring_metric = 'accuracy'
        elif self.target_type == 'regression':
            if self.scoring_metric == 'accuracy':
                self.scoring_metric = 'neg_mean_squared_error'

        # Ensure model save directory exists
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        self._scoring_metric_set = True

    def get_algorithm_params(self) -> Dict[str, Any]:
        """
        Get default parameters for the selected algorithm.

        Returns:
            Dictionary of algorithm-specific parameters
        """
        params = {
            'random_state': self.random_state,
            'n_jobs': self.n_jobs if self.algorithm in ['random_forest', 'extra_trees'] else 1
        }

        # Add algorithm-specific parameters
        if self.algorithm == 'random_forest':
            params.update({
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'bootstrap': True,
                'oob_score': True
            })
        elif self.algorithm == 'linear_regression':
            if self.target_type == 'classification':
                params.update({
                    'max_iter': self.max_iter,
                    'solver': 'lbfgs'
                })
        elif self.algorithm == 'svm':
            params.update({
                'cache_size': self.cache_size,
                'max_iter': self.max_iter
            })
            if self.target_type == 'regression':
                params['epsilon'] = 0.1
        elif self.algorithm == 'gradient_boosting':
            params.update({
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'validation_fraction': self.validation_fraction,
                'n_iter_no_change': self.n_iter_no_change if self.early_stopping else None
            })

        return params

    def get_algorithm_parameters(self, algorithm_name: str) -> Dict[str, Any]:
        """
        Get configured parameters for a specific algorithm.

        Args:
            algorithm_name: Name of the algorithm

        Returns:
            Dictionary of configured algorithm-specific parameters
        """
        params = {
            'random_state': self.random_state,
            'n_jobs': self.n_jobs if algorithm_name in ['random_forest', 'extra_trees'] else 1
        }

        # Add algorithm-specific configured parameters
        if algorithm_name == 'random_forest':
            params.update({
                'n_estimators': getattr(self, 'random_forest_n_estimators', 100),
                'max_depth': getattr(self, 'random_forest_max_depth', None),
                'min_samples_split': getattr(self, 'random_forest_min_samples_split', 2),
                'min_samples_leaf': getattr(self, 'random_forest_min_samples_leaf', 1),
                'bootstrap': getattr(self, 'random_forest_bootstrap', True),
                'oob_score': getattr(self, 'random_forest_oob_score', True)
            })
        elif algorithm_name == 'linear_regression':
            if self.target_type == 'classification':
                params.update({
                    'max_iter': self.max_iter,
                    'solver': getattr(self, 'linear_regression_solver', 'lbfgs')
                })
        elif algorithm_name == 'svm':
            params.update({
                'cache_size': self.cache_size,
                'max_iter': self.max_iter
            })
            if self.target_type == 'regression':
                params['epsilon'] = getattr(self, 'svm_epsilon', 0.1)
        elif algorithm_name == 'gradient_boosting':
            params.update({
                'n_estimators': getattr(self, 'gradient_boosting_n_estimators', 100),
                'learning_rate': getattr(self, 'gradient_boosting_learning_rate', 0.1),
                'max_depth': getattr(self, 'gradient_boosting_max_depth', 3)
            })

        return params

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """
        Get preprocessing configuration.

        Returns:
            Dictionary of preprocessing settings
        """
        return {
            'scaling_enabled': self.scaling_enabled,
            'feature_engineering_enabled': self.feature_engineering_enabled,
            'feature_selection_enabled': self.feature_selection_enabled,
            'feature_selection_k': self.feature_selection_k,
            'handle_missing_values': self.handle_missing_values,
            'missing_value_strategy': self.missing_value_strategy,
            'outlier_detection': self.outlier_detection,
            'outlier_threshold': self.outlier_threshold
        }

    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training configuration.

        Returns:
            Dictionary of training settings
        """
        return {
            'hyperparameter_optimization': self.hyperparameter_optimization,
            'optimization_method': self.optimization_method,
            'optimization_iterations': self.optimization_iterations,
            'cross_validation_folds': self.cross_validation_folds,
            'scoring_metric': self.scoring_metric,
            'early_stopping': self.early_stopping,
            'validation_fraction': self.validation_fraction,
            'n_iter_no_change': self.n_iter_no_change
        }

    def validate(self) -> List[str]:
        """
        Validate configuration settings.

        Returns:
            List of validation error messages (empty if valid)

        Raises:
            ValueError: If any validation errors are found
        """
        errors = []

        # Validate algorithm
        valid_algorithms = [
            'linear_regression', 'ridge', 'lasso', 'elastic_net',
            'random_forest', 'extra_trees', 'gradient_boosting',
            'svm', 'decision_tree', 'naive_bayes', 'knn'
        ]
        if self.algorithm not in valid_algorithms:
            errors.append(f"Invalid algorithm: {self.algorithm}. Must be one of {valid_algorithms}")

        # Validate target type
        if self.target_type not in ['regression', 'classification']:
            errors.append(f"Invalid target_type: {self.target_type}. Must be 'regression' or 'classification'")

        # Validate optimization method
        valid_methods = ['grid_search', 'random_search', 'bayesian']
        if self.optimization_method not in valid_methods:
            errors.append(f"Invalid optimization_method: {self.optimization_method}. Must be one of {valid_methods}")

        # Validate ranges
        if self.cross_validation_folds < 2:
            errors.append("cross_validation_folds must be >= 2")

        if self.optimization_iterations < 1:
            errors.append("optimization_iterations must be >= 1")

        if not 0 < self.validation_fraction < 1:
            errors.append("validation_fraction must be between 0 and 1")

        # Raise ValueError if there are errors
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

        return errors

    @classmethod
    def for_trading_regression(cls) -> 'MLConfig':
        """
        Create configuration optimized for trading regression tasks.

        Returns:
            MLConfig instance with trading-specific settings
        """
        return cls(
            algorithm='random_forest',
            target_type='regression',
            hyperparameter_optimization=True,
            optimization_method='random_search',
            optimization_iterations=100,
            scoring_metric='neg_mean_squared_error',
            scaling_enabled=True,
            feature_engineering_enabled=True,
            early_stopping=True,
            n_jobs=-1
        )

    @classmethod
    def for_trading_classification(cls) -> 'MLConfig':
        """
        Create configuration optimized for trading classification tasks.

        Returns:
            MLConfig instance with trading-specific settings
        """
        return cls(
            algorithm='random_forest',
            target_type='classification',
            hyperparameter_optimization=True,
            optimization_method='random_search',
            optimization_iterations=100,
            scoring_metric='f1_weighted',
            scaling_enabled=True,
            feature_engineering_enabled=True,
            early_stopping=True,
            n_jobs=-1
        )

    @classmethod
    def for_fast_prototyping(cls) -> 'MLConfig':
        """
        Create configuration for fast prototyping and testing.

        Returns:
            MLConfig instance with fast execution settings
        """
        return cls(
            algorithm='linear_regression',
            hyperparameter_optimization=False,
            optimization_iterations=10,
            cross_validation_folds=3,
            scaling_enabled=True,
            feature_engineering_enabled=False,
            early_stopping=False,
            n_jobs=1
        )
