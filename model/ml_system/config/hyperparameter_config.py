"""
Hyperparameter Configuration for ML System

Provides comprehensive hyperparameter optimization configuration and search spaces
based on modern scikit-learn best practices.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from scipy.stats import randint, uniform


@dataclass
class HyperparameterConfig:
    """
    Configuration for hyperparameter optimization.

    Supports grid search, random search, and Bayesian optimization
    with predefined search spaces for common algorithms.
    """

    # Optimization method
    method: str = 'random_search'  # 'grid_search', 'random_search', 'bayesian'
    n_iter: int = 50  # Number of iterations for random/bayesian search
    cv_folds: int = 5  # Cross-validation folds
    scoring: str = 'neg_mean_squared_error'  # Scoring metric
    n_jobs: int = -1  # Parallel jobs
    random_state: int = 42

    # Search space customization
    custom_search_space: Optional[Dict[str, Any]] = None
    enable_early_stopping: bool = True
    early_stopping_rounds: int = 10

    # Advanced options
    refit: bool = True
    return_train_score: bool = False
    verbose: int = 1

    def get_search_space(self, algorithm: str, target_type: str) -> Dict[str, Any]:
        """
        Get hyperparameter search space for an algorithm.

        Args:
            algorithm: Algorithm name
            target_type: 'regression' or 'classification'

        Returns:
            Dictionary containing hyperparameter search space
        """
        if self.custom_search_space:
            return self.custom_search_space

        # Default search spaces based on scikit-learn best practices
        search_spaces = {
            'random_forest': {
                'model__n_estimators': [50, 100, 200, 300, 500],
                'model__max_depth': [None, 10, 20, 30, 50],
                'model__min_samples_split': [2, 5, 10, 20],
                'model__min_samples_leaf': [1, 2, 4, 8],
                'model__max_features': ['sqrt', 'log2', None, 0.5, 0.8],
                'model__bootstrap': [True, False]
            },

            'extra_trees': {
                'model__n_estimators': [50, 100, 200, 300, 500],
                'model__max_depth': [None, 10, 20, 30, 50],
                'model__min_samples_split': [2, 5, 10, 20],
                'model__min_samples_leaf': [1, 2, 4, 8],
                'model__max_features': ['sqrt', 'log2', None, 0.5, 0.8]
            },

            'gradient_boosting': {
                'model__n_estimators': [50, 100, 200, 300],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'model__max_depth': [3, 5, 7, 9],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__subsample': [0.8, 0.9, 1.0]
            },

            'svm': {
                'model__C': [0.01, 0.1, 1, 10, 100, 1000],
                'model__kernel': ['linear', 'rbf', 'poly'],
                'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            },

            'logistic_regression': {
                'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'model__solver': ['lbfgs', 'liblinear', 'saga'],
                'model__penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'model__max_iter': [1000, 2000, 5000]
            },

            'knn': {
                'model__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
                'model__weights': ['uniform', 'distance'],
                'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'model__leaf_size': [20, 30, 40, 50],
                'model__p': [1, 2]
            },

            'ridge': {
                'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            },

            'lasso': {
                'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                'model__max_iter': [1000, 2000, 5000],
                'model__selection': ['cyclic', 'random']
            },

            'elastic_net': {
                'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
                'model__max_iter': [1000, 2000, 5000],
                'model__selection': ['cyclic', 'random']
            },

            'decision_tree': {
                'model__max_depth': [None, 5, 10, 15, 20, 25],
                'model__min_samples_split': [2, 5, 10, 20],
                'model__min_samples_leaf': [1, 2, 4, 8],
                'model__max_features': ['sqrt', 'log2', None],
                'model__criterion': ['gini', 'entropy'] if target_type == 'classification' else ['mse', 'friedman_mse', 'mae']
            },

            'naive_bayes': {
                'model__var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
            }
        }

        # Add scaling parameters if applicable
        base_space = search_spaces.get(algorithm, {})

        # Add scaler parameters for algorithms that benefit from scaling
        if algorithm in ['svm', 'logistic_regression', 'ridge', 'lasso', 'elastic_net', 'knn']:
            base_space['scaler__with_mean'] = [True, False]
            base_space['scaler__with_std'] = [True, False]

        return base_space

    def get_random_search_space(self, algorithm: str, target_type: str) -> Dict[str, Any]:
        """
        Get search space optimized for RandomizedSearchCV.

        Args:
            algorithm: Algorithm name
            target_type: 'regression' or 'classification'

        Returns:
            Dictionary with scipy.stats distributions for random search
        """
        random_spaces = {
            'random_forest': {
                'model__n_estimators': randint(50, 500),
                'model__max_depth': [None] + list(range(5, 51, 5)),
                'model__min_samples_split': randint(2, 21),
                'model__min_samples_leaf': randint(1, 11),
                'model__max_features': ['sqrt', 'log2', None],
                'model__bootstrap': [True, False]
            },

            'gradient_boosting': {
                'model__n_estimators': randint(50, 301),
                'model__learning_rate': uniform(0.01, 0.29),
                'model__max_depth': randint(3, 10),
                'model__min_samples_split': randint(2, 21),
                'model__min_samples_leaf': randint(1, 11),
                'model__subsample': uniform(0.6, 0.4)
            },

            'svm': {
                'model__C': uniform(0.1, 999.9),
                'model__kernel': ['linear', 'rbf', 'poly'],
                'model__gamma': ['scale', 'auto']
            },

            'logistic_regression': {
                'model__C': uniform(0.001, 999.999),
                'model__solver': ['lbfgs', 'liblinear', 'saga'],
                'model__max_iter': randint(1000, 5001)
            },

            'knn': {
                'model__n_neighbors': randint(3, 22),
                'model__weights': ['uniform', 'distance'],
                'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'model__leaf_size': randint(10, 51),
                'model__p': [1, 2]
            }
        }

        return random_spaces.get(algorithm, self.get_search_space(algorithm, target_type))

    def get_bayesian_search_space(self, algorithm: str, target_type: str) -> Dict[str, Any]:
        """
        Get search space optimized for Bayesian optimization.

        Args:
            algorithm: Algorithm name
            target_type: 'regression' or 'classification'

        Returns:
            Dictionary with skopt search space definitions
        """
        try:
            from skopt.space import Categorical, Integer, Real

            bayesian_spaces = {
                'random_forest': {
                    'model__n_estimators': Integer(50, 500),
                    'model__max_depth': Categorical([None] + list(range(5, 51, 5))),
                    'model__min_samples_split': Integer(2, 20),
                    'model__min_samples_leaf': Integer(1, 10),
                    'model__max_features': Categorical(['sqrt', 'log2', None]),
                    'model__bootstrap': Categorical([True, False])
                },

                'gradient_boosting': {
                    'model__n_estimators': Integer(50, 300),
                    'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'model__max_depth': Integer(3, 9),
                    'model__min_samples_split': Integer(2, 20),
                    'model__min_samples_leaf': Integer(1, 10),
                    'model__subsample': Real(0.6, 1.0)
                },

                'svm': {
                    'model__C': Real(0.1, 1000, prior='log-uniform'),
                    'model__kernel': Categorical(['linear', 'rbf', 'poly']),
                    'model__gamma': Categorical(['scale', 'auto'])
                },

                'logistic_regression': {
                    'model__C': Real(0.001, 1000, prior='log-uniform'),
                    'model__solver': Categorical(['lbfgs', 'liblinear', 'saga']),
                    'model__max_iter': Integer(1000, 5000)
                }
            }

            return bayesian_spaces.get(algorithm, {})

        except ImportError:
            # Fall back to regular search space if skopt not available
            return self.get_search_space(algorithm, target_type)

    def validate_search_space(self, search_space: Dict[str, Any]) -> bool:
        """
        Validate hyperparameter search space.

        Args:
            search_space: Search space to validate

        Returns:
            True if valid, False otherwise
        """
        if not search_space:
            return False

        # Check that all values are lists or distributions
        for param, values in search_space.items():
            if not isinstance(values, (list, tuple)) and not hasattr(values, 'rvs'):
                return False

        return True

    def estimate_search_time(self, algorithm: str, n_samples: int, n_features: int) -> Dict[str, float]:
        """
        Estimate hyperparameter search time based on algorithm and data size.

        Args:
            algorithm: Algorithm name
            n_samples: Number of training samples
            n_features: Number of features

        Returns:
            Dictionary with time estimates in seconds
        """
        # Base time estimates (seconds per CV fold per parameter combination)
        base_times = {
            'linear_regression': 0.01,
            'ridge': 0.01,
            'lasso': 0.05,
            'elastic_net': 0.05,
            'logistic_regression': 0.1,
            'random_forest': 0.5,
            'extra_trees': 0.4,
            'gradient_boosting': 1.0,
            'svm': 2.0,
            'knn': 0.2,
            'decision_tree': 0.1,
            'naive_bayes': 0.01
        }

        base_time = base_times.get(algorithm, 1.0)

        # Scale by data size
        size_factor = (n_samples / 1000) * (n_features / 100)
        estimated_time_per_fit = base_time * max(1.0, size_factor)

        # Calculate total search time
        search_space = self.get_search_space(algorithm, 'regression')

        if self.method == 'grid_search':
            n_combinations = 1
            for values in search_space.values():
                n_combinations *= len(values) if isinstance(values, list) else 1
            total_fits = n_combinations
        else:
            total_fits = self.n_iter

        total_time = estimated_time_per_fit * total_fits * self.cv_folds

        return {
            'time_per_fit': estimated_time_per_fit,
            'total_fits': total_fits,
            'estimated_total_time': total_time,
            'estimated_total_time_formatted': f"{total_time/60:.1f} minutes" if total_time > 60 else f"{total_time:.1f} seconds"
        }
