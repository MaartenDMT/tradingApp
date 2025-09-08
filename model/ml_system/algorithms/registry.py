"""
Algorithm Registry for ML System

Provides a centralized registry for all available ML algorithms with proper
configuration and instantiation based on scikit-learn best practices.
"""

import logging
from typing import Any, Dict, Optional

from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (ElasticNet, ElasticNetCV, Lasso, LassoCV,
                                  LinearRegression, LogisticRegression, Ridge,
                                  RidgeCV)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

logger = logging.getLogger(__name__)


class AlgorithmRegistry:
    """
    Registry for available ML algorithms with proper configuration.

    Provides a centralized way to access and configure scikit-learn algorithms
    with sensible defaults and proper parameter handling.
    """

    def __init__(self):
        """Initialize the algorithm registry."""
        self._algorithms = {
            'regression': {
                'linear_regression': LinearRegression,
                'ridge': Ridge,
                'ridge_cv': RidgeCV,
                'lasso': Lasso,
                'lasso_cv': LassoCV,
                'elastic_net': ElasticNet,
                'elastic_net_cv': ElasticNetCV,
                'random_forest': RandomForestRegressor,
                'extra_trees': ExtraTreesRegressor,
                'gradient_boosting': GradientBoostingRegressor,
                'decision_tree': DecisionTreeRegressor,
                'svm': SVR,
                'knn': KNeighborsRegressor
            },
            'classification': {
                'logistic_regression': LogisticRegression,
                'random_forest': RandomForestClassifier,
                'extra_trees': ExtraTreesClassifier,
                'gradient_boosting': GradientBoostingClassifier,
                'decision_tree': DecisionTreeClassifier,
                'svm': SVC,
                'naive_bayes': GaussianNB,
                'naive_bayes_multinomial': MultinomialNB,
                'knn': KNeighborsClassifier
            }
        }

        # Default parameters for each algorithm
        self._default_params = {
            'linear_regression': {},
            'ridge': {'alpha': 1.0},
            'ridge_cv': {'alphas': [0.1, 1.0, 10.0], 'cv': 5},
            'lasso': {'alpha': 1.0},
            'lasso_cv': {'cv': 5},
            'elastic_net': {'alpha': 1.0, 'l1_ratio': 0.5},
            'elastic_net_cv': {'cv': 5},
            'logistic_regression': {
                'max_iter': 1000,
                'solver': 'lbfgs',
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1
            },
            'extra_trees': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'bootstrap': False,
                'random_state': 42,
                'n_jobs': -1
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            'decision_tree': {
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            'svm': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'cache_size': 200
            },
            'naive_bayes': {},
            'naive_bayes_multinomial': {},
            'knn': {
                'n_neighbors': 5,
                'weights': 'uniform',
                'algorithm': 'auto'
            }
        }

    def get_algorithm(self, name: str, target_type: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get an instantiated algorithm with proper configuration.

        Args:
            name: Algorithm name
            target_type: 'regression' or 'classification'
            params: Optional parameters to override defaults

        Returns:
            Instantiated sklearn estimator

        Raises:
            ValueError: If algorithm or target type is invalid
        """
        if target_type not in self._algorithms:
            raise ValueError(f"Invalid target_type: {target_type}")

        if name not in self._algorithms[target_type]:
            available = list(self._algorithms[target_type].keys())
            raise ValueError(f"Algorithm '{name}' not available for {target_type}. "
                           f"Available algorithms: {available}")

        # Get the algorithm class
        algorithm_class = self._algorithms[target_type][name]

        # Get default parameters and merge with user parameters
        algorithm_params = self._default_params.get(name, {}).copy()
        if params:
            algorithm_params.update(params)

        # Handle special cases for specific algorithms
        if name == 'svm' and target_type == 'regression':
            # For SVR, add epsilon parameter if not specified
            if 'epsilon' not in algorithm_params:
                algorithm_params['epsilon'] = 0.1

        if name in ['random_forest', 'extra_trees']:
            # Add oob_score for ensemble methods with bootstrap
            if target_type == 'classification' and algorithm_params.get('bootstrap', True):
                algorithm_params['oob_score'] = True

        try:
            algorithm = algorithm_class(**algorithm_params)
            logger.info(f"Created {name} algorithm for {target_type} with params: {algorithm_params}")
            return algorithm
        except Exception as e:
            logger.error(f"Failed to create algorithm {name}: {str(e)}")
            raise

    def get_available_algorithms(self, target_type: Optional[str] = None) -> Dict[str, list]:
        """
        Get list of available algorithms.

        Args:
            target_type: Optional filter by target type

        Returns:
            Dictionary of available algorithms by target type
        """
        if target_type:
            if target_type not in self._algorithms:
                return {}
            return {target_type: list(self._algorithms[target_type].keys())}

        return {
            task_type: list(algorithms.keys())
            for task_type, algorithms in self._algorithms.items()
        }

    def get_algorithm_info(self, name: str, target_type: str) -> Dict[str, Any]:
        """
        Get detailed information about an algorithm.

        Args:
            name: Algorithm name
            target_type: 'regression' or 'classification'

        Returns:
            Dictionary containing algorithm information
        """
        if target_type not in self._algorithms:
            raise ValueError(f"Invalid target_type: {target_type}")

        if name not in self._algorithms[target_type]:
            raise ValueError(f"Algorithm '{name}' not available for {target_type}")

        algorithm_class = self._algorithms[target_type][name]
        default_params = self._default_params.get(name, {})

        return {
            'name': name,
            'target_type': target_type,
            'class': algorithm_class.__name__,
            'module': algorithm_class.__module__,
            'default_parameters': default_params,
            'supports_feature_importance': hasattr(algorithm_class(), 'feature_importances_') or
                                          hasattr(algorithm_class(), 'coef_'),
            'supports_probability': hasattr(algorithm_class(), 'predict_proba'),
            'supports_multioutput': getattr(algorithm_class, '_more_tags', lambda: {})().get('multioutput', False)
        }

    def validate_algorithm_params(self, name: str, target_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize algorithm parameters.

        Args:
            name: Algorithm name
            target_type: 'regression' or 'classification'
            params: Parameters to validate

        Returns:
            Validated and sanitized parameters
        """
        if target_type not in self._algorithms:
            raise ValueError(f"Invalid target_type: {target_type}")

        if name not in self._algorithms[target_type]:
            raise ValueError(f"Algorithm '{name}' not available for {target_type}")

        algorithm_class = self._algorithms[target_type][name]

        try:
            # Create a temporary instance to validate parameters
            temp_instance = algorithm_class()
            valid_params = temp_instance.get_params()

            # Filter out invalid parameters
            validated_params = {
                key: value for key, value in params.items()
                if key in valid_params
            }

            # Warn about invalid parameters
            invalid_params = set(params.keys()) - set(valid_params.keys())
            if invalid_params:
                logger.warning(f"Invalid parameters for {name}: {invalid_params}")

            return validated_params

        except Exception as e:
            logger.error(f"Parameter validation failed for {name}: {str(e)}")
            return {}

    def get_hyperparameter_space(self, name: str, target_type: str) -> Dict[str, Any]:
        """
        Get hyperparameter search space for an algorithm.

        Args:
            name: Algorithm name
            target_type: 'regression' or 'classification'

        Returns:
            Dictionary defining hyperparameter search space
        """
        spaces = {
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'extra_trees': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            },
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'liblinear', 'saga'],
                'penalty': ['l1', 'l2', 'elasticnet', 'none']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            },
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
            },
            'lasso': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            },
            'elastic_net': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        }

        return spaces.get(name, {})

    def get_hyperparameter_search_space(self, name: str, target_type: str) -> Dict[str, Any]:
        """
        Get hyperparameter search space for an algorithm (for backward compatibility).

        Args:
            name: Algorithm name
            target_type: 'regression' or 'classification'

        Returns:
            Dictionary defining hyperparameter search space
        """
        return self.get_hyperparameter_space(name, target_type)
