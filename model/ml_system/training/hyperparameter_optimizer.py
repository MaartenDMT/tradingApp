"""
Hyperparameter Optimization for ML System

Provides comprehensive hyperparameter optimization using Grid Search,
Random Search, and Bayesian Optimization based on scikit-learn best practices.
"""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from ..config.hyperparameter_config import HyperparameterConfig
from ..config.ml_config import MLConfig

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Comprehensive hyperparameter optimization for ML algorithms.

    Supports multiple optimization strategies including grid search,
    random search, and Bayesian optimization with proper cross-validation
    and performance tracking.
    """

    def __init__(self, config: MLConfig):
        """
        Initialize hyperparameter optimizer.

        Args:
            config: ML configuration object
        """
        self.config = config
        self.hp_config = HyperparameterConfig(
            method=config.optimization_method,
            n_iter=config.optimization_iterations,
            cv_folds=config.cross_validation_folds,
            scoring=config.scoring_metric,
            random_state=config.random_state
        )
        self.optimization_results = None

    def optimize(
        self,
        pipeline: Pipeline,
        X: np.ndarray,
        y: np.ndarray,
        custom_search_space: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization.

        Args:
            pipeline: Scikit-learn pipeline to optimize
            X: Training features
            y: Training targets
            custom_search_space: Optional custom search space

        Returns:
            Dictionary containing best parameters
        """
        start_time = time.time()

        try:
            # Get search space
            if custom_search_space:
                search_space = custom_search_space
            else:
                search_space = self.hp_config.get_search_space(
                    self.config.algorithm,
                    self.config.target_type
                )

            if not search_space:
                logger.warning(f"No search space defined for {self.config.algorithm}")
                return {}

            logger.info(f"Starting {self.hp_config.method} optimization for {self.config.algorithm}")
            logger.info(f"Search space: {list(search_space.keys())}")

            # Estimate optimization time
            time_estimate = self.hp_config.estimate_search_time(
                self.config.algorithm, len(X), X.shape[1]
            )
            logger.info(f"Estimated optimization time: {time_estimate['estimated_total_time_formatted']}")

            # Perform optimization based on method
            if self.hp_config.method == 'grid_search':
                optimizer = self._create_grid_search(pipeline, search_space)
            elif self.hp_config.method == 'random_search':
                optimizer = self._create_random_search(pipeline, search_space)
            elif self.hp_config.method == 'bayesian':
                optimizer = self._create_bayesian_search(pipeline, search_space)
            else:
                raise ValueError(f"Unknown optimization method: {self.hp_config.method}")

            # Fit optimizer
            optimizer.fit(X, y)

            # Extract results
            optimization_time = time.time() - start_time

            self.optimization_results = {
                'best_params': optimizer.best_params_,
                'best_score': optimizer.best_score_,
                'best_estimator': optimizer.best_estimator_,
                'cv_results': optimizer.cv_results_ if hasattr(optimizer, 'cv_results_') else None,
                'optimization_time': optimization_time,
                'method': self.hp_config.method,
                'search_space_size': self._calculate_search_space_size(search_space),
                'n_splits': self.hp_config.cv_folds
            }

            logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
            logger.info(f"Best score: {optimizer.best_score_:.4f}")
            logger.info(f"Best parameters: {optimizer.best_params_}")

            return optimizer.best_params_

        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {str(e)}")
            raise

    def _create_grid_search(self, pipeline: Pipeline, search_space: Dict[str, Any]) -> GridSearchCV:
        """Create GridSearchCV optimizer."""
        return GridSearchCV(
            estimator=pipeline,
            param_grid=search_space,
            cv=self.hp_config.cv_folds,
            scoring=self.hp_config.scoring,
            n_jobs=self.hp_config.n_jobs,
            refit=self.hp_config.refit,
            return_train_score=self.hp_config.return_train_score,
            verbose=self.hp_config.verbose
        )

    def _create_random_search(self, pipeline: Pipeline, search_space: Dict[str, Any]) -> RandomizedSearchCV:
        """Create RandomizedSearchCV optimizer."""
        # Use random search space if available
        random_space = self.hp_config.get_random_search_space(
            self.config.algorithm,
            self.config.target_type
        )

        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=random_space if random_space != search_space else search_space,
            n_iter=self.hp_config.n_iter,
            cv=self.hp_config.cv_folds,
            scoring=self.hp_config.scoring,
            n_jobs=self.hp_config.n_jobs,
            refit=self.hp_config.refit,
            return_train_score=self.hp_config.return_train_score,
            verbose=self.hp_config.verbose,
            random_state=self.hp_config.random_state
        )

    def _create_bayesian_search(self, pipeline: Pipeline, search_space: Dict[str, Any]):
        """Create Bayesian optimization using scikit-optimize."""
        try:
            from skopt import BayesSearchCV

            # Get Bayesian search space
            bayesian_space = self.hp_config.get_bayesian_search_space(
                self.config.algorithm,
                self.config.target_type
            )

            if not bayesian_space:
                logger.warning("Bayesian search space not available, falling back to random search")
                return self._create_random_search(pipeline, search_space)

            return BayesSearchCV(
                estimator=pipeline,
                search_spaces=bayesian_space,
                n_iter=self.hp_config.n_iter,
                cv=self.hp_config.cv_folds,
                scoring=self.hp_config.scoring,
                n_jobs=self.hp_config.n_jobs,
                refit=self.hp_config.refit,
                return_train_score=self.hp_config.return_train_score,
                verbose=self.hp_config.verbose,
                random_state=self.hp_config.random_state
            )

        except ImportError:
            logger.warning("scikit-optimize not available, falling back to random search")
            return self._create_random_search(pipeline, search_space)

    def _calculate_search_space_size(self, search_space: Dict[str, Any]) -> int:
        """Calculate the total size of the search space."""
        if self.hp_config.method == 'grid_search':
            size = 1
            for values in search_space.values():
                if isinstance(values, list):
                    size *= len(values)
            return size
        else:
            return self.hp_config.n_iter

    def get_optimization_results(self) -> Optional[Dict[str, Any]]:
        """
        Get detailed optimization results.

        Returns:
            Dictionary containing optimization results or None if not run
        """
        return self.optimization_results

    def analyze_parameter_importance(self) -> Optional[Dict[str, float]]:
        """
        Analyze parameter importance based on optimization results.

        Returns:
            Dictionary mapping parameters to importance scores
        """
        if not self.optimization_results or not self.optimization_results['cv_results']:
            return None

        try:
            cv_results = self.optimization_results['cv_results']
            param_importance = {}

            # Analyze correlation between parameters and scores
            mean_scores = cv_results['mean_test_score']

            for param_name in cv_results.keys():
                if param_name.startswith('param_'):
                    param_values = cv_results[param_name]

                    # Calculate correlation if parameter is numeric
                    if all(isinstance(v, (int, float)) for v in param_values):
                        correlation = np.corrcoef(param_values, mean_scores)[0, 1]
                        param_importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0

            return param_importance

        except Exception as e:
            logger.warning(f"Could not analyze parameter importance: {str(e)}")
            return None

    def get_best_parameters_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get a summary of the best parameters found.

        Returns:
            Dictionary containing parameter summary
        """
        if not self.optimization_results:
            return None

        best_params = self.optimization_results['best_params']

        summary = {
            'algorithm': self.config.algorithm,
            'target_type': self.config.target_type,
            'optimization_method': self.hp_config.method,
            'best_score': self.optimization_results['best_score'],
            'optimization_time': self.optimization_results['optimization_time'],
            'parameters': {}
        }

        # Categorize parameters
        for param, value in best_params.items():
            if param.startswith('model__'):
                param_name = param.replace('model__', '')
                summary['parameters'][param_name] = value
            elif param.startswith('scaler__'):
                param_name = param.replace('scaler__', '')
                summary['parameters'][f'scaler_{param_name}'] = value
            else:
                summary['parameters'][param] = value

        return summary

    def save_optimization_results(self, filepath: str) -> None:
        """
        Save optimization results to file.

        Args:
            filepath: Path to save results
        """
        if not self.optimization_results:
            logger.warning("No optimization results to save")
            return

        import json

        # Prepare results for JSON serialization
        results_to_save = {
            'best_params': self.optimization_results['best_params'],
            'best_score': float(self.optimization_results['best_score']),
            'optimization_time': self.optimization_results['optimization_time'],
            'method': self.optimization_results['method'],
            'search_space_size': self.optimization_results['search_space_size'],
            'n_splits': self.optimization_results['n_splits'],
            'algorithm': self.config.algorithm,
            'target_type': self.config.target_type
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(results_to_save, f, indent=2)
            logger.info(f"Optimization results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save optimization results: {str(e)}")

    def load_optimization_results(self, filepath: str) -> None:
        """
        Load optimization results from file.

        Args:
            filepath: Path to load results from
        """
        import json

        try:
            with open(filepath, 'r') as f:
                self.optimization_results = json.load(f)
            logger.info(f"Optimization results loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load optimization results: {str(e)}")
