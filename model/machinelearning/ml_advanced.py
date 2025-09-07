"""
Advanced ML features including ensemble methods, online learning, and automated model selection.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor,
                              VotingClassifier, VotingRegressor)
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

# Import our ML models
try:
    from .ml_models import get_model
    from .ml_util import classifier, regression
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    classifier = []
    regression = []


class EnsembleManager:
    """Manage ensemble methods for improved prediction accuracy."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.ensembles = {}
        self.performance_history = {}

    def create_voting_ensemble(self, algorithms: List[str], ensemble_type: str = 'hard',
                              weights: Optional[List[float]] = None) -> Any:
        """
        Create a voting ensemble from multiple algorithms.

        :param algorithms: List of algorithm names
        :param ensemble_type: 'hard' or 'soft' voting
        :param weights: Optional weights for each algorithm
        :return: Voting ensemble model
        """
        if not MODELS_AVAILABLE:
            raise ImportError("ML models not available")

        estimators = []
        for algo in algorithms:
            try:
                model = get_model(algo)
                estimators.append((algo.replace(' ', '_').lower(), model))
                self.logger.info(f"Added {algo} to ensemble")
            except ValueError as e:
                self.logger.warning(f"Skipping {algo}: {e}")

        if not estimators:
            raise ValueError("No valid estimators for ensemble")

        # Determine if this is classification or regression
        is_classifier = any(algo in classifier for algo in algorithms)

        if is_classifier:
            ensemble = VotingClassifier(
                estimators=estimators,
                voting=ensemble_type,
                weights=weights
            )
        else:
            ensemble = VotingRegressor(
                estimators=estimators,
                weights=weights
            )

        ensemble_name = f"voting_{'_'.join(algo.replace(' ', '_').lower() for algo in algorithms)}"
        self.ensembles[ensemble_name] = ensemble

        self.logger.info(f"Created {ensemble_type} voting ensemble with {len(estimators)} models")
        return ensemble

    def create_bagging_ensemble(self, base_algorithm: str, n_estimators: int = 10,
                               max_samples: float = 1.0, max_features: float = 1.0) -> Any:
        """
        Create a bagging ensemble from a base algorithm.

        :param base_algorithm: Base algorithm name
        :param n_estimators: Number of base estimators
        :param max_samples: Fraction of samples to draw
        :param max_features: Fraction of features to draw
        :return: Bagging ensemble model
        """
        if not MODELS_AVAILABLE:
            raise ImportError("ML models not available")

        base_model = get_model(base_algorithm)
        is_classifier = base_algorithm in classifier

        if is_classifier:
            ensemble = BaggingClassifier(
                estimator=base_model,
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features,
                random_state=42,
                n_jobs=-1
            )
        else:
            ensemble = BaggingRegressor(
                estimator=base_model,
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features,
                random_state=42,
                n_jobs=-1
            )

        ensemble_name = f"bagging_{base_algorithm.replace(' ', '_').lower()}_{n_estimators}"
        self.ensembles[ensemble_name] = ensemble

        self.logger.info(f"Created bagging ensemble with {n_estimators} {base_algorithm} estimators")
        return ensemble

    def create_custom_ensemble(self, models: List[Any], weights: Optional[List[float]] = None,
                              combination_method: str = 'average') -> 'CustomEnsemble':
        """
        Create a custom ensemble with flexible combination methods.

        :param models: List of fitted models
        :param weights: Optional weights for each model
        :param combination_method: 'average', 'weighted_average', 'median', 'max_vote'
        :return: Custom ensemble model
        """
        ensemble = CustomEnsemble(models, weights, combination_method)
        ensemble_name = f"custom_{combination_method}_{len(models)}_models"
        self.ensembles[ensemble_name] = ensemble

        self.logger.info(f"Created custom ensemble with {len(models)} models using {combination_method}")
        return ensemble

    def evaluate_ensemble(self, ensemble: Any, X_test: np.ndarray, y_test: np.ndarray,
                         cv_folds: int = 5) -> Dict[str, float]:
        """
        Evaluate ensemble performance using cross-validation.

        :param ensemble: Ensemble model to evaluate
        :param X_test: Test features
        :param y_test: Test targets
        :param cv_folds: Number of CV folds
        :return: Performance metrics
        """
        # Determine scoring metric
        is_classifier = hasattr(ensemble, 'predict_proba') or isinstance(ensemble, (VotingClassifier, BaggingClassifier))
        scoring = 'accuracy' if is_classifier else 'neg_mean_squared_error'

        # Cross-validation
        if is_classifier:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        cv_scores = cross_val_score(ensemble, X_test, y_test, cv=cv, scoring=scoring, n_jobs=-1)

        # Additional metrics
        y_pred = ensemble.predict(X_test)

        if is_classifier:
            test_score = accuracy_score(y_test, y_pred)
            metrics = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_score,
                'cv_scores': cv_scores.tolist()
            }
        else:
            test_score = -mean_squared_error(y_test, y_pred)  # Negative for consistency
            metrics = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_neg_mse': test_score,
                'test_rmse': np.sqrt(-test_score),
                'cv_scores': cv_scores.tolist()
            }

        return metrics

    def get_best_ensemble(self) -> Tuple[str, Any, Dict[str, float]]:
        """
        Get the best performing ensemble based on historical performance.

        :return: Tuple of (ensemble_name, ensemble_model, performance_metrics)
        """
        if not self.performance_history:
            raise ValueError("No ensemble performance history available")

        # Find best based on CV mean score
        best_name = max(self.performance_history.keys(),
                       key=lambda k: self.performance_history[k]['cv_mean'])

        return best_name, self.ensembles[best_name], self.performance_history[best_name]


class CustomEnsemble(BaseEstimator):
    """Custom ensemble implementation with flexible combination methods."""

    def __init__(self, models: List[Any], weights: Optional[List[float]] = None,
                 combination_method: str = 'average'):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.combination_method = combination_method
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all models in the ensemble."""
        for model in self.models:
            model.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble."""
        if not self.is_fitted_:
            raise ValueError("Ensemble not fitted yet")

        predictions = np.array([model.predict(X) for model in self.models])

        if self.combination_method == 'average':
            return np.mean(predictions, axis=0)
        elif self.combination_method == 'weighted_average':
            return np.average(predictions, weights=self.weights, axis=0)
        elif self.combination_method == 'median':
            return np.median(predictions, axis=0)
        elif self.combination_method == 'max_vote':
            return np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, predictions)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")


class OnlineLearningManager:
    """Manage online learning for continuous model adaptation."""

    def __init__(self, initial_model: Any, buffer_size: int = 1000,
                 update_frequency: int = 100, logger: Optional[logging.Logger] = None):
        self.model = initial_model
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency
        self.logger = logger or logging.getLogger(__name__)

        self.data_buffer = []
        self.label_buffer = []
        self.update_count = 0
        self.performance_history = []

    def add_sample(self, X: np.ndarray, y: Any):
        """Add a new sample to the learning buffer."""
        self.data_buffer.append(X)
        self.label_buffer.append(y)

        # Maintain buffer size
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)
            self.label_buffer.pop(0)

        # Check if update is needed
        if len(self.data_buffer) >= self.update_frequency:
            self._update_model()

    def _update_model(self):
        """Update the model with buffered data."""
        if len(self.data_buffer) < 10:  # Need minimum samples
            return

        X_buffer = np.array(self.data_buffer)
        y_buffer = np.array(self.label_buffer)

        try:
            # Retrain the model with buffered data
            self.model.fit(X_buffer, y_buffer)
            self.update_count += 1

            # Evaluate performance
            y_pred = self.model.predict(X_buffer)
            if hasattr(self.model, 'predict_proba'):
                score = accuracy_score(y_buffer, y_pred)
                metric_name = 'accuracy'
            else:
                score = -mean_squared_error(y_buffer, y_pred)
                metric_name = 'neg_mse'

            self.performance_history.append({
                'update_count': self.update_count,
                'timestamp': time.time(),
                metric_name: score,
                'buffer_size': len(self.data_buffer)
            })

            self.logger.info(f"Model updated (#{self.update_count}): {metric_name}={score:.4f}")

        except Exception as e:
            self.logger.error(f"Failed to update model: {e}")

    def get_performance_trend(self) -> Dict[str, Any]:
        """Get performance trend analysis."""
        if len(self.performance_history) < 2:
            return {"message": "Not enough updates for trend analysis"}

        recent_scores = [h.get('accuracy', h.get('neg_mse', 0)) for h in self.performance_history[-5:]]
        overall_scores = [h.get('accuracy', h.get('neg_mse', 0)) for h in self.performance_history]

        return {
            'total_updates': self.update_count,
            'recent_avg': np.mean(recent_scores),
            'overall_avg': np.mean(overall_scores),
            'trend': 'improving' if recent_scores[-1] > overall_scores[0] else 'degrading',
            'last_score': recent_scores[-1] if recent_scores else 0
        }


class AutoMLPipeline:
    """Automated machine learning pipeline for model selection and hyperparameter tuning."""

    def __init__(self, logger: Optional[logging.Logger] = None, max_time_minutes: int = 30):
        self.logger = logger or logging.getLogger(__name__)
        self.max_time_minutes = max_time_minutes
        self.results = {}
        self.best_model = None
        self.best_score = float('-inf')
        self.best_params = None

    def auto_select_model(self, X: np.ndarray, y: np.ndarray,
                         algorithms: Optional[List[str]] = None,
                         cv_folds: int = 5) -> Dict[str, Any]:
        """
        Automatically select the best model from available algorithms.

        :param X: Training features
        :param y: Training targets
        :param algorithms: List of algorithms to try (None for all)
        :param cv_folds: Number of cross-validation folds
        :return: Results dictionary with best model and performance
        """
        if not MODELS_AVAILABLE:
            raise ImportError("ML models not available")

        if algorithms is None:
            # Use a subset of algorithms for speed
            algorithms = ["Random Forest", "Logistic Regression", "SVM", "XGBoost"]

        # Determine problem type
        is_classification = len(np.unique(y)) < 20 and np.issubdtype(y.dtype, np.integer)
        scoring = 'accuracy' if is_classification else 'neg_mean_squared_error'
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if is_classification else KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        self.logger.info(f"Starting automated model selection with {len(algorithms)} algorithms")
        start_time = time.time()

        for algorithm in algorithms:
            if time.time() - start_time > self.max_time_minutes * 60:
                self.logger.warning("Time limit exceeded, stopping model selection")
                break

            try:
                self.logger.info(f"Evaluating {algorithm}...")

                # Get model
                model = get_model(algorithm)

                # Cross-validation
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                mean_score = scores.mean()
                std_score = scores.std()

                # Fit model for additional metrics
                model.fit(X, y)
                y_pred = model.predict(X)

                if is_classification:
                    train_score = accuracy_score(y, y_pred)
                else:
                    train_score = -mean_squared_error(y, y_pred)

                result = {
                    'model': model,
                    'cv_mean': mean_score,
                    'cv_std': std_score,
                    'train_score': train_score,
                    'cv_scores': scores.tolist(),
                    'algorithm': algorithm
                }

                self.results[algorithm] = result

                # Track best model
                if mean_score > self.best_score:
                    self.best_score = mean_score
                    self.best_model = model
                    self.best_params = result

                self.logger.info(f"{algorithm}: CV={mean_score:.4f}±{std_score:.4f}")

            except Exception as e:
                self.logger.error(f"Failed to evaluate {algorithm}: {e}")
                continue

        duration = time.time() - start_time
        self.logger.info(f"Model selection completed in {duration:.2f} seconds")

        return {
            'best_algorithm': self.best_params['algorithm'] if self.best_params else None,
            'best_model': self.best_model,
            'best_score': self.best_score,
            'all_results': self.results,
            'duration_seconds': duration
        }

    def parallel_model_evaluation(self, X: np.ndarray, y: np.ndarray,
                                 algorithms: List[str], max_workers: int = 4) -> Dict[str, Any]:
        """
        Evaluate multiple models in parallel for faster execution.

        :param X: Training features
        :param y: Training targets
        :param algorithms: List of algorithms to evaluate
        :param max_workers: Maximum number of parallel workers
        :return: Evaluation results
        """
        if not MODELS_AVAILABLE:
            raise ImportError("ML models not available")

        def evaluate_single_model(algorithm):
            try:
                model = get_model(algorithm)

                # Simple train-test evaluation for speed
                scores = cross_val_score(model, X, y, cv=3, n_jobs=1)  # Reduced CV for speed

                return {
                    'algorithm': algorithm,
                    'model': model,
                    'cv_mean': scores.mean(),
                    'cv_std': scores.std(),
                    'cv_scores': scores.tolist(),
                    'status': 'success'
                }
            except Exception as e:
                return {
                    'algorithm': algorithm,
                    'error': str(e),
                    'status': 'failed'
                }

        self.logger.info(f"Starting parallel evaluation of {len(algorithms)} algorithms")
        start_time = time.time()

        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_algorithm = {
                executor.submit(evaluate_single_model, algo): algo
                for algo in algorithms
            }

            for future in as_completed(future_to_algorithm):
                result = future.result()
                algorithm = result['algorithm']
                results[algorithm] = result

                if result['status'] == 'success':
                    self.logger.info(f"✓ {algorithm}: {result['cv_mean']:.4f}±{result['cv_std']:.4f}")

                    # Track best model
                    if result['cv_mean'] > self.best_score:
                        self.best_score = result['cv_mean']
                        self.best_model = result['model']
                        self.best_params = result
                else:
                    self.logger.error(f"✗ {algorithm}: {result['error']}")

        duration = time.time() - start_time
        successful_models = sum(1 for r in results.values() if r['status'] == 'success')

        self.logger.info(f"Parallel evaluation completed: {successful_models}/{len(algorithms)} successful in {duration:.2f}s")

        return {
            'results': results,
            'best_algorithm': self.best_params['algorithm'] if self.best_params else None,
            'best_model': self.best_model,
            'best_score': self.best_score,
            'duration_seconds': duration,
            'success_count': successful_models
        }


class FeatureImportanceAnalyzer:
    """Analyze feature importance across different models and methods."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.importance_results = {}

    def analyze_model_importance(self, model: Any, X: pd.DataFrame,
                                feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Extract feature importance from a fitted model.

        :param model: Fitted model
        :param X: Feature data
        :param feature_names: Feature names
        :return: Feature importance dictionary
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        importance_dict = {}

        # Try different methods to get importance
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            self.logger.info("Used feature_importances_ attribute")

        elif hasattr(model, 'coef_'):
            # Linear models
            if model.coef_.ndim > 1:
                # Multi-class classification
                importance_values = np.abs(model.coef_).mean(axis=0)
            else:
                importance_values = np.abs(model.coef_)
            importance_dict = dict(zip(feature_names, importance_values))
            self.logger.info("Used coef_ attribute")

        else:
            self.logger.warning("Model does not support feature importance extraction")
            return {}

        # Normalize to sum to 1
        total_importance = sum(importance_dict.values())
        if total_importance > 0:
            importance_dict = {k: v/total_importance for k, v in importance_dict.items()}

        return importance_dict

    def permutation_importance(self, model: Any, X: np.ndarray, y: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              n_repeats: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Calculate permutation importance for features.

        :param model: Fitted model
        :param X: Features
        :param y: Targets
        :param feature_names: Feature names
        :param n_repeats: Number of permutation repeats
        :return: Permutation importance results
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Baseline score
        baseline_score = model.score(X, y)

        importance_scores = {}

        for i, feature_name in enumerate(feature_names):
            scores = []

            for _ in range(n_repeats):
                # Create permuted version
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])

                # Calculate score with permuted feature
                permuted_score = model.score(X_permuted, y)
                importance = baseline_score - permuted_score
                scores.append(importance)

            importance_scores[feature_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }

        self.logger.info(f"Calculated permutation importance for {len(feature_names)} features")
        return importance_scores

    def rank_features(self, importance_dict: Dict[str, float], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Rank features by importance.

        :param importance_dict: Feature importance dictionary
        :param top_k: Number of top features to return
        :return: List of (feature_name, importance) tuples
        """
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_k]

    def get_feature_importance_summary(self, model: Any, X: pd.DataFrame, y: np.ndarray,
                                     feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get comprehensive feature importance summary.

        :param model: Fitted model
        :param X: Features
        :param y: Targets
        :param feature_names: Feature names
        :return: Comprehensive importance summary
        """
        if feature_names is None:
            feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])]

        X_array = X.values if isinstance(X, pd.DataFrame) else X

        summary = {}

        # Model-based importance
        model_importance = self.analyze_model_importance(model, X, feature_names)
        if model_importance:
            summary['model_importance'] = model_importance
            summary['top_features_model'] = self.rank_features(model_importance, 5)

        # Permutation importance (with reduced repeats for speed)
        try:
            perm_importance = self.permutation_importance(model, X_array, y, feature_names, n_repeats=5)
            perm_mean = {k: v['mean'] for k, v in perm_importance.items()}
            summary['permutation_importance'] = perm_importance
            summary['top_features_permutation'] = self.rank_features(perm_mean, 5)
        except Exception as e:
            self.logger.warning(f"Permutation importance failed: {e}")

        return summary


# Export classes for use in other modules
__all__ = [
    'EnsembleManager',
    'CustomEnsemble',
    'OnlineLearningManager',
    'AutoMLPipeline',
    'FeatureImportanceAnalyzer'
]
