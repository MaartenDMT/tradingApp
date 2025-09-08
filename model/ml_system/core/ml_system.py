"""
Core ML System Implementation

This module provides the main MLSystem class that orchestrates all ML operations
including training, prediction, model management, and evaluation. Based on modern
scikit-learn best practices and patterns.
"""

import logging
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..algorithms.registry import AlgorithmRegistry
from ..config.ml_config import MLConfig
from ..training.hyperparameter_optimizer import HyperparameterOptimizer
from ..training.model_evaluator import ModelEvaluator
from ..training.model_persistence import ModelPersistence
from ..utils.data_validator import DataValidator
from ..utils.feature_engineering import FeatureEngineer
from ..utils.performance_tracker import PerformanceTracker

# Suppress sklearn convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = logging.getLogger(__name__)


class MLSystem:
    """
    Comprehensive Machine Learning System for traditional ML algorithms.

    Provides a unified interface for training, evaluating, and deploying
    traditional machine learning models with modern best practices.

    Features:
    - Support for regression and classification tasks
    - Automated hyperparameter optimization
    - Cross-validation and model evaluation
    - Model persistence and versioning
    - Feature engineering and selection
    - Performance monitoring and logging

    Example:
        >>> config = MLConfig(
        ...     algorithm='random_forest',
        ...     target_type='regression',
        ...     hyperparameter_optimization=True
        ... )
        >>> ml_system = MLSystem(config)
        >>> results = ml_system.train(X_train, y_train)
        >>> predictions = ml_system.predict(X_test)
    """

    def __init__(self, config: MLConfig):
        """
        Initialize ML System with configuration.

        Args:
            config: ML configuration object containing all settings
        """
        self.config = config
        self.model = None
        self.pipeline = None
        self.is_fitted = False
        self.feature_names = None
        self.training_history = []

        # Initialize components
        self.algorithm_registry = AlgorithmRegistry()
        self.hyperparameter_optimizer = HyperparameterOptimizer(config)
        self.model_evaluator = ModelEvaluator(config)
        self.model_persistence = ModelPersistence(config)
        self.feature_engineer = FeatureEngineer(config)
        self.data_validator = DataValidator()
        self.performance_tracker = PerformanceTracker()

        logger.info(f"MLSystem initialized with algorithm: {config.algorithm}")

    def train(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        validation_split: float = 0.2,
        cross_validation: bool = True,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train the ML model with comprehensive evaluation.

        Args:
            X: Training features
            y: Training targets
            validation_split: Fraction of data for validation
            cross_validation: Whether to perform cross-validation
            save_model: Whether to save the trained model

        Returns:
            Dictionary containing training results and metrics
        """
        try:
            logger.info(f"Starting ML training with {self.config.algorithm}")

            # Validate input data
            X, y = self.data_validator.validate_and_clean(X, y)

            # Store feature names
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            # Feature engineering
            if self.config.feature_engineering_enabled:
                X = self.feature_engineer.engineer_features(X)

            # Split data for validation
            if validation_split > 0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=validation_split,
                    random_state=self.config.random_state,
                    stratify=y if self.config.target_type == 'classification' else None
                )
            else:
                X_train, y_train = X, y
                X_val, y_val = None, None

            # Get base algorithm
            algorithm = self.algorithm_registry.get_algorithm(
                self.config.algorithm,
                self.config.target_type
            )

            # Create pipeline with preprocessing
            pipeline_steps = []
            if self.config.scaling_enabled:
                pipeline_steps.append(('scaler', StandardScaler()))
            pipeline_steps.append(('model', algorithm))

            self.pipeline = Pipeline(pipeline_steps)

            # Hyperparameter optimization
            optimization_results = None
            best_params = None
            if self.config.hyperparameter_optimization:
                logger.info("Performing hyperparameter optimization")
                best_params = self.hyperparameter_optimizer.optimize(
                    self.pipeline, X_train, y_train
                )
                optimization_results = self.hyperparameter_optimizer.get_optimization_results()

                # Update pipeline with best parameters
                for param, value in best_params.items():
                    if param.startswith('model__'):
                        param_name = param.replace('model__', '')
                        setattr(self.pipeline.named_steps['model'], param_name, value)

            # Train final model
            logger.info("Training final model")
            self.pipeline.fit(X_train, y_train)
            self.model = self.pipeline.named_steps['model']
            self.is_fitted = True

            # Evaluate model
            results = {
                'algorithm': self.config.algorithm,
                'target_type': self.config.target_type,
                'training_samples': len(X_train),
                'features': len(self.feature_names),
                'timestamp': datetime.now().isoformat()
            }

            # Add optimization results if available
            if best_params is not None:
                results['best_params'] = best_params
            if optimization_results is not None:
                results['optimization_results'] = optimization_results

            # Training metrics
            train_predictions = self.pipeline.predict(X_train)
            results['train_metrics'] = self.model_evaluator.evaluate(
                y_train, train_predictions, self.config.target_type
            )

            # Validation metrics
            if X_val is not None:
                val_predictions = self.pipeline.predict(X_val)
                results['validation_metrics'] = self.model_evaluator.evaluate(
                    y_val, val_predictions, self.config.target_type
                )

            # Cross-validation
            if cross_validation:
                cv_scores = cross_val_score(
                    self.pipeline, X, y,
                    cv=self.config.cross_validation_folds,
                    scoring=self.config.scoring_metric
                )
                results['cross_validation'] = {
                    'mean_score': float(np.mean(cv_scores)),
                    'std_score': float(np.std(cv_scores)),
                    'scores': cv_scores.tolist()
                }

            # Save model
            if save_model:
                model_path = self.model_persistence.save_model(
                    self.pipeline, results
                )
                results['model_path'] = str(model_path)

            # Track performance
            self.performance_tracker.log_training(results)
            self.training_history.append(results)

            logger.info("Training completed successfully")
            return results

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        return_probabilities: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using the trained model.

        Args:
            X: Input features for prediction
            return_probabilities: Whether to return class probabilities (classification only)

        Returns:
            Predictions array, optionally with probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        try:
            # Validate and prepare data
            X = self.data_validator.validate_features(X, self.feature_names)

            # Feature engineering (if enabled during training)
            if self.config.feature_engineering_enabled:
                X = self.feature_engineer.transform_features(X)

            # Make predictions
            predictions = self.pipeline.predict(X)

            # Return probabilities if requested and available
            if (return_probabilities and
                self.config.target_type == 'classification' and
                hasattr(self.pipeline.named_steps['model'], 'predict_proba')):
                probabilities = self.pipeline.predict_proba(X)
                return predictions, probabilities

            return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.

        Args:
            X: Test features
            y: True test targets
            detailed: Whether to include detailed metrics

        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")

        try:
            predictions = self.predict(X)
            results = self.model_evaluator.evaluate(
                y, predictions, self.config.target_type, detailed=detailed
            )

            # Track evaluation performance
            self.performance_tracker.log_evaluation(results)

            return results

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

    def save_model(self, path: Optional[Path] = None) -> Path:
        """
        Save the trained model to disk.

        Args:
            path: Optional custom save path

        Returns:
            Path where model was saved
        """
        if not self.is_fitted:
            raise ValueError("No trained model to save")

        return self.model_persistence.save_model(self.pipeline, path=path)

    def load_model(self, path: Path) -> None:
        """
        Load a previously trained model from disk.

        Args:
            path: Path to the saved model
        """
        self.pipeline = self.model_persistence.load_model(path)
        self.model = self.pipeline.named_steps['model']
        self.is_fitted = True

        logger.info(f"Model loaded from {path}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores if available.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            return None

        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_scores = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importance_scores = np.abs(self.model.coef_).flatten()
            else:
                return None

            return dict(zip(self.feature_names, importance_scores))

        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return None

    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all training sessions.

        Returns:
            List of training result dictionaries
        """
        return self.training_history.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the current model.

        Returns:
            Dictionary containing model information
        """
        if not self.is_fitted:
            return {'status': 'not_fitted'}

        info = {
            'status': 'fitted',
            'algorithm': self.config.algorithm,
            'target_type': self.config.target_type,
            'features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'model_type': type(self.model).__name__,
            'preprocessing_steps': [step[0] for step in self.pipeline.steps[:-1]],
            'hyperparameters': self.model.get_params() if hasattr(self.model, 'get_params') else {}
        }

        # Add feature importance if available
        feature_importance = self.get_feature_importance()
        if feature_importance:
            info['feature_importance'] = feature_importance

        return info
