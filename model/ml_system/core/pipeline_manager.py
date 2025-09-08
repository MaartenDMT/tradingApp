"""
Pipeline Manager for ML System

Provides comprehensive pipeline management with preprocessing,
model training, and evaluation in a unified workflow.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..algorithms.registry import AlgorithmRegistry
from ..config.ml_config import MLConfig
from ..utils.data_validator import DataValidator
from ..utils.feature_engineering import FeatureEngineer
from .base_estimator import BaseMLEstimator, SklearnEstimatorWrapper

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Comprehensive pipeline management for ML workflows.

    Manages the entire ML pipeline from data preprocessing through
    model training and evaluation with proper validation and logging.
    """

    def __init__(self, config: MLConfig):
        """
        Initialize pipeline manager.

        Args:
            config: ML configuration object
        """
        self.config = config
        self.pipeline = None
        self.is_fitted = False

        # Initialize components
        self.algorithm_registry = AlgorithmRegistry()
        self.feature_engineer = FeatureEngineer(config)
        self.data_validator = DataValidator()

        # Pipeline components
        self.preprocessing_steps = []
        self.model_step = None
        self.pipeline_metadata = {}

    def create_pipeline(
        self,
        custom_steps: Optional[List[Tuple[str, Any]]] = None,
        custom_model: Optional[BaseMLEstimator] = None
    ) -> Pipeline:
        """
        Create a complete ML pipeline.

        Args:
            custom_steps: Optional custom preprocessing steps
            custom_model: Optional custom model to use

        Returns:
            Configured scikit-learn Pipeline
        """
        try:
            logger.info(f"Creating pipeline for {self.config.algorithm}")

            # Build preprocessing steps
            steps = []

            # Add custom preprocessing steps
            if custom_steps:
                steps.extend(custom_steps)

            # Add standard preprocessing
            if self.config.scaling_enabled:
                steps.append(('scaler', StandardScaler()))

            # Add model step
            if custom_model:
                if isinstance(custom_model, BaseMLEstimator):
                    model = custom_model
                else:
                    model = SklearnEstimatorWrapper(custom_model)
            else:
                # Get model from registry
                sklearn_model = self.algorithm_registry.get_algorithm(
                    self.config.algorithm,
                    self.config.target_type
                )
                model = SklearnEstimatorWrapper(sklearn_model)

            steps.append(('model', model))

            # Create pipeline
            self.pipeline = Pipeline(steps)
            self.preprocessing_steps = steps[:-1]
            self.model_step = steps[-1]

            # Store metadata
            self.pipeline_metadata = {
                'algorithm': self.config.algorithm,
                'target_type': self.config.target_type,
                'preprocessing_steps': [step[0] for step in self.preprocessing_steps],
                'model_type': type(model).__name__,
                'scaling_enabled': self.config.scaling_enabled,
                'feature_engineering_enabled': self.config.feature_engineering_enabled
            }

            logger.info(f"Pipeline created with {len(steps)} steps")
            return self.pipeline

        except Exception as e:
            logger.error(f"Failed to create pipeline: {str(e)}")
            raise

    def fit_pipeline(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        validation_split: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Fit the complete pipeline.

        Args:
            X: Training features
            y: Training targets
            validation_split: Optional validation split ratio

        Returns:
            Dictionary containing fit results
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be created before fitting")

        try:
            import time
            start_time = time.time()

            logger.info("Starting pipeline fit")

            # Validate and clean data
            X_clean, y_clean = self.data_validator.validate_and_clean(X, y)

            # Feature engineering if enabled
            if self.config.feature_engineering_enabled:
                X_clean = self.feature_engineer.engineer_features(X_clean, y_clean, fit=True)

            # Split data if validation requested
            if validation_split:
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_clean, y_clean,
                    test_size=validation_split,
                    random_state=self.config.random_state,
                    stratify=y_clean if self.config.target_type == 'classification' else None
                )
            else:
                X_train, y_train = X_clean, y_clean
                X_val, y_val = None, None

            # Fit pipeline
            self.pipeline.fit(X_train, y_train)
            self.is_fitted = True

            fit_time = time.time() - start_time

            # Prepare results
            results = {
                'fit_time': fit_time,
                'training_samples': len(X_train),
                'features': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0]),
                'pipeline_steps': [step[0] for step in self.pipeline.steps],
                'model_params': self.get_model_params()
            }

            # Add validation results if available
            if X_val is not None:
                val_predictions = self.pipeline.predict(X_val)
                results['validation_predictions'] = val_predictions
                results['validation_samples'] = len(X_val)

            logger.info(f"Pipeline fit completed in {fit_time:.3f}s")
            return results

        except Exception as e:
            logger.error(f"Pipeline fit failed: {str(e)}")
            raise

    def predict_pipeline(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        return_probabilities: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using the fitted pipeline.

        Args:
            X: Features for prediction
            return_probabilities: Whether to return probabilities (classification only)

        Returns:
            Predictions, optionally with probabilities
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")

        try:
            # Validate input data
            X_clean = self.data_validator.validate_features(X)

            # Feature engineering if enabled during training
            if self.config.feature_engineering_enabled:
                X_clean = self.feature_engineer.transform_features(X_clean)

            # Make predictions
            predictions = self.pipeline.predict(X_clean)

            # Return probabilities if requested
            if (return_probabilities and
                self.config.target_type == 'classification' and
                hasattr(self.pipeline.named_steps['model'], 'predict_proba')):
                probabilities = self.pipeline.predict_proba(X_clean)
                return predictions, probabilities

            return predictions

        except Exception as e:
            logger.error(f"Pipeline prediction failed: {str(e)}")
            raise

    def evaluate_pipeline(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate pipeline performance.

        Args:
            X: Test features
            y: True test targets
            detailed: Whether to include detailed metrics

        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before evaluation")

        try:
            from ..training.model_evaluator import ModelEvaluator

            # Make predictions
            predictions = self.predict_pipeline(X)

            # Evaluate using model evaluator
            evaluator = ModelEvaluator(self.config)
            results = evaluator.evaluate(y, predictions, self.config.target_type, detailed)

            # Add pipeline-specific information
            results['pipeline_info'] = {
                'algorithm': self.config.algorithm,
                'preprocessing_steps': [step[0] for step in self.preprocessing_steps],
                'model_type': type(self.pipeline.named_steps['model']).__name__
            }

            return results

        except Exception as e:
            logger.error(f"Pipeline evaluation failed: {str(e)}")
            raise

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from the pipeline model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            return None

        try:
            model = self.pipeline.named_steps['model']

            # Get feature names (considering feature engineering)
            if self.config.feature_engineering_enabled:
                feature_names = self.feature_engineer.get_feature_names()
            else:
                feature_names = getattr(self, 'feature_names_', None)

            # Get importance scores
            if hasattr(model, 'get_feature_importance'):
                importance_scores = model.get_feature_importance()
            elif hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_scores = np.abs(model.coef_).flatten()
            else:
                return None

            if feature_names and len(feature_names) == len(importance_scores):
                return dict(zip(feature_names, importance_scores))
            else:
                # Fallback to generic names
                generic_names = [f"feature_{i}" for i in range(len(importance_scores))]
                return dict(zip(generic_names, importance_scores))

        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return None

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get parameters of the pipeline model.

        Returns:
            Dictionary containing model parameters
        """
        if not self.is_fitted:
            return {}

        try:
            model = self.pipeline.named_steps['model']
            if hasattr(model, 'get_params'):
                return model.get_params()
            else:
                return {}
        except Exception as e:
            logger.warning(f"Could not extract model parameters: {str(e)}")
            return {}

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the pipeline.

        Returns:
            Dictionary containing pipeline information
        """
        info = {
            'is_fitted': self.is_fitted,
            'config': {
                'algorithm': self.config.algorithm,
                'target_type': self.config.target_type,
                'scaling_enabled': self.config.scaling_enabled,
                'feature_engineering_enabled': self.config.feature_engineering_enabled
            }
        }

        if self.pipeline:
            info['pipeline_steps'] = [step[0] for step in self.pipeline.steps]
            info['model_type'] = type(self.pipeline.named_steps['model']).__name__

        info.update(self.pipeline_metadata)

        return info

    def clone_pipeline(self) -> 'PipelineManager':
        """
        Create a clone of the pipeline manager.

        Returns:
            New PipelineManager instance with same configuration
        """
        clone = PipelineManager(self.config)
        if self.pipeline:
            from sklearn.base import clone
            clone.pipeline = clone(self.pipeline)
        return clone

    def reset_pipeline(self) -> None:
        """Reset the pipeline to unfitted state."""
        self.pipeline = None
        self.is_fitted = False
        self.preprocessing_steps = []
        self.model_step = None
        self.pipeline_metadata = {}
        self.feature_engineer.reset()

        logger.info("Pipeline reset to unfitted state")
