"""
Base ML Estimator for ML System

Provides base class for all ML estimators with common functionality
including validation, logging, and performance tracking.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

logger = logging.getLogger(__name__)


class BaseMLEstimator(BaseEstimator, ABC):
    """
    Base class for all ML estimators in the system.

    Provides common functionality including validation, logging,
    and performance tracking that all estimators should inherit.
    """

    def __init__(self, **kwargs):
        """
        Initialize base ML estimator.

        Args:
            **kwargs: Estimator-specific parameters
        """
        self.is_fitted = False
        self.feature_names = None
        self.n_features = None
        self.training_time = None
        self.model_metadata = {}

        # Set parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'BaseMLEstimator':
        """
        Fit the estimator to training data.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Self (fitted estimator)
        """
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the fitted estimator.

        Args:
            X: Features for prediction

        Returns:
            Predictions array
        """
        pass

    def validate_input(self, X: Union[np.ndarray, pd.DataFrame], training: bool = True) -> None:
        """
        Validate input data format and consistency.

        Args:
            X: Input features
            training: Whether this is training data
        """
        # Basic validation
        if X is None or (hasattr(X, '__len__') and len(X) == 0):
            raise ValueError("Input data is empty")

        # Convert to numpy for shape checking
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            current_features = X.columns.tolist()
        else:
            X_array = np.array(X)
            current_features = None

        # Check dimensions
        if X_array.ndim != 2:
            raise ValueError(f"Input must be 2D, got {X_array.ndim}D")

        if training:
            # Store feature information for training
            self.n_features = X_array.shape[1]
            if current_features:
                self.feature_names = current_features
        else:
            # Validate consistency for prediction
            if not self.is_fitted:
                raise ValueError("Estimator must be fitted before prediction")

            if X_array.shape[1] != self.n_features:
                raise ValueError(f"Feature count mismatch: expected {self.n_features}, got {X_array.shape[1]}")

            if self.feature_names and current_features:
                if current_features != self.feature_names:
                    logger.warning("Feature names don't match training data")

    def validate_target(self, y: Union[np.ndarray, pd.Series]) -> None:
        """
        Validate target data format.

        Args:
            y: Target values
        """
        if y is None or (hasattr(y, '__len__') and len(y) == 0):
            raise ValueError("Target data is empty")

        # Convert to numpy for validation
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)

        # Check for all NaN
        if np.isnan(y_array).all():
            raise ValueError("All target values are NaN")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the fitted estimator.

        Returns:
            Dictionary containing estimator metadata
        """
        metadata = {
            'estimator_type': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'training_time': self.training_time
        }
        metadata.update(self.model_metadata)
        return metadata

    def set_metadata(self, **kwargs) -> None:
        """
        Set custom metadata for the estimator.

        Args:
            **kwargs: Metadata key-value pairs
        """
        self.model_metadata.update(kwargs)


class BaseMLClassifier(BaseMLEstimator, ClassifierMixin):
    """
    Base class for classification estimators.

    Extends BaseMLEstimator with classification-specific functionality.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classes_ = None
        self.n_classes_ = None

    def validate_target(self, y: Union[np.ndarray, pd.Series]) -> None:
        """
        Validate classification target data.

        Args:
            y: Target values
        """
        super().validate_target(y)

        # Convert to numpy
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)

        # Store class information
        self.classes_ = np.unique(y_array[~np.isnan(y_array)])
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ < 2:
            raise ValueError(f"Need at least 2 classes, got {self.n_classes_}")

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features for prediction

        Returns:
            Class probabilities array
        """
        raise NotImplementedError("predict_proba not implemented")

    def predict_log_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict log class probabilities.

        Args:
            X: Features for prediction

        Returns:
            Log class probabilities array
        """
        return np.log(self.predict_proba(X))


class BaseMLRegressor(BaseMLEstimator, RegressorMixin):
    """
    Base class for regression estimators.

    Extends BaseMLEstimator with regression-specific functionality.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_mean_ = None
        self.target_std_ = None

    def validate_target(self, y: Union[np.ndarray, pd.Series]) -> None:
        """
        Validate regression target data.

        Args:
            y: Target values
        """
        super().validate_target(y)

        # Convert to numpy
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)

        # Check for infinite values
        if np.isinf(y_array).any():
            raise ValueError("Target contains infinite values")

        # Store target statistics
        valid_targets = y_array[~np.isnan(y_array)]
        if len(valid_targets) > 0:
            self.target_mean_ = np.mean(valid_targets)
            self.target_std_ = np.std(valid_targets)


class SklearnEstimatorWrapper(BaseMLEstimator):
    """
    Wrapper for scikit-learn estimators to provide base functionality.

    Allows existing sklearn estimators to be used with the ML system
    while providing consistent interface and functionality.
    """

    def __init__(self, estimator, **kwargs):
        """
        Initialize wrapper around sklearn estimator.

        Args:
            estimator: Scikit-learn estimator instance
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.estimator = estimator
        self.estimator_type = 'sklearn_wrapper'

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'SklearnEstimatorWrapper':
        """
        Fit the wrapped estimator.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Self (fitted wrapper)
        """
        import time

        # Validate inputs
        self.validate_input(X, training=True)
        self.validate_target(y)

        # Convert to appropriate format
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)

        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)

        # Fit estimator
        start_time = time.time()
        self.estimator.fit(X_array, y_array)
        self.training_time = time.time() - start_time

        # Update state
        self.is_fitted = True

        # Copy relevant attributes
        if hasattr(self.estimator, 'classes_'):
            self.classes_ = self.estimator.classes_
            self.n_classes_ = len(self.classes_)

        logger.info(f"Fitted {self.estimator.__class__.__name__} in {self.training_time:.3f}s")

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the wrapped estimator.

        Args:
            X: Features for prediction

        Returns:
            Predictions array
        """
        # Validate input
        self.validate_input(X, training=False)

        # Convert to appropriate format
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)

        # Make predictions
        return self.estimator.predict(X_array)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities using the wrapped estimator.

        Args:
            X: Features for prediction

        Returns:
            Class probabilities array
        """
        if not hasattr(self.estimator, 'predict_proba'):
            raise NotImplementedError("Wrapped estimator doesn't support predict_proba")

        # Validate input
        self.validate_input(X, training=False)

        # Convert to appropriate format
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)

        return self.estimator.predict_proba(X_array)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters of the wrapped estimator.

        Args:
            deep: Whether to get nested parameters

        Returns:
            Parameters dictionary
        """
        params = self.estimator.get_params(deep=deep)
        params['estimator'] = self.estimator
        return params

    def set_params(self, **params) -> 'SklearnEstimatorWrapper':
        """
        Set parameters of the wrapped estimator.

        Args:
            **params: Parameters to set

        Returns:
            Self
        """
        if 'estimator' in params:
            self.estimator = params.pop('estimator')

        if params:
            self.estimator.set_params(**params)

        return self

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance from the wrapped estimator.

        Returns:
            Feature importance array or None if not available
        """
        if hasattr(self.estimator, 'feature_importances_'):
            return self.estimator.feature_importances_
        elif hasattr(self.estimator, 'coef_'):
            return np.abs(self.estimator.coef_).flatten()
        else:
            return None
