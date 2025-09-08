"""
Feature Engineering for ML System

Provides comprehensive feature engineering capabilities including
scaling, transformation, selection, and creation of new features.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..config.ml_config import MLConfig

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for ML systems.

    Provides scaling, transformation, selection, and creation of new features
    with proper handling of different data types and missing values.
    """

    def __init__(self, config: MLConfig):
        """
        Initialize feature engineer.

        Args:
            config: ML configuration object
        """
        self.config = config
        self.scalers = {}
        self.feature_selectors = {}
        self.encoders = {}
        self.is_fitted = False
        self.feature_names = None
        self.selected_features = None

    def engineer_features(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        fit: bool = True
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Apply comprehensive feature engineering.

        Args:
            X: Input features
            y: Target values (needed for supervised feature selection)
            fit: Whether to fit transformers or use existing ones

        Returns:
            Engineered features
        """
        try:
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
                X_processed = X.copy()
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                X_processed = pd.DataFrame(X, columns=self.feature_names)

            logger.info(f"Starting feature engineering with {len(self.feature_names)} features")

            # Handle missing values
            if self.config.handle_missing_values:
                X_processed = self._handle_missing_values(X_processed, fit)

            # Create new features
            X_processed = self._create_features(X_processed)

            # Encode categorical features
            X_processed = self._encode_categorical_features(X_processed, fit)

            # Scale features if enabled
            if self.config.scaling_enabled:
                X_processed = self._scale_features(X_processed, fit)

            # Feature selection
            if self.config.feature_selection_enabled and y is not None:
                X_processed = self._select_features(X_processed, y, fit)

            # Handle outliers
            if self.config.outlier_detection:
                X_processed = self._handle_outliers(X_processed, fit)

            if fit:
                self.is_fitted = True

            logger.info(f"Feature engineering completed. Output shape: {X_processed.shape}")

            # Return in original format
            if isinstance(X, np.ndarray):
                return X_processed.values
            return X_processed

        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise

    def transform_features(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform features using fitted transformers.

        Args:
            X: Input features to transform

        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transforming")

        return self.engineer_features(X, fit=False)

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        scaling: bool = None,
        feature_selection: bool = None,
        n_features: int = None,
        polynomial_features: bool = None,
        polynomial_degree: int = None,
        interaction_features: bool = None,
        create_features: bool = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Fit feature engineer and transform features in one step.

        Args:
            X: Input features
            y: Target values (needed for supervised feature selection)
            scaling: Whether to enable scaling (overrides config if provided)
            feature_selection: Whether to enable feature selection
            n_features: Number of features to select (if feature_selection=True)
            polynomial_features: Whether to create polynomial features
            polynomial_degree: Degree for polynomial features
            interaction_features: Whether to create interaction features
            create_features: Whether to create new features (if False, skips feature creation)

        Returns:
            Transformed features
        """
        # Store original config values
        original_scaling = self.config.scaling_enabled
        original_feature_selection = self.config.feature_selection_enabled
        original_n_features = getattr(self.config, 'feature_selection_k', 10)
        original_polynomial = getattr(self.config, 'polynomial_features_enabled', False)
        original_polynomial_degree = getattr(self.config, 'polynomial_degree', 2)
        original_interaction = getattr(self.config, 'interaction_features_enabled', False)
        original_feature_engineering = getattr(self.config, 'feature_engineering_enabled', True)

        # Temporarily override config values if parameters are provided
        if scaling is not None:
            self.config.scaling_enabled = scaling
        if feature_selection is not None:
            self.config.feature_selection_enabled = feature_selection
        if n_features is not None:
            self.config.feature_selection_k = n_features
        if polynomial_features is not None:
            self.config.polynomial_features_enabled = polynomial_features
        if polynomial_degree is not None:
            self.config.polynomial_degree = polynomial_degree
        if interaction_features is not None:
            self.config.interaction_features_enabled = interaction_features
        if create_features is not None:
            self.config.feature_engineering_enabled = create_features

        try:
            result = self.engineer_features(X, y, fit=True)
        finally:
            # Restore original config values
            self.config.scaling_enabled = original_scaling
            self.config.feature_selection_enabled = original_feature_selection
            self.config.feature_selection_k = original_n_features
            self.config.polynomial_features_enabled = original_polynomial
            self.config.polynomial_degree = original_polynomial_degree
            self.config.interaction_features_enabled = original_interaction
            self.config.feature_engineering_enabled = original_feature_engineering

        return result

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None):
        """
        Fit feature engineer transformers.

        Args:
            X: Input features
            y: Target values (needed for supervised feature selection)

        Returns:
            Self for method chaining
        """
        self.engineer_features(X, y, fit=True)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform features using fitted transformers.

        Args:
            X: Input features to transform

        Returns:
            Transformed features
        """
        return self.transform_features(X)

    def _handle_missing_values(
        self,
        X: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if not X.isnull().any().any():
            return X

        logger.info("Handling missing values")
        X_processed = X.copy()

        for column in X_processed.columns:
            if X_processed[column].isnull().any():
                if X_processed[column].dtype in ['object', 'category']:
                    # Categorical columns
                    if fit:
                        mode_value = X_processed[column].mode()
                        fill_value = mode_value[0] if len(mode_value) > 0 else 'unknown'
                        self.scalers[f'{column}_fill'] = fill_value
                    else:
                        fill_value = self.scalers.get(f'{column}_fill', 'unknown')
                    X_processed[column].fillna(fill_value, inplace=True)
                else:
                    # Numerical columns
                    if self.config.missing_value_strategy == 'mean':
                        if fit:
                            fill_value = X_processed[column].mean()
                            self.scalers[f'{column}_fill'] = fill_value
                        else:
                            fill_value = self.scalers.get(f'{column}_fill', 0)
                    elif self.config.missing_value_strategy == 'median':
                        if fit:
                            fill_value = X_processed[column].median()
                            self.scalers[f'{column}_fill'] = fill_value
                        else:
                            fill_value = self.scalers.get(f'{column}_fill', 0)
                    else:  # most_frequent
                        if fit:
                            fill_value = X_processed[column].mode()[0] if len(X_processed[column].mode()) > 0 else 0
                            self.scalers[f'{column}_fill'] = fill_value
                        else:
                            fill_value = self.scalers.get(f'{column}_fill', 0)

                    X_processed[column].fillna(fill_value, inplace=True)

        return X_processed

    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones."""
        logger.info("Creating new features")
        X_processed = X.copy()

        # Get numerical columns
        numerical_cols = X_processed.select_dtypes(include=[np.number]).columns.tolist()

        # Interaction features
        if getattr(self.config, 'interaction_features_enabled', False) and len(numerical_cols) >= 2:
            # Feature interactions (only create a few to avoid explosion)
            important_cols = numerical_cols[:5]  # Limit to first 5 numerical columns

            for i, col1 in enumerate(important_cols):
                for j, col2 in enumerate(important_cols[i+1:], i+1):
                    if len(X_processed.columns) < 100:  # Limit total features
                        # Multiplication
                        X_processed[f'{col1}_x_{col2}'] = X_processed[col1] * X_processed[col2]

                        # Ratio (avoid division by zero)
                        mask = X_processed[col2] != 0
                        ratio_col = f'{col1}_div_{col2}'
                        X_processed[ratio_col] = 0.0  # Use float to avoid dtype warning
                        X_processed.loc[mask, ratio_col] = X_processed.loc[mask, col1] / X_processed.loc[mask, col2]

        # Statistical features for numerical columns (only if basic feature engineering is enabled)
        if getattr(self.config, 'feature_engineering_enabled', True) and len(numerical_cols) >= 3:
            # Mean of all numerical features
            X_processed['numerical_mean'] = X_processed[numerical_cols].mean(axis=1)
            # Standard deviation
            X_processed['numerical_std'] = X_processed[numerical_cols].std(axis=1)
            # Sum
            X_processed['numerical_sum'] = X_processed[numerical_cols].sum(axis=1)

        # Polynomial features
        if getattr(self.config, 'polynomial_features_enabled', False):
            polynomial_degree = getattr(self.config, 'polynomial_degree', 2)
            for col in numerical_cols[:3]:  # Only for first 3 columns to avoid explosion
                if polynomial_degree >= 2:
                    X_processed[f'{col}_squared'] = X_processed[col] ** 2
                if polynomial_degree >= 3:
                    X_processed[f'{col}_cubed'] = X_processed[col] ** 3
                # Always add sqrt for polynomial features
                X_processed[f'{col}_sqrt'] = np.sqrt(np.abs(X_processed[col]))

        return X_processed

    def _encode_categorical_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        if not categorical_cols:
            return X

        logger.info(f"Encoding {len(categorical_cols)} categorical features")
        X_processed = X.copy()

        for col in categorical_cols:
            if fit:
                encoder = LabelEncoder()
                X_processed[col] = encoder.fit_transform(X_processed[col].astype(str))
                self.encoders[col] = encoder
            else:
                encoder = self.encoders.get(col)
                if encoder:
                    # Handle unseen categories
                    unique_values = set(encoder.classes_)
                    X_processed[col] = X_processed[col].astype(str)
                    X_processed[col] = X_processed[col].apply(
                        lambda x: x if x in unique_values else encoder.classes_[0]
                    )
                    X_processed[col] = encoder.transform(X_processed[col])

        return X_processed

    def _scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if not numerical_cols:
            return X

        logger.info(f"Scaling {len(numerical_cols)} numerical features")
        X_processed = X.copy()

        if fit:
            scaler = StandardScaler()
            X_processed[numerical_cols] = scaler.fit_transform(X_processed[numerical_cols])
            self.scalers['standard_scaler'] = scaler
        else:
            scaler = self.scalers.get('standard_scaler')
            if scaler:
                X_processed[numerical_cols] = scaler.transform(X_processed[numerical_cols])

        return X_processed

    def _select_features(
        self,
        X: pd.DataFrame,
        y: Union[np.ndarray, pd.Series],
        fit: bool = True
    ) -> pd.DataFrame:
        """Select best features using statistical tests."""
        logger.info(f"Selecting {self.config.feature_selection_k} best features")

        if fit:
            # Choose selector based on target type
            if self.config.target_type == 'regression':
                selector = SelectKBest(score_func=f_regression, k=min(self.config.feature_selection_k, X.shape[1]))
            else:
                selector = SelectKBest(score_func=f_classif, k=min(self.config.feature_selection_k, X.shape[1]))

            X_selected = selector.fit_transform(X, y)
            self.feature_selectors['kbest'] = selector
            self.selected_features = X.columns[selector.get_support()].tolist()
        else:
            selector = self.feature_selectors.get('kbest')
            if selector and self.selected_features:
                X_selected = selector.transform(X)
            else:
                return X

        # Create DataFrame with selected features
        selected_df = pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
        return selected_df

    def _handle_outliers(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle outliers using clipping."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if not numerical_cols:
            return X

        logger.info("Handling outliers")
        X_processed = X.copy()

        for col in numerical_cols:
            if fit:
                Q1 = X_processed[col].quantile(0.25)
                Q3 = X_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.config.outlier_threshold * IQR
                upper_bound = Q3 + self.config.outlier_threshold * IQR

                self.scalers[f'{col}_lower_bound'] = lower_bound
                self.scalers[f'{col}_upper_bound'] = upper_bound
            else:
                lower_bound = self.scalers.get(f'{col}_lower_bound', X_processed[col].min())
                upper_bound = self.scalers.get(f'{col}_upper_bound', X_processed[col].max())

            X_processed[col] = X_processed[col].clip(lower_bound, upper_bound)

        return X_processed

    def get_feature_names(self) -> Optional[List[str]]:
        """Get names of features after engineering."""
        if self.selected_features:
            return self.selected_features
        return self.feature_names

    def get_feature_importance_from_selection(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores from feature selection."""
        selector = self.feature_selectors.get('kbest')
        if selector and hasattr(selector, 'scores_'):
            feature_names = self.get_feature_names()
            if feature_names and len(feature_names) == len(selector.scores_):
                return dict(zip(feature_names, selector.scores_))
        return None

    def reset(self) -> None:
        """Reset the feature engineer to unfitted state."""
        self.scalers.clear()
        self.feature_selectors.clear()
        self.encoders.clear()
        self.is_fitted = False
        self.feature_names = None
        self.selected_features = None
