"""
Data Validation for ML System

Provides comprehensive data validation and cleaning capabilities
to ensure data quality before model training and prediction.
"""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validation for ML systems.

    Validates data quality, consistency, and format to ensure
    reliable model training and prediction.
    """

    def __init__(self):
        """Initialize data validator."""
        self.validation_history = []

    def validate_and_clean(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> tuple:
        """
        Validate and clean input data.

        Args:
            X: Input features
            y: Target values (optional)

        Returns:
            Tuple of (cleaned_X, cleaned_y)
        """
        logger.info("Starting data validation and cleaning")

        # Convert to appropriate format
        X_clean = self._convert_to_dataframe(X)
        y_clean = y

        # Basic validation
        self._validate_basic_requirements(X_clean, y_clean)

        # Clean data
        X_clean = self._clean_features(X_clean)
        if y_clean is not None:
            y_clean = self._clean_targets(y_clean)

        # Final validation
        self._validate_final_data(X_clean, y_clean)

        logger.info(f"Data validation completed. Shape: {X_clean.shape}")

        # Convert back to original format if needed
        if isinstance(X, np.ndarray):
            X_clean = X_clean.values
        if isinstance(y, np.ndarray) and y_clean is not None:
            y_clean = np.array(y_clean) if not isinstance(y_clean, np.ndarray) else y_clean

        return X_clean, y_clean

    def validate_features(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        expected_features: Optional[List[str]] = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Validate features for prediction.

        Args:
            X: Input features
            expected_features: Expected feature names

        Returns:
            Validated features
        """
        X_clean = self._convert_to_dataframe(X)

        # Validate feature consistency
        if expected_features:
            self._validate_feature_consistency(X_clean, expected_features)

        # Clean features
        X_clean = self._clean_features(X_clean)

        # Convert back to original format if needed
        if isinstance(X, np.ndarray):
            return X_clean.values
        return X_clean

    def _convert_to_dataframe(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """Convert input to DataFrame for consistent processing."""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        elif isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        else:
            raise ValueError(f"Unsupported data type: {type(X)}")

    def _validate_basic_requirements(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> None:
        """Validate basic data requirements."""
        # Check if data is empty
        if X.empty:
            raise ValueError("Input features are empty")

        # Check for all NaN columns
        all_nan_cols = X.columns[X.isnull().all()].tolist()
        if all_nan_cols:
            logger.warning(f"Columns with all NaN values will be dropped: {all_nan_cols}")

        # Check for constant columns
        constant_cols = []
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].nunique() <= 1:
                constant_cols.append(col)

        if constant_cols:
            logger.warning(f"Constant columns detected: {constant_cols}")

        # Validate targets if provided
        if y is not None:
            if len(y) != len(X):
                raise ValueError(f"Feature and target lengths don't match: {len(X)} vs {len(y)}")

            if isinstance(y, (pd.Series, np.ndarray)) and len(y) == 0:
                raise ValueError("Target values are empty")

    def _clean_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean feature data."""
        X_clean = X.copy()

        # Drop columns with all NaN values
        all_nan_cols = X_clean.columns[X_clean.isnull().all()].tolist()
        if all_nan_cols:
            X_clean = X_clean.drop(columns=all_nan_cols)
            logger.info(f"Dropped {len(all_nan_cols)} columns with all NaN values")

        # Drop constant columns
        constant_cols = []
        for col in X_clean.select_dtypes(include=[np.number]).columns:
            if X_clean[col].nunique() <= 1:
                constant_cols.append(col)

        if constant_cols:
            X_clean = X_clean.drop(columns=constant_cols)
            logger.info(f"Dropped {len(constant_cols)} constant columns")

        # Handle infinite values
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(X_clean[col]).any():
                logger.warning(f"Infinite values found in column {col}, replacing with NaN")
                X_clean[col] = X_clean[col].replace([np.inf, -np.inf], np.nan)

        # Handle extremely large values
        for col in numeric_cols:
            if X_clean[col].dtype in ['float64', 'int64']:
                max_val = np.finfo(np.float32).max if X_clean[col].dtype == 'float64' else np.iinfo(np.int32).max
                if (X_clean[col].abs() > max_val).any():
                    logger.warning(f"Extremely large values found in column {col}, clipping")
                    X_clean[col] = X_clean[col].clip(-max_val, max_val)

        return X_clean

    def _clean_targets(self, y: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
        """Clean target data."""
        if isinstance(y, pd.Series):
            y_clean = y.copy()
        else:
            y_clean = np.array(y)

        # Handle infinite values in targets
        if isinstance(y_clean, pd.Series):
            if np.isinf(y_clean).any():
                logger.warning("Infinite values found in targets, replacing with NaN")
                y_clean = y_clean.replace([np.inf, -np.inf], np.nan)
        else:
            if np.isinf(y_clean).any():
                logger.warning("Infinite values found in targets, replacing with NaN")
                y_clean = np.where(np.isinf(y_clean), np.nan, y_clean)

        return y_clean

    def _validate_feature_consistency(
        self,
        X: pd.DataFrame,
        expected_features: List[str]
    ) -> None:
        """Validate feature consistency for prediction."""
        current_features = X.columns.tolist()

        # Check for missing features
        missing_features = set(expected_features) - set(current_features)
        if missing_features:
            raise ValueError(f"Missing expected features: {missing_features}")

        # Check for extra features
        extra_features = set(current_features) - set(expected_features)
        if extra_features:
            logger.warning(f"Extra features found (will be ignored): {extra_features}")

        # Reorder columns to match expected order
        if current_features != expected_features:
            available_expected = [f for f in expected_features if f in current_features]
            X = X[available_expected]

    def _validate_final_data(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> None:
        """Perform final validation checks."""
        # Check if any data remains
        if X.empty:
            raise ValueError("No valid features remain after cleaning")

        # Check for excessive missing values
        missing_percentage = X.isnull().sum() / len(X)
        high_missing_cols = missing_percentage[missing_percentage > 0.8].index.tolist()

        if high_missing_cols:
            logger.warning(f"Columns with >80% missing values: {high_missing_cols}")

        # Check data types
        problematic_cols = []
        for col in X.columns:
            if X[col].dtype == 'object':
                # Check if string column has too many unique values
                if X[col].nunique() > len(X) * 0.8:
                    problematic_cols.append(col)

        if problematic_cols:
            logger.warning(f"High-cardinality categorical columns: {problematic_cols}")

        # Validate targets
        if y is not None:
            if isinstance(y, pd.Series):
                if y.isnull().all():
                    raise ValueError("All target values are missing")
            else:
                if np.isnan(y).all():
                    raise ValueError("All target values are missing")

    def get_data_quality_report(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> dict:
        """
        Generate a comprehensive data quality report.

        Args:
            X: Input features
            y: Target values (optional)

        Returns:
            Dictionary containing data quality metrics
        """
        X_df = self._convert_to_dataframe(X)

        report = {
            'n_samples': len(X_df),
            'n_features': len(X_df.columns),
            'missing_values': {},
            'data_types': {},
            'feature_stats': {},
            'potential_issues': []
        }

        # Missing values analysis
        missing_counts = X_df.isnull().sum()
        missing_percentages = (missing_counts / len(X_df) * 100).round(2)

        for col in X_df.columns:
            if missing_counts[col] > 0:
                report['missing_values'][col] = {
                    'count': int(missing_counts[col]),
                    'percentage': float(missing_percentages[col])
                }

        # Data types
        for col in X_df.columns:
            report['data_types'][col] = str(X_df[col].dtype)

        # Feature statistics for numeric columns
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats = X_df[col].describe()
            report['feature_stats'][col] = {
                'mean': float(stats['mean']) if not np.isnan(stats['mean']) else None,
                'std': float(stats['std']) if not np.isnan(stats['std']) else None,
                'min': float(stats['min']) if not np.isnan(stats['min']) else None,
                'max': float(stats['max']) if not np.isnan(stats['max']) else None,
                'unique_values': int(X_df[col].nunique())
            }

        # Identify potential issues
        # High missing values
        high_missing = missing_percentages[missing_percentages > 50].index.tolist()
        if high_missing:
            report['potential_issues'].append(f"High missing values (>50%): {high_missing}")

        # Constant features
        constant_features = [col for col in numeric_cols if X_df[col].nunique() <= 1]
        if constant_features:
            report['potential_issues'].append(f"Constant features: {constant_features}")

        # High cardinality categorical features
        categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns
        high_cardinality = []
        for col in categorical_cols:
            if X_df[col].nunique() > len(X_df) * 0.8:
                high_cardinality.append(col)

        if high_cardinality:
            report['potential_issues'].append(f"High cardinality categorical: {high_cardinality}")

        # Target analysis if provided
        if y is not None:
            if isinstance(y, pd.Series):
                y_array = y.values
            else:
                y_array = np.array(y)

            report['target_stats'] = {
                'missing_count': int(np.isnan(y_array).sum()),
                'missing_percentage': float(np.isnan(y_array).sum() / len(y_array) * 100),
                'unique_values': int(pd.Series(y_array).nunique())
            }

            if not np.isnan(y_array).all():
                if pd.Series(y_array).dtype in [np.number]:
                    target_stats = pd.Series(y_array).describe()
                    report['target_stats'].update({
                        'mean': float(target_stats['mean']) if not np.isnan(target_stats['mean']) else None,
                        'std': float(target_stats['std']) if not np.isnan(target_stats['std']) else None,
                        'min': float(target_stats['min']) if not np.isnan(target_stats['min']) else None,
                        'max': float(target_stats['max']) if not np.isnan(target_stats['max']) else None
                    })

        return report
