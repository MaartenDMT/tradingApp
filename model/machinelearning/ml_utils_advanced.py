"""
Advanced ML utilities for data preprocessing, validation, and performance optimization.
"""
import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


@dataclass
class PerformanceMetrics:
    """Data class for storing performance metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class DataPreprocessor:
    """Advanced data preprocessing utilities."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.scalers = {}
        self.fitted_columns = {}

    def smart_scaling(self, X: pd.DataFrame, method: str = 'auto',
                     columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply intelligent scaling based on data distribution.

        :param X: Input data
        :param method: Scaling method ('auto', 'standard', 'minmax', 'robust')
        :param columns: Specific columns to scale
        :return: Scaled data
        """
        X_scaled = X.copy()

        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in X.columns:
                continue

            data = X[col].dropna()
            if len(data) == 0:
                continue

            # Determine best scaling method if auto
            if method == 'auto':
                # Check for outliers
                q1, q3 = data.quantile([0.25, 0.75])
                iqr = q3 - q1
                outlier_ratio = ((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).sum() / len(data)

                # Check distribution
                skewness = abs(data.skew())

                if outlier_ratio > 0.1:
                    chosen_method = 'robust'
                elif skewness > 2:
                    chosen_method = 'robust'
                elif data.min() >= 0 and data.max() <= 1:
                    chosen_method = 'none'  # Already normalized
                else:
                    chosen_method = 'standard'
            else:
                chosen_method = method

            # Apply scaling
            if chosen_method == 'standard':
                scaler = StandardScaler()
            elif chosen_method == 'minmax':
                scaler = MinMaxScaler()
            elif chosen_method == 'robust':
                scaler = RobustScaler()
            else:
                continue  # No scaling

            X_scaled[col] = scaler.fit_transform(X[[col]])
            self.scalers[col] = scaler

            self.logger.debug(f"Applied {chosen_method} scaling to {col}")

        self.fitted_columns = columns
        return X_scaled

    def handle_missing_values(self, X: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing values with intelligent strategies.

        :param X: Input data
        :param strategy: Strategy ('auto', 'drop', 'mean', 'median', 'mode', 'forward_fill')
        :return: Data with handled missing values
        """
        X_filled = X.copy()

        for col in X.columns:
            missing_ratio = X[col].isnull().sum() / len(X)

            if missing_ratio == 0:
                continue

            if strategy == 'auto':
                if missing_ratio > 0.5:
                    # Too many missing values, consider dropping
                    chosen_strategy = 'drop'
                elif X[col].dtype in ['object', 'category']:
                    chosen_strategy = 'mode'
                elif X[col].dtype in ['datetime64[ns]']:
                    chosen_strategy = 'forward_fill'
                else:
                    # Numeric data
                    if abs(X[col].skew()) > 2:
                        chosen_strategy = 'median'
                    else:
                        chosen_strategy = 'mean'
            else:
                chosen_strategy = strategy

            # Apply strategy
            if chosen_strategy == 'drop':
                X_filled = X_filled.drop(columns=[col])
            elif chosen_strategy == 'mean':
                X_filled[col].fillna(X_filled[col].mean(), inplace=True)
            elif chosen_strategy == 'median':
                X_filled[col].fillna(X_filled[col].median(), inplace=True)
            elif chosen_strategy == 'mode':
                mode_value = X_filled[col].mode()
                if not mode_value.empty:
                    X_filled[col].fillna(mode_value[0], inplace=True)
            elif chosen_strategy == 'forward_fill':
                X_filled[col].fillna(method='ffill', inplace=True)
                X_filled[col].fillna(method='bfill', inplace=True)  # Fill remaining

            self.logger.debug(f"Applied {chosen_strategy} to {col} (missing: {missing_ratio:.2%})")

        return X_filled

    def detect_outliers(self, X: pd.DataFrame, method: str = 'iqr',
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Detect outliers in the data.

        :param X: Input data
        :param method: Detection method ('iqr', 'zscore', 'isolation_forest')
        :param columns: Columns to check for outliers
        :return: Boolean DataFrame indicating outliers
        """
        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()

        outliers = pd.DataFrame(False, index=X.index, columns=X.columns)

        for col in columns:
            if col not in X.columns:
                continue

            data = X[col].dropna()
            if len(data) == 0:
                continue

            if method == 'iqr':
                q1, q3 = data.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers[col] = (X[col] < lower_bound) | (X[col] > upper_bound)

            elif method == 'zscore':
                z_scores = np.abs((data - data.mean()) / data.std())
                outliers.loc[data.index, col] = z_scores > 3

            elif method == 'isolation_forest':
                try:
                    from sklearn.ensemble import IsolationForest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_pred = iso_forest.fit_predict(X[[col]].dropna())
                    outliers.loc[data.index, col] = outlier_pred == -1
                except ImportError:
                    self.logger.warning("IsolationForest not available, using IQR method")
                    # Fallback to IQR
                    q1, q3 = data.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers[col] = (X[col] < lower_bound) | (X[col] > upper_bound)

        return outliers


class ModelValidator:
    """Advanced model validation utilities."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         problem_type: str = 'auto') -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        :param y_true: True values
        :param y_pred: Predicted values
        :param problem_type: 'classification', 'regression', or 'auto'
        :return: PerformanceMetrics object
        """
        metrics = PerformanceMetrics()

        # Auto-detect problem type
        if problem_type == 'auto':
            unique_values = len(np.unique(y_true))
            is_integer = np.issubdtype(y_true.dtype, np.integer)
            problem_type = 'classification' if unique_values < 20 and is_integer else 'regression'

        try:
            if problem_type == 'classification':
                metrics.accuracy = accuracy_score(y_true, y_pred)
                metrics.precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics.recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics.f1_score = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            else:  # regression
                metrics.mse = mean_squared_error(y_true, y_pred)
                metrics.rmse = np.sqrt(metrics.mse)
                metrics.mae = mean_absolute_error(y_true, y_pred)
                metrics.r2_score = r2_score(y_true, y_pred)

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")

        return metrics

    def cross_validate_with_metrics(self, model: Any, X: np.ndarray, y: np.ndarray,
                                   cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation with comprehensive metrics.

        :param model: Model to validate
        :param X: Features
        :param y: Targets
        :param cv_folds: Number of CV folds
        :return: Cross-validation results
        """
        from sklearn.model_selection import KFold, StratifiedKFold

        # Determine problem type
        unique_values = len(np.unique(y))
        is_classification = unique_values < 20 and np.issubdtype(y.dtype, np.integer)

        if is_classification:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_val)

            # Calculate metrics
            metrics = self.calculate_metrics(y_val, y_pred,
                                           'classification' if is_classification else 'regression')

            fold_result = {
                'fold': fold,
                'metrics': metrics.to_dict()
            }
            fold_metrics.append(fold_result)

        # Aggregate results
        all_metrics = [fold['metrics'] for fold in fold_metrics]
        aggregated = {}

        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics]
            aggregated[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }

        return {
            'fold_results': fold_metrics,
            'aggregated_metrics': aggregated,
            'cv_folds': cv_folds,
            'problem_type': 'classification' if is_classification else 'regression'
        }


class ResourceMonitor:
    """Monitor system resources during ML operations."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.start_memory = None

    def start_monitoring(self):
        """Start resource monitoring."""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        self.logger.info("Started resource monitoring")

    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        current_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        cpu_percent = psutil.cpu_percent(interval=1)

        usage = {
            'cpu_percent': cpu_percent,
            'memory_mb': current_memory,
            'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
        }

        if self.start_time:
            usage['elapsed_seconds'] = time.time() - self.start_time

        if self.start_memory:
            usage['memory_increase_mb'] = current_memory - self.start_memory

        return usage

    def log_usage(self):
        """Log current resource usage."""
        usage = self.get_current_usage()
        self.logger.info(f"CPU: {usage['cpu_percent']:.1f}%, "
                        f"Memory: {usage['memory_mb']:.0f}MB, "
                        f"Available: {usage['memory_available_mb']:.0f}MB")

        if 'elapsed_seconds' in usage:
            self.logger.info(f"Elapsed: {usage['elapsed_seconds']:.1f}s")

    def cleanup_memory(self):
        """Force garbage collection to free memory."""
        gc.collect()
        self.logger.debug("Performed garbage collection")


class DataSplitter:
    """Advanced data splitting utilities."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def time_series_split(self, X: pd.DataFrame, y: pd.Series,
                         test_size: float = 0.2, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time series cross-validation splits.

        :param X: Features with datetime index
        :param y: Targets
        :param test_size: Proportion of data for testing
        :param n_splits: Number of splits
        :return: List of (train_idx, test_idx) tuples
        """
        n_samples = len(X)
        test_size_samples = int(n_samples * test_size)

        splits = []
        for i in range(n_splits):
            # Progressive training set size
            train_end = n_samples - test_size_samples - (n_splits - i - 1) * (test_size_samples // n_splits)
            test_start = train_end
            test_end = min(test_start + test_size_samples, n_samples)

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        self.logger.info(f"Created {len(splits)} time series splits")
        return splits

    def stratified_group_split(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                              test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data while maintaining group integrity and target distribution.

        :param X: Features
        :param y: Targets
        :param groups: Group identifiers
        :param test_size: Test set proportion
        :param random_state: Random seed
        :return: Train and test indices
        """
        from sklearn.model_selection import GroupShuffleSplit

        # Use GroupShuffleSplit as base
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))

        self.logger.info(f"Group split: {len(train_idx)} train, {len(test_idx)} test samples")
        return train_idx, test_idx


# Utility functions
def save_model_metadata(model: Any, metadata: Dict[str, Any], filepath: Union[str, Path]):
    """Save model metadata to file."""
    import json

    metadata_path = Path(filepath).with_suffix('.metadata.json')

    # Make metadata JSON serializable
    serializable_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            serializable_metadata[key] = value
        elif isinstance(value, np.ndarray):
            serializable_metadata[key] = value.tolist()
        else:
            serializable_metadata[key] = str(value)

    with open(metadata_path, 'w') as f:
        json.dump(serializable_metadata, f, indent=2)


def load_model_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load model metadata from file."""
    import json

    metadata_path = Path(filepath).with_suffix('.metadata.json')

    if not metadata_path.exists():
        return {}

    with open(metadata_path, 'r') as f:
        return json.load(f)


def create_feature_engineering_pipeline(X: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Create an automated feature engineering pipeline.

    :param X: Input data
    :param target_col: Target column name if present in X
    :return: Feature engineering recommendations
    """
    recommendations = {
        'scaling_recommendations': {},
        'encoding_recommendations': {},
        'feature_creation_recommendations': [],
        'outlier_handling_recommendations': {}
    }

    # Analyze each column
    for col in X.columns:
        if col == target_col:
            continue

        col_data = X[col].dropna()
        if len(col_data) == 0:
            continue

        # Numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            # Scaling recommendation
            skewness = abs(col_data.skew())
            q1, q3 = col_data.quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_ratio = ((col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))).sum() / len(col_data)

            if outlier_ratio > 0.1 or skewness > 2:
                recommendations['scaling_recommendations'][col] = 'robust'
            elif col_data.min() >= 0 and col_data.max() <= 1:
                recommendations['scaling_recommendations'][col] = 'none'
            else:
                recommendations['scaling_recommendations'][col] = 'standard'

            # Outlier handling
            if outlier_ratio > 0.05:
                recommendations['outlier_handling_recommendations'][col] = {
                    'method': 'iqr_clipping' if outlier_ratio < 0.2 else 'transformation',
                    'outlier_ratio': outlier_ratio
                }

        # Categorical columns
        elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
            unique_values = col_data.nunique()
            cardinality_ratio = unique_values / len(col_data)

            if cardinality_ratio < 0.05:  # Low cardinality
                recommendations['encoding_recommendations'][col] = 'one_hot'
            elif unique_values < 100:  # Medium cardinality
                recommendations['encoding_recommendations'][col] = 'target_encoding'
            else:  # High cardinality
                recommendations['encoding_recommendations'][col] = 'frequency_encoding'

    # Feature creation recommendations
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        recommendations['feature_creation_recommendations'].extend([
            'polynomial_features',
            'interaction_features',
            'feature_ratios'
        ])

    return recommendations


# Export all classes and functions
__all__ = [
    'PerformanceMetrics',
    'DataPreprocessor',
    'ModelValidator',
    'ResourceMonitor',
    'DataSplitter',
    'save_model_metadata',
    'load_model_metadata',
    'create_feature_engineering_pipeline'
]
