"""
Data Manager for the trading reinforcement learning system.
Handles data loading, preprocessing, and feature engineering.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import util.loggers as loggers
from util.utils import convert_df, features, tradex_features

logger = loggers.setup_loggers()
env_logger = logger['env']
rl_logger = logger['rl']


class DataManager:
    """
    Centralized data management for trading environments.
    """

    def __init__(self, config: Dict, symbol: str, percentage: float = 15.0):
        """
        Initialize the data manager.

        Args:
            config: Configuration dictionary
            symbol: Trading symbol
            percentage: Percentage of data to use for training
        """
        self.config = config
        self.symbol = symbol
        self.percentage = percentage / 100.0

        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.feature_data: Optional[pd.DataFrame] = None
        self.normalized_data: Optional[pd.DataFrame] = None

        self._data_loaded = False
        self._features_created = False

    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load trading data from pickle file.

        Args:
            data_path: Optional custom path to data file

        Returns:
            Loaded DataFrame
        """
        if data_path is None:
            data_path = 'data/pickle/all/30m_data_all.pkl'

        if not os.path.exists(data_path):
            env_logger.error(f'Data file not found: {data_path}')
            return pd.DataFrame()

        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)

            if data.empty:
                env_logger.error("Loaded data is empty")
                return pd.DataFrame()

            # Convert and preprocess
            data = convert_df(data)

            if data.empty or data.isnull().values.any():
                env_logger.warning("Data contains NaN values, cleaning...")
                data = data.dropna()

            # Apply percentage filter
            if self.percentage < 1.0:
                rows_to_keep = int(len(data) * self.percentage)
                data = data.head(rows_to_keep)

            self.raw_data = data
            self._data_loaded = True

            env_logger.info(f'Data loaded successfully. Shape: {data.shape}')
            rl_logger.info(f'Using {self.percentage*100:.1f}% of data: {len(data)} rows')

            return data

        except Exception as e:
            env_logger.error(f"Error loading data: {e}")
            return pd.DataFrame()

    def create_features(self, feature_list: List[str] = None) -> pd.DataFrame:
        """
        Create technical indicators and features.

        Args:
            feature_list: Optional list of specific features to create

        Returns:
            DataFrame with features
        """
        if not self._data_loaded or self.raw_data is None or self.raw_data.empty:
            env_logger.error("Data must be loaded before creating features")
            return pd.DataFrame()

        try:
            # Ensure volume is properly formatted
            data_copy = self.raw_data.copy()
            data_copy['volume'] = data_copy['volume'].astype(float).round(6)

            # Generate basic features
            basic_features = features(data_copy.copy())

            # Generate trading-specific features
            trading_features = tradex_features(self.symbol, data_copy.copy())

            if basic_features.empty or trading_features.empty:
                env_logger.error("Feature generation failed")
                return pd.DataFrame()

            # Combine features
            combined_data = pd.concat([trading_features, basic_features], axis=1)
            combined_data.dropna(inplace=True)

            # Add derived features
            combined_data = self._add_derived_features(combined_data)

            # Filter to specific features if provided
            if feature_list:
                available_features = [f for f in feature_list if f in combined_data.columns]
                if len(available_features) != len(feature_list):
                    missing = set(feature_list) - set(available_features)
                    env_logger.warning(f"Missing features: {missing}")
                combined_data = combined_data[available_features]

            self.processed_data = combined_data
            self._features_created = True

            env_logger.info(f"Features created successfully. Shape: {combined_data.shape}")
            return combined_data

        except Exception as e:
            env_logger.error(f"Error creating features: {e}")
            return pd.DataFrame()

    def _add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features and calculations.

        Args:
            data: DataFrame with basic features

        Returns:
            DataFrame with additional derived features
        """
        try:
            # Add lagged price for comparison
            data['last_price'] = data['close'].shift(1)

            # Add returns calculation
            data['returns'] = np.log(data['close'] / data['close'].shift(5).replace(0, np.nan))
            data['returns'] = data['returns'].fillna(0)

            # Add portfolio balance placeholder
            data['portfolio_balance'] = 10000.0  # Will be updated by environment

            # Add price change indicators
            data['price_change'] = data['close'] - data['last_price']
            data['price_change_pct'] = (data['price_change'] / data['last_price']).fillna(0)

            # Add volatility measures
            data['volatility_5'] = data['returns'].rolling(window=5).std().fillna(0)
            data['volatility_20'] = data['returns'].rolling(window=20).std().fillna(0)

            # Clean any remaining NaN values
            data = data.fillna(method='bfill').fillna(0)

            return data

        except Exception as e:
            env_logger.error(f"Error adding derived features: {e}")
            return data

    def normalize_features(self, feature_columns: List[str], method: str = 'standard') -> pd.DataFrame:
        """
        Normalize specified feature columns.

        Args:
            feature_columns: List of columns to normalize
            method: Normalization method ('standard' or 'minmax')

        Returns:
            DataFrame with normalized features
        """
        if not self._features_created or self.processed_data is None:
            env_logger.error("Features must be created before normalization")
            return pd.DataFrame()

        try:
            data_copy = self.processed_data.copy()

            # Select only the features that exist in the data
            available_features = [f for f in feature_columns if f in data_copy.columns]

            if not available_features:
                env_logger.error("No valid features found for normalization")
                return data_copy

            if method == 'standard':
                # Z-score normalization
                for feature in available_features:
                    mean_val = data_copy[feature].mean()
                    std_val = data_copy[feature].std()
                    if std_val != 0:
                        data_copy[feature] = (data_copy[feature] - mean_val) / std_val
                    else:
                        data_copy[feature] = 0

            elif method == 'minmax':
                # Min-max normalization
                for feature in available_features:
                    min_val = data_copy[feature].min()
                    max_val = data_copy[feature].max()
                    if max_val != min_val:
                        data_copy[feature] = (data_copy[feature] - min_val) / (max_val - min_val)
                    else:
                        data_copy[feature] = 0

            # Ensure no infinite or NaN values
            data_copy = data_copy.replace([np.inf, -np.inf], 0).fillna(0)

            self.normalized_data = data_copy
            env_logger.info(f"Features normalized using {method} method")

            return data_copy

        except Exception as e:
            env_logger.error(f"Error normalizing features: {e}")
            return self.processed_data

    def get_feature_subset(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Get a subset of features from the processed data.

        Args:
            feature_names: List of feature names to extract

        Returns:
            DataFrame with selected features
        """
        if self.processed_data is None:
            env_logger.error("No processed data available")
            return pd.DataFrame()

        available_features = [f for f in feature_names if f in self.processed_data.columns]

        if len(available_features) != len(feature_names):
            missing = set(feature_names) - set(available_features)
            env_logger.warning(f"Missing features: {missing}")

        return self.processed_data[available_features] if available_features else pd.DataFrame()

    def create_training_data(self,
                           feature_columns: List[str],
                           target_column: str = 'returns',
                           normalize: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create training data with features and targets.

        Args:
            feature_columns: List of feature column names
            target_column: Name of target column
            normalize: Whether to normalize features

        Returns:
            Tuple of (features_df, target_series)
        """
        if not self._features_created or self.processed_data is None:
            env_logger.error("Features must be created first")
            return pd.DataFrame(), pd.Series()

        try:
            # Get feature subset
            if normalize:
                self.normalize_features(feature_columns)
                feature_data = self.normalized_data[feature_columns]
            else:
                feature_data = self.get_feature_subset(feature_columns)

            # Get target data
            if target_column in self.processed_data.columns:
                target_data = self.processed_data[target_column]
            else:
                env_logger.error(f"Target column '{target_column}' not found")
                return feature_data, pd.Series()

            # Ensure same index
            feature_data = feature_data.loc[target_data.index]

            env_logger.info(f"Training data created: Features {feature_data.shape}, Target {target_data.shape}")
            return feature_data, target_data

        except Exception as e:
            env_logger.error(f"Error creating training data: {e}")
            return pd.DataFrame(), pd.Series()

    def get_data_info(self) -> Dict:
        """
        Get information about the loaded and processed data.

        Returns:
            Dictionary with data information
        """
        info = {
            'data_loaded': self._data_loaded,
            'features_created': self._features_created,
            'raw_data_shape': self.raw_data.shape if self.raw_data is not None else None,
            'processed_data_shape': self.processed_data.shape if self.processed_data is not None else None,
            'normalized_data_shape': self.normalized_data.shape if self.normalized_data is not None else None,
            'available_columns': list(self.processed_data.columns) if self.processed_data is not None else [],
            'symbol': self.symbol,
            'percentage_used': self.percentage * 100
        }

        return info

    def validate_data_quality(self) -> Dict[str, bool]:
        """
        Validate the quality of loaded and processed data.

        Returns:
            Dictionary with validation results
        """
        validation = {
            'data_exists': False,
            'no_empty_data': False,
            'no_null_values': False,
            'sufficient_rows': False,
            'price_data_valid': False
        }

        if self.processed_data is not None and not self.processed_data.empty:
            validation['data_exists'] = True
            validation['no_empty_data'] = True
            validation['no_null_values'] = not self.processed_data.isnull().any().any()
            validation['sufficient_rows'] = len(self.processed_data) >= 100

            # Check if basic price columns exist and are valid
            price_cols = ['open', 'high', 'low', 'close', 'volume']
            if all(col in self.processed_data.columns for col in price_cols):
                price_data = self.processed_data[price_cols]
                validation['price_data_valid'] = (price_data > 0).all().all()

        return validation

    def get_feature_statistics(self) -> Dict:
        """
        Get statistical information about features.

        Returns:
            Dictionary with feature statistics
        """
        if self.processed_data is None:
            return {}

        try:
            stats = {
                'feature_count': len(self.processed_data.columns),
                'mean_values': self.processed_data.mean().to_dict(),
                'std_values': self.processed_data.std().to_dict(),
                'min_values': self.processed_data.min().to_dict(),
                'max_values': self.processed_data.max().to_dict(),
                'null_counts': self.processed_data.isnull().sum().to_dict()
            }

            return stats

        except Exception as e:
            env_logger.error(f"Error calculating feature statistics: {e}")
            return {}

    def save_processed_data(self, filepath: str):
        """
        Save processed data to pickle file.

        Args:
            filepath: Path to save the data
        """
        if self.processed_data is None:
            env_logger.error("No processed data to save")
            return

        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'wb') as f:
                pickle.dump(self.processed_data, f)

            env_logger.info(f"Processed data saved to {filepath}")

        except Exception as e:
            env_logger.error(f"Error saving processed data: {e}")

    def reset(self):
        """Reset the data manager to initial state."""
        self.raw_data = None
        self.processed_data = None
        self.feature_data = None
        self.normalized_data = None
        self._data_loaded = False
        self._features_created = False
        env_logger.info("Data manager reset")
        self._features_created = False
        env_logger.info("Data manager reset")
