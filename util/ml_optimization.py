"""
Optimized Machine Learning Pipeline with Advanced Performance Features

This module provides enhanced ML pipeline with:
- Pipeline caching using joblib memory
- Parallel processing optimization
- Model persistence with pickle protocol 5
- Grid search with successive halving
- Memory-efficient data handling
- Performance monitoring and profiling
"""

import logging
import os
import shutil
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib
import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, HalvingGridSearchCV,
                                     cross_val_score)
from sklearn.pipeline import Pipeline

matplotlib.use('Agg')  # Use non-interactive backend

# Memory profiling (optional)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class MLConfig:
    """Configuration for ML pipeline optimization"""
    # Caching
    cache_dir: str = "data/ml/cache"
    cache_size_limit: str = "2GB"

    # Parallel processing
    n_jobs: int = -1  # Use all available cores
    backend: str = "threading"  # or "loky" for CPU-intensive tasks
    batch_size: int = 1000

    # Model persistence
    pickle_protocol: int = 5  # Latest protocol for better performance
    compression_level: int = 3  # Compression for model files

    # Grid search optimization
    use_halving_search: bool = True
    cv_folds: int = 5
    scoring: str = "accuracy"

    # Memory management
    chunk_size: int = 10000
    memory_threshold: float = 0.8  # 80% memory usage threshold

    # Performance monitoring
    enable_profiling: bool = True
    profile_output_dir: str = "data/ml/profiles"


class MemoryMonitor:
    """Monitor memory usage and provide warnings"""

    def __init__(self, threshold: float = 0.8, logger: Optional[logging.Logger] = None):
        self.threshold = threshold
        self.logger = logger or logging.getLogger(__name__)
        self.peak_memory = 0
        self.start_memory = 0

    def start_monitoring(self):
        """Start memory monitoring"""
        if HAS_PSUTIL:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB

    def check_memory(self) -> Dict[str, float]:
        """Check current memory usage"""
        if not HAS_PSUTIL:
            return {"available": True, "usage_percent": 0, "memory_mb": 0}

        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_percent = process.memory_percent()

        self.peak_memory = max(self.peak_memory, memory_mb)

        if memory_percent / 100 > self.threshold:
            self.logger.warning(f"High memory usage: {memory_percent:.1f}% ({memory_mb:.1f} MB)")

        return {
            "available": memory_percent / 100 < self.threshold,
            "usage_percent": memory_percent,
            "memory_mb": memory_mb,
            "peak_memory_mb": self.peak_memory
        }

    def get_memory_report(self) -> Dict[str, float]:
        """Get memory usage report"""
        current = self.check_memory()
        return {
            **current,
            "start_memory_mb": self.start_memory,
            "memory_increase_mb": current["memory_mb"] - self.start_memory
        }


class ChunkedDataProcessor:
    """Process large datasets in chunks to manage memory"""

    def __init__(self, chunk_size: int = 10000, logger: Optional[logging.Logger] = None):
        self.chunk_size = chunk_size
        self.logger = logger or logging.getLogger(__name__)

    def process_in_chunks(self, data: pd.DataFrame, processor_func, **kwargs) -> pd.DataFrame:
        """Process dataframe in chunks"""
        if len(data) <= self.chunk_size:
            return processor_func(data, **kwargs)

        results = []
        n_chunks = (len(data) + self.chunk_size - 1) // self.chunk_size

        self.logger.info(f"Processing {len(data)} rows in {n_chunks} chunks")

        for i in range(0, len(data), self.chunk_size):
            chunk = data.iloc[i:i + self.chunk_size]
            result = processor_func(chunk, **kwargs)
            results.append(result)

            if i // self.chunk_size % 10 == 0:  # Log every 10 chunks
                self.logger.debug(f"Processed chunk {i // self.chunk_size + 1}/{n_chunks}")

        return pd.concat(results, ignore_index=True)


class CachedTransformer(BaseEstimator, TransformerMixin):
    """Transformer wrapper with caching capability"""

    def __init__(self, transformer, memory=None, cache_key_prefix=""):
        self.transformer = transformer
        self.memory = memory
        self.cache_key_prefix = cache_key_prefix

    def fit(self, X, y=None):
        if self.memory:
            # Cache the fit operation
            cached_fit = self.memory.cache(self.transformer.fit)
            cached_fit(X, y)
        else:
            self.transformer.fit(X, y)
        return self

    def transform(self, X):
        if self.memory:
            # Cache the transform operation
            cached_transform = self.memory.cache(self.transformer.transform)
            return cached_transform(X)
        else:
            return self.transformer.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class OptimizedMLPipeline:
    """
    High-performance machine learning pipeline with advanced optimizations
    """

    def __init__(self, config: Optional[MLConfig] = None, logger: Optional[logging.Logger] = None):
        self.config = config or MLConfig()
        self.logger = logger or logging.getLogger(__name__)

        # Initialize components
        self.memory_monitor = MemoryMonitor(self.config.memory_threshold, self.logger)
        self.data_processor = ChunkedDataProcessor(self.config.chunk_size, self.logger)

        # Set up caching
        self._setup_caching()

        # Performance tracking
        self.metrics = {
            "fit_times": [],
            "predict_times": [],
            "memory_usage": [],
            "cache_hits": 0,
            "cache_misses": 0
        }

        # Configure joblib
        self._configure_joblib()

    def _setup_caching(self):
        """Set up joblib memory caching"""
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.memory = Memory(
            location=str(cache_dir),
            verbose=1 if self.logger.level <= logging.DEBUG else 0
        )

        self.logger.info(f"Cache directory: {cache_dir}")

    def _configure_joblib(self):
        """Configure joblib for optimal performance"""
        # Set BLAS threading for sklearn
        os.environ['SKLEARN_PAIRWISE_DIST_CHUNK_SIZE'] = str(self.config.chunk_size)

        # Configure joblib backend
        joblib.parallel.DEFAULT_BACKEND = self.config.backend

        self.logger.info(f"Joblib configured with backend: {self.config.backend}, n_jobs: {self.config.n_jobs}")

    @contextmanager
    def performance_monitor(self, operation_name: str):
        """Context manager for performance monitoring"""
        start_time = time.time()
        self.memory_monitor.start_monitoring()

        try:
            yield
        finally:
            duration = time.time() - start_time
            memory_report = self.memory_monitor.get_memory_report()

            self.logger.info(
                f"{operation_name} completed in {duration:.2f}s, "
                f"memory: {memory_report['memory_mb']:.1f}MB "
                f"(peak: {memory_report['peak_memory_mb']:.1f}MB)"
            )

            # Store metrics
            if operation_name.startswith("fit"):
                self.metrics["fit_times"].append(duration)
            elif operation_name.startswith("predict"):
                self.metrics["predict_times"].append(duration)

            self.metrics["memory_usage"].append(memory_report)

    def create_optimized_pipeline(self,
                                  preprocessors: List[Tuple[str, Any]],
                                  estimator: Any,
                                  use_cache: bool = True) -> Pipeline:
        """Create an optimized pipeline with caching"""
        steps = []

        for name, transformer in preprocessors:
            if use_cache:
                cached_transformer = CachedTransformer(
                    transformer,
                    memory=self.memory,
                    cache_key_prefix=name
                )
                steps.append((name, cached_transformer))
            else:
                steps.append((name, transformer))

        steps.append(("estimator", estimator))

        pipeline = Pipeline(
            steps,
            memory=self.memory if use_cache else None
        )

        return pipeline

    def optimized_grid_search(self,
                              pipeline: Pipeline,
                              param_grid: Dict[str, List[Any]],
                              X: np.ndarray,
                              y: np.ndarray,
                              use_halving: bool = None) -> Union[GridSearchCV, HalvingGridSearchCV]:
        """Perform optimized grid search"""
        use_halving = use_halving if use_halving is not None else self.config.use_halving_search

        if use_halving:
            self.logger.info("Using HalvingGridSearchCV for faster hyperparameter search")
            search = HalvingGridSearchCV(
                pipeline,
                param_grid,
                cv=self.config.cv_folds,
                scoring=self.config.scoring,
                n_jobs=self.config.n_jobs,
                verbose=1 if self.logger.level <= logging.DEBUG else 0,
                factor=3,  # More aggressive pruning
                min_resources='exhaust'  # Use all available data in final iteration
            )
        else:
            self.logger.info("Using traditional GridSearchCV")
            search = GridSearchCV(
                pipeline,
                param_grid,
                cv=self.config.cv_folds,
                scoring=self.config.scoring,
                n_jobs=self.config.n_jobs,
                verbose=1 if self.logger.level <= logging.DEBUG else 0
            )

        with self.performance_monitor("grid_search_fit"):
            search.fit(X, y)

        return search

    def parallel_model_evaluation(self,
                                  models: Dict[str, Any],
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate multiple models in parallel"""

        def evaluate_single_model(name, model):
            with self.performance_monitor(f"evaluate_{name}"):
                # Fit model
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Calculate metrics
                if hasattr(model, "predict_proba"):  # Classification
                    accuracy = accuracy_score(y_test, y_pred)
                    try:
                        auc = roc_auc_score(y_test, y_pred)
                    except ValueError:
                        auc = np.nan

                    return {
                        "accuracy": accuracy,
                        "auc": auc,
                        "type": "classification"
                    }
                else:  # Regression
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)

                    return {
                        "r2": r2,
                        "mae": mae,
                        "mse": mse,
                        "rmse": np.sqrt(mse),
                        "type": "regression"
                    }

        # Run evaluations in parallel
        results = Parallel(n_jobs=self.config.n_jobs, backend=self.config.backend)(
            delayed(evaluate_single_model)(name, model)
            for name, model in models.items()
        )

        return dict(zip(models.keys(), results))

    def save_model_optimized(self, model: Any, filename: str) -> None:
        """Save model with optimized settings"""
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with self.performance_monitor("model_save"):
            joblib.dump(
                model,
                filepath,
                protocol=self.config.pickle_protocol,
                compress=self.config.compression_level
            )

        file_size = filepath.stat().st_size / (1024 * 1024)  # MB
        self.logger.info(f"Model saved to {filepath} ({file_size:.2f} MB)")

    def load_model_optimized(self, filename: str) -> Any:
        """Load model with performance monitoring"""
        with self.performance_monitor("model_load"):
            model = joblib.load(filename)

        file_size = Path(filename).stat().st_size / (1024 * 1024)  # MB
        self.logger.info(f"Model loaded from {filename} ({file_size:.2f} MB)")

        return model

    def batch_predict(self, model: Any, X: np.ndarray, batch_size: int = None) -> np.ndarray:
        """Perform batch prediction for large datasets"""
        batch_size = batch_size or self.config.batch_size

        if len(X) <= batch_size:
            with self.performance_monitor("predict_single"):
                return model.predict(X)

        predictions = []
        n_batches = (len(X) + batch_size - 1) // batch_size

        self.logger.info(f"Processing {len(X)} samples in {n_batches} batches")

        with self.performance_monitor("predict_batch"):
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                batch_pred = model.predict(batch)
                predictions.append(batch_pred)

        return np.concatenate(predictions)

    def get_feature_importance(self,
                               model: Any,
                               feature_names: List[str],
                               method: str = "auto") -> pd.DataFrame:
        """Get feature importance with multiple methods"""

        # Extract the actual estimator if it's in a pipeline
        estimator = model
        if hasattr(model, 'best_estimator_'):
            estimator = model.best_estimator_

        if hasattr(estimator, 'named_steps'):
            # It's a pipeline, get the final estimator
            estimator = estimator.named_steps[list(estimator.named_steps.keys())[-1]]

        importance_data = []

        # Try different methods based on estimator type
        if hasattr(estimator, 'feature_importances_'):
            importance_data.append({
                'method': 'tree_importance',
                'importance': estimator.feature_importances_
            })

        if hasattr(estimator, 'coef_'):
            coef = estimator.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            importance_data.append({
                'method': 'coefficient',
                'importance': np.abs(coef)
            })

        # Create DataFrame
        if importance_data:
            df_list = []
            for data in importance_data:
                df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': data['importance'],
                    'method': data['method']
                })
                df_list.append(df)

            return pd.concat(df_list, ignore_index=True)

        return pd.DataFrame(columns=['feature', 'importance', 'method'])

    def cross_validate_parallel(self,
                                estimator: Any,
                                X: np.ndarray,
                                y: np.ndarray,
                                cv: int = None,
                                scoring: str = None) -> Dict[str, float]:
        """Parallel cross-validation with performance monitoring"""
        cv = cv or self.config.cv_folds
        scoring = scoring or self.config.scoring

        with self.performance_monitor("cross_validation"):
            scores = cross_val_score(
                estimator, X, y,
                cv=cv,
                scoring=scoring,
                n_jobs=self.config.n_jobs
            )

        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }

    def clear_cache(self):
        """Clear the joblib cache"""
        if hasattr(self.memory, 'clear'):
            self.memory.clear()
            self.logger.info("Cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        cache_dir = Path(self.config.cache_dir)

        if cache_dir.exists():
            # Calculate cache size
            total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            file_count = len(list(cache_dir.rglob('*')))

            return {
                'cache_dir': str(cache_dir),
                'total_size_mb': total_size / (1024 * 1024),
                'file_count': file_count,
                'hits': self.metrics.get('cache_hits', 0),
                'misses': self.metrics.get('cache_misses', 0)
            }

        return {'cache_dir': str(cache_dir), 'exists': False}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        fit_times = self.metrics.get('fit_times', [])
        predict_times = self.metrics.get('predict_times', [])

        return {
            'fit_times': {
                'mean': np.mean(fit_times) if fit_times else 0,
                'std': np.std(fit_times) if fit_times else 0,
                'count': len(fit_times)
            },
            'predict_times': {
                'mean': np.mean(predict_times) if predict_times else 0,
                'std': np.std(predict_times) if predict_times else 0,
                'count': len(predict_times)
            },
            'memory_usage': self.metrics.get('memory_usage', []),
            'cache_info': self.get_cache_info(),
            'config': {
                'n_jobs': self.config.n_jobs,
                'backend': self.config.backend,
                'chunk_size': self.config.chunk_size,
                'use_halving_search': self.config.use_halving_search
            }
        }

    def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear memory cache
            self.clear_cache()

            # Remove temporary cache directory if configured
            cache_dir = Path(self.config.cache_dir)
            if cache_dir.exists() and 'temp' in str(cache_dir):
                shutil.rmtree(cache_dir)
                self.logger.info(f"Temporary cache directory {cache_dir} removed")

        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")


# Factory functions
def create_optimized_pipeline(config: Optional[MLConfig] = None, **kwargs) -> OptimizedMLPipeline:
    """Create an optimized ML pipeline"""
    return OptimizedMLPipeline(config, **kwargs)


def create_cached_transformer(transformer: Any, cache_dir: str = None) -> CachedTransformer:
    """Create a cached transformer"""
    memory = None
    if cache_dir:
        memory = Memory(location=cache_dir, verbose=0)

    return CachedTransformer(transformer, memory)


# Configuration presets
FAST_CONFIG = MLConfig(
    n_jobs=-1,
    backend="threading",
    use_halving_search=True,
    cv_folds=3,
    chunk_size=5000
)

MEMORY_EFFICIENT_CONFIG = MLConfig(
    n_jobs=2,
    backend="threading",
    chunk_size=1000,
    memory_threshold=0.7,
    compression_level=9
)

HIGH_PERFORMANCE_CONFIG = MLConfig(
    n_jobs=-1,
    backend="loky",
    use_halving_search=True,
    cv_folds=5,
    chunk_size=10000,
    pickle_protocol=5,
    compression_level=3
)
