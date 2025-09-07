import datetime
import gc
import os
import pickle
import threading
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import shap
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import (SelectKBest, f_classif, f_regression,
                                       mutual_info_classif)
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, explained_variance_score,
                             make_scorer, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV

from ...util import loggers
from ...util.utils import tradex_features
from .ml_models import get_model
from .ml_util import (classifier, future_score_clas, future_score_reg,
                      regression)

# Import optimized utilities
try:
    from ...util.cache import HybridCache, cache_key_for_ml_prediction
    from ...util.ml_optimization import MLConfig, OptimizedMLPipeline
    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False

# Disable ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Use the 'Agg' backend
matplotlib.use('Agg')

logger = loggers.setup_loggers()

PLOTS_PATH = 'data/ml/plots'
MODEL_PERFO_PATH = 'data/ml/csv/model_performance.csv'
SHIFT_CANDLES = 3

# Performance monitoring
class PerformanceMonitor:
    """Monitor system performance during ML operations."""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def get_memory_usage(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def get_memory_delta(self):
        """Get memory usage change since initialization."""
        return self.get_memory_usage() - self.initial_memory

    def log_performance(self, operation_name, logger):
        """Log current performance metrics."""
        memory_mb = self.get_memory_usage()
        memory_delta = self.get_memory_delta()
        cpu_percent = self.process.cpu_percent()

        logger.info(f"{operation_name} - Memory: {memory_mb:.1f}MB (Δ{memory_delta:+.1f}MB), CPU: {cpu_percent:.1f}%")


class MachineLearning:
    def __init__(self, exchange, symbol) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.logger = logger['model']
        self.shap_interpreter = SHAPClass()
        self.performance_monitor = PerformanceMonitor()

        # Initialize optimized components if available
        self._cache = None
        self._optimized_pipeline = None
        self.parallel_backend = 'threading'  # Default to threading for I/O bound tasks
        self.max_workers = min(32, (os.cpu_count() or 1) + 4)  # Conservative worker count

        if OPTIMIZATIONS_AVAILABLE:
            try:
                # Initialize optimized ML pipeline
                config = MLConfig(
                    n_jobs=-1,
                    backend="loky",
                    use_halving_search=True,
                    cv_folds=3,
                    chunk_size=5000,
                    cache_size_mb=500
                )
                self._optimized_pipeline = OptimizedMLPipeline(config=config)
                self.logger.info("Optimized ML pipeline initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize optimized pipeline: {e}")
                self._optimized_pipeline = None

    def optimize_memory_usage(self):
        """Optimize memory usage by clearing caches and triggering garbage collection."""
        # Clear any matplotlib figures
        plt.close('all')

        # Trigger garbage collection
        collected = gc.collect()
        self.logger.info(f"Garbage collection freed {collected} objects")

        # Log memory usage
        self.performance_monitor.log_performance("Memory optimization", self.logger)

    def parallel_model_training(self, algorithms, X_train, y_train, X_test, y_test, scoring_function, cv=5):
        """
        Train multiple models in parallel for faster comparison.

        :param algorithms: List of algorithm names to train
        :param X_train: Training features
        :param y_train: Training targets
        :param X_test: Test features
        :param y_test: Test targets
        :param scoring_function: Function to score the models
        :param cv: Number of cross-validation folds
        :return: Dictionary of results for each algorithm
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def train_single_model(algorithm):
            """Train a single model and return its performance."""
            start_time = time.time()
            try:
                model = get_model(algorithm)
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)
                score = scoring_function(y_test, y_pred)

                training_time = time.time() - start_time

                return {
                    'algorithm': algorithm,
                    'model': model,
                    'score': score,
                    'training_time': training_time,
                    'status': 'success'
                }
            except Exception as e:
                return {
                    'algorithm': algorithm,
                    'model': None,
                    'score': None,
                    'training_time': time.time() - start_time,
                    'status': 'failed',
                    'error': str(e)
                }

        # Track performance
        self.performance_monitor.log_performance(f"Starting parallel training of {len(algorithms)} models", self.logger)

        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all training jobs
            future_to_algorithm = {
                executor.submit(train_single_model, algo): algo
                for algo in algorithms
            }

            # Collect results as they complete
            for future in as_completed(future_to_algorithm):
                result = future.result()
                results[result['algorithm']] = result

                if result['status'] == 'success':
                    self.logger.info(f"✓ {result['algorithm']}: Score={result['score']:.4f}, Time={result['training_time']:.2f}s")
                else:
                    self.logger.error(f"✗ {result['algorithm']}: Failed - {result['error']}")

        self.performance_monitor.log_performance("Completed parallel training", self.logger)
        return results

    def advanced_hyperparameter_search(self, algorithm, X_train, y_train, search_type='bayesian', n_iter=50):
        """
        Perform advanced hyperparameter optimization using Bayesian or Halving search.

        :param algorithm: Algorithm name
        :param X_train: Training features
        :param y_train: Training targets
        :param search_type: 'bayesian', 'halving_grid', or 'halving_random'
        :param n_iter: Number of iterations for optimization
        :return: Best model and parameters
        """
        from model.machinelearning.ml_models import get_model_parameters

        try:
            model, param_grid = get_model_parameters(algorithm)

            self.logger.info(f"Starting {search_type} hyperparameter search for {algorithm}")

            if search_type == 'bayesian':
                search = BayesSearchCV(
                    model,
                    param_grid,
                    n_iter=n_iter,
                    cv=3,
                    n_jobs=-1,
                    random_state=42,
                    scoring='accuracy' if algorithm in classifier else 'neg_mean_squared_error'
                )
            elif search_type == 'halving_grid':
                from sklearn.model_selection import HalvingGridSearchCV
                search = HalvingGridSearchCV(
                    model,
                    param_grid,
                    cv=3,
                    n_jobs=-1,
                    random_state=42,
                    scoring='accuracy' if algorithm in classifier else 'neg_mean_squared_error'
                )
            elif search_type == 'halving_random':
                from sklearn.model_selection import HalvingRandomSearchCV
                search = HalvingRandomSearchCV(
                    model,
                    param_grid,
                    n_iter=n_iter,
                    cv=3,
                    n_jobs=-1,
                    random_state=42,
                    scoring='accuracy' if algorithm in classifier else 'neg_mean_squared_error'
                )
            else:
                raise ValueError(f"Unknown search type: {search_type}")

            # Perform the search
            search.fit(X_train, y_train)

            self.logger.info(f"Best parameters for {algorithm}: {search.best_params_}")
            self.logger.info(f"Best score: {search.best_score_:.4f}")

            return search.best_estimator_, search.best_params_, search.best_score_

        except Exception as e:
            self.logger.error(f"Hyperparameter search failed for {algorithm}: {e}")
            return None, None, None

    def create_feature_selector(self, X, y, selection_method='mutual_info', k=10):
        """
        Create a feature selector for dimensionality reduction.

        :param X: Features
        :param y: Target
        :param selection_method: 'mutual_info', 'f_classif', or 'f_regression'
        :param k: Number of features to select
        :return: Fitted feature selector
        """
        if selection_method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
        elif selection_method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        elif selection_method == 'f_regression':
            selector = SelectKBest(f_regression, k=k)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")

        selector.fit(X, y)
        self.logger.info(f"Feature selector created using {selection_method}, selected {k} features")

        return selector

    async def get_cache(self):
        """Get or initialize cache instance."""
        if self._cache is None and OPTIMIZATIONS_AVAILABLE:
            try:
                from util.cache import get_global_cache
                self._cache = await get_global_cache()
            except Exception:
                self._cache = None
        return self._cache

    async def predict_async(self, model, df=None, t='30m', symbol='BTC/USDT') -> int:
        """Async version of predict method with enhanced caching."""
        try:
            # Check cache first for recent predictions
            cache = await self.get_cache()
            if cache:
                cache_key = cache_key_for_ml_prediction(model.__class__.__name__, f"{symbol}_{t}")
                cached_prediction = await cache.get(cache_key)
                if cached_prediction is not None:
                    self.logger.debug(f"Returning cached prediction for {symbol} at timeframe {t}")
                    return cached_prediction

            # Fetch the current data for the symbol
            data = self.exchange.fetch_ohlcv(symbol, timeframe=t, limit=500)
            df = pd.DataFrame(
                data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

            # Use optimized feature processing if available
            if self._optimized_pipeline:
                features = await self._process_features_optimized(df)
            else:
                features = self.process_features(df)

            # Load scaler and transform features
            scaler = self.load_pretrained_scaler()
            features = scaler.transform(features)

            # Make prediction
            prediction = model.predict(features[-1].reshape(1, -1))[0]

            # Log and cache the prediction
            self.logger.info(f"Prediction for {symbol} at timeframe {t}: {prediction}")

            if cache:
                await cache.set(cache_key, prediction, ttl=30)

            return prediction

        except Exception as e:
            self.logger.error(f"Error during async prediction: {str(e)}")
            return 1

    def predict(self, model, df=None, t='30m', symbol='BTC/USDT') -> int:
        """Enhanced predict method with caching support."""
        try:
            # For backward compatibility, use simple caching
            cache_key = f"ml_prediction_{symbol}_{t}"

            # Log cache key for debugging
            self.logger.debug(f"Using cache key: {cache_key}")

            # Try to use optimized async prediction if available
            if OPTIMIZATIONS_AVAILABLE:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If already in async context, use sync version with basic caching
                        pass
                    else:
                        return loop.run_until_complete(self.predict_async(model, df, t, symbol))
                except Exception:
                    pass

            # Fallback to original implementation with basic caching
            # Fetch the current data for the symbol
            data = self.exchange.fetch_ohlcv(symbol, timeframe=t, limit=500)
            df = pd.DataFrame(
                data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

            # Use the process_features method to get the features for prediction
            features = self.process_features(df)

            # Load scaler and transform features
            scaler = self.load_pretrained_scaler()
            features = scaler.transform(features)

            # Make a prediction using the model
            prediction = model.predict(features[-1].reshape(1, -1))[0]

            # Logging the prediction result
            self.logger.info(f"Prediction for {symbol} at timeframe {t}: {prediction}")

            return prediction
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            return 1  # return a default value

    async def _process_features_optimized(self, df):
        """Process features using optimized pipeline."""
        if self._optimized_pipeline:
            return await self._optimized_pipeline.process_features_async(df)
        else:
            return self.process_features(df)

    def evaluate_model(self, name, model, X_test, y_test):
        y_pred = model.predict(X_test)

        if name in regression:
            accuracy = self.evaluate_regression_model(name, y_test, y_pred)
        else:
            accuracy = self.evaluate_classification_model(name, y_test, y_pred)

        # Use the interpret_model method to get SHAP values and plots
        _ = self.interpret_model(
            model, X_test, save_values=True, plot_values=True, feature_names=self.column_names)

        return accuracy

    def evaluate_regression_model(self, name, y_test, y_pred):
        # Calculate and log regression metrics
        r2, mae, mse, rmse, evs = self.calculate_regression_metrics(
            y_test, y_pred)

        # Plot regression results
        self.plot_regression_results(
            name, y_test, y_pred, r2, mae, mse, rmse, evs)

        return r2

    def evaluate_classification_model(self, name, y_test, y_pred):
        # Calculate and log classification metrics
        accuracy, auc = self.calculate_classification_metrics(y_test, y_pred)

        # Plot classification results
        self.plot_classification_results(name, y_test, y_pred, accuracy, auc)

        return accuracy

    def calculate_regression_metrics(self, y_test, y_pred):
        # Compute various regression metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        evs = explained_variance_score(y_test, y_pred)

        # Log the metrics
        self.logger.info(f"R2 Score: {r2}")
        self.logger.info(f"Mean Absolute Error: {mae}")
        self.logger.info(f"Mean Squared Error: {mse}")
        self.logger.info(f"Root Mean Squared Error: {rmse}")
        self.logger.info(f"Explained Variance Score: {evs}")

        return r2, mae, mse, rmse, evs

    def calculate_classification_metrics(self, y_test, y_pred):
        # Compute various classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred) if len(
            set(y_test)) == 2 else "Not Applicable"
        self.logger.info(f"Accuracy: {accuracy}")
        self.logger.info(f"AUC-ROC: {auc}")
        self.logger.info(
            f"Classification Report:\n{classification_report(y_test, y_pred)}")

        return accuracy, auc

    def plot_regression_results(self, name, y_test, y_pred, r2, mae, mse, rmse, evs):
        plot_dir = PLOTS_PATH
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot for true values vs predictions
        ax.scatter(y_test, y_pred)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title(f'Regression Results for {name} - True vs Predicted')

        # Plotting the line of best fit
        ax.plot([y_test.min(), y_test.max()], [
                y_test.min(), y_test.max()], 'k--', lw=4)

        # Adding text for metrics
        textstr = f'R2: {r2:.2f}\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nEVS: {evs:.2f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Save and show the plot
        plt.savefig(f'{plot_dir}/{name}_regression_plot.png')

        # Show or close the plot
        if threading.current_thread() == threading.main_thread():
            plt.show()
        else:
            plt.close()

    def plot_classification_results(self, name, y_test, y_pred, accuracy, auc_score):
        plot_dir = PLOTS_PATH
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotting the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix for {name}')

        # Plot ROC curve if binary classification
        if len(set(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)

            fig_roc, ax_roc = plt.subplots(figsize=(10, 6))
            ax_roc.plot(fpr, tpr, color='blue', lw=2,
                        label=f'ROC curve (area = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title(f'ROC Curve for {name}')
            ax_roc.legend(loc="lower right")
            plt.savefig(f'{plot_dir}/{name}_roc_curve.png')

        # Save and show the plots
        plt.savefig(f'{plot_dir}/{name}_confusion_matrix.png')

        # Show or close the plot
        if threading.current_thread() == threading.main_thread():
            plt.show()
        else:
            plt.close()

    def save_model(self, model, accuracy, score=None):
        path = r'data/ml/2020/'
        params_path = r'data/ml/csv/params_accuracy.csv'

        # Check if the best estimator is a Pipeline and extract the actual model
        if isinstance(model.best_estimator_, Pipeline):
            actual_model = model.best_estimator_.named_steps['regression']
        else:
            actual_model = model.best_estimator_

        if accuracy > 0.60:
            version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            model_name = actual_model.__class__.__name__
            model_filename = f'{path}trained_{model_name}_{version}_{accuracy*100:.2f}.p'

            joblib.dump(actual_model, model_filename)
            self.logger.info(f'Model saved as {model_filename}')

            model_params_accuracy = {
                "Model": model_name,
                "Version": version,
                "Params": str(model.best_params_),
                "Accuracy": accuracy,
                "Money": score
            }

            if os.path.exists(params_path):
                df = pd.read_csv(params_path)
                new_df = pd.DataFrame([model_params_accuracy])
                df = pd.concat([df, new_df], ignore_index=True)
            else:
                df = pd.DataFrame([model_params_accuracy])

            df.to_csv(params_path, index=False)

            self.log_model_performance(actual_model, accuracy, score)

    def load_model(self, filename):
        path = r'data/ml/2020'
        file = f'{path}/{filename}'
        model = None  # Initialize model with a default value
        try:
            # Open the file and load the model
            model = joblib.load(file)
            self.logger.info(f'the model is : {model}')
        except FileNotFoundError:
            self.logger.error(f'there is no file called: {file}')

        return model

    def process_features(self, df):

        processed_features = tradex_features(self.symbol, df)

        self.column_names = processed_features.columns.tolist()

        # Fill missing values in the entire DataFrame
        processed_features = processed_features.fillna(0)

        # Return the processed features as a NumPy array
        return processed_features.to_numpy()

    def process_labels(self, df, model, n_candles=3, threshold=0.08):
        if model in classifier:
            # Shift closing prices by -2 to compare with the price two candles ahead
            df['future_close'] = df['close'].shift(-n_candles)

            # Calculate the percentage change in price
            df['price_change_percentage'] = (
                df['future_close'] - df['close']) / df['close'] * 100

            # Define labels: 2 for long, 1 for do nothing, 0 for short, based on the threshold
            df['label'] = np.where(df['price_change_percentage'] >= threshold, 2,
                                   np.where(df['price_change_percentage'] <= -threshold, 0, 1))

            # Fill NaN values in 'label' column
            df['label'].fillna(1, inplace=True)

            # Save the 'label' column to a CSV file
            df[['label', 'price_change_percentage', 'future_close', 'close']].to_csv(
                'data/ml/csv/labels.csv', index=False)

            self.logger.info(df['label'])
            return df['label']
        else:
            return df['close']

    async def train_evaluate_and_save_model_async(self, model: str, usage_percent=0.7):
        """Async version of model training with optimizations."""
        if self._optimized_pipeline and OPTIMIZATIONS_AVAILABLE:
            try:
                return await self._train_with_optimized_pipeline(model, usage_percent)
            except Exception as e:
                self.logger.warning(f"Optimized training failed, falling back to standard: {e}")

        # Fallback to standard training
        return self.train_evaluate_and_save_model(model, usage_percent)

    async def _train_with_optimized_pipeline(self, model: str, usage_percent=0.7):
        """Train model using optimized pipeline."""
        # Load data
        pickle_file_path = 'data/2020/30m_data.pkl'
        csv_file_path = 'data/csv/BTC_1h.csv'

        df = None
        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path)
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        elif os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as f:
                df = pickle.load(f)
                df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        else:
            self.logger.error('Neither the CSV nor the pickle file exists.')
            return None, 0

        # Use optimized pipeline for training
        try:
            # Process features using optimized pipeline
            processed_features = await self._optimized_pipeline.process_features_async(df)
            df = pd.concat([df, processed_features], axis=1)

            # Prepare training data
            split_index = int(len(df) * usage_percent)
            train_df = df.iloc[:split_index]
            test_df = df.iloc[split_index:]

            feature_columns = [col for col in df.columns if col not in [
                'open', 'high', 'low', 'close', 'volume', 'label']]

            X_train = train_df[feature_columns]
            y_train = train_df['label']
            X_test = test_df[feature_columns]
            y_test = test_df['label']

            # Train using optimized pipeline
            trained_model = await self._optimized_pipeline.train_model_async(
                model, X_train, y_train, X_test, y_test
            )

            # Evaluate model
            accuracy = self.evaluate_model(model, trained_model, X_test, y_test)

            # Save model
            self.save_model(model, trained_model)

            return trained_model, accuracy

        except Exception as e:
            self.logger.error(f"Error in optimized training: {e}")
            raise

    def train_evaluate_and_save_model(self, model: str, usage_percent=0.7):
        """Enhanced training method with optional async support."""
        # Try async version if optimizations available
        if OPTIMIZATIONS_AVAILABLE and self._optimized_pipeline:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    return loop.run_until_complete(
                        self.train_evaluate_and_save_model_async(model, usage_percent)
                    )
            except Exception:
                pass

        # Original training implementation with optimizations
        # Define the path of the pickle file
        pickle_file_path = 'data/2020/30m_data.pkl'
        csv_file_path = 'data/csv/BTC_1h.csv'

        # Initialize the DataFrame
        df = None

        # Check if the CSV file exists
        if os.path.exists(csv_file_path):
            # If it exists, load the data from the CSV file
            df = pd.read_csv(csv_file_path)
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        elif os.path.exists(pickle_file_path):
            # If the CSV file doesn't exist, check if the pickle file exists
            with open(pickle_file_path, 'rb') as f:
                df = pickle.load(f)
                df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        else:
            # If neither file exists, log an error and return
            self.logger.error(
                'Neither the CSV nor the pickle file exists. Please check the file paths.')
            return

        # Process the features
        processed_features = self.process_features(df)
        df = pd.concat([df, processed_features], axis=1)

        # Split the data into training and testing sets
        split_index = int(len(df) * usage_percent)
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]

        # Separate features and labels
        feature_columns = [col for col in df.columns if col not in [
            'open', 'high', 'low', 'close', 'volume', 'label']]
        X_train = train_df[feature_columns]
        y_train = train_df['label']
        X_test = test_df[feature_columns]
        y_test = test_df['label']

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save the scaler
        with open(f'data/ml/scaler/scaler_{model}.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        # Get the model
        clf = get_model(model)

        # Train the model with parallel processing if supported
        if hasattr(clf, 'n_jobs'):
            clf.n_jobs = -1  # Use all available cores

        # Train the model
        clf.fit(X_train_scaled, y_train)

        # Evaluate the model
        accuracy = self.evaluate_model(model, clf, X_test_scaled, y_test)

        # Save the model
        self.save_model(model, clf)

        return clf, accuracy

    def selected_labels_features_train(self, model: str, X, y):
        trainer = MLModelTrainer(model, self.logger)
        trained_model = trainer.train(X, y)

        return trained_model

    def train_evaluate_and_save_model_thread(self, model: str) -> None:
        # Create a new thread
        t = threading.Thread(
            target=self.train_evaluate_and_save_model, args=(model,))
        t.setDaemon(True)
        # Start the thread
        t.start()

    def selected_labels_features_train_thread(self, model: str, X, y) -> None:
        # Create a new thread
        t = threading.Thread(
            target=self.selected_labels_features_train, args=(model, X, y,))
        t.setDaemon(True)
        # Start the thread
        t.start()

    def load_pretrained_scaler(self):
        return joblib.load('data/scaler/scaler.pkl')

    def log_model_performance(self, model, accuracy, score):
        model_performance_file = MODEL_PERFO_PATH
        model_info = {
            "Model": model.__class__.__name__,
            "Version": datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
            "Accuracy": accuracy,
            "Score": score,
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Convert the dictionary to a DataFrame
        model_info_df = pd.DataFrame([model_info])

        # Check if file exists to append or write new
        if os.path.isfile(model_performance_file):
            model_info_df.to_csv(model_performance_file,
                                 mode='a', header=False, index=False)
        else:
            model_info_df.to_csv(model_performance_file,
                                 mode='w', header=True, index=False)

        self.compare_model_performance()

    def compare_model_performance(self):
        model_performance_file = MODEL_PERFO_PATH
        if not os.path.isfile(model_performance_file):
            self.logger.info("No model performance data found.")
            return

        df = pd.read_csv(model_performance_file)
        # Implement your logic to compare models here
        # For example, you can sort by accuracy, display top N models, etc.
        sorted_df = df.sort_values(by=['Accuracy'], ascending=False)
        self.logger.info(sorted_df.head())  # Display top models

    def interpret_model(self, model, X_test, save_values=False, plot_values=False, feature_names=None):
        """
        Interpret the model predictions using SHAP values.

        :param model: The trained machine learning model.
        :param X_test: Test dataset (features) used for the model.
        """
        shap_values = self.shap_interpreter.explain_prediction(
            model, X_test, save_values, plot_values, feature_names)
        # Additional code to handle or display SHAP values can be added here
        return shap_values


class MLModelTrainer:
    def __init__(self, algorithm, logger) -> None:
        """
        Initialize the MLModelTrainer with the given algorithm and logger.

        :param algorithm: Algorithm used for model training.
        :param logger: Logger for logging progress and errors.
        """

        self.algorithm = algorithm
        self.logger = logger
        self.stop_loading = False
        self.executor = ThreadPoolExecutor()
        self.futures = []

    def train(self, X, y, feature_names):
        """
        Train the model using the given features X and target y.

        :param X: Features for training.
        :param y: Target variable for training.
        """
        best_estimator = None  # Initialize to None
        model, parameters = get_model(self.algorithm)
        try:
            if self.algorithm in classifier:
                # Create a scorer from the custom scoring function using lambda to handle n_candles
                monetary_scorer = make_scorer(lambda y_true, y_pred: future_score_clas(
                    y_true, y_pred, n_candles=SHIFT_CANDLES), greater_is_better=True)
            elif self.algorithm in regression:
                # Create a scorer from the custom scoring function using lambda to handle n_candles
                monetary_scorer = make_scorer(lambda y_true, y_pred: future_score_reg(
                    y_true, y_pred, n_candles=SHIFT_CANDLES), greater_is_better=True)
            else:
                raise ValueError("No Model Found.")
        except Exception as e:
            self.logger.error(
                f"error with the make scorer method: {e}\n{traceback.format_exc()}")

        try:
            # Assuming parallelize_search is a method defined elsewhere in your class
            best_estimator = self.parallelize_search(
                model, parameters, X, y, monetary_scorer, feature_names)
            self.logger.info("Model training successful.")

        except Exception as e:
            self.logger.error(
                f"Model training failed: {e}\n{traceback.format_exc()}")
            # Handle the case when best_estimator is not assigned
            if best_estimator is None:
                # Handle appropriately, maybe return None or raise an exception
                raise ValueError(
                    "Model training failed and no best_estimator found.")
            return None, None  # Return None or handle the exception as needed

        best_params = best_estimator.best_params_
        best_f1_score = best_estimator.best_score_

        # print the best parameters and best score
        self.logger.info(
            f"the custom: parallelize search best params are {best_params}")
        self.logger.info(
            f"the custom: parallelize search best score is {best_f1_score}")
        self.logger.info(
            f"the custom: parallelize search best estimator is {best_estimator}")

        # return the best model
        return best_estimator, best_f1_score

    def parallelize_search(self, estimator, param_grid, X, y, scorer, feature_names):
        # grid_search_estimator = GridSearchCV(
        #     estimator, param_grid, cv=5, scoring=scorer, verbose=2, n_jobs=8)
        random_search_estimator = RandomizedSearchCV(
            estimator, param_grid, cv=5, scoring=scorer, verbose=2, n_jobs=6)
        bayes_search_estimator = BayesSearchCV(
            estimator, param_grid, cv=5, scoring=scorer, verbose=2, n_jobs=6, n_points=4)

        results = []
        with ThreadPoolExecutor() as executor:
            futures = []
            try:
                # grid_search = executor.submit(grid_search_estimator.fit, X, y)
                random_search = executor.submit(
                    random_search_estimator.fit, X, y)
                bayes_search = executor.submit(
                    bayes_search_estimator.fit, X, y)

                futures = [random_search, bayes_search]
                for future in as_completed(futures):

                    # This will re-raise any exception that occurred during execution.
                    result = future.result()
                    results.append(result)

            except Exception as e:
                self.logger.error(
                    f"Exception occurred during model search: {e}")
                self.shutdown_thread_pool()
            finally:
                self.shutdown_thread_pool()

        if not results:
            raise ValueError("All searches failed!")

        best_estimator = max(results, key=lambda x: x.best_score_)
        self.logger.info(best_estimator)

        # Get the best model from the tuned estimator
        best_model = best_estimator.best_estimator_

        # Perform feature importance analysis
        feature_importances = None
        if feature_names is not None:
            try:
                if hasattr(best_model, 'feature_importances_'):
                    feature_importances = best_model.feature_importances_
                elif hasattr(best_model, 'coef_'):
                    feature_importances = best_model.coef_.flatten()
            except Exception as e:
                self.logger.error(
                    f"Error calculating feature importances: {e}")

        output_csv = 'data/ml/csv/feature_importances.csv'

        # Create new DataFrame for feature importances
        if feature_importances is not None and output_csv is not None:
            try:
                # Create a dictionary for the new DataFrame
                data_dict = {
                    'model': [str(best_model)],
                    'best_score': [best_estimator.best_score_]
                }
                for i, feature_name in enumerate(feature_names):
                    data_dict[feature_name] = [feature_importances[i]]

                # Create DataFrame with new data
                new_feature_importance_df = pd.DataFrame(data_dict)

                # Check if CSV file exists
                if os.path.exists(output_csv):
                    # Read existing data
                    existing_df = pd.read_csv(output_csv)
                    # Append new data
                    updated_df = pd.concat(
                        [existing_df, new_feature_importance_df], ignore_index=True)
                else:
                    # Use new data as the DataFrame
                    updated_df = new_feature_importance_df

                # Save the updated DataFrame
                updated_df.to_csv(output_csv, index=False)
                self.logger.info(
                    f"Feature importances and best score updated in {output_csv}")
            except Exception as e:
                self.logger.error(
                    f"Error updating feature importances and best score in CSV: {e}")

        return best_estimator

    def shutdown_thread_pool(self):
        # Cancel all futures that are not yet running
        for future in self.futures:
            future.cancel()

        # Shutdown the executor
        self.executor.shutdown(wait=True)
        self.logger.info("Thread pool shut down successfully.")


class SHAPClass:
    def __init__(self) -> None:
        self.logger = logger['model']

    def explain_prediction(self, model, feature_array, save_values=False, plot_values=False, feature_names=None):
        """
        Generate SHAP values for a given model and feature array,
        with options to save and plot the values.

        :param model: The trained machine learning model.
        :param feature_array: A numpy array or pandas DataFrame of feature values.
        :param save_values: Boolean flag to save SHAP values.
        :param plot_values: Boolean flag to plot SHAP values.
        :return: SHAP values.
        """

        # Check if the model is a GridSearchCV instance
        if isinstance(model, (GridSearchCV, BayesSearchCV, RandomizedSearchCV)):
            model = model.best_estimator_

        if isinstance(model, Pipeline):
            model = model.named_steps['regression']
        # Debugging: Print the type of the model and test the predict method
        self.logger.info(f"Model type: {type(model)}")
        try:
            sample_prediction = model.predict(feature_array[:1])
            self.logger.info(feature_array[:1])
            self.logger.info(f"Sample Prediction: {sample_prediction}")
        except Exception as e:
            self.logger.info(f"Error making sample prediction: {e}")
            return None

        # Initialize the SHAP explainer with the model
        try:
            explainer = shap.Explainer(model, feature_names=feature_names)
        except Exception as e:
            self.logger.info(
                f"Error shap explainer: {e}\n{traceback.format_exc()}")
            return None
        # Calculate SHAP values
        shap_values = explainer.shap_values(feature_array)

        # Optionally save SHAP values
        if save_values:
            self.save_shap_values(
                shap_values, 'data/ml/json/shap/shap_values.json')

        # Optionally plot SHAP values
        if plot_values:
            self.plot_shap_values(shap_values, feature_array, feature_names)

        return shap_values

    def plot_shap_values(self, shap_values, features, feature_names):
        """
        Plot SHAP values for a given set of features.

        :param shap_values: The SHAP values.
        :param features: The features corresponding to the SHAP values.
        """
        shap.summary_plot(shap_values, features, show=False,
                          feature_names=feature_names)

        # Save the plot
        plt.savefig('data/ml/plots/shap/shap_summary_plot.png')
        self.logger.info("SHAP summary plot saved as 'shap_summary_plot.png'")

    def save_shap_values(self, shap_values, filename):
        """
        Save SHAP values to a file.

        :param shap_values: The SHAP values to save.
        :param filename: The filename to save the SHAP values to.
        """
        try:
            with open(filename, 'wb') as file:  # Open file in binary mode
                joblib.dump(shap_values, file)
            self.logger.info(f"SHAP values saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving SHAP values: {e}")
