import asyncio
from functools import lru_cache
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              GradientBoostingRegressor, IsolationForest,
                              RandomForestClassifier)
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectFromModel
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (BayesianRidge, ElasticNet, HuberRegressor,
                                  Lars, Lasso, LassoCV, LinearRegression,
                                  LogisticRegression, RANSACRegressor, Ridge,
                                  SGDClassifier, SGDRegressor,
                                  TheilSenRegressor)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR, LinearSVR, NuSVR
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier

# Import optimized utilities
try:
    from ...util.cache import HybridCache, get_global_cache
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False

# Global model cache
_model_cache: Dict[str, Any] = {}
_parameter_cache: Dict[str, Dict] = {}

# Optimized parameter ranges for efficient hyperparameter tuning
max_iter = np.arange(2000, 10_000, 2000)
n_estimators = np.arange(1000, 5000, 1000)
learning_rate = np.logspace(-3, 0, 4)
tol = np.logspace(-3, -1, 3)
alpha = np.logspace(-3, 0, 4)

# Model configuration for performance optimization
MODEL_CONFIGS = {
    "high_performance": {
        "n_jobs": -1,
        "random_state": 42,
        "max_features": "sqrt"
    },
    "memory_efficient": {
        "n_jobs": 2,
        "random_state": 42,
        "max_features": "log2"
    },
    "fast_training": {
        "n_jobs": 4,
        "random_state": 42,
        "warm_start": True
    }
}


@lru_cache(maxsize=128)
def get_model(algorithm: str, config_type: str = "high_performance"):
    """
    Get the model for the training with caching and optimization.

    :param algorithm: the name of the model to be selected.
    :param config_type: performance configuration type
    :return: SKlearn Model with optimized parameters.
    """
    # Check cache first
    cache_key = f"{algorithm}_{config_type}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    model = None
    base_config = MODEL_CONFIGS.get(config_type, MODEL_CONFIGS["high_performance"])

    # Filter config based on algorithm requirements
    def filter_config(valid_params):
        return {k: v for k, v in base_config.items() if k in valid_params}

    # Regression Models
    if algorithm == "Linear Regression":
        model = LinearRegression(**filter_config(['n_jobs']))
    elif algorithm == "Ridge Regression":
        model = Ridge(**filter_config(['random_state']))
    elif algorithm == "Lasso Regression":
        model = Lasso(**filter_config(['random_state']))
    elif algorithm == "Elastic Net Regression":
        model = ElasticNet(**filter_config(['random_state']))
    elif algorithm == "Bayesian Ridge Regression":
        model = BayesianRidge()
    elif algorithm == "Gradient Boosting Regressor":
        config = filter_config(['random_state'])
        config['n_estimators'] = 100 if config_type == "fast_training" else 500
        model = GradientBoostingRegressor(**config)
    elif algorithm == "SVR":
        model = SVR()
    elif algorithm == "LinearSVR":
        model = LinearSVR(**filter_config(['random_state']))
    elif algorithm == "NuSVR":
        model = NuSVR()
    elif algorithm == "SGD Regressor":
        model = SGDRegressor(**filter_config(['random_state', 'n_jobs']))
    elif algorithm == "Huber Regressor":
        model = HuberRegressor()
    elif algorithm == "Lars":
        model = Lars()
    elif algorithm == "RANSAC Regressor":
        model = RANSACRegressor(**filter_config(['random_state']))
    elif algorithm == "Theil-Sen Regressor":
        model = TheilSenRegressor(**filter_config(['random_state', 'n_jobs']))

    # Classification Models
    elif algorithm == "Logistic Regression":
        config = filter_config(['random_state', 'n_jobs'])
        config['max_iter'] = 1000 if config_type == "fast_training" else 5000
        model = LogisticRegression(**config)
    elif algorithm == "MLPClassifier":
        model = MLPClassifier(**filter_config(['random_state']))
    elif algorithm == "BernoulliRBM":
        model = BernoulliRBM(**filter_config(['random_state']))
    elif algorithm == "Decision Tree Classifier":
        model = DecisionTreeClassifier(**base_config)
    elif algorithm == "Random Forest Classifier":
        config = dict(base_config)
        config['n_estimators'] = 50 if config_type == "fast_training" else 200
        model = RandomForestClassifier(**config)
    elif algorithm == "SVC":
        model = SVC(**filter_config(['random_state']))
    elif algorithm == "Isolation Forest":
        config = filter_config(['random_state', 'n_jobs'])
        config['n_estimators'] = 50 if config_type == "fast_training" else 100
        model = IsolationForest(**config)
    elif algorithm == "Gradient Boosting Classifier":
        config = filter_config(['random_state'])
        config['n_estimators'] = 50 if config_type == "fast_training" else 200
        model = GradientBoostingClassifier(**config)
    elif algorithm == "Extra Tree Classifier":
        model = ExtraTreeClassifier(**base_config)
    elif algorithm == "XGBoost Classifier":
        config = filter_config(['random_state', 'n_jobs'])
        config['n_estimators'] = 50 if config_type == "fast_training" else 200
        config['eval_metric'] = 'logloss'
        model = XGBClassifier(**config)
    elif algorithm == "Gaussian Naive Bayes":
        model = GaussianNB()
    elif algorithm == "Radius Neighbors Classifier":
        model = RadiusNeighborsClassifier(**filter_config(['n_jobs']))
    elif algorithm == "K-Nearest Neighbors":
        model = KNeighborsClassifier(**filter_config(['n_jobs']))
    elif algorithm == "AdaBoost Classifier":
        config = filter_config(['random_state'])
        config['n_estimators'] = 25 if config_type == "fast_training" else 100
        model = AdaBoostClassifier(**config)
    elif algorithm == "Gaussian Process Classifier":
        model = GaussianProcessClassifier(**filter_config(['random_state', 'n_jobs']))
    elif algorithm == "Quadratic Discriminant Analysis":
        model = QuadraticDiscriminantAnalysis()
    elif algorithm == "SGD Classifier":
        model = SGDClassifier(**filter_config(['random_state', 'n_jobs']))
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Cache the model
    _model_cache[cache_key] = model
    return model


def get_model_parameters(algorithm: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Get optimized hyperparameters for the specified algorithm with comprehensive parameter grids.

    :param algorithm: the name of the model
    :return: Tuple of (model, parameters) for grid search
    """
    # Check cache first
    if algorithm in _parameter_cache:
        return _parameter_cache[algorithm]

    model = None
    parameters = {}

    # Regression Models with comprehensive parameter grids
    if algorithm == "Linear Regression":
        model = LinearRegression()
        parameters = {
            "fit_intercept": [True, False],
            "copy_X": [True, False],
            "n_jobs": [-1]
        }

    elif algorithm == "Ridge Regression":
        model = Ridge()
        parameters = {
            "alpha": [0.001, 0.01, 0.1, 1.0],
            "fit_intercept": [True],
            "positive": [False],
            "tol": [0.1, 0.01],
            "max_iter": [3000, 6000],
            "solver": ["sag", "saga"]
        }

    elif algorithm == "Lasso Regression":
        model = Lasso()
        parameters = {
            "alpha": [1, 0.5],
            "fit_intercept": [True],
            "precompute": [True, False],
            "positive": [False],
            "tol": [0.001],
            "max_iter": [4000, 5000],
            "selection": ["cyclic", "random"]
        }

    elif algorithm == "Elastic Net Regression":
        model = ElasticNet()
        parameters = {
            "alpha": [0.1, 0.01],
            "l1_ratio": [0.7, 0.6],
            "fit_intercept": [True],
            "precompute": [True],
            "positive": [True, False],
            "selection": ["random"],
            "tol": [0.1],
            "max_iter": [4000, 5000]
        }

    elif algorithm == "Bayesian Ridge Regression":
        model = BayesianRidge()
        parameters = {
            "n_iter": [300, 500],
            "tol": [1e-3, 1e-4],
            "alpha_1": [1e-6, 1e-5],
            "alpha_2": [1e-6, 1e-5],
            "lambda_1": [1e-6, 1e-5],
            "lambda_2": [1e-6, 1e-5]
        }

    elif algorithm == "SVR":
        model = SVR()
        parameters = {
            "C": np.logspace(-3, 1, 4),
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": np.arange(2, 8),
            "gamma": ["auto", "scale"],
            "tol": tol,
            "epsilon": tol,
            "shrinking": [True, False]
        }

    elif algorithm == "NuSVR":
        model = NuSVR()
        parameters = {
            "nu": [0.25, 0.5, 0.75],
            "C": np.logspace(-3, 1, 4),
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": np.arange(2, 8),
            "gamma": ["auto", "scale"],
            "tol": tol,
            "epsilon": tol,
            "shrinking": [True, False]
        }

    elif algorithm == "LinearSVR":
        model = LinearSVR()
        parameters = {
            "C": np.logspace(-3, 1, 4),
            "epsilon": tol,
            "tol": tol,
            "fit_intercept": [True, False],
            "max_iter": [1000, 2000, 3000]
        }

    elif algorithm == "SGD Regressor":
        model = SGDRegressor()
        parameters = {
            "loss": ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
            "penalty": ["l2", "l1", "elasticnet"],
            "alpha": tol,
            "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
            "shuffle": [True, False],
            "max_iter": max_iter
        }

    elif algorithm == "Huber Regressor":
        model = HuberRegressor()
        parameters = {
            "epsilon": [1.35, 1.5, 1.75],
            "max_iter": max_iter,
            "alpha": tol,
            "fit_intercept": [True, False]
        }

    elif algorithm == "Lars":
        model = Lars()
        parameters = {
            "n_nonzero_coefs": [1, 5, 10],
            "eps": [1e-4, 1e-3, 1e-2]
        }

    elif algorithm == "RANSAC Regressor":
        model = RANSACRegressor()
        parameters = {
            "min_samples": [None, 0.1, 0.5, 1.0],
            "max_trials": [100, 500, 1000],
            "residual_threshold": [1.0, 2.0, 3.0]
        }

    elif algorithm == "Theil-Sen Regressor":
        model = TheilSenRegressor()
        parameters = {
            "max_subpopulation": [100, 500, 1000],
            "max_iter": max_iter,
            "tol": tol
        }

    elif algorithm == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor()
        parameters = {
            "loss": ["huber", "quantile"],
            "learning_rate": learning_rate,
            "n_estimators": np.arange(10, 310, 50),
            "subsample": [1.0, 0.9, 0.8],
            "criterion": ["friedman_mse", "squared_error"],
            "max_depth": [3, 5, 7]
        }

    # Classification Models with comprehensive parameter grids
    elif algorithm == "Logistic Regression":
        model = LogisticRegression()
        # Define valid solver-penalty combinations
        solver_penalty_combinations = [
            {'solver': 'lbfgs', 'penalty': 'l2'},
            {'solver': 'lbfgs', 'penalty': 'none'},
            {'solver': 'liblinear', 'penalty': 'l1'},
            {'solver': 'liblinear', 'penalty': 'l2'},
            {'solver': 'newton-cg', 'penalty': 'l2'},
            {'solver': 'newton-cg', 'penalty': 'none'},
            {'solver': 'sag', 'penalty': 'l2'},
            {'solver': 'sag', 'penalty': 'none'},
            {'solver': 'saga', 'penalty': 'l1'},
            {'solver': 'saga', 'penalty': 'l2'},
            {'solver': 'saga', 'penalty': 'elasticnet'},
            {'solver': 'saga', 'penalty': 'none'}
        ]

        # Expand parameter grid
        parameters = []
        for combo in solver_penalty_combinations:
            params = {
                "C": np.logspace(-2, 1, 4),
                "max_iter": max_iter,
                "tol": tol
            }
            params.update(combo)
            parameters.append(params)

    elif algorithm == "Random Forest Classifier":
        model = RandomForestClassifier()
        parameters = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 3],
            "min_samples_leaf": [3, 4],
            "criterion": ["gini", "entropy"],
            "bootstrap": [True],
            "max_features": ['sqrt', 'log2', None]
        }

    elif algorithm == "XGBoost Classifier":
        model = XGBClassifier()
        parameters = {
            "max_depth": [3, 4, 6, 8, 10],
            "eta": [0.1, 0.2, 0.3, 0.4, 0.5],
            "objective": ["binary:logistic", "multi:softmax"],
            "num_class": [3],
            "eval_metric": ["mlogloss"]
        }

    elif algorithm == "SVC":
        model = SVC()
        parameters = {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": ["auto", "scale"],
            "tol": tol,
            "C": [0.1, 1, 10]
        }

    elif algorithm == "Decision Tree Classifier":
        model = DecisionTreeClassifier()
        parameters = {
            "min_samples_leaf": [1, 2, 3, 4, 10],
            "criterion": ["gini", "entropy"],
            "max_features": ['sqrt', 'log2'],
            "min_impurity_decrease": [0.1, 0.2],
            "class_weight": ['balanced'],
            "ccp_alpha": [0.1, 1, 0]
        }

    elif algorithm == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier()
        parameters = {
            "loss": ["log_loss"],
            "learning_rate": [0.1, 0.01, 0.001],
            "n_estimators": [50, 100, 200],
            "subsample": [1, 0.9, 0.8],
            "criterion": ["friedman_mse", "squared_error"],
            "max_depth": [3, 4, 6, 8, 10],
            "min_samples_split": [2, 5, 10],
            "max_features": ['sqrt', 'log2', None],
            "min_weight_fraction_leaf": [0, 0.1, 0.2],
            "max_leaf_nodes": [None, 10, 20, 30],
            "min_impurity_decrease": [0, 0.01, 0.1],
            "validation_fraction": [0.1, 0.2],
            "n_iter_no_change": [None, 10, 20],
            "tol": [0.0001, 0.001],
            "ccp_alpha": [0, 0.01, 0.1]
        }

    elif algorithm == "MLPClassifier":
        model = MLPClassifier()
        parameters = {
            "activation": ['relu', 'identity', 'logistic', 'tanh'],
            "hidden_layer_sizes": [(10,), (20,), (10, 10)],
            "alpha": learning_rate,
            "solver": ["sgd", "adam"],
            "learning_rate_init": learning_rate,
            "momentum": [0.9, 0.95, 0.99],
            "max_iter": max_iter
        }

    elif algorithm == "K-Nearest Neighbors":
        model = KNeighborsClassifier()
        parameters = {
            "n_neighbors": [3, 5, 11, 19],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [30, 40, 50],
            "p": [1, 2],
            "n_jobs": [-1]
        }

    elif algorithm == "AdaBoost Classifier":
        model = AdaBoostClassifier()
        parameters = {
            "n_estimators": np.arange(10, 310, 50),
            "learning_rate": learning_rate,
            "algorithm": ["SAMME", "SAMME.R"]
        }

    elif algorithm == "Gaussian Process Classifier":
        from sklearn.gaussian_process.kernels import RBF, Matern
        model = GaussianProcessClassifier()
        Rbf_ = 1.0 * RBF(length_scale=1.0)
        Mattern_ = 1.0 * Matern(length_scale=1.0)
        parameters = {
            "kernel": [Rbf_, Mattern_],
            "n_restarts_optimizer": [0, 1, 2],
            "max_iter_predict": np.arange(100, 500, 100),
            "optimizer": ["fmin_l_bfgs_b"],
            "n_jobs": [-1],
            "copy_X_train": [True, False],
            "multi_class": ["one_vs_rest", "one_vs_one"],
            "random_state": [0, 10, 20, 50]
        }

    elif algorithm == "SGD Classifier":
        model = SGDClassifier()
        parameters = {
            "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
            "penalty": ["l2", "l1", "elasticnet"],
            "alpha": tol,
            "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
            "shuffle": [True, False],
            "n_jobs": [-1],
            "max_iter": max_iter
        }

    elif algorithm == "Gaussian Naive Bayes":
        model = GaussianNB()
        parameters = {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        }

    elif algorithm == "Isolation Forest":
        model = IsolationForest()
        parameters = {
            "n_estimators": [50, 100, 200],
            "contamination": [0.1, 0.2, 0.3],
            "max_features": [5, 10, 15, 20],
            "n_jobs": [-1]
        }

    else:
        # Default case - return basic model with empty parameters
        model = get_model(algorithm)
        parameters = {}

    # Pipeline support for feature selection (Linear Regression example)
    if algorithm == "Linear Regression":
        try:
            # Wrap model in pipeline with feature selection
            model = Pipeline([
                ('feature_selection', SelectFromModel(LassoCV())),
                ('regression', model)
            ])
            # Update parameters for pipeline
            new_parameters = {
                f'regression__{key}': value for key, value in parameters.items()
            }
            new_parameters['feature_selection__estimator__cv'] = [5]
            new_parameters['feature_selection__threshold'] = [1e-5, 1e-4, 1e-3]
            parameters = new_parameters
        except NotFittedError:
            print("Model not fitted or incompatible with SelectFromModel.")

    # Cache the result
    result = (model, parameters)
    _parameter_cache[algorithm] = result
    return result


async def get_model_async(algorithm: str, config_type: str = "high_performance"):
    """
    Async version of get_model for non-blocking model instantiation.

    :param algorithm: the name of the model to be selected
    :param config_type: performance configuration type
    :return: SKlearn Model with optimized parameters
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_model, algorithm, config_type)


# Enhanced exports list
__all__ = [
    # Core functions
    'get_model',
    'get_model_parameters',
    'get_model_async',

    # Model configurations
    'MODEL_CONFIGS',

    # Parameter ranges
    'max_iter',
    'n_estimators',
    'learning_rate',
    'tol',
    'alpha',

    # Cache availability flag
    'CACHING_AVAILABLE',

    # Async support
    'async_get_model',
    'async_get_model_parameters',
    'create_model_pipeline',
    'save_model_cache',
    'load_model_cache',
    'clear_model_cache',
    'get_cache_stats'
]


def get_model_info(algorithm: str) -> Dict[str, Any]:
    """
    Get information about a model including its type and capabilities.

    :param algorithm: the name of the model
    :return: Dictionary with model information
    """
    classification_models = [
        "Logistic Regression", "MLPClassifier", "BernoulliRBM",
        "Decision Tree Classifier", "Random Forest Classifier", "SVC",
        "Isolation Forest", "Gradient Boosting Classifier",
        "Extra Tree Classifier", "XGBoost Classifier",
        "Gaussian Naive Bayes", "Radius Neighbors Classifier",
        "K-Nearest Neighbors", "AdaBoost Classifier",
        "Gaussian Process Classifier", "Quadratic Discriminant Analysis",
        "SGD Classifier"
    ]

    model_type = "classification" if algorithm in classification_models else "regression"
    supports_parallel = algorithm in [
        "Random Forest Classifier", "Extra Tree Classifier",
        "Logistic Regression", "SGD Classifier", "SGD Regressor",
        "K-Nearest Neighbors", "Radius Neighbors Classifier",
        "Linear Regression", "Theil-Sen Regressor"
    ]

    return {
        "algorithm": algorithm,
        "type": model_type,
        "supports_parallel": supports_parallel,
        "available_configs": list(MODEL_CONFIGS.keys())
    }


# Export functions and utilities
__all__ = [
    # Core model functions
    'get_model',
    'get_model_parameters',
    'get_model_info',

    # Cache management
    'clear_model_cache',
    'save_model_cache',
    'load_model_cache',
    'get_cache_stats',

    # Async support
    'get_model_async',

    # Configuration
    'MODEL_CONFIGS',

    # Parameter ranges
    'max_iter',
    'n_estimators',
    'learning_rate',
    'tol',
    'alpha',

    # Cache availability flag
    'CACHING_AVAILABLE',

    # Async support
    'async_get_model',
    'async_get_model_parameters',
    'create_model_pipeline',
    'save_model_cache',
    'load_model_cache',
    'clear_model_cache',
    'get_cache_stats'
]


# Async model creation for non-blocking operations
async def async_get_model(algorithm: str, config_type: str = "high_performance"):
    """
    Asynchronously get the model for training with caching and optimization.

    :param algorithm: the name of the model to be selected.
    :param config_type: performance configuration type
    :return: SKlearn Model with optimized parameters.
    """
    import asyncio

    # Run the synchronous model creation in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_model, algorithm, config_type)


async def async_get_model_parameters(algorithm: str):
    """
    Asynchronously get optimized hyperparameters for the specified algorithm.

    :param algorithm: the name of the model
    :return: Tuple of (model, parameters) for grid search
    """
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_model_parameters, algorithm)


def create_model_pipeline(algorithm: str, scaler=None, feature_selector=None, config_type: str = "high_performance"):
    """
    Create a scikit-learn pipeline with preprocessing and model.

    :param algorithm: the name of the model to be selected
    :param scaler: optional scaler for feature preprocessing
    :param feature_selector: optional feature selector
    :param config_type: performance configuration type
    :return: sklearn Pipeline object
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = []

    # Add scaler if provided
    if scaler is not None:
        steps.append(('scaler', scaler))
    elif algorithm in ['SVM', 'SGD Classifier', 'SGD Regressor', 'Logistic Regression']:
        # These algorithms benefit from scaling
        steps.append(('scaler', StandardScaler()))

    # Add feature selector if provided
    if feature_selector is not None:
        steps.append(('feature_selection', feature_selector))

    # Add the model
    model = get_model(algorithm, config_type)
    steps.append(('model', model))

    return Pipeline(steps)


def save_model_cache(filepath: str = "data/pickle/model_cache.pkl"):
    """
    Save the current model cache to disk for persistence.

    :param filepath: path to save the cache
    """
    import os
    import pickle

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        cache_data = {
            'model_cache': dict(_model_cache),
            'parameter_cache': dict(_parameter_cache)
        }

        with open(filepath, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"Model cache saved to {filepath}")
    except Exception as e:
        print(f"Error saving model cache: {e}")


def load_model_cache(filepath: str = "data/pickle/model_cache.pkl"):
    """
    Load model cache from disk.

    :param filepath: path to load the cache from
    """
    import pickle
    global _model_cache, _parameter_cache

    try:
        with open(filepath, 'rb') as f:
            cache_data = pickle.load(f)

        _model_cache.update(cache_data.get('model_cache', {}))
        _parameter_cache.update(cache_data.get('parameter_cache', {}))

        print(f"Model cache loaded from {filepath}")
        print(f"Loaded {len(_model_cache)} models and {len(_parameter_cache)} parameter sets")
    except FileNotFoundError:
        print(f"Cache file {filepath} not found. Starting with empty cache.")
    except Exception as e:
        print(f"Error loading model cache: {e}")


def clear_model_cache():
    """
    Clear the in-memory model cache.
    """
    global _model_cache, _parameter_cache
    _model_cache.clear()
    _parameter_cache.clear()

    # Clear LRU cache
    get_model.cache_clear()

    print("Model cache cleared")


def get_cache_stats():
    """
    Get statistics about the current cache usage.

    :return: dictionary with cache statistics
    """
    cache_info = get_model.cache_info()

    return {
        'lru_cache_hits': cache_info.hits,
        'lru_cache_misses': cache_info.misses,
        'lru_cache_size': cache_info.currsize,
        'lru_cache_max_size': cache_info.maxsize,
        'model_cache_size': len(_model_cache),
        'parameter_cache_size': len(_parameter_cache),
        'hit_rate': cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0
    }


# Enhanced caching with HybridCache integration
if CACHING_AVAILABLE:
    # Global hybrid cache instance
    _hybrid_cache = None

    def get_hybrid_cache():
        """Get or create the global hybrid cache instance."""
        global _hybrid_cache
        if _hybrid_cache is None:
            try:
                _hybrid_cache = get_global_cache()
            except Exception:
                _hybrid_cache = HybridCache(max_memory_items=256, disk_cache_size="100MB")
        return _hybrid_cache

    def cached_get_model(algorithm: str, config_type: str = "high_performance"):
        """
        Get model with hybrid caching (memory + disk).

        :param algorithm: the name of the model to be selected
        :param config_type: performance configuration type
        :return: SKlearn Model with optimized parameters
        """
        cache_key = f"model_{algorithm}_{config_type}"
        cache = get_hybrid_cache()

        # Try to get from cache
        cached_model = cache.get(cache_key)
        if cached_model is not None:
            return cached_model

        # Create new model and cache it
        model = get_model(algorithm, config_type)
        cache.set(cache_key, model, ttl=3600)  # Cache for 1 hour

        return model
