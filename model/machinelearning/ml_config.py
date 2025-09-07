"""
Configuration file for advanced ML features.
"""

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    'voting': {
        'default_algorithms': [
            'Random Forest',
            'Logistic Regression',
            'SVM',
            'Gradient Boosting'
        ],
        'voting_type': 'soft',  # 'hard' or 'soft'
        'weights': None,  # Equal weights if None
    },

    'bagging': {
        'n_estimators': 10,
        'max_samples': 1.0,
        'max_features': 1.0,
        'random_state': 42,
    },

    'custom': {
        'combination_methods': [
            'average',
            'weighted_average',
            'median',
            'max_vote'
        ],
        'default_method': 'average',
    }
}

# Online Learning Configuration
ONLINE_LEARNING_CONFIG = {
    'buffer_size': 1000,
    'update_frequency': 100,
    'min_samples_for_update': 10,
    'performance_window': 5,  # Number of recent updates to consider for trend
}

# AutoML Configuration
AUTOML_CONFIG = {
    'max_time_minutes': 30,
    'default_algorithms': [
        'Random Forest',
        'Logistic Regression',
        'SVM',
        'XGBoost',
        'Gradient Boosting',
        'Decision Tree',
        'Naive Bayes',
        'K-Nearest Neighbors'
    ],
    'cv_folds': 5,
    'parallel_workers': 4,
    'quick_algorithms': [  # For fast evaluation
        'Random Forest',
        'Logistic Regression',
        'Decision Tree'
    ]
}

# Feature Importance Configuration
FEATURE_IMPORTANCE_CONFIG = {
    'permutation_repeats': 10,
    'top_features_count': 10,
    'importance_threshold': 0.01,  # Minimum importance to consider
}

# Performance Monitoring
PERFORMANCE_CONFIG = {
    'metrics': {
        'classification': ['accuracy', 'precision', 'recall', 'f1_score'],
        'regression': ['mse', 'rmse', 'mae', 'r2_score']
    },
    'validation': {
        'test_size': 0.2,
        'random_state': 42,
        'stratify': True  # For classification
    }
}

# Model Selection Criteria
MODEL_SELECTION_CONFIG = {
    'primary_metric': 'cv_mean',  # Primary metric for model selection
    'secondary_metrics': ['cv_std', 'train_score'],
    'score_improvement_threshold': 0.01,  # Minimum improvement to consider
    'max_models_to_evaluate': 10,
}

# Advanced Features Flags
ADVANCED_FEATURES_FLAGS = {
    'enable_ensemble_methods': True,
    'enable_online_learning': True,
    'enable_automl': True,
    'enable_feature_importance': True,
    'enable_parallel_processing': True,
    'enable_performance_monitoring': True,
}

# Resource Limits
RESOURCE_LIMITS = {
    'max_memory_gb': 8,
    'max_cpu_cores': 4,
    'max_training_time_minutes': 60,
    'max_prediction_time_seconds': 30,
}

# Logging Configuration for Advanced ML
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': {
        'file': {
            'filename': 'data/logs/ml_advanced.log',
            'max_bytes': 10485760,  # 10MB
            'backup_count': 5,
        },
        'console': {
            'enabled': True,
            'level': 'INFO',
        }
    }
}

# Export all configurations
__all__ = [
    'ENSEMBLE_CONFIG',
    'ONLINE_LEARNING_CONFIG',
    'AUTOML_CONFIG',
    'FEATURE_IMPORTANCE_CONFIG',
    'PERFORMANCE_CONFIG',
    'MODEL_SELECTION_CONFIG',
    'ADVANCED_FEATURES_FLAGS',
    'RESOURCE_LIMITS',
    'LOGGING_CONFIG'
]
