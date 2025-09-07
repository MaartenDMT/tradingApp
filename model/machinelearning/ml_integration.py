"""
Advanced ML Integration Module - Combines all advanced ML features into a unified interface.
"""
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Import advanced ML components
try:
    from .ml_advanced import (AutoMLPipeline, EnsembleManager,
                              FeatureImportanceAnalyzer, OnlineLearningManager)
    from .ml_config import (AUTOML_CONFIG, ENSEMBLE_CONFIG,
                            ONLINE_LEARNING_CONFIG)
    from .ml_utils_advanced import (DataPreprocessor, DataSplitter,
                                    ModelValidator, ResourceMonitor)
    ADVANCED_ML_AVAILABLE = True
except ImportError as e:
    ADVANCED_ML_AVAILABLE = False
    import warnings
    warnings.warn(f"Advanced ML features not available: {e}")


class AdvancedMLManager:
    """
    Unified manager for all advanced ML features.
    Provides a high-level interface for ensemble methods, AutoML, online learning, etc.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the Advanced ML Manager.

        :param config: Configuration dictionary
        :param logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}

        # Initialize components
        self.ensemble_manager = None
        self.automl_pipeline = None
        self.online_learning_manager = None
        self.feature_analyzer = None
        self.data_preprocessor = None
        self.model_validator = None
        self.resource_monitor = None
        self.data_splitter = None

        # State tracking
        self.is_initialized = False
        self.current_models = {}
        self.performance_history = []

        # Initialize if advanced ML is available
        if ADVANCED_ML_AVAILABLE:
            self._initialize_components()
        else:
            self.logger.warning("Advanced ML features not available")

    def _initialize_components(self):
        """Initialize all ML components."""
        try:
            self.ensemble_manager = EnsembleManager(self.logger)
            self.automl_pipeline = AutoMLPipeline(
                self.logger,
                max_time_minutes=self.config.get('max_time_minutes', AUTOML_CONFIG['max_time_minutes'])
            )
            self.feature_analyzer = FeatureImportanceAnalyzer(self.logger)
            self.data_preprocessor = DataPreprocessor(self.logger)
            self.model_validator = ModelValidator(self.logger)
            self.resource_monitor = ResourceMonitor(self.logger)
            self.data_splitter = DataSplitter(self.logger)

            self.is_initialized = True
            self.logger.info("Advanced ML Manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            self.is_initialized = False

    def preprocess_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                       preprocessing_config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Comprehensive data preprocessing pipeline.

        :param X: Feature data
        :param y: Target data (optional)
        :param preprocessing_config: Preprocessing configuration
        :return: Preprocessed features and targets
        """
        if not self.is_initialized:
            raise RuntimeError("Advanced ML Manager not initialized")

        config = preprocessing_config or {}
        self.logger.info("Starting comprehensive data preprocessing")

        # Handle missing values
        X_processed = self.data_preprocessor.handle_missing_values(
            X, strategy=config.get('missing_value_strategy', 'auto')
        )

        # Detect and optionally handle outliers
        if config.get('handle_outliers', True):
            outliers = self.data_preprocessor.detect_outliers(
                X_processed, method=config.get('outlier_method', 'iqr')
            )
            outlier_ratio = outliers.any(axis=1).sum() / len(X_processed)
            self.logger.info(f"Detected {outlier_ratio:.2%} rows with outliers")

            # Optionally remove severe outliers
            if outlier_ratio < 0.1 and config.get('remove_outliers', False):
                mask = ~outliers.any(axis=1)
                X_processed = X_processed[mask]
                if y is not None:
                    y = y[mask]
                self.logger.info(f"Removed {(~mask).sum()} outlier rows")

        # Smart scaling
        X_processed = self.data_preprocessor.smart_scaling(
            X_processed, method=config.get('scaling_method', 'auto')
        )

        self.logger.info("Data preprocessing completed")
        return X_processed, y

    def auto_model_selection(self, X: pd.DataFrame, y: pd.Series,
                           selection_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Automated model selection with comprehensive evaluation.

        :param X: Features
        :param y: Targets
        :param selection_config: Model selection configuration
        :return: Selection results
        """
        if not self.is_initialized:
            raise RuntimeError("Advanced ML Manager not initialized")

        config = selection_config or {}
        self.resource_monitor.start_monitoring()

        try:
            # Preprocess data
            X_processed, y_processed = self.preprocess_data(X, y, config.get('preprocessing', {}))

            # Auto model selection
            algorithms = config.get('algorithms', AUTOML_CONFIG['default_algorithms'])

            if config.get('use_parallel', True):
                results = self.automl_pipeline.parallel_model_evaluation(
                    X_processed.values, y_processed.values, algorithms,
                    max_workers=config.get('max_workers', AUTOML_CONFIG['parallel_workers'])
                )
            else:
                results = self.automl_pipeline.auto_select_model(
                    X_processed.values, y_processed.values, algorithms,
                    cv_folds=config.get('cv_folds', AUTOML_CONFIG['cv_folds'])
                )

            # Feature importance analysis for best model
            if results['best_model'] is not None:
                feature_importance = self.feature_analyzer.get_feature_importance_summary(
                    results['best_model'], X_processed, y_processed.values,
                    feature_names=X_processed.columns.tolist()
                )
                results['feature_importance'] = feature_importance

            # Resource usage
            results['resource_usage'] = self.resource_monitor.get_current_usage()

            # Store results
            self.current_models['automl_best'] = results['best_model']
            self.performance_history.append({
                'timestamp': time.time(),
                'method': 'automl',
                'results': results
            })

            self.logger.info(f"AutoML completed: Best model = {results.get('best_algorithm', 'None')}")
            return results

        except Exception as e:
            self.logger.error(f"AutoML failed: {e}")
            raise
        finally:
            self.resource_monitor.cleanup_memory()

    def create_ensemble(self, X: pd.DataFrame, y: pd.Series,
                       ensemble_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create and evaluate ensemble models.

        :param X: Features
        :param y: Targets
        :param ensemble_config: Ensemble configuration
        :return: Ensemble results
        """
        if not self.is_initialized:
            raise RuntimeError("Advanced ML Manager not initialized")

        config = ensemble_config or {}
        self.logger.info("Creating ensemble models")

        try:
            # Preprocess data
            X_processed, y_processed = self.preprocess_data(X, y, config.get('preprocessing', {}))

            # Split data for evaluation
            train_idx, test_idx = self.data_splitter.stratified_group_split(
                X_processed, y_processed,
                groups=config.get('groups', pd.Series(range(len(X_processed)))),
                test_size=config.get('test_size', 0.2)
            )

            X_train, X_test = X_processed.iloc[train_idx], X_processed.iloc[test_idx]
            y_train, y_test = y_processed.iloc[train_idx], y_processed.iloc[test_idx]

            ensemble_results = {}

            # Voting ensemble
            if config.get('create_voting', True):
                algorithms = config.get('voting_algorithms', ENSEMBLE_CONFIG['voting']['default_algorithms'])
                voting_ensemble = self.ensemble_manager.create_voting_ensemble(
                    algorithms,
                    ensemble_type=config.get('voting_type', ENSEMBLE_CONFIG['voting']['voting_type'])
                )

                # Train and evaluate
                voting_ensemble.fit(X_train.values, y_train.values)
                voting_metrics = self.ensemble_manager.evaluate_ensemble(
                    voting_ensemble, X_test.values, y_test.values
                )

                ensemble_results['voting'] = {
                    'model': voting_ensemble,
                    'metrics': voting_metrics,
                    'algorithms': algorithms
                }

            # Bagging ensemble
            if config.get('create_bagging', True):
                base_algorithm = config.get('bagging_algorithm', 'Random Forest')
                bagging_ensemble = self.ensemble_manager.create_bagging_ensemble(
                    base_algorithm,
                    n_estimators=config.get('bagging_estimators', ENSEMBLE_CONFIG['bagging']['n_estimators'])
                )

                # Train and evaluate
                bagging_ensemble.fit(X_train.values, y_train.values)
                bagging_metrics = self.ensemble_manager.evaluate_ensemble(
                    bagging_ensemble, X_test.values, y_test.values
                )

                ensemble_results['bagging'] = {
                    'model': bagging_ensemble,
                    'metrics': bagging_metrics,
                    'base_algorithm': base_algorithm
                }

            # Find best ensemble
            if ensemble_results:
                best_ensemble_name = max(ensemble_results.keys(),
                                      key=lambda k: ensemble_results[k]['metrics']['cv_mean'])

                ensemble_results['best_ensemble'] = {
                    'name': best_ensemble_name,
                    'model': ensemble_results[best_ensemble_name]['model'],
                    'metrics': ensemble_results[best_ensemble_name]['metrics']
                }

                # Store best ensemble
                self.current_models['ensemble_best'] = ensemble_results['best_ensemble']['model']

            self.logger.info(f"Ensemble creation completed: {len(ensemble_results)} ensembles created")
            return ensemble_results

        except Exception as e:
            self.logger.error(f"Ensemble creation failed: {e}")
            raise

    def setup_online_learning(self, initial_model: Any,
                             online_config: Optional[Dict[str, Any]] = None) -> OnlineLearningManager:
        """
        Setup online learning for continuous model adaptation.

        :param initial_model: Initial model for online learning
        :param online_config: Online learning configuration
        :return: Online learning manager
        """
        if not self.is_initialized:
            raise RuntimeError("Advanced ML Manager not initialized")

        config = online_config or {}

        self.online_learning_manager = OnlineLearningManager(
            initial_model,
            buffer_size=config.get('buffer_size', ONLINE_LEARNING_CONFIG['buffer_size']),
            update_frequency=config.get('update_frequency', ONLINE_LEARNING_CONFIG['update_frequency']),
            logger=self.logger
        )

        self.current_models['online_learning'] = self.online_learning_manager
        self.logger.info("Online learning setup completed")

        return self.online_learning_manager

    def comprehensive_model_analysis(self, X: pd.DataFrame, y: pd.Series,
                                   analysis_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive model analysis including feature importance, performance metrics, etc.

        :param X: Features
        :param y: Targets
        :param analysis_config: Analysis configuration
        :return: Comprehensive analysis results
        """
        if not self.is_initialized:
            raise RuntimeError("Advanced ML Manager not initialized")

        config = analysis_config or {}
        self.logger.info("Starting comprehensive model analysis")

        analysis_results = {
            'timestamp': time.time(),
            'data_summary': {},
            'preprocessing_analysis': {},
            'model_comparison': {},
            'feature_analysis': {},
            'performance_analysis': {}
        }

        try:
            # Data summary
            analysis_results['data_summary'] = {
                'n_samples': len(X),
                'n_features': len(X.columns),
                'missing_ratio': X.isnull().sum().sum() / (len(X) * len(X.columns)),
                'target_distribution': y.value_counts().to_dict() if y.dtype == 'object' else {
                    'mean': float(y.mean()),
                    'std': float(y.std()),
                    'min': float(y.min()),
                    'max': float(y.max())
                }
            }

            # AutoML analysis
            if config.get('run_automl', True):
                automl_results = self.auto_model_selection(X, y, config.get('automl_config', {}))
                analysis_results['model_comparison'] = automl_results

            # Ensemble analysis
            if config.get('run_ensemble', True):
                ensemble_results = self.create_ensemble(X, y, config.get('ensemble_config', {}))
                analysis_results['ensemble_analysis'] = ensemble_results

            # Feature importance analysis (using best model)
            best_model = self.current_models.get('automl_best') or self.current_models.get('ensemble_best')
            if best_model is not None:
                X_processed, y_processed = self.preprocess_data(X, y)
                best_model.fit(X_processed.values, y_processed.values)

                feature_analysis = self.feature_analyzer.get_feature_importance_summary(
                    best_model, X_processed, y_processed.values, X_processed.columns.tolist()
                )
                analysis_results['feature_analysis'] = feature_analysis

            # Performance summary
            analysis_results['performance_analysis'] = {
                'total_models_evaluated': len(self.performance_history),
                'resource_usage': self.resource_monitor.get_current_usage(),
                'best_model_type': type(best_model).__name__ if best_model else None
            }

            self.logger.info("Comprehensive analysis completed")
            return analysis_results

        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            raise

    def get_model_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate model recommendations based on analysis results.

        :param analysis_results: Results from comprehensive_model_analysis
        :return: Model recommendations
        """
        recommendations = {
            'best_model': None,
            'model_type_recommendation': None,
            'preprocessing_recommendations': [],
            'feature_recommendations': [],
            'performance_recommendations': [],
            'deployment_recommendations': []
        }

        try:
            # Model recommendations
            if 'model_comparison' in analysis_results:
                best_score = analysis_results['model_comparison'].get('best_score', 0)
                best_algorithm = analysis_results['model_comparison'].get('best_algorithm')

                if best_score > 0.8:
                    recommendations['model_type_recommendation'] = 'single_model'
                    recommendations['best_model'] = best_algorithm
                elif 'ensemble_analysis' in analysis_results:
                    recommendations['model_type_recommendation'] = 'ensemble'

            # Feature recommendations
            if 'feature_analysis' in analysis_results:
                feature_importance = analysis_results['feature_analysis']
                if 'top_features_model' in feature_importance:
                    top_features = [f[0] for f in feature_importance['top_features_model'][:5]]
                    recommendations['feature_recommendations'].append(
                        f"Focus on top features: {', '.join(top_features)}"
                    )

            # Performance recommendations
            resource_usage = analysis_results.get('performance_analysis', {}).get('resource_usage', {})
            if resource_usage.get('memory_mb', 0) > 1000:
                recommendations['performance_recommendations'].append(
                    "Consider model optimization for memory usage"
                )

            # Deployment recommendations
            if recommendations['model_type_recommendation'] == 'ensemble':
                recommendations['deployment_recommendations'].append(
                    "Use ensemble for better accuracy, single model for speed"
                )

            return recommendations

        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            return recommendations

    def export_results(self, results: Dict[str, Any], filepath: Union[str, Path]):
        """
        Export analysis results to file.

        :param results: Results to export
        :param filepath: Export file path
        """
        import json
        from datetime import datetime

        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'advanced_ml_version': '1.0.0',
            'results': results
        }

        # Make results JSON serializable
        def make_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return obj

        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items() if k != 'model'}  # Exclude model objects
            elif isinstance(d, list):
                return [clean_dict(item) for item in d]
            else:
                return make_serializable(d)

        clean_results = clean_dict(export_data)

        with open(filepath, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)

        self.logger.info(f"Results exported to {filepath}")


# Convenience function for quick analysis
def quick_ml_analysis(X: pd.DataFrame, y: pd.Series,
                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Perform a quick ML analysis with default settings.

    :param X: Features
    :param y: Targets
    :param config: Optional configuration
    :return: Analysis results
    """
    if not ADVANCED_ML_AVAILABLE:
        raise RuntimeError("Advanced ML features not available")

    manager = AdvancedMLManager(config)
    results = manager.comprehensive_model_analysis(X, y, config)
    recommendations = manager.get_model_recommendations(results)

    return {
        'analysis_results': results,
        'recommendations': recommendations,
        'manager': manager  # For continued use
    }


# Export main classes and functions
__all__ = [
    'AdvancedMLManager',
    'quick_ml_analysis',
    'ADVANCED_ML_AVAILABLE'
]
    'quick_ml_analysis',
    'ADVANCED_ML_AVAILABLE'
]
