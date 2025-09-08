"""
Comprehensive Test Suite for ML System

Tests all major components of the ML system including
configuration, algorithms, training, evaluation, and utilities.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


class TestMLConfig(unittest.TestCase):
    """Test ML configuration classes."""

    def setUp(self):
        """Set up test fixtures."""
        from model.ml_system.config.ml_config import MLConfig
        self.MLConfig = MLConfig

    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = self.MLConfig()

        self.assertEqual(config.algorithm, 'random_forest')
        self.assertEqual(config.target_type, 'regression')
        self.assertTrue(config.scaling_enabled)
        self.assertFalse(config.hyperparameter_optimization)

    def test_trading_regression_preset(self):
        """Test trading regression preset configuration."""
        config = self.MLConfig.for_trading_regression()

        self.assertEqual(config.algorithm, 'random_forest')
        self.assertEqual(config.target_type, 'regression')
        self.assertTrue(config.scaling_enabled)
        self.assertEqual(config.cross_validation_folds, 5)

    def test_trading_classification_preset(self):
        """Test trading classification preset configuration."""
        config = self.MLConfig.for_trading_classification()

        self.assertEqual(config.algorithm, 'random_forest')
        self.assertEqual(config.target_type, 'classification')
        self.assertTrue(config.scaling_enabled)
        self.assertEqual(config.cross_validation_folds, 5)

    def test_config_validation(self):
        """Test configuration validation."""
        config = self.MLConfig()

        # Test invalid algorithm
        with self.assertRaises(ValueError):
            config.algorithm = 'invalid_algorithm'
            config.validate()

        # Test invalid target type
        config = self.MLConfig()
        config.target_type = 'invalid_type'
        with self.assertRaises(ValueError):
            config.validate()

    def test_algorithm_parameters_extraction(self):
        """Test algorithm parameters extraction."""
        config = self.MLConfig()
        config.random_forest_n_estimators = 200
        config.random_forest_max_depth = 15

        params = config.get_algorithm_parameters('random_forest')

        self.assertEqual(params['n_estimators'], 200)
        self.assertEqual(params['max_depth'], 15)


class TestAlgorithmRegistry(unittest.TestCase):
    """Test algorithm registry functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from model.ml_system.algorithms.registry import AlgorithmRegistry
        self.registry = AlgorithmRegistry()

    def test_available_algorithms(self):
        """Test available algorithms listing."""
        regression_algorithms = self.registry.get_available_algorithms('regression')
        classification_algorithms = self.registry.get_available_algorithms('classification')

        self.assertIn('linear_regression', regression_algorithms['regression'])
        self.assertIn('random_forest', regression_algorithms['regression'])
        self.assertIn('logistic_regression', classification_algorithms['classification'])
        self.assertIn('random_forest', classification_algorithms['classification'])

    def test_algorithm_creation(self):
        """Test algorithm instance creation."""
        # Test regression algorithm
        rf_regressor = self.registry.get_algorithm('random_forest', 'regression')
        self.assertIsNotNone(rf_regressor)

        # Test classification algorithm
        rf_classifier = self.registry.get_algorithm('random_forest', 'classification')
        self.assertIsNotNone(rf_classifier)

        # Test with parameters
        rf_custom = self.registry.get_algorithm(
            'random_forest',
            'regression',
            {'n_estimators': 200, 'max_depth': 10}
        )
        self.assertEqual(rf_custom.n_estimators, 200)
        self.assertEqual(rf_custom.max_depth, 10)

    def test_invalid_algorithm(self):
        """Test handling of invalid algorithms."""
        with self.assertRaises(ValueError):
            self.registry.get_algorithm('invalid_algorithm', 'regression')

    def test_hyperparameter_search_space(self):
        """Test hyperparameter search space generation."""
        search_space = self.registry.get_hyperparameter_search_space('random_forest', 'regression')

        self.assertIn('n_estimators', search_space)
        self.assertIn('max_depth', search_space)
        self.assertIsInstance(search_space['n_estimators'], list)


class TestMLSystem(unittest.TestCase):
    """Test main ML system functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from model.ml_system import MLConfig, MLSystem
        self.MLSystem = MLSystem
        self.MLConfig = MLConfig

        # Generate test data
        self.X_reg, self.y_reg = make_regression(
            n_samples=300, n_features=10, noise=0.1, random_state=42
        )
        self.X_clf, self.y_clf = make_classification(
            n_samples=300, n_features=10, n_classes=2, random_state=42
        )

        # Convert to DataFrames
        self.X_reg_df = pd.DataFrame(self.X_reg, columns=[f"feature_{i}" for i in range(10)])
        self.X_clf_df = pd.DataFrame(self.X_clf, columns=[f"feature_{i}" for i in range(10)])

    def test_regression_training(self):
        """Test regression model training."""
        config = self.MLConfig(
            algorithm='random_forest',
            target_type='regression',
            hyperparameter_optimization=False
        )

        ml_system = self.MLSystem(config)
        results = ml_system.train(self.X_reg_df, self.y_reg, save_model=False)

        self.assertIn('train_metrics', results)
        self.assertIn('r2_score', results['train_metrics'])
        self.assertGreater(results['train_metrics']['r2_score'], 0.5)

    def test_classification_training(self):
        """Test classification model training."""
        config = self.MLConfig(
            algorithm='random_forest',
            target_type='classification',
            hyperparameter_optimization=False
        )

        ml_system = self.MLSystem(config)
        results = ml_system.train(self.X_clf_df, self.y_clf, save_model=False)

        self.assertIn('train_metrics', results)
        self.assertIn('accuracy', results['train_metrics'])
        self.assertGreater(results['train_metrics']['accuracy'], 0.7)

    def test_predictions(self):
        """Test model predictions."""
        config = self.MLConfig(
            algorithm='random_forest',
            target_type='regression',
            hyperparameter_optimization=False
        )

        ml_system = self.MLSystem(config)
        ml_system.train(self.X_reg_df, self.y_reg, save_model=False)

        # Test predictions
        predictions = ml_system.predict(self.X_reg_df)
        self.assertEqual(len(predictions), len(self.y_reg))
        self.assertTrue(isinstance(predictions, np.ndarray))

    def test_classification_probabilities(self):
        """Test classification probability predictions."""
        config = self.MLConfig(
            algorithm='random_forest',
            target_type='classification',
            hyperparameter_optimization=False
        )

        ml_system = self.MLSystem(config)
        ml_system.train(self.X_clf_df, self.y_clf, save_model=False)

        # Test probability predictions
        predictions, probabilities = ml_system.predict(self.X_clf_df, return_probabilities=True)

        self.assertEqual(len(predictions), len(self.y_clf))
        self.assertEqual(probabilities.shape[0], len(self.y_clf))
        self.assertEqual(probabilities.shape[1], 2)  # Binary classification

        # Check probabilities sum to 1
        prob_sums = np.sum(probabilities, axis=1)
        np.testing.assert_array_almost_equal(prob_sums, np.ones(len(prob_sums)))

    def test_model_evaluation(self):
        """Test model evaluation functionality."""
        config = self.MLConfig(
            algorithm='random_forest',
            target_type='regression',
            hyperparameter_optimization=False
        )

        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_reg_df, self.y_reg, test_size=0.3, random_state=42
        )

        ml_system = self.MLSystem(config)
        ml_system.train(X_train, y_train, save_model=False)

        # Evaluate model
        evaluation_results = ml_system.evaluate(X_test, y_test, detailed=True)

        self.assertIn('r2_score', evaluation_results)
        self.assertIn('rmse', evaluation_results)
        self.assertIn('mae', evaluation_results)
        self.assertIn('mse', evaluation_results)

    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization."""
        config = self.MLConfig(
            algorithm='random_forest',
            target_type='regression',
            hyperparameter_optimization=True,
            optimization_method='random_search',
            optimization_iterations=5  # Small number for testing
        )

        ml_system = self.MLSystem(config)
        results = ml_system.train(self.X_reg_df, self.y_reg, save_model=False)

        self.assertIn('best_params', results)
        self.assertIn('optimization_results', results)
        self.assertGreater(len(results['best_params']), 0)

    def test_cross_validation(self):
        """Test cross-validation functionality."""
        config = self.MLConfig(
            algorithm='random_forest',
            target_type='regression',
            hyperparameter_optimization=False,
            cross_validation_folds=3
        )

        ml_system = self.MLSystem(config)
        results = ml_system.train(
            self.X_reg_df, self.y_reg,
            cross_validation=True,
            save_model=False
        )

        self.assertIn('cross_validation', results)
        self.assertIn('mean_score', results['cross_validation'])
        self.assertIn('std_score', results['cross_validation'])
        self.assertIn('scores', results['cross_validation'])

    def test_feature_importance(self):
        """Test feature importance extraction."""
        config = self.MLConfig(
            algorithm='random_forest',
            target_type='regression',
            hyperparameter_optimization=False
        )

        ml_system = self.MLSystem(config)
        ml_system.train(self.X_reg_df, self.y_reg, save_model=False)

        feature_importance = ml_system.get_feature_importance()

        self.assertIsNotNone(feature_importance)
        self.assertEqual(len(feature_importance), len(self.X_reg_df.columns))

        # Check all importance values are non-negative
        for importance in feature_importance.values():
            self.assertGreaterEqual(importance, 0)


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering utilities."""

    def setUp(self):
        """Set up test fixtures."""
        from model.ml_system.utils.feature_engineering import FeatureEngineer
        self.FeatureEngineer = FeatureEngineer

        # Generate test data
        self.X, self.y = make_regression(n_samples=200, n_features=8, random_state=42)
        self.X_df = pd.DataFrame(self.X, columns=[f"feature_{i}" for i in range(8)])

    def test_basic_scaling(self):
        """Test basic feature scaling."""
        from model.ml_system.config.ml_config import MLConfig
        config = MLConfig()
        engineer = self.FeatureEngineer(config)
        X_scaled = engineer.fit_transform(self.X_df, scaling=True)

        # Check scaling (mean ~0, std ~1)
        means = np.mean(X_scaled, axis=0)
        stds = np.std(X_scaled, axis=0)

        np.testing.assert_array_almost_equal(means, np.zeros(X_scaled.shape[1]), decimal=10)
        np.testing.assert_array_almost_equal(stds, np.ones(X_scaled.shape[1]), decimal=10)

    def test_feature_selection(self):
        """Test feature selection functionality."""
        from model.ml_system.config.ml_config import MLConfig
        config = MLConfig()
        engineer = self.FeatureEngineer(config)
        X_selected = engineer.fit_transform(
            self.X_df,
            self.y,
            feature_selection=True,
            n_features=5,
            create_features=False
        )

        self.assertEqual(X_selected.shape[1], 5)
        self.assertEqual(X_selected.shape[0], self.X_df.shape[0])

    def test_polynomial_features(self):
        """Test polynomial feature generation."""
        from model.ml_system.config.ml_config import MLConfig
        config = MLConfig()
        engineer = self.FeatureEngineer(config)
        X_poly = engineer.fit_transform(
            self.X_df,
            polynomial_features=True,
            polynomial_degree=2
        )

        # Polynomial features should increase feature count
        self.assertGreater(X_poly.shape[1], self.X_df.shape[1])

    def test_interaction_features(self):
        """Test interaction feature generation."""
        from model.ml_system.config.ml_config import MLConfig
        config = MLConfig()
        engineer = self.FeatureEngineer(config)
        X_interact = engineer.fit_transform(
            self.X_df,
            interaction_features=True
        )

        # Interaction features should increase feature count
        self.assertGreater(X_interact.shape[1], self.X_df.shape[1])

    def test_combined_transformations(self):
        """Test combined feature transformations."""
        from model.ml_system.config.ml_config import MLConfig
        config = MLConfig()
        engineer = self.FeatureEngineer(config)
        X_transformed = engineer.fit_transform(
            self.X_df,
            self.y,
            scaling=True,
            feature_selection=True,
            n_features=6,
            polynomial_features=True,
            polynomial_degree=2
        )

        self.assertIsNotNone(X_transformed)
        self.assertEqual(X_transformed.shape[0], self.X_df.shape[0])

        # Check scaling on transformed features
        means = np.mean(X_transformed, axis=0)
        stds = np.std(X_transformed, axis=0)

        # Should be approximately scaled
        self.assertTrue(np.all(np.abs(means) < 1e-10))
        self.assertTrue(np.all(np.abs(stds - 1) < 1e-10))


class TestModelPersistence(unittest.TestCase):
    """Test model persistence functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from model.ml_system import MLConfig, MLSystem
        from model.ml_system.training.model_persistence import ModelPersistence

        self.ModelPersistence = ModelPersistence
        self.MLSystem = MLSystem
        self.MLConfig = MLConfig

        # Create temporary directory for testing on F: drive (has space)
        import os
        if os.path.exists('F:\\'):
            self.temp_dir = tempfile.mkdtemp(dir='F:\\')
        else:
            self.temp_dir = tempfile.mkdtemp()

        # Generate test data and train a model (use same feature count as FeatureEngineering tests)
        X, y = make_regression(n_samples=100, n_features=8, random_state=42)
        self.X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
        self.y = y

        config = self.MLConfig(algorithm='random_forest', target_type='regression')
        config.feature_engineering_enabled = False  # Disable feature engineering to avoid feature count mismatch
        self.ml_system = self.MLSystem(config)
        self.ml_system.train(self.X_df, self.y, save_model=False)

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_model_saving_and_loading(self):
        """Test model saving and loading."""
        config = self.MLConfig()
        config.model_save_dir = Path(self.temp_dir)
        config.model_versioning = False  # Disable versioning for this test
        persistence = self.ModelPersistence(config)

        # Save model
        model_path = persistence.save_model(
            self.ml_system.model,
            'test_model',
            metadata={'test': True}
        )

        self.assertTrue(os.path.exists(model_path))

        # Load model
        loaded_model = persistence.load_model('test_model')
        self.assertIsNotNone(loaded_model)

        # Test predictions are the same (use numpy arrays to avoid feature name warning)
        X_test = self.X_df.values
        original_predictions = self.ml_system.model.predict(X_test)
        loaded_predictions = loaded_model.predict(X_test)

        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)

    def test_metadata_handling(self):
        """Test metadata saving and loading."""
        config = self.MLConfig()
        config.model_save_dir = Path(self.temp_dir)
        config.model_versioning = False  # Disable versioning for this test
        persistence = self.ModelPersistence(config)

        metadata = {
            'algorithm': 'random_forest',
            'target_type': 'regression',
            'feature_count': 5
        }

        # Save model with metadata
        persistence.save_model(self.ml_system.model, 'test_model_meta', metadata=metadata)

        # Load metadata
        loaded_metadata = persistence.load_metadata('test_model_meta')

        self.assertEqual(loaded_metadata['algorithm'], 'random_forest')
        self.assertEqual(loaded_metadata['target_type'], 'regression')
        self.assertEqual(loaded_metadata['feature_count'], 5)

    def test_model_versioning(self):
        """Test model versioning functionality."""
        config = self.MLConfig()
        config.model_save_dir = Path(self.temp_dir)
        persistence = self.ModelPersistence(config)

        # Save multiple versions
        path1 = persistence.save_model(
            self.ml_system.model,
            'versioned_model',
            create_version=True
        )

        # Add small delay to ensure different timestamp
        import time
        time.sleep(0.1)

        path2 = persistence.save_model(
            self.ml_system.model,
            'versioned_model',
            create_version=True
        )

        # Paths should be different
        self.assertNotEqual(path1, path2)

        # Both models should exist
        self.assertTrue(os.path.exists(path1))
        self.assertTrue(os.path.exists(path2))

        # Get versions
        versions = persistence.get_model_versions('versioned_model')
        self.assertGreaterEqual(len(versions), 2)

    def test_list_saved_models(self):
        """Test listing saved models."""
        config = self.MLConfig()
        config.model_save_dir = Path(self.temp_dir)
        config.model_versioning = False  # Disable versioning for this test
        persistence = self.ModelPersistence(config)

        # Save multiple models
        persistence.save_model(self.ml_system.model, 'model_1')
        persistence.save_model(self.ml_system.model, 'model_2')
        persistence.save_model(self.ml_system.model, 'model_3')

        # List models
        models = persistence.list_saved_models()

        self.assertIn('model_1', models)
        self.assertIn('model_2', models)
        self.assertIn('model_3', models)


class TestPerformanceTracker(unittest.TestCase):
    """Test performance tracking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from model.ml_system.utils.performance_tracker import \
            PerformanceTracker
        self.PerformanceTracker = PerformanceTracker
        self.tracker = PerformanceTracker()

    def test_performance_tracking(self):
        """Test basic performance tracking."""
        # Track some performance data
        performance_data = {
            'algorithm': 'random_forest',
            'accuracy': 0.85,
            'training_time': 10.5,
            'feature_count': 15
        }

        self.tracker.track_performance('experiment_1', performance_data)

        # Get summary
        summary = self.tracker.get_performance_summary('experiment_1')

        self.assertIsNotNone(summary)
        self.assertEqual(len(summary['metrics']), 1)
        self.assertEqual(summary['metrics'][0]['algorithm'], 'random_forest')

    def test_multiple_experiments(self):
        """Test tracking multiple experiments."""
        # Track multiple experiments
        for i in range(3):
            performance_data = {
                'algorithm': f'algorithm_{i}',
                'accuracy': 0.8 + i * 0.05,
                'training_time': 10 + i * 2
            }
            self.tracker.track_performance('multi_experiment', performance_data)

        # Get summary
        summary = self.tracker.get_performance_summary('multi_experiment')

        self.assertEqual(len(summary['metrics']), 3)
        self.assertEqual(summary['experiment_count'], 3)

    def test_data_export(self):
        """Test performance data export."""
        # Track some data
        for i in range(5):
            performance_data = {
                'run': i,
                'score': 0.7 + i * 0.05
            }
            self.tracker.track_performance('export_test', performance_data)

        # Export data
        exported_data = self.tracker.export_data('export_test')

        self.assertEqual(len(exported_data), 5)
        self.assertIn('run', exported_data[0])
        self.assertIn('score', exported_data[0])


def run_test_suite():
    """Run the complete test suite."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestMLConfig,
        TestAlgorithmRegistry,
        TestMLSystem,
        TestFeatureEngineering,
        TestModelPersistence,
        TestPerformanceTracker
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result


if __name__ == "__main__":
    print("Running ML System Test Suite")
    print("=" * 50)

    # Run all tests
    test_result = run_test_suite()

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUITE SUMMARY")
    print("=" * 50)
    print(f"Tests run: {test_result.testsRun}")
    print(f"Failures: {len(test_result.failures)}")
    print(f"Errors: {len(test_result.errors)}")
    print(f"Success rate: {(test_result.testsRun - len(test_result.failures) - len(test_result.errors)) / test_result.testsRun * 100:.1f}%")

    if test_result.failures:
        print(f"\nFailures: {len(test_result.failures)}")
        for test, traceback in test_result.failures:
            print(f"  - {test}: {traceback}")

    if test_result.errors:
        print(f"\nErrors: {len(test_result.errors)}")
        for test, traceback in test_result.errors:
            print(f"  - {test}: {traceback}")

    # Exit with appropriate code
    exit_code = 0 if len(test_result.failures) == 0 and len(test_result.errors) == 0 else 1
    exit(exit_code)
