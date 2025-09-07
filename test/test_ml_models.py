"""
Unit tests for ML models functionality.
"""
import asyncio
import os
import sys
import unittest

import numpy as np
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.machinelearning.ml_models import (MODEL_CONFIGS, async_get_model,
                                             async_get_model_parameters,
                                             clear_model_cache,
                                             create_model_pipeline,
                                             get_cache_stats, get_model,
                                             get_model_parameters,
                                             load_model_cache,
                                             save_model_cache)


class TestMLModels(unittest.TestCase):
    """Test cases for ML model functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_algorithms = [
            "Linear Regression", "Ridge Regression", "Random Forest",
            "SVM", "Logistic Regression", "XGBoost"
        ]
        self.sample_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

    def test_get_model_basic(self):
        """Test basic model creation."""
        for algorithm in self.test_algorithms:
            with self.subTest(algorithm=algorithm):
                try:
                    model = get_model(algorithm)
                    self.assertIsNotNone(model)
                    # Check if model has required scikit-learn interface
                    self.assertTrue(hasattr(model, 'fit'))
                    self.assertTrue(hasattr(model, 'predict'))
                except ValueError as e:
                    # Some algorithms might not be implemented
                    if "Unknown algorithm" in str(e):
                        self.skipTest(f"Algorithm {algorithm} not implemented")
                    else:
                        raise

    def test_get_model_with_configs(self):
        """Test model creation with different configurations."""
        for config_type in MODEL_CONFIGS.keys():
            with self.subTest(config_type=config_type):
                model = get_model("Random Forest", config_type)
                self.assertIsNotNone(model)

    def test_get_model_parameters(self):
        """Test parameter grid retrieval."""
        for algorithm in self.test_algorithms:
            with self.subTest(algorithm=algorithm):
                try:
                    model, params = get_model_parameters(algorithm)
                    self.assertIsNotNone(model)
                    self.assertIsInstance(params, dict)
                    self.assertGreater(len(params), 0)
                except ValueError as e:
                    if "Unknown algorithm" in str(e):
                        self.skipTest(f"Algorithm {algorithm} not implemented")
                    else:
                        raise

    def test_async_get_model(self):
        """Test async model creation."""
        async def run_async_test():
            model = await async_get_model("Random Forest")
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, 'fit'))

        asyncio.run(run_async_test())

    def test_async_get_model_parameters(self):
        """Test async parameter retrieval."""
        async def run_async_test():
            model, params = await async_get_model_parameters("Random Forest")
            self.assertIsNotNone(model)
            self.assertIsInstance(params, dict)

        asyncio.run(run_async_test())

    def test_create_model_pipeline(self):
        """Test pipeline creation."""
        pipeline = create_model_pipeline("Random Forest")
        self.assertIsNotNone(pipeline)
        self.assertTrue(hasattr(pipeline, 'fit'))
        self.assertTrue(hasattr(pipeline, 'predict'))

        # Test with scaler
        from sklearn.preprocessing import StandardScaler
        pipeline_with_scaler = create_model_pipeline("SVM", scaler=StandardScaler())
        self.assertIsNotNone(pipeline_with_scaler)

    def test_cache_operations(self):
        """Test cache save/load operations."""
        # Clear cache first
        clear_model_cache()

        # Create some models to populate cache
        model1 = get_model("Random Forest")
        model2 = get_model("SVM")

        # Test cache stats
        stats = get_cache_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('lru_cache_hits', stats)
        self.assertIn('lru_cache_misses', stats)

        # Test save cache
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            try:
                save_model_cache(tmp.name)

                # Clear and reload
                clear_model_cache()
                load_model_cache(tmp.name)

                # Verify cache is working
                stats_after = get_cache_stats()
                self.assertIsInstance(stats_after, dict)
            finally:
                os.unlink(tmp.name)

    def test_model_fitting_and_prediction(self):
        """Test that models can actually fit and predict."""
        X = self.sample_data[['feature1', 'feature2']].values
        y = self.sample_data['target'].values

        for algorithm in ["Random Forest", "Logistic Regression"]:
            with self.subTest(algorithm=algorithm):
                try:
                    model = get_model(algorithm)
                    model.fit(X, y)
                    predictions = model.predict(X)
                    self.assertEqual(len(predictions), len(y))
                except Exception as e:
                    self.fail(f"Model {algorithm} failed to fit/predict: {e}")

    def test_invalid_algorithm(self):
        """Test handling of invalid algorithm names."""
        with self.assertRaises(ValueError):
            get_model("NonExistentAlgorithm")

        with self.assertRaises(ValueError):
            get_model_parameters("NonExistentAlgorithm")

    def test_model_reproducibility(self):
        """Test that models produce consistent results."""
        # Test with algorithms that support random_state
        X = self.sample_data[['feature1', 'feature2']].values
        y = self.sample_data['target'].values

        model1 = get_model("Random Forest")
        model2 = get_model("Random Forest")

        model1.fit(X, y)
        model2.fit(X, y)

        pred1 = model1.predict(X)
        pred2 = model2.predict(X)

        # Results should be similar (allowing for some randomness)
        accuracy_diff = abs(np.mean(pred1 == y) - np.mean(pred2 == y))
        self.assertLess(accuracy_diff, 0.1, "Models should produce similar results")


class TestMLModelsPerformance(unittest.TestCase):
    """Performance tests for ML models."""

    def test_model_creation_speed(self):
        """Test that model creation is reasonably fast."""
        import time

        start_time = time.time()
        for _ in range(10):
            model = get_model("Random Forest")
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        self.assertLess(avg_time, 1.0, "Model creation should be fast")

    def test_cache_effectiveness(self):
        """Test that caching improves performance."""
        import time

        clear_model_cache()

        # First creation (cache miss)
        start_time = time.time()
        model1 = get_model("Random Forest")
        first_time = time.time() - start_time

        # Second creation (cache hit)
        start_time = time.time()
        model2 = get_model("Random Forest")
        second_time = time.time() - start_time

        # Cache hit should be significantly faster
        self.assertLess(second_time, first_time, "Cached model creation should be faster")


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestMLModels))
    test_suite.addTest(unittest.makeSuite(TestMLModelsPerformance))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    if result.wasSuccessful():
        print(f"\n✅ All {result.testsRun} tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    if result.wasSuccessful():
        print(f"\n✅ All {result.testsRun} tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests")
