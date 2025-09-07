"""
Performance benchmarking tests for the ML trading system.
"""
import asyncio
import os
import statistics
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock

from model.machinelearning.machinelearning import MachineLearning
from model.machinelearning.ml_models import (async_get_model, get_model,
                                             get_model_parameters)


class PerformanceBenchmark:
    """Performance benchmarking utilities."""

    @staticmethod
    def time_function(func, *args, **kwargs):
        """Time a function execution."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time

    @staticmethod
    async def time_async_function(func, *args, **kwargs):
        """Time an async function execution."""
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time

    @staticmethod
    def run_multiple_times(func, times=10, *args, **kwargs):
        """Run a function multiple times and collect timing statistics."""
        times_list = []
        results = []

        for _ in range(times):
            result, elapsed_time = PerformanceBenchmark.time_function(func, *args, **kwargs)
            times_list.append(elapsed_time)
            results.append(result)

        return {
            'results': results,
            'times': times_list,
            'avg_time': statistics.mean(times_list),
            'min_time': min(times_list),
            'max_time': max(times_list),
            'std_dev': statistics.stdev(times_list) if len(times_list) > 1 else 0
        }


class TestMLModelPerformance(unittest.TestCase):
    """Performance tests for ML models."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.algorithms = [
            "Linear Regression", "Ridge Regression", "Random Forest",
            "SVM", "Logistic Regression"
        ]

        # Create test datasets of different sizes
        self.small_data = self._create_test_data(100)
        self.medium_data = self._create_test_data(1000)
        self.large_data = self._create_test_data(10000)

    def _create_test_data(self, size):
        """Create test dataset of specified size."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(size),
            'feature2': np.random.randn(size),
            'feature3': np.random.randn(size),
            'feature4': np.random.randn(size),
            'target': np.random.randint(0, 2, size)
        })

    def test_model_creation_performance(self):
        """Test model creation performance across algorithms."""
        print("\n📊 Model Creation Performance Benchmark")
        print("=" * 50)

        results = {}
        for algorithm in self.algorithms:
            try:
                benchmark = PerformanceBenchmark.run_multiple_times(
                    get_model, times=5, algorithm=algorithm
                )
                results[algorithm] = benchmark

                print(f"{algorithm:20} | Avg: {benchmark['avg_time']:.4f}s | "
                      f"Min: {benchmark['min_time']:.4f}s | "
                      f"Max: {benchmark['max_time']:.4f}s")

                # Assert reasonable performance
                self.assertLess(benchmark['avg_time'], 2.0,
                               f"{algorithm} creation too slow")

            except ValueError as e:
                if "Unknown algorithm" in str(e):
                    print(f"{algorithm:20} | SKIPPED - Not implemented")
                else:
                    raise

        # Find fastest and slowest
        if results:
            fastest = min(results.keys(), key=lambda k: results[k]['avg_time'])
            slowest = max(results.keys(), key=lambda k: results[k]['avg_time'])

            print(f"\n🏆 Fastest: {fastest} ({results[fastest]['avg_time']:.4f}s)")
            print(f"🐌 Slowest: {slowest} ({results[slowest]['avg_time']:.4f}s)")

    def test_model_training_performance(self):
        """Test model training performance with different data sizes."""
        print("\n📊 Model Training Performance Benchmark")
        print("=" * 50)

        algorithm = "Random Forest"  # Use a representative algorithm
        datasets = {
            'Small (100)': self.small_data,
            'Medium (1K)': self.medium_data,
            'Large (10K)': self.large_data
        }

        for dataset_name, data in datasets.items():
            X = data[['feature1', 'feature2', 'feature3', 'feature4']].values
            y = data['target'].values

            def train_model():
                model = get_model(algorithm)
                model.fit(X, y)
                return model

            benchmark = PerformanceBenchmark.run_multiple_times(train_model, times=3)

            print(f"{dataset_name:15} | Avg: {benchmark['avg_time']:.4f}s | "
                  f"Min: {benchmark['min_time']:.4f}s | "
                  f"Max: {benchmark['max_time']:.4f}s")

            # Performance assertions based on data size
            if '100' in dataset_name:
                self.assertLess(benchmark['avg_time'], 1.0)
            elif '1K' in dataset_name:
                self.assertLess(benchmark['avg_time'], 5.0)
            elif '10K' in dataset_name:
                self.assertLess(benchmark['avg_time'], 30.0)

    def test_async_vs_sync_performance(self):
        """Compare async vs sync model creation performance."""
        print("\n📊 Async vs Sync Performance Comparison")
        print("=" * 50)

        algorithm = "Random Forest"
        times = 5

        # Test sync performance
        sync_benchmark = PerformanceBenchmark.run_multiple_times(
            get_model, times=times, algorithm=algorithm
        )

        # Test async performance
        async def test_async():
            times_list = []
            for _ in range(times):
                _, elapsed_time = await PerformanceBenchmark.time_async_function(
                    async_get_model, algorithm=algorithm
                )
                times_list.append(elapsed_time)

            return {
                'times': times_list,
                'avg_time': statistics.mean(times_list),
                'min_time': min(times_list),
                'max_time': max(times_list)
            }

        async_benchmark = asyncio.run(test_async())

        print(f"Sync  | Avg: {sync_benchmark['avg_time']:.4f}s | "
              f"Min: {sync_benchmark['min_time']:.4f}s | "
              f"Max: {sync_benchmark['max_time']:.4f}s")

        print(f"Async | Avg: {async_benchmark['avg_time']:.4f}s | "
              f"Min: {async_benchmark['min_time']:.4f}s | "
              f"Max: {async_benchmark['max_time']:.4f}s")

        # Async should be similar to sync for single operations
        # (main benefit is in concurrent operations)
        ratio = async_benchmark['avg_time'] / sync_benchmark['avg_time']
        self.assertLess(ratio, 2.0, "Async overhead too high")

    def test_concurrent_model_creation(self):
        """Test concurrent model creation performance."""
        print("\n📊 Concurrent Model Creation Performance")
        print("=" * 50)

        algorithms = self.algorithms[:3]  # Use first 3 algorithms
        workers = [1, 2, 4, 8]

        def create_all_models():
            for alg in algorithms:
                try:
                    get_model(alg)
                except ValueError:
                    pass  # Skip unknown algorithms

        for worker_count in workers:
            start_time = time.perf_counter()

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(create_all_models) for _ in range(worker_count)]
                for future in futures:
                    future.result()

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            print(f"Workers: {worker_count:2} | Time: {elapsed_time:.4f}s | "
                  f"Models/sec: {(len(algorithms) * worker_count) / elapsed_time:.2f}")

    def test_parameter_grid_generation_performance(self):
        """Test parameter grid generation performance."""
        print("\n📊 Parameter Grid Generation Performance")
        print("=" * 50)

        for algorithm in self.algorithms:
            try:
                benchmark = PerformanceBenchmark.run_multiple_times(
                    get_model_parameters, times=10, algorithm=algorithm
                )

                print(f"{algorithm:20} | Avg: {benchmark['avg_time']:.6f}s | "
                      f"Min: {benchmark['min_time']:.6f}s")

                # Parameter generation should be very fast
                self.assertLess(benchmark['avg_time'], 0.01,
                               f"{algorithm} parameter generation too slow")

            except ValueError as e:
                if "Unknown algorithm" in str(e):
                    print(f"{algorithm:20} | SKIPPED - Not implemented")
                else:
                    raise


class TestMachineLearningPerformance(unittest.TestCase):
    """Performance tests for MachineLearning class."""

    def setUp(self):
        """Set up ML performance test fixtures."""
        self.mock_exchange = Mock()
        self.ml = MachineLearning(self.mock_exchange, "BTC/USDT")

        # Create test data
        self.test_data = self._create_test_data(1000)

    def _create_test_data(self, size):
        """Create test dataset."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(size),
            'feature2': np.random.randn(size),
            'feature3': np.random.randn(size),
            'target': np.random.randint(0, 2, size)
        })

    def test_parallel_model_training_performance(self):
        """Test parallel model training performance."""
        print("\n📊 Parallel Model Training Performance")
        print("=" * 50)

        algorithms = ["Random Forest", "Logistic Regression", "SVM"]
        X = self.test_data[['feature1', 'feature2', 'feature3']].values
        y = self.test_data['target'].values

        # Split data for training and testing
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        def accuracy_score(y_true, y_pred):
            return np.mean(y_true == y_pred)

        # Test parallel training
        start_time = time.perf_counter()
        results = self.ml.parallel_model_training(
            algorithms, X_train, y_train, X_test, y_test, accuracy_score
        )
        end_time = time.perf_counter()

        parallel_time = end_time - start_time

        print(f"Parallel training of {len(algorithms)} models: {parallel_time:.4f}s")

        # Verify all models were trained
        successful_models = sum(1 for r in results.values() if r['status'] == 'success')
        self.assertGreater(successful_models, 0, "No models trained successfully")

        print(f"Successfully trained: {successful_models}/{len(algorithms)} models")

        # Test sequential training for comparison
        start_time = time.perf_counter()
        for algorithm in algorithms:
            try:
                model = get_model(algorithm)
                model.fit(X_train, y_train)
                model.predict(X_test)
            except:
                pass
        end_time = time.perf_counter()

        sequential_time = end_time - start_time
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0

        print(f"Sequential training time: {sequential_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")

    def test_memory_optimization_performance(self):
        """Test memory optimization performance."""
        print("\n📊 Memory Optimization Performance")
        print("=" * 50)

        # Monitor memory usage before optimization
        self.ml.performance_monitor.log_performance("Before optimization", self.ml.logger)

        initial_memory = self.ml.performance_monitor.get_memory_usage()

        # Create some objects to optimize
        large_objects = []
        for i in range(100):
            large_objects.append(np.random.randn(1000, 1000))

        memory_after_allocation = self.ml.performance_monitor.get_memory_usage()

        # Run memory optimization
        start_time = time.perf_counter()
        self.ml.optimize_memory_usage()
        end_time = time.perf_counter()

        optimization_time = end_time - start_time
        final_memory = self.ml.performance_monitor.get_memory_usage()

        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"After allocation: {memory_after_allocation:.1f} MB")
        print(f"After optimization: {final_memory:.1f} MB")
        print(f"Optimization time: {optimization_time:.4f}s")

        # Memory optimization should be fast
        self.assertLess(optimization_time, 1.0, "Memory optimization too slow")


class TestOverallSystemPerformance(unittest.TestCase):
    """Overall system performance tests."""

    def test_end_to_end_prediction_performance(self):
        """Test end-to-end prediction performance."""
        print("\n📊 End-to-End Prediction Performance")
        print("=" * 50)

        mock_exchange = Mock()
        ml = MachineLearning(mock_exchange, "BTC/USDT")

        # Create test data
        test_df = pd.DataFrame({
            'open': np.random.randn(100) + 50000,
            'high': np.random.randn(100) + 50100,
            'low': np.random.randn(100) + 49900,
            'close': np.random.randn(100) + 50000,
            'volume': np.random.randn(100) + 1000
        })

        algorithms = ["Random Forest", "Logistic Regression"]

        for algorithm in algorithms:
            try:
                model = get_model(algorithm)

                # Time the prediction process
                def make_prediction():
                    return ml.predict(model, test_df, '1h', 'BTC/USDT')

                benchmark = PerformanceBenchmark.run_multiple_times(
                    make_prediction, times=5
                )

                print(f"{algorithm:20} | Avg: {benchmark['avg_time']:.4f}s | "
                      f"Min: {benchmark['min_time']:.4f}s")

                # Predictions should be fast
                self.assertLess(benchmark['avg_time'], 5.0,
                               f"{algorithm} prediction too slow")

            except Exception as e:
                print(f"{algorithm:20} | ERROR: {str(e)}")


if __name__ == '__main__':
    print("🚀 Starting ML Trading System Performance Benchmarks")
    print("=" * 60)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add performance test cases
    test_suite.addTest(unittest.makeSuite(TestMLModelPerformance))
    test_suite.addTest(unittest.makeSuite(TestMachineLearningPerformance))
    test_suite.addTest(unittest.makeSuite(TestOverallSystemPerformance))

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)

    # Print final summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print(f"✅ All {result.testsRun} performance tests passed!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests")

    print("🏁 Performance benchmarking complete!")
    if result.wasSuccessful():
        print(f"✅ All {result.testsRun} performance tests passed!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests")

    print("🏁 Performance benchmarking complete!")
