"""
Comprehensive test runner for the ML trading system.
Enhanced to include RL algorithm testing.
"""
import sys
import time
import unittest
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from test.test_autobot import (
    TestAutoBot,
    TestAutoBotIntegration,
    TestConnectionManager,
    TestRiskManager,
)

# Import test modules
from test.test_ml_models import TestMLModels, TestMLModelsPerformance
from test.test_performance import (
    TestMachineLearningPerformance,
    TestMLModelPerformance,
    TestOverallSystemPerformance,
)

# Check if RL tests are available
try:
    import test.test_rl_algorithms  # noqa: F401
    import test.test_rl_training  # noqa: F401
    RL_TESTS_AVAILABLE = True
    print("✅ RL tests loaded successfully")
except ImportError as e:
    print(f"⚠️ RL tests not available: {e}")
    RL_TESTS_AVAILABLE = False


def create_test_suite():
    """Create comprehensive test suite."""
    test_suite = unittest.TestSuite()

    # Unit tests
    print("📦 Adding unit tests...")
    test_suite.addTest(unittest.makeSuite(TestMLModels))
    test_suite.addTest(unittest.makeSuite(TestRiskManager))
    test_suite.addTest(unittest.makeSuite(TestConnectionManager))
    test_suite.addTest(unittest.makeSuite(TestAutoBot))

    # Integration tests
    print("🔗 Adding integration tests...")
    test_suite.addTest(unittest.makeSuite(TestAutoBotIntegration))

    # Performance tests
    print("⚡ Adding performance tests...")
    test_suite.addTest(unittest.makeSuite(TestMLModelsPerformance))
    test_suite.addTest(unittest.makeSuite(TestMLModelPerformance))
    test_suite.addTest(unittest.makeSuite(TestMachineLearningPerformance))
    test_suite.addTest(unittest.makeSuite(TestOverallSystemPerformance))

    # Add RL tests if available
    if RL_TESTS_AVAILABLE:
        print("🤖 Adding RL algorithm tests...")
        try:
            from test.test_rl_algorithms import (
                TestA3CAlgorithm,
                TestICMModule,
                TestPPOAlgorithm,
                TestRLIntegration,
                TestRLPerformanceBenchmarks,
                TestSACAlgorithm,
            )
            test_suite.addTest(unittest.makeSuite(TestPPOAlgorithm))
            test_suite.addTest(unittest.makeSuite(TestSACAlgorithm))
            test_suite.addTest(unittest.makeSuite(TestA3CAlgorithm))
            test_suite.addTest(unittest.makeSuite(TestICMModule))
            test_suite.addTest(unittest.makeSuite(TestRLIntegration))
            test_suite.addTest(unittest.makeSuite(TestRLPerformanceBenchmarks))
        except ImportError:
            print("⚠️ Could not load RL algorithm tests")

        print("🏋️ Adding RL training tests...")
        try:
            from test.test_rl_training import (
                TestAlgorithmConfigs,
                TestMetricsTracker,
                TestRLTrainer,
                TestTrainingConfig,
                TestTrainingVisualizer,
            )
            test_suite.addTest(unittest.makeSuite(TestTrainingConfig))
            test_suite.addTest(unittest.makeSuite(TestRLTrainer))
            test_suite.addTest(unittest.makeSuite(TestMetricsTracker))
            test_suite.addTest(unittest.makeSuite(TestTrainingVisualizer))
            test_suite.addTest(unittest.makeSuite(TestAlgorithmConfigs))
        except ImportError:
            print("⚠️ Could not load RL training tests")

    return test_suite


def run_test_category(category_name, test_classes, verbose=True):
    """Run a specific category of tests."""
    print(f"\n{'='*60}")
    print(f"🧪 Running {category_name}")
    print(f"{'='*60}")

    suite = unittest.TestSuite()
    for test_class in test_classes:
        suite.addTest(unittest.makeSuite(test_class))

    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()

    # Print category summary
    duration = end_time - start_time
    print(f"\n📊 {category_name} Summary:")
    print(f"   ⏱️  Duration: {duration:.2f} seconds")
    print(f"   ✅ Tests run: {result.testsRun}")
    print(f"   ❌ Failures: {len(result.failures)}")
    print(f"   🔥 Errors: {len(result.errors)}")

    if result.failures:
        print(f"\n❌ Failures in {category_name}:")
        for test, traceback in result.failures:
            print(f"   - {test}")

    if result.errors:
        print(f"\n🔥 Errors in {category_name}:")
        for test, traceback in result.errors:
            print(f"   - {test}")

    return result


def main():
    """Main test runner function."""
    print("🚀 ML Trading System Test Suite")
    print("="*60)
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python version: {sys.version}")
    print()

    # Test categories
    categories = {
        "Unit Tests": [TestMLModels, TestRiskManager, TestConnectionManager, TestAutoBot],
        "Integration Tests": [TestAutoBotIntegration],
        "Performance Tests": [TestMLModelsPerformance, TestMLModelPerformance,
                             TestMachineLearningPerformance, TestOverallSystemPerformance]
    }

    total_start_time = time.time()
    all_results = []

    # Run test categories
    for category_name, test_classes in categories.items():
        try:
            result = run_test_category(category_name, test_classes)
            all_results.append((category_name, result))
        except Exception as e:
            print(f"❌ Failed to run {category_name}: {e}")
            continue

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # Overall summary
    print(f"\n{'='*60}")
    print("📈 OVERALL TEST SUMMARY")
    print(f"{'='*60}")
    print(f"⏱️  Total duration: {total_duration:.2f} seconds")

    total_tests = sum(result.testsRun for _, result in all_results)
    total_failures = sum(len(result.failures) for _, result in all_results)
    total_errors = sum(len(result.errors) for _, result in all_results)

    print(f"📊 Total tests: {total_tests}")
    print(f"✅ Passed: {total_tests - total_failures - total_errors}")
    print(f"❌ Failed: {total_failures}")
    print(f"🔥 Errors: {total_errors}")

    success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
    print(f"📈 Success rate: {success_rate:.1f}%")

    # Final verdict
    if total_failures == 0 and total_errors == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("✨ The ML trading system is ready for action!")
        return 0
    else:
        print(f"\n💥 {total_failures + total_errors} tests failed or had errors")
        print("🔧 Please fix the issues before deploying")
        return 1


def run_quick_tests():
    """Run a quick subset of tests for development."""
    print("⚡ Running Quick Test Suite")
    print("="*40)

    quick_tests = [TestMLModels, TestRiskManager, TestAutoBot]
    return run_test_category("Quick Tests", quick_tests)


def run_performance_only():
    """Run only performance tests."""
    print("🏃‍♂️ Running Performance Tests Only")
    print("="*40)

    perf_tests = [TestMLModelPerformance, TestMachineLearningPerformance, TestOverallSystemPerformance]
    return run_test_category("Performance Tests", perf_tests)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ML Trading System Test Runner')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance tests only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.quick:
        result = run_quick_tests()
        exit_code = 0 if result.wasSuccessful() else 1
    elif args.performance:
        result = run_performance_only()
        exit_code = 0 if result.wasSuccessful() else 1
    else:
        exit_code = main()

    sys.exit(exit_code)
