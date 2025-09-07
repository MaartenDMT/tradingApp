"""
Enhanced test runner for the complete ML trading system.

This module provides comprehensive testing for all system components
including traditional ML models, RL algorithms, and integration tests.
"""

import sys
import unittest
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

"""
Enhanced test runner for the complete ML trading system.

This module provides comprehensive testing for all system components
including traditional ML models, RL algorithms, and integration tests.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


"""
Enhanced test runner for the complete ML trading system.

This module provides comprehensive testing for all system components
including traditional ML models, RL algorithms, and integration tests.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_comprehensive_test_suite():
    """Create comprehensive test suite for all system components."""

    # Import test modules after path setup
    from test.test_autobot import (
        TestAutoBot,
        TestAutoBotIntegration,
        TestConnectionManager,
        TestRiskManager,
    )
    from test.test_ml_models import TestMLModels, TestMLModelsPerformance
    from test.test_performance import (
        TestMachineLearningPerformance,
        TestMLModelPerformance,
        TestOverallSystemPerformance,
    )

    test_suite = unittest.TestSuite()

    print("=" * 60)
    print("🧪 COMPREHENSIVE ML TRADING SYSTEM TEST SUITE")
    print("=" * 60)

    # Traditional ML and system tests
    print("\n📦 Adding Traditional ML Tests...")
    test_suite.addTest(unittest.makeSuite(TestMLModels))
    test_suite.addTest(unittest.makeSuite(TestRiskManager))
    test_suite.addTest(unittest.makeSuite(TestConnectionManager))
    test_suite.addTest(unittest.makeSuite(TestAutoBot))

    print("🔗 Adding Integration Tests...")
    test_suite.addTest(unittest.makeSuite(TestAutoBotIntegration))

    print("⚡ Adding Performance Tests...")
    test_suite.addTest(unittest.makeSuite(TestMLModelsPerformance))
    test_suite.addTest(unittest.makeSuite(TestMLModelPerformance))
    test_suite.addTest(unittest.makeSuite(TestMachineLearningPerformance))
    test_suite.addTest(unittest.makeSuite(TestOverallSystemPerformance))

    # RL system tests (if available)
    try:
        from test.test_rl_algorithms import (
            TestA3CAlgorithm,
            TestICMModule,
            TestPPOAlgorithm,
            TestRLIntegration,
            TestRLPerformanceBenchmarks,
            TestSACAlgorithm,
        )
        from test.test_rl_training import (
            TestAlgorithmConfigs,
            TestMetricsTracker,
            TestRLTrainer,
            TestTrainingConfig,
            TestTrainingVisualizer,
        )

        print("\n🤖 Adding RL Algorithm Tests...")
        test_suite.addTest(unittest.makeSuite(TestPPOAlgorithm))
        test_suite.addTest(unittest.makeSuite(TestSACAlgorithm))
        test_suite.addTest(unittest.makeSuite(TestA3CAlgorithm))
        test_suite.addTest(unittest.makeSuite(TestICMModule))

        print("🔄 Adding RL Integration Tests...")
        test_suite.addTest(unittest.makeSuite(TestRLIntegration))

        print("🏋️ Adding RL Training Tests...")
        test_suite.addTest(unittest.makeSuite(TestTrainingConfig))
        test_suite.addTest(unittest.makeSuite(TestRLTrainer))
        test_suite.addTest(unittest.makeSuite(TestMetricsTracker))
        test_suite.addTest(unittest.makeSuite(TestTrainingVisualizer))
        test_suite.addTest(unittest.makeSuite(TestAlgorithmConfigs))

        print("🏃 Adding RL Performance Benchmarks...")
        test_suite.addTest(unittest.makeSuite(TestRLPerformanceBenchmarks))

    except ImportError as e:
        print(f"\n⚠️  RL Tests Skipped: {e}")

    return test_suite


def run_test_category(category_name, test_classes, verbose=True):
    """Run a specific category of tests."""
    print(f"\n{'='*60}")
    print(f"🧪 Running {category_name}")
    print("="*60)

    suite = unittest.TestSuite()
    for test_class in test_classes:
        suite.addTest(unittest.makeSuite(test_class))

    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)

    print(f"\n📊 {category_name} Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100
    print(f"   Success rate: {success_rate:.1f}%")

    return result


def run_individual_test_categories():
    """Run tests by category for detailed analysis."""

    results = {}

    # Traditional ML tests
    ml_tests = [TestMLModels, TestMLModelsPerformance]
    results['ML Models'] = run_test_category("ML Models", ml_tests)

    # System tests
    system_tests = [TestAutoBot, TestRiskManager, TestConnectionManager]
    results['System Components'] = run_test_category("System Components", system_tests)

    # Integration tests
    integration_tests = [TestAutoBotIntegration]
    results['Integration'] = run_test_category("Integration", integration_tests)

    # Performance tests
    performance_tests = [
        TestMLModelPerformance,
        TestMachineLearningPerformance,
        TestOverallSystemPerformance
    ]
    results['Performance'] = run_test_category("Performance", performance_tests)

    # RL tests (if available)
    if RL_TESTS_AVAILABLE:
        rl_algorithm_tests = [
            TestPPOAlgorithm, TestSACAlgorithm,
            TestA3CAlgorithm, TestICMModule
        ]
        results['RL Algorithms'] = run_test_category("RL Algorithms", rl_algorithm_tests)

        rl_training_tests = [
            TestTrainingConfig, TestRLTrainer,
            TestMetricsTracker, TestTrainingVisualizer,
            TestAlgorithmConfigs
        ]
        results['RL Training'] = run_test_category("RL Training", rl_training_tests)

        rl_integration_tests = [TestRLIntegration]
        results['RL Integration'] = run_test_category("RL Integration", rl_integration_tests)

        rl_benchmark_tests = [TestRLPerformanceBenchmarks]
        results['RL Benchmarks'] = run_test_category("RL Benchmarks", rl_benchmark_tests)

    return results


def print_comprehensive_summary(results):
    """Print comprehensive test summary."""
    print("\n" + "="*80)
    print("📋 COMPREHENSIVE TEST SUMMARY")
    print("="*80)

    total_tests = 0
    total_failures = 0
    total_errors = 0

    for category, result in results.items():
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)

        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100
        status = "✅ PASS" if success_rate == 100 else "⚠️ ISSUES" if success_rate >= 80 else "❌ FAIL"

        print(f"{status} {category:20} | Tests: {result.testsRun:3d} | "
              f"Failures: {len(result.failures):2d} | Errors: {len(result.errors):2d} | "
              f"Success: {success_rate:5.1f}%")

    print("-" * 80)
    overall_success = (total_tests - total_failures - total_errors) / max(total_tests, 1) * 100
    overall_status = "✅ PASS" if overall_success == 100 else "⚠️ ISSUES" if overall_success >= 80 else "❌ FAIL"

    print(f"{overall_status} {'OVERALL':20} | Tests: {total_tests:3d} | "
          f"Failures: {total_failures:2d} | Errors: {total_errors:2d} | "
          f"Success: {overall_success:5.1f}%")

    print("="*80)

    # Additional analysis
    if total_failures > 0 or total_errors > 0:
        print(f"\n🔍 Issues found in {len([r for r in results.values() if len(r.failures) + len(r.errors) > 0])} categories")
        print("   Consider reviewing failed tests for system stability")

    if RL_TESTS_AVAILABLE:
        rl_categories = [k for k in results.keys() if 'RL' in k]
        if rl_categories:
            rl_total = sum(results[cat].testsRun for cat in rl_categories)
            rl_failures = sum(len(results[cat].failures) for cat in rl_categories)
            rl_errors = sum(len(results[cat].errors) for cat in rl_categories)
            rl_success = (rl_total - rl_failures - rl_errors) / max(rl_total, 1) * 100
            print(f"\n🤖 RL System Health: {rl_success:.1f}% ({rl_total} tests)")

    return overall_success >= 80  # Return True if overall health is good


def main():
    """Main test execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced ML Trading System Test Runner')
    parser.add_argument('--category', choices=['all', 'ml', 'system', 'integration', 'performance', 'rl'],
                       default='all', help='Test category to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--individual', '-i', action='store_true',
                       help='Run tests by individual categories')

    args = parser.parse_args()

    if args.category == 'all' and args.individual:
        # Run individual categories for detailed analysis
        results = run_individual_test_categories()
        success = print_comprehensive_summary(results)
        return 0 if success else 1

    elif args.category == 'all':
        # Run all tests together
        print("Running comprehensive test suite...")
        suite = create_comprehensive_test_suite()
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(suite)

        # Print summary
        print(f"\n{'='*60}")
        print("📊 FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")

        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100
        print(f"Success rate: {success_rate:.1f}%")

        if result.failures:
            print("\n❌ FAILURES:")
            for test, traceback in result.failures:
                print(f"  - {test}")

        if result.errors:
            print("\n💥 ERRORS:")
            for test, traceback in result.errors:
                print(f"  - {test}")

        overall_status = "PASSED" if success_rate >= 80 else "FAILED"
        print(f"\nOverall result: {overall_status}")

        return 0 if success_rate >= 80 else 1

    else:
        # Run specific category
        category_map = {
            'ml': [TestMLModels, TestMLModelsPerformance],
            'system': [TestAutoBot, TestRiskManager, TestConnectionManager],
            'integration': [TestAutoBotIntegration],
            'performance': [TestMLModelPerformance, TestMachineLearningPerformance, TestOverallSystemPerformance],
            'rl': ([TestPPOAlgorithm, TestSACAlgorithm, TestA3CAlgorithm, TestICMModule,
                   TestRLIntegration, TestRLTrainer] if RL_TESTS_AVAILABLE else [])
        }

        if args.category in category_map:
            test_classes = category_map[args.category]
            if not test_classes:
                print(f"No tests available for category: {args.category}")
                return 1

            result = run_test_category(args.category.upper(), test_classes, args.verbose)
            success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100
            return 0 if success_rate >= 80 else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
