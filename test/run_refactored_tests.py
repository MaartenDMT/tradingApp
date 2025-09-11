"""
Test Runner for Refactored Components

This script provides a comprehensive test runner for the refactored components
of the trading application.
"""

import sys
import time
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_refactored_tests():
    """Run all tests for refactored components."""
    print("🚀 Running Tests for Refactored Components")
    print("=" * 50)
    
    # Import test modules
    try:
        from test.test_refactored_components import create_test_suite
        print("✅ Test modules loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to load test modules: {e}")
        return False
    
    # Create and run test suite
    print("\n🧪 Running refactored component tests...")
    suite = create_test_suite()
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Calculate duration
    duration = end_time - start_time
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"⏱️  Duration: {duration:.2f} seconds")
    print(f"✅ Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"🔥 Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}")
            print(f"     {traceback}")
    
    if result.errors:
        print("\n🔥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}")
            print(f"     {traceback}")
    
    # Success rate
    total_tests = result.testsRun
    passed_tests = total_tests - len(result.failures) - len(result.errors)
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n📈 Success rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
    
    # Final result
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("✨ Refactored components are working correctly!")
        return True
    else:
        print(f"\n💥 {len(result.failures) + len(result.errors)} tests failed or had errors")
        print("🔧 Please fix the issues before proceeding")
        return False


def run_quick_refactored_tests():
    """Run a quick subset of refactored component tests."""
    print("⚡ Running Quick Refactored Component Tests")
    print("=" * 45)
    
    # Import and run specific test classes
    try:
        from test.test_refactored_components import (
            TestRefactoredModels, 
            TestRefactoredPresenters,
            TestStandardizedLoggers
        )
        
        # Create suite with selected tests
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestRefactoredModels))
        suite.addTest(unittest.makeSuite(TestRefactoredPresenters))
        suite.addTest(unittest.makeSuite(TestStandardizedLoggers))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"❌ Failed to load quick test modules: {e}")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Refactored Component Test Runner')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_refactored_tests()
    else:
        success = run_refactored_tests()
    
    sys.exit(0 if success else 1)