"""
Test validation script for the enhanced testing framework.

This script validates that our testing framework works correctly
and provides a summary of testing capabilities.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def validate_test_structure():
    """Validate the test structure and report capabilities."""

    print("🔍 TESTING FRAMEWORK VALIDATION")
    print("=" * 50)

    # Check for test files
    test_files = {
        'Traditional ML Tests': [
            'test_autobot.py',
            'test_ml_models.py',
            'test_performance.py'
        ],
        'RL Algorithm Tests': [
            'test_rl_algorithms.py',
            'test_rl_training.py'
        ],
        'Test Infrastructure': [
            'run_tests.py',
            'run_enhanced_tests.py'
        ]
    }

    total_files = 0
    found_files = 0

    for category, files in test_files.items():
        print(f"\n📋 {category}:")
        for file_name in files:
            file_path = os.path.join(os.path.dirname(__file__), file_name)
            total_files += 1
            if os.path.exists(file_path):
                print(f"  ✅ {file_name}")
                found_files += 1
            else:
                print(f"  ❌ {file_name} (missing)")

    print(f"\n📊 Test File Coverage: {found_files}/{total_files} ({found_files/total_files*100:.1f}%)")

    # Check for RL system availability
    print("\n🤖 RL System Availability:")
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        import model.rl_system  # noqa: F401
        print("  ✅ RL system modules available")

        # Check specific components
        rl_components = [
            'model.rl_system.algorithms.policy_based.ppo',
            'model.rl_system.algorithms.actor_critic.sac',
            'model.rl_system.algorithms.policy_based.a3c',
            'model.rl_system.algorithms.exploration.icm',
            'model.rl_system.integration'
        ]

        available_components = 0
        for component in rl_components:
            try:
                __import__(component)
                available_components += 1
                print(f"    ✅ {component.split('.')[-1].upper()}")
            except ImportError:
                print(f"    ❌ {component.split('.')[-1].upper()}")

        print(f"  📊 RL Components: {available_components}/{len(rl_components)} available")

    except ImportError as e:
        print(f"  ❌ RL system not available: {e}")

    # Test runner validation
    print("\n🏃 Test Runner Validation:")
    try:
        from test.run_tests import create_test_suite, RL_TESTS_AVAILABLE
        suite = create_test_suite()
        test_count = suite.countTestCases()
        print("  ✅ Main test runner functional")
        print(f"  📊 Total test cases: {test_count}")
        print(f"  🤖 RL tests available: {RL_TESTS_AVAILABLE}")
    except Exception as e:
        print(f"  ❌ Test runner error: {e}")

    # Examples validation
    print("\n📚 Examples Framework:")
    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    if os.path.exists(examples_dir):
        example_categories = ['basic_usage', 'advanced_features', 'trading_scenarios']
        for category in example_categories:
            category_path = os.path.join(examples_dir, category)
            if os.path.exists(category_path):
                files = [f for f in os.listdir(category_path) if f.endswith('.py')]
                print(f"  ✅ {category}: {len(files)} examples")
            else:
                print(f"  ❌ {category}: missing")
    else:
        print("  ❌ Examples directory not found")

    return found_files == total_files


def run_sample_tests():
    """Run a small sample of tests to validate functionality."""

    print("\n\n🧪 SAMPLE TEST EXECUTION")
    print("=" * 50)

    try:
        from test.run_tests import run_quick_tests

        print("Running quick test suite...")
        result = run_quick_tests()

        if result.wasSuccessful():
            print("✅ Sample tests passed!")
            return True
        else:
            print(f"❌ Sample tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
            return False

    except Exception as e:
        print(f"❌ Could not run sample tests: {e}")
        return False


def print_testing_summary():
    """Print comprehensive testing framework summary."""

    print("\n\n📋 TESTING FRAMEWORK SUMMARY")
    print("=" * 60)

    print("🎯 Testing Capabilities:")
    print("  • Traditional ML model testing")
    print("  • System component testing (AutoBot, Risk Manager)")
    print("  • Performance benchmarking")
    print("  • RL algorithm testing (PPO, SAC, A3C, ICM)")
    print("  • RL training infrastructure testing")
    print("  • Integration testing")

    print("\n🔧 Test Categories:")
    print("  1. Unit Tests - Individual component testing")
    print("  2. Integration Tests - Cross-component interactions")
    print("  3. Performance Tests - Benchmarking and optimization")
    print("  4. Algorithm Tests - RL algorithm validation")
    print("  5. Training Tests - Training infrastructure validation")

    print("\n🚀 Usage Examples:")
    print("  • python test/run_tests.py - Run all available tests")
    print("  • python test/test_rl_algorithms.py - Run RL algorithm tests only")
    print("  • python test/test_rl_training.py - Run training system tests only")
    print("  • python examples/ - Explore usage examples")

    print("\n✨ Key Features:")
    print("  • Automatic RL test detection and inclusion")
    print("  • Comprehensive performance benchmarking")
    print("  • Detailed error reporting and diagnostics")
    print("  • Integration with existing test infrastructure")
    print("  • Support for both traditional ML and RL components")


def main():
    """Main validation function."""

    print("🔬 ENHANCED TESTING FRAMEWORK VALIDATION")
    print("=" * 70)

    # Validate structure
    structure_valid = validate_test_structure()

    # Run sample tests
    tests_pass = run_sample_tests()

    # Print summary
    print_testing_summary()

    # Final assessment
    print("\n" + "=" * 70)
    print("🏁 VALIDATION RESULTS")
    print("=" * 70)

    if structure_valid and tests_pass:
        print("🎉 SUCCESS: Enhanced testing framework is fully operational!")
        print("✅ All components validated and functional")
        print("🚀 Ready for comprehensive system testing")
        return 0
    else:
        issues = []
        if not structure_valid:
            issues.append("test structure incomplete")
        if not tests_pass:
            issues.append("sample tests failed")

        print(f"⚠️ ISSUES DETECTED: {', '.join(issues)}")
        print("🔧 Please review and fix the identified issues")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
