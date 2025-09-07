"""
Test script to verify the optimizations made to the trading application.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_caching():
    """Test the caching utility."""
    print("Testing caching utility...")
    try:
        from util.cache import get_cache, Cache
        
        # Test basic cache operations
        cache = get_cache()
        cache.set("test_key", "test_value", ttl=10)
        value = cache.get("test_key")
        assert value == "test_value", f"Expected 'test_value', got {value}"
        
        # Test cache size
        size = cache.size()
        assert size >= 1, f"Expected cache size >= 1, got {size}"
        
        print("✅ Caching utility tests passed")
        return True
    except Exception as e:
        print(f"❌ Caching utility tests failed: {e}")
        return False

def test_validation():
    """Test the validation utility."""
    print("Testing validation utility...")
    try:
        from util.validation import validate_symbol, validate_percentage, validate_leverage
        
        # Test symbol validation
        assert validate_symbol("BTC/USD:USD") == True, "Valid symbol should pass"
        assert validate_symbol("ETH/USDT") == True, "Valid symbol should pass"
        assert validate_symbol("INVALID_SYMBOL") == False, "Invalid symbol should fail"
        
        # Test percentage validation
        assert validate_percentage(50) == True, "Valid percentage should pass"
        assert validate_percentage(100) == True, "Valid percentage should pass"
        assert validate_percentage(-1) == False, "Invalid percentage should fail"
        assert validate_percentage(101) == False, "Invalid percentage should fail"
        
        # Test leverage validation
        assert validate_leverage(10) == True, "Valid leverage should pass"
        assert validate_leverage(100) == True, "Valid leverage should pass"
        assert validate_leverage(0) == False, "Invalid leverage should fail"
        assert validate_leverage(101) == False, "Invalid leverage should fail"
        
        print("✅ Validation utility tests passed")
        return True
    except Exception as e:
        print(f"❌ Validation utility tests failed: {e}")
        return False

def test_error_handling():
    """Test the error handling utility."""
    print("Testing error handling utility...")
    try:
        import logging
        from util.error_handling import handle_exception, safe_execute
        
        # Set up a logger for testing
        logger = logging.getLogger("test")
        logger.setLevel(logging.ERROR)
        
        # Test safe_execute
        def test_func():
            return "success"
        
        result = safe_execute(logger, "test operation", test_func)
        assert result == "success", f"Expected 'success', got {result}"
        
        print("✅ Error handling utility tests passed")
        return True
    except Exception as e:
        print(f"❌ Error handling utility tests failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running optimization tests...\\n")
    
    tests = [
        test_caching,
        test_validation,
        test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All optimization tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())