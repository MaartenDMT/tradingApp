"""
Test script to verify database connection pooling improvements.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_db_pooling():
    """Test the database connection pooling improvements."""
    print("Testing database connection pooling...")
    try:
        # Import the models module to test database pooling
        from model.models import initialize_db_pool, db_pool
        
        # Test that the pool initialization function exists
        assert callable(initialize_db_pool), "initialize_db_pool should be callable"
        
        # Test that db_pool is properly defined
        print(f"Database pool initialized: {db_pool is not None}")
        
        print("✅ Database connection pooling tests completed")
        return True
    except Exception as e:
        print(f"❌ Database connection pooling tests failed: {e}")
        return False

def test_login_model():
    """Test the LoginModel improvements."""
    print("Testing LoginModel improvements...")
    try:
        from model.models import LoginModel
        
        # Create an instance of LoginModel
        login_model = LoginModel()
        
        # Test that it was created successfully
        assert login_model is not None, "LoginModel should be created successfully"
        
        # Test credential setting
        login_model.set_credentials("test_user", "test_pass")
        assert login_model._username == "test_user", "Username should be set correctly"
        assert login_model._password == "test_pass", "Password should be set correctly"
        
        print("✅ LoginModel tests passed")
        return True
    except Exception as e:
        print(f"❌ LoginModel tests failed: {e}")
        return False

def main():
    """Run all database-related tests."""
    print("Running database optimization tests...\\n")
    
    tests = [
        test_db_pooling,
        test_login_model
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
    
    print(f"Database Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All database optimization tests passed!")
        return 0
    else:
        print("⚠️  Some database tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())