#!/usr/bin/env python3
"""
Test script to validate ML and RL system implementations.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_ml_system():
    """Test ML system functionality."""
    print("Testing ML System...")
    
    try:
        from model.ml_system.core.ml_system import MLSystem
        from model.ml_system.config.ml_config import MLConfig
        
        print("✓ ML System imports successful")
        
        # Test configuration
        config = MLConfig(algorithm='random_forest', target_type='regression')
        config_dict = config.to_dict()
        config_from_dict = MLConfig.from_dict(config_dict)
        
        print("✓ ML Config serialization/deserialization working")
        
        # Test ML system creation
        ml_system = MLSystem(config)
        print("✓ ML System initialization successful")
        
        return True
        
    except Exception as e:
        print(f"✗ ML System test failed: {e}")
        return False

def test_rl_system():
    """Test RL system functionality."""
    print("\nTesting RL System...")
    
    try:
        from model.rl_system.integration.rl_system import RLSystemManager
        
        print("✓ RL System imports successful")
        
        # Test RL system manager creation
        rl_manager = RLSystemManager()
        print("✓ RL System Manager initialization successful")
        
        # Test getting available algorithms
        algorithms = list(rl_manager.agent_registry.keys())
        print(f"✓ Available RL algorithms: {algorithms}")
        
        return True
        
    except Exception as e:
        print(f"✗ RL System test failed: {e}")
        return False

def test_optimized_models():
    """Test optimized models integration."""
    print("\nTesting Optimized Models...")
    
    try:
        from model.models_optimized import OptimizedModels
        
        print("✓ Optimized Models import successful")
        
        # Create optimized models instance (no presenter required)
        models = OptimizedModels()
        
        print("✓ Optimized Models initialization successful")
        
        # Test system availability
        status = models.get_health_status()
        print(f"✓ System health status: {status}")
        
        return True
        
    except Exception as e:
        print(f"✗ Optimized Models test failed: {e}")
        return False

def test_optimized_views():
    """Test optimized views."""
    print("\nTesting Optimized Views...")
    
    try:
        from view.ml_system_tab import MLSystemTab
        from view.rl_system_tab import RLSystemTab
        
        print("✓ ML and RL System tabs import successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Optimized Views test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running Trading App ML/RL System Integration Tests\n")
    
    tests = [
        test_ml_system,
        test_rl_system,
        test_optimized_models,
        test_optimized_views
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
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
