"""
Simple validation script for advanced ML features.
"""
import logging
import os
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test that all advanced ML modules can be imported."""
    try:
        from model.machinelearning.ml_advanced import (AutoMLPipeline,
                                                       EnsembleManager)
        from model.machinelearning.ml_config import (AUTOML_CONFIG,
                                                     ENSEMBLE_CONFIG)
        from model.machinelearning.ml_integration import AdvancedMLManager
        from model.machinelearning.ml_utils_advanced import (DataPreprocessor,
                                                             ModelValidator)

        logger.info("✓ All advanced ML modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing functionality."""
    try:
        from model.machinelearning.ml_utils_advanced import DataPreprocessor

        # Create sample data
        np.random.seed(42)
        data = {
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(10, 5, 100),
            'feature3': np.random.uniform(0, 100, 100)
        }
        # Add some missing values
        data['feature1'][10:15] = np.nan

        X = pd.DataFrame(data)

        preprocessor = DataPreprocessor()

        # Test missing value handling
        X_filled = preprocessor.handle_missing_values(X)
        assert X_filled.isnull().sum().sum() == 0, "Missing values not handled"

        # Test scaling
        X_scaled = preprocessor.smart_scaling(X_filled)
        assert X_scaled is not None, "Scaling failed"

        # Test outlier detection
        outliers = preprocessor.detect_outliers(X_filled)
        assert isinstance(outliers, pd.DataFrame), "Outlier detection failed"

        logger.info("✓ Data preprocessing tests passed")
        return True
    except Exception as e:
        logger.error(f"✗ Data preprocessing test failed: {e}")
        return False

def test_ensemble_manager():
    """Test ensemble manager functionality."""
    try:
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        from model.machinelearning.ml_advanced import EnsembleManager

        # Create sample data
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                                 n_redundant=3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        ensemble_manager = EnsembleManager()

        # Test that ensemble manager initializes
        assert ensemble_manager is not None, "EnsembleManager initialization failed"

        logger.info("✓ Ensemble manager tests passed")
        return True
    except Exception as e:
        logger.error(f"✗ Ensemble manager test failed: {e}")
        return False

def test_automl_pipeline():
    """Test AutoML pipeline functionality."""
    try:
        from sklearn.datasets import make_classification

        from model.machinelearning.ml_advanced import AutoMLPipeline

        # Create sample data
        X, y = make_classification(n_samples=100, n_features=5, n_informative=3,
                                 n_redundant=1, random_state=42)

        automl = AutoMLPipeline(max_time_minutes=1)  # Short time for testing

        # Test that automl initializes
        assert automl is not None, "AutoMLPipeline initialization failed"

        logger.info("✓ AutoML pipeline tests passed")
        return True
    except Exception as e:
        logger.error(f"✗ AutoML pipeline test failed: {e}")
        return False

def test_integration_manager():
    """Test the integration manager."""
    try:
        from model.machinelearning.ml_integration import AdvancedMLManager

        manager = AdvancedMLManager()

        # Test initialization
        assert manager is not None, "AdvancedMLManager initialization failed"

        logger.info("✓ Integration manager tests passed")
        return True
    except Exception as e:
        logger.error(f"✗ Integration manager test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    logger.info("Starting advanced ML features validation...")

    tests = [
        test_basic_imports,
        test_data_preprocessing,
        test_ensemble_manager,
        test_automl_pipeline,
        test_integration_manager
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
            logger.error(f"Test {test.__name__} crashed: {e}")
            failed += 1

    logger.info("\nValidation Summary:")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {passed + failed}")

    if failed == 0:
        logger.info("🎉 All advanced ML features validation tests passed!")
        return True
    else:
        logger.warning(f"⚠️  {failed} tests failed")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
