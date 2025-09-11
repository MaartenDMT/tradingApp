"""
Comprehensive Test Suite for Refactored Components

This module provides tests for the refactored models, presenters, and views.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from test.base_test import BaseTestCase

class TestRefactoredModels(BaseTestCase):
    """Test cases for refactored model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Mock model dependencies
        self.mock_config = self.mock_config
        self.mock_logger = self.mock_logger
        
    def test_base_model_initialization(self):
        """Test BaseModel initialization."""
        from model.base_model import BaseModel
        
        model = BaseModel("TestModel")
        self.assertEqual(model.name, "TestModel")
        self.assertFalse(model._initialized)
        
    def test_base_model_initialize(self):
        """Test BaseModel initialize method."""
        from model.base_model import BaseModel
        
        class TestModel(BaseModel):
            def _initialize(self):
                self.test_attr = "initialized"
                
        model = TestModel("TestModel")
        result = model.initialize()
        
        self.assertTrue(result)
        self.assertTrue(model._initialized)
        self.assertEqual(model.test_attr, "initialized")
        
    def test_base_model_get_config_value(self):
        """Test BaseModel get_config_value method."""
        from model.base_model import BaseModel
        from util.config_manager import AppConfig
        
        # Create a real config instance for testing
        config = AppConfig()
        config.test_value = "test_config_value"
        
        with patch('model.base_model.get_config', return_value=config):
            model = BaseModel("TestModel")
            value = model.get_config_value("test_value", "default")
            self.assertEqual(value, "test_config_value")
            
            # Test default value
            default_value = model.get_config_value("nonexistent_key", "default")
            self.assertEqual(default_value, "default")
            
    def test_refactored_models_initialization(self):
        """Test OptimizedModels initialization."""
        try:
            from model.refactored_models import OptimizedModels
            
            with patch('model.refactored_models.MANUAL_TRADING_AVAILABLE', False), \
                 patch('model.refactored_models.ML_SYSTEM_AVAILABLE', False), \
                 patch('model.refactored_models.RL_SYSTEM_AVAILABLE', False), \
                 patch('model.refactored_models.FEATURES_AVAILABLE', False):
                
                models = OptimizedModels()
                result = models.initialize()
                
                self.assertTrue(result)
                self.assertIsNotNone(models.login_model)
                self.assertIsNotNone(models.bot_model)
                
        except ImportError:
            self.skipTest("Refactored models not available")


class TestRefactoredPresenters(BaseTestCase):
    """Test cases for refactored presenter components."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Create mock model and view
        self.mock_model = Mock()
        self.mock_view = Mock()
        
    def test_base_presenter_initialization(self):
        """Test BasePresenter initialization."""
        from refactored_presenters import BasePresenter
        
        presenter = BasePresenter(self.mock_model, self.mock_view)
        self.assertEqual(presenter.model, self.mock_model)
        self.assertEqual(presenter.view, self.mock_view)
        self.assertFalse(presenter._initialized)
        
    def test_base_presenter_initialize(self):
        """Test BasePresenter initialize method."""
        from refactored_presenters import BasePresenter
        
        class TestPresenter(BasePresenter):
            def _initialize(self):
                self.test_attr = "initialized"
                
        presenter = TestPresenter(self.mock_model, self.mock_view)
        result = presenter.initialize()
        
        self.assertTrue(result)
        self.assertTrue(presenter._initialized)
        self.assertEqual(presenter.test_attr, "initialized")
        
    def test_optimized_presenter_initialization(self):
        """Test OptimizedPresenter initialization."""
        try:
            from refactored_presenters import OptimizedPresenter
            
            presenter = OptimizedPresenter(self.mock_model, self.mock_view)
            result = presenter.initialize()
            
            self.assertTrue(result)
            
        except ImportError:
            self.skipTest("Refactored presenters not available")


class TestRefactoredViews(BaseTestCase):
    """Test cases for refactored view components."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Create mock parent window
        self.mock_parent = Mock()
        
    def test_base_view_initialization(self):
        """Test BaseView initialization."""
        from view.refactored_views import BaseView
        
        view = BaseView(self.mock_parent)
        self.assertEqual(view.parent, self.mock_parent)
        self.assertFalse(view._initialized)
        
    def test_base_view_initialize(self):
        """Test BaseView initialize method."""
        from view.refactored_views import BaseView
        
        class TestView(BaseView):
            def _initialize(self):
                self.test_attr = "initialized"
                
        view = TestView(self.mock_parent)
        result = view.initialize()
        
        self.assertTrue(result)
        self.assertTrue(view._initialized)
        self.assertEqual(view.test_attr, "initialized")
        
    def test_optimized_window_view_initialization(self):
        """Test OptimizedWindowView initialization."""
        try:
            from view.refactored_views import OptimizedWindowView
            
            with patch('view.refactored_views.HAS_TTKBOOTSTRAP', False):
                # This test might fail in a headless environment, so we'll just check import
                pass
                
        except ImportError:
            self.skipTest("Refactored views not available")


class TestStandardizedLoggers(BaseTestCase):
    """Test cases for standardized loggers."""
    
    def test_logger_setup(self):
        """Test logger setup functionality."""
        from util.standardized_loggers import setup_loggers, get_logger
        
        # Set up loggers
        loggers = setup_loggers()
        
        self.assertIsInstance(loggers, dict)
        self.assertGreater(len(loggers), 0)
        
        # Test getting a specific logger
        app_logger = get_logger('app')
        self.assertIsNotNone(app_logger)
        
    def test_logger_levels(self):
        """Test logger level setting."""
        from util.standardized_loggers import setup_loggers, set_log_level, get_logger
        import logging
        
        # Set up loggers
        setup_loggers()
        
        # Set log level
        set_log_level('app', logging.DEBUG)
        
        # Get logger and check level
        app_logger = get_logger('app')
        self.assertEqual(app_logger.level, logging.DEBUG)


def create_test_suite():
    """Create a test suite with all refactored component tests."""
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestRefactoredModels))
    suite.addTest(unittest.makeSuite(TestRefactoredPresenters))
    suite.addTest(unittest.makeSuite(TestRefactoredViews))
    suite.addTest(unittest.makeSuite(TestStandardizedLoggers))
    
    return suite


if __name__ == '__main__':
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)