"""
Main entry point for the Trading Application.

This script initializes and runs the trading application with proper error handling,
async optimization, and enhanced performance monitoring.
"""

import asyncio
import os
import sys
import traceback
import warnings
from pathlib import Path

import util.loggers as loggers

# Suppress pkg_resources deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*pkg_resources")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated as an API.*")

# Suppress TensorFlow deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*tensorflow")
warnings.filterwarnings("ignore", message=".*tf.losses.sparse_softmax_cross_entropy is deprecated.*")
warnings.filterwarnings("ignore", message=".*tf.logging.TaskLevelStatusMessage is deprecated.*")
warnings.filterwarnings("ignore", message=".*tf.control_flow_v2_enabled is deprecated.*")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# Suppress Gym deprecation warning
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained since 2022.*")

# Suppress TensorFlow warnings more broadly
warnings.filterwarnings("ignore", module=".*tf_keras.*")
warnings.filterwarnings("ignore", module=".*tensorflow_probability.*")

# Suppress pandas_ta SyntaxWarning
warnings.filterwarnings("ignore", category=SyntaxWarning, module=".*pandas_ta.*")

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import application components
try:
    from model.models import OptimizedModels as Models
    from presenters import OptimizedPresenter as Presenter
    from view.views import OptimizedWindowView as WindowView
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure all dependencies are installed.")
    print("Run 'python install_deps.py' or '.\\run.ps1 setup' to install dependencies.")
    sys.exit(1)

# Set up logging
logger = loggers.setup_loggers()
app_logger = logger['app']


def setup_exception_handling():
    """Set up global exception handling to log uncaught exceptions"""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupts
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        app_logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception


async def initialize_async_components():
    """Initialize async components for enhanced performance."""
    try:
        # Import async utilities here to avoid circular imports
        from util.cache import get_global_cache

        app_logger.info("Initializing async components")

        # Initialize global cache
        cache = await get_global_cache()
        app_logger.info("Global cache initialized")

        return {"cache": cache}

    except Exception as e:
        app_logger.error(f"Failed to initialize async components: {e}")
        return {}


async def shutdown_async_components():
    """Shutdown async components gracefully."""
    try:
        from util.cache import shutdown_global_cache

        app_logger.info("Shutting down async components")
        await shutdown_global_cache()
        app_logger.info("Async components shut down successfully")

    except Exception as e:
        app_logger.error(f"Error shutting down async components: {e}")


def main():
    """Initialize and run the trading application."""
    try:
        app_logger.info("Starting Trading Application with async optimizations")

        # Set up global exception handling
        setup_exception_handling()

        # Initialize async components
        async_components = {}

        # Run async initialization
        try:
            async_components = asyncio.run(initialize_async_components())
        except Exception as e:
            app_logger.warning(f"Failed to initialize async components: {e}")
            app_logger.info("Continuing with standard initialization")

        # Initialize application components
        app_logger.info("Initializing models")
        model = Models()

        app_logger.info("Initializing view")
        view = WindowView()

        app_logger.info("Initializing presenter")
        app = Presenter(model, view)

        # For development, automatically trigger login with test credentials
        # This will bypass the login screen
        if os.getenv('DEV_MODE', 'false').lower() == 'true':
            app._model.login_model.set_credentials('test', 't')
            # Directly show the main view
            app.get_main_view()

        app_logger.info("Running application")

        try:
            app.run()
        finally:
            # Ensure async components are properly shut down
            if async_components:
                try:
                    asyncio.run(shutdown_async_components())
                except Exception as e:
                    app_logger.error(f"Error during async shutdown: {e}")

    except KeyboardInterrupt:
        app_logger.info("Application interrupted by user")
        print("\nApplication interrupted by user. Exiting...")
        # Graceful shutdown
        try:
            asyncio.run(shutdown_async_components())
        except Exception:
            pass
        sys.exit(0)
    except Exception as e:
        app_logger.error(f"Error starting application: {e}")
        app_logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"Error starting application: {e}")
        print("Check the logs for more details.")
        # Attempt cleanup
        try:
            asyncio.run(shutdown_async_components())
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
