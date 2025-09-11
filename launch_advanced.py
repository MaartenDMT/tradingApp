"""
Enhanced Trading Application Launcher

This script provides a comprehensive launcher for the advanced trading application with:
- Dependency checking and installation
- Environment validation
- Production configuration management
- Advanced feature initialization
"""

import os
import sys
import subprocess
import asyncio
import warnings
from pathlib import Path

# Load configuration
try:
    from util.config_manager import get_config
    config = get_config()
except ImportError:
    # Fallback if config manager is not available yet
    config = None

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """Check and install required dependencies."""
    required_packages = [
        'torch',
        'torchvision', 
        'torchaudio',
        'ttkbootstrap',
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'asyncio',
        'websockets',
        'requests',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - missing")
    
    if missing_packages:
        print(f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")
        try:
            # Try our new install_deps.py script first
            import install_deps
            if install_deps.main():
                print("✅ Dependencies installed successfully with install_deps.py")
                return True
            else:
                print("❌ install_deps.py failed")
                return False
        except Exception as e:
            print(f"⚠️  install_deps.py not available or failed: {e}")
            try:
                # Try uv first (faster)
                subprocess.run([
                    "uv", "pip", "install"
                ] + missing_packages, check=True)
                print("✅ Dependencies installed successfully with uv")
            except (subprocess.CalledProcessError, FileNotFoundError):
                try:
                    # Fallback to pip
                    subprocess.run([
                        sys.executable, "-m", "pip", "install"
                    ] + missing_packages, check=True)
                    print("✅ Dependencies installed successfully with pip")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to install dependencies: {e}")
                    return False
    
    return True

def validate_advanced_features():
    """Validate that advanced features are available."""
    try:
        # Check production config
        from production_config import PRODUCTION_CONFIG, ProductionManager
        print("✅ Production configuration - available")
        
        # Check market data integration
        from market_data_integration import market_data_manager
        print("✅ Market data integration - available")
        
        # Check algorithm research
        from algorithm_research import algorithm_comparator, research_runner
        print("✅ Algorithm research tools - available")
        
        # Check tournament system
        from multi_agent_tournaments import tournament_engine
        print("✅ Multi-agent tournaments - available")
        
        # Check RL system
        from model.rl_system.factory import RLAlgorithmFactory
        print("✅ RL algorithm system - available")
        
        # Check advanced UI
        from view.advanced_rl_system_tab import AdvancedRLSystemTab
        print("✅ Advanced UI system - available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Advanced feature validation failed: {e}")
        return False

def initialize_environment():
    """Initialize environment variables and configuration."""
    try:
        # Set development mode
        os.environ['DEV_MODE'] = 'true'
        
        # Set default API keys if not present (for demo)
        if not os.getenv('BINANCE_API_KEY'):
            print("⚠️  BINANCE_API_KEY not set - using demo mode")
            os.environ['BINANCE_API_KEY'] = 'demo_key'
            os.environ['BINANCE_SECRET_KEY'] = 'demo_secret'
        
        if not os.getenv('ALPHA_VANTAGE_API_KEY'):
            print("⚠️  ALPHA_VANTAGE_API_KEY not set - using demo mode")
            os.environ['ALPHA_VANTAGE_API_KEY'] = 'demo_key'
        
        # Create required directories
        required_dirs = [
            'data/cache',
            'data/logs',
            'data/ML',
            'data/RL',
            'model_saves',
            'tournament_results',
            'research_results'
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print("✅ Environment initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Environment initialization failed: {e}")
        return False

def run_application():
    """Run the main trading application."""
    try:
        print("\n🚀 Starting Advanced Trading Application...")
        print("=" * 50)
        
        # Import and run main application
        from main import main
        main()
        
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()

def show_feature_summary():
    """Show summary of available features."""
    print("\n📋 ADVANCED FEATURES AVAILABLE:")
    print("=" * 50)
    print("🚀 Production Configuration")
    print("   • Environment validation")
    print("   • Deployment readiness checks")
    print("   • Configuration management")
    print()
    print("📈 Market Data Integration")
    print("   • Binance real-time crypto data")
    print("   • Alpha Vantage stocks/forex data")
    print("   • WebSocket real-time feeds")
    print("   • Historical data fetching")
    print()
    print("🔬 Algorithm Research")
    print("   • 11+ RL algorithms comparison")
    print("   • Hyperparameter optimization")
    print("   • Statistical analysis")
    print("   • Performance visualization")
    print()
    print("🏆 Multi-Agent Tournaments")
    print("   • Round-robin tournaments")
    print("   • Elimination brackets")
    print("   • Swiss system")
    print("   • Ladder competitions")
    print()
    print("🧠 Machine Learning System")
    print("   • Advanced ML algorithms")
    print("   • Model optimization")
    print("   • Performance tracking")
    print()
    print("📊 Monitoring & Analytics")
    print("   • Real-time system metrics")
    print("   • Performance analytics")
    print("   • Report generation")
    print("   • Visual dashboards")
    print("=" * 50)

def main():
    """Main launcher function."""
    print("🎯 ADVANCED TRADING APPLICATION LAUNCHER")
    print("=" * 50)
    
    # Step 1: Check dependencies
    print("\n1️⃣ Checking dependencies...")
    if not check_dependencies():
        print("❌ Dependency check failed. Please install missing packages manually.")
        return
    
    # Step 2: Initialize environment
    print("\n2️⃣ Initializing environment...")
    if not initialize_environment():
        print("❌ Environment initialization failed.")
        return
    
    # Step 3: Validate advanced features
    print("\n3️⃣ Validating advanced features...")
    if not validate_advanced_features():
        print("❌ Advanced features validation failed.")
        print("Some advanced features may not be available.")
    
    # Step 4: Show feature summary
    show_feature_summary()
    
    # Step 5: Run application
    input("\n🎮 Press Enter to launch the application...")
    run_application()

if __name__ == "__main__":
    main()
