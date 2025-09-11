"""
Environment Setup Script

Sets up the environment for the advanced trading application
"""

import os
from pathlib import Path
from util.config_manager import AppConfig

def create_env_file():
    """Create .env file with default configuration."""
    
    env_content = """# Advanced Trading Application Configuration

# Development Mode
DEV_MODE=true

# Database Configuration (disabled for development)
DATABASE_URL=postgresql://localhost:5432/trading_app
DATABASE_ENABLED=false

# API Keys (replace with your actual keys)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379
REDIS_ENABLED=false

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=data/logs/app.log

# Performance Configuration
CACHE_SIZE=1000
MAX_WORKERS=4
BATCH_SIZE=32

# Security Configuration
SECRET_KEY=your_secret_key_here
ENCRYPTION_KEY=your_encryption_key_here

# Production Configuration
PRODUCTION_MODE=false
MONITORING_ENABLED=true
"""
    
    env_path = Path('.env')
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("✅ Created .env file with default configuration")
    else:
        print("ℹ️  .env file already exists")

def setup_directories():
    """Create required directory structure."""
    
    # Get data paths from configuration
    try:
        config = AppConfig.from_files()
        directories = list(config.data_paths.keys())
    except:
        # Fallback if config is not available
        directories = [
            'data/cache',
            'data/logs', 
            'data/csv',
            'data/ML',
            'data/RL',
            'data/pickle',
            'data/scaler',
            'model_saves',
            'tournament_results',
            'research_results',
            'experiments',
            'backups'
        ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_gitignore():
    """Create .gitignore file for the project."""
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Environment Variables
.env
.env.local
.env.development
.env.test
.env.production

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data Files
data/cache/
data/logs/
data/pickle/
*.pkl
*.csv
*.json

# Model Files
model_saves/
*.pth
*.pt
*.h5

# Results
tournament_results/
research_results/
experiments/
backups/

# System
.DS_Store
Thumbs.db

# Logs
*.log
logs/
"""
    
    gitignore_path = Path('.gitignore')
    if not gitignore_path.exists():
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print("✅ Created .gitignore file")
    else:
        print("ℹ️  .gitignore file already exists")

def main():
    """Main setup function."""
    print("🔧 SETTING UP ADVANCED TRADING APPLICATION")
    print("=" * 50)
    
    print("\n1️⃣ Creating environment configuration...")
    create_env_file()
    
    print("\n2️⃣ Setting up directory structure...")
    setup_directories()
    
    print("\n3️⃣ Creating .gitignore...")
    create_gitignore()
    
    print("\n✅ SETUP COMPLETE!")
    print("\n📋 Next steps:")
    print("   1. Edit .env file with your API keys")
    print("   2. Run 'python launch_advanced.py' to start the application")
    print("   3. Check ADVANCED_FEATURES_GUIDE.md for detailed documentation")

if __name__ == "__main__":
    main()
