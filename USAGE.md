# Trading Application - Setup and Usage Guide

## Project Structure

```
tradingApp/
├── config.ini                 # Application configuration
├── main.py                    # Main entry point
├── init_app.py                # Initialization checker
├── install_deps.py            # Dependency installer
├── presenters.py              # Application presenters
├── refactored_presenters.py   # Refactored application presenters
├── hyper.py                   # Hyperparameter optimization
├── predict_tester.py          # ML model tester
├── requirements.txt           # Python dependencies
├── requirements_simple.txt    # Simplified dependencies for clean installation
├── .env.example              # Environment variables template
├── README.md                 # Project overview
├── USAGE.md                  # This file
├── Makefile                  # Unix/Linux/macOS build script
├── run.ps1                   # Windows PowerShell script
├── run.bat                   # Windows batch script
├── docs/                     # Documentation
│   ├── architecture.md       # Architecture details
│   └── setup.md              # Setup instructions
├── data/                     # Data storage (git-ignored)
├── model/                    # Business logic
│   ├── refactored_models.py  # Refactored models
│   └── base_model.py         # Base model class
├── view/                     # User interface
│   └── refactored_views.py   # Refactored views
├── util/                     # Utility functions
│   └── standardized_loggers.py # Enhanced logging
└── test/                     # Test suite
    └── test_refactored_components.py # Tests for refactored components
```

## Setup Instructions

### Method 1: Using Helper Scripts (Recommended)

#### On Unix/Linux/macOS:
```bash
make setup
```

#### On Windows (PowerShell):
```powershell
.\\run.ps1 setup
```

#### On Windows (Command Prompt):
```cmd
run.bat setup
```

### Method 2: Manual Setup

1. **Install Python**: Ensure you have Python 3.7 or higher installed

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**:
   ```bash
   python install_deps.py
   ```
   Or manually:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env  # On Windows: copy .env.example .env
   ```
   Then edit `.env` and add your actual API keys and database credentials

5. **Initialize the application**:
   ```bash
   python init_app.py
   ```

## Configuration

- `config.ini`: Adjust application settings
- `.env`: Add your API keys and database credentials

## Documentation

- `README.md`: Project overview
- `docs/architecture.md`: Detailed architecture information
- `docs/setup.md`: Detailed setup instructions

## Testing

You can test the application with default credentials:
- Username: test
- Password: t

## Components

1. **Manual Trading**: Direct trade execution
2. **Machine Learning**: Automated trading with ML models
3. **Reinforcement Learning**: Advanced DQL-based trading
4. **Bot Management**: Create and manage trading bots
5. **Charting**: Visual market data analysis
6. **Exchange Integration**: Connect to multiple exchanges

## Running the Application

### Using Helper Scripts (Recommended)

#### On Unix/Linux/macOS:
```bash
make run
```

#### On Windows (PowerShell):
```powershell
.\\run.ps1 run
```

#### On Windows (Command Prompt):
```cmd
run.bat run
```

### Manual Method
```bash
python main.py
```

## Script Commands

### Makefile (Unix/Linux/macOS)
- `make setup` - Set up the virtual environment and install dependencies
- `make run` - Run the trading application
- `make install-deps` - Install all required dependencies
- `make clean` - Remove the virtual environment
- `make test` - Run tests (if available)
- `make lint` - Run code linting (if available)
- `make update-deps` - Update dependencies to latest versions
- `make dev-setup` - Install development tools and setup environment

### PowerShell Script (Windows)
- `.\\run.ps1 setup` - Set up the virtual environment and install dependencies
- `.\\run.ps1 run` - Run the trading application
- `.\\run.ps1 install-deps` - Install all required dependencies
- `.\\run.ps1 clean` - Remove the virtual environment
- `.\\run.ps1 test` - Run tests (if available)
- `.\\run.ps1 lint` - Run code linting (if available)
- `.\\run.ps1 update-deps` - Update dependencies to latest versions
- `.\\run.ps1 dev-setup` - Install development tools and setup environment

### Batch Script (Windows)
- `run.bat setup` - Set up the virtual environment and install dependencies
- `run.bat run` - Run the trading application
- `run.bat install-deps` - Install all required dependencies
- `run.bat clean` - Remove the virtual environment
- `run.bat test` - Run tests (if available)
- `run.bat lint` - Run code linting (if available)
- `run.bat update-deps` - Update dependencies to latest versions
- `run.bat dev-setup` - Install development tools and setup environment

## CUDA Support

This application uses CUDA-compatible PyTorch (torch-cu124) which provides GPU acceleration for machine learning tasks. This is especially beneficial for the reinforcement learning components and machine learning models that can take advantage of GPU processing power.

To use CUDA:
1. Ensure you have NVIDIA CUDA 12.4 installed on your system
2. Run the setup scripts as usual
3. The application will automatically use GPU acceleration when available

If you don't have CUDA installed or want to use CPU-only PyTorch, you can modify the requirements_simple.txt file to use the CPU version instead:
```
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2
```

## Troubleshooting

If you encounter issues:
1. Run `python init_app.py` to check your setup
2. Ensure all environment variables are set correctly
3. Check that your database is accessible
4. Verify API keys have necessary permissions