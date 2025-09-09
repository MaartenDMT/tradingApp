# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a comprehensive Python-based trading application that provides multiple trading strategies including manual trading, machine learning-based trading, and reinforcement learning-based trading. The application features a GUI built with Tkinter/ttkbootstrap and follows the Model-View-Presenter (MVP) architectural pattern.

## Common Development Commands

### Environment Setup
```bash
# Setup using UV (recommended)
uv venv
uv sync

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Unix-like:
source .venv/bin/activate

# Manual setup for PyTorch CUDA (after UV sync)
uv pip install --no-build torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Build and Run
```bash
# Run application using UV
uv run python main.py

# Run using Makefile
make run

# Direct execution
python main.py
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test files
uv run pytest test/test_ml_models.py
uv run pytest test/test_reinforcement/
uv run pytest test_modern_algorithms_clean.py

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov
```

### Development Tools
```bash
# Install development dependencies
uv sync --group dev

# Code formatting
uv run black .

# Linting
uv run flake8
uv run pylint model/ view/ util/

# Auto-formatting
uv run autopep8 --in-place --recursive .
```

### Database Operations
```bash
# Initialize application (checks setup)
python init_app.py

# Test database optimizations
python test_db_optimizations.py

# Hyperparameter optimization
python hyper.py
```

### ML/RL Specific Commands
```bash
# Test ML models
python predict_tester.py

# Test modern algorithms
python test_modern_algorithms_clean.py

# Run RL training tests
python test/test_rl_training.py

# Performance testing
python test/test_performance.py
```

## Architecture Overview

### High-Level Structure
The application follows **Model-View-Presenter (MVP)** pattern with lazy initialization for performance:

**Entry Point**: `main.py` → `Presenter` → `Models` + `WindowView`

### Core Components

1. **Models Layer** (`model/`)
   - **models.py**: Central coordinator with lazy-loaded tab models
   - **manualtrading/**: Direct trading functionality via `Trading` class
   - **machinelearning/**: ML strategies including `AutoBot`, `MachineLearning`, and advanced ML pipelines
   - **reinforcement/**: RL trading agents using stable-baselines3 and custom DQN implementations
   - **features.py**: Technical analysis indicators via `Tradex_indicator`

2. **Views Layer** (`view/`)
   - **views.py**: Main window with `WindowView`, `LoginView`, `MainView`
   - **tradetab.py**: Manual trading interface
   - **mltab.py**: Machine learning controls and training interface
   - **rltab.py**: Reinforcement learning training and testing
   - **bottab.py**: Bot management (start/stop/configure bots)
   - **charttab.py**: Real-time charting with matplotlib integration
   - **exchangetab.py**: Exchange connection management

3. **Presenters Layer** (`presenters.py`)
   - Central `Presenter` class with lazy-loaded tab presenters
   - Tab-specific presenters: `TradePresenter`, `MLPresenter`, `RLPresenter`, etc.
   - Handles all UI logic and model coordination

4. **Utilities Layer** (`util/`)
   - **async_client.py**, **async_trading.py**: Asynchronous exchange operations
   - **cache.py**: Global caching system with `HybridCache`
   - **db_pool.py**: PostgreSQL connection pooling
   - **ml_optimization.py**: ML pipeline optimizations with chunking
   - **secure_credentials.py**: API key masking and encryption
   - **error_handling.py**: Centralized exception handling
   - **loggers.py**: Multi-component logging system

### Key Architectural Patterns

**Lazy Initialization**: All models and presenters are initialized only when accessed via properties, improving startup time.

**Connection Pooling**: Database connections use `psycopg2.pool.ThreadedConnectionPool` for performance.

**Async Operations**: Exchange API calls and data fetching use async/await patterns with `aiohttp` and custom async utilities.

**ML Pipeline Architecture**:
- **OptimizedMLPipeline** with configurable backends (loky, threading)
- **HybridCache** for ML predictions and data caching
- Parallel model training using `ThreadPoolExecutor`
- SHAP integration for model interpretability

**RL Architecture**:
- **StablebaselineModel** wrapper for PPO, DQN, A2C algorithms  
- Custom trading environments with portfolio simulation
- Optuna-based hyperparameter optimization
- Multi-environment training support

**Exchange Integration**:
- CCXT library for multiple exchange support
- Sandbox mode for testing (configurable via `test_mode`)
- WebSocket connections for real-time data
- Rate limiting and retry logic built-in

## Key Configuration

### Environment Variables (.env)
- `API_KEY_PHE_TEST`, `API_SECRET_PHE_TEST`: Phemex testnet credentials
- `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`: PostgreSQL settings
- `DEV_MODE=true`: Enables auto-login with test credentials

### Configuration Files
- **config.ini**: Application settings (window size, WebSocket params, model hyperparameters)
- **pyproject.toml**: Dependencies, build config, tool settings (Black, pytest)

### Test Credentials
- Default login: username=`test`, password=`t`

## Important Development Notes

### CUDA Support
- Project uses PyTorch with CUDA 12.4 support
- CPU versions installed via UV, CUDA versions require manual installation
- GPU acceleration available for ML/RL training

### Database Requirements  
- PostgreSQL required for user management and data storage
- Connection pooling configured for concurrent operations
- Database schema initialized via `init_app.py`

### Performance Optimizations
- Memory monitoring via `PerformanceMonitor` class
- Garbage collection optimization in ML pipelines
- Async component initialization and shutdown
- Chunked data processing for large datasets

### Logging System
Multiple loggers: `app`, `model`, `view`, `presenter`, `autobot`, `tradex`, `rl_trading`, `rl`, `env`, `agent`, `manual`

### ML/RL Model Management
- Models saved to `data/ml/models/` and `data/rl/models/`
- Performance metrics in `data/ml/csv/model_performance.csv`
- SHAP plots generated in `data/ml/plots/`

## New Optimized Trading System

### Overview
A new `trading/` module has been implemented that provides a modern, optimized alternative to the existing `manualtrading/` system. This system incorporates best practices from both `ml_system/` and `rl_system/` architectures.

### Key Improvements
- **Modular Architecture**: Clear separation of concerns with dedicated modules for execution, risk management, portfolio management, and strategies
- **Async/Await Support**: Full async implementation for high-performance concurrent trading operations
- **Type Safety**: Comprehensive type definitions with enums and dataclasses for all trading entities
- **Professional Configuration**: Environment-aware configuration system with validation
- **Risk Management**: Built-in risk assessment, position sizing, and portfolio risk controls
- **Performance Tracking**: Comprehensive metrics tracking and performance analysis

### Architecture
```
trading/
├── core/                      # Core system components
│   ├── trading_system.py      # Main system orchestrator
│   ├── config.py             # Configuration management
│   └── types.py              # Type definitions and enums
├── execution/                # Order execution and management
├── risk_management/          # Risk assessment and controls
├── strategies/               # Trading strategy framework
├── market_data/              # Data providers and real-time feeds
├── portfolio/                # Portfolio and performance tracking
└── utils/                    # Trading utilities and helpers
```

### Key Features
- **TradingSystem**: Main orchestrator with lifecycle management
- **TradingConfig**: Comprehensive configuration with environment detection
- **Type Definitions**: Strong typing with OrderType, OrderSide, Position, MarketData classes
- **Async Operations**: Non-blocking execution with concurrent order management
- **Risk Controls**: Position sizing, drawdown monitoring, VaR calculations
- **Performance Metrics**: Real-time P&L, Sharpe ratio, win rate tracking

### Usage Example
```python
from trading import TradingSystem, TradingConfig, TradingSignal, OrderSide

# Create configuration
config = TradingConfig()
config.paper_trading = True
config.risk.max_risk_per_trade = 0.02  # 2% per trade

# Initialize system
async with TradingSystem(config) as system:
    # Execute a trade
    signal = TradingSignal(
        symbol="BTC/USD:USD",
        signal_type=OrderSide.BUY,
        strength=0.8,
        confidence=0.9
    )
    
    result = await system.execute_trade(signal)
    print(f"Trade result: {result.success}")
```

## Dependencies Management

This project uses **UV** for fast Python package management. Key dependency groups:
- **Main**: Trading libraries (ccxt, pandas, numpy, scikit-learn, tensorflow, pytorch)
- **Dev**: Development tools (pytest, black, flake8, pylint)
- **ML**: Extended ML stack (xgboost, shap, optuna, stable-baselines3)

After `uv sync`, manually install CUDA PyTorch versions:
```bash
uv pip install --no-build torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
