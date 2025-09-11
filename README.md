# Trading Application

A comprehensive trading application built with Python that provides multiple trading strategies including manual trading, machine learning-based trading, and reinforcement learning-based trading.

## Recent Enhancements

This application has undergone significant enhancements to improve its functionality, maintainability, security, and performance. See [ENHANCEMENT_SUMMARY.md](ENHANCEMENT_SUMMARY.md) for a complete overview of all improvements made.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Setup with UV](#quick-setup-with-uv)
- [Traditional Setup](#traditional-setup)
- [Configuration](#configuration)
- [Security](#security)
- [Usage](#usage)
- [Scripts](#scripts)
- [Documentation](#documentation)
- [Components](#components)
- [Dependencies](#dependencies)

## Overview

This is a comprehensive trading application built with Python that provides multiple trading strategies including manual trading, machine learning-based trading, and reinforcement learning-based trading. The application features a graphical user interface built with Tkinter and ttkbootstrap for an enhanced user experience.

## Features

1. **Manual Trading**: Direct control over trades with real-time market data
2. **Machine Learning Trading**: Automated trading based on ML models
3. **Reinforcement Learning Trading**: Advanced trading strategies using deep Q-learning
4. **Multiple Exchange Support**: Connect to various cryptocurrency exchanges
5. **Real-time Data**: Live market data and price updates
6. **Technical Analysis**: Comprehensive set of trading indicators
7. **Risk Management**: Stop loss, take profit, and position sizing controls
8. **Charting**: Visual representation of market data
9. **Bot Management**: Create, start, stop, and manage trading bots

## Architecture

The application follows the Model-View-Presenter (MVP) architectural pattern:

### Models
The model layer contains all business logic and data handling:
- `models.py`: Main model classes that coordinate different aspects of the application
- `machinelearning/`: Implementation of machine learning trading strategies
- `manualtrading/`: Implementation of manual trading functionality
- `reinforcement/`: Implementation of reinforcement learning trading strategies

### Views
The view layer handles the user interface:
- `views.py`: Main application window and login view
- `tradetab.py`: Manual trading interface
- `bottab.py`: Bot management interface
- `charttab.py`: Charting interface
- `exchangetab.py`: Exchange management interface
- `mltab.py`: Machine learning interface
- `rltab.py`: Reinforcement learning interface

### Presenters
The presenter layer acts as an intermediary between models and views:
- `presenters.py`: Contains all presenter classes that handle the logic between models and views

## Quick Setup with UV

This project now supports [UV](https://github.com/astral-sh/uv), an extremely fast Python package installer and resolver, as the recommended way to set up and manage dependencies.

### Prerequisites

- Python 3.8 or higher
- pip (to install UV initially)

### Setup Steps

1. **Install UV** (if not already installed):
   ```bash
   pip install uv
   ```

2. **Create virtual environment and install dependencies**:
   ```bash
   # On Windows
   .\run.ps1 setup
   
   # On Unix-like systems
   ./setup.sh
   ```

   Or manually:
   ```bash
   uv venv
   uv pip install -e .
   ```

3. **Activate the virtual environment**:
   ```bash
   # On Windows
   .venv\Scripts\activate
   
   # On Unix-like systems
   source .venv/bin/activate
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your actual API keys and database credentials

5. **Initialize the application**:
   ```bash
   python init_app.py
   ```

6. **Run the application**:
   ```bash
   python main.py
   ```

### Benefits of Using UV

- **Speed**: UV is up to 10-100x faster than pip
- **Reliability**: Better dependency resolution
- **Reproducibility**: Lock file support for consistent environments
- **Simplicity**: Single binary installation

## Traditional Setup

If you prefer not to use UV, you can still use traditional pip-based setup:

1. **Create a virtual environment**:
   ```bash
   python -m venv trading_venv
   source trading_venv/bin/activate  # On Windows: trading_venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   # Option 1: Using our custom install_deps.py script (recommended)
   python install_deps.py
   
   # Option 2: Using pip directly
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your actual API keys and database credentials

4. **Initialize the application**:
   ```bash
   python init_app.py
   ```

5. **Run the application**:
   ```bash
   python main.py
   ```

## Configuration

### Environment Variables (.env)
The application requires several environment variables for API keys and database connections:
- `API_KEY_PHE_TEST`: Phemex testnet API key
- `API_SECRET_PHE_TEST`: Phemex testnet API secret
- Database credentials (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD)

### Configuration (config.ini)
The application uses `config.ini` for various settings:
- Window size
- WebSocket parameters
- Data timeframes
- Trading parameters
- Model hyperparameters
- Environment settings

### Centralized Configuration Management
The application now uses a centralized configuration management system (`util/config_manager.py`) that:
- Consolidates settings from multiple sources (config.ini, .env, production_config.py)
- Provides type-safe access to configuration values
- Supports dynamic configuration reloading
- Handles API key management securely
- Integrates with the production configuration system

## Security

For enhanced security, this application includes utilities for:
- Secure credential storage and retrieval
- API key masking in logs
- Encryption/decryption capabilities for sensitive data

See `util/secure_credentials.py` for implementation details.

## Usage

### Running the Application

1. **With PowerShell script** (Windows):
   ```bash
   .\run.ps1 run
   ```

2. **With shell script** (Unix-like systems):
   ```bash
   ./run.sh
   ```

3. **Direct execution**:
   ```bash
   python main.py
   ```

### Development Commands

- **Setup development environment**:
  ```bash
  .\run.ps1 dev-setup
  ```

- **Run tests**:
  ```bash
  .\run.ps1 test
  ```

- **Run linting**:
  ```bash
  .\run.ps1 lint
  ```

- **Update dependencies**:
  ```bash
  .\run.ps1 update-deps
  ```

## Scripts

The project includes several helper scripts:

- `run.ps1`: PowerShell script for Windows users
- `setup.sh`: Shell script for Unix-like systems
- `setup.bat`: Batch script for Windows command prompt
- `install_deps.py`: Python script for robust dependency installation using pyproject.toml
- `init_app.py`: Application initialization checker
- `test_uv_setup.py`: UV setup verification script

## Documentation

See the `docs/` directory for detailed documentation:
- `architecture.md`: Detailed architecture information
- `setup.md`: Extended setup instructions
- `security.md`: Security best practices and implementation details

### Utility Modules

The `util/` directory contains various utility modules that provide common functionality:

- `async_client.py`: Async CCXT client with WebSocket support and connection optimization
- `async_trading.py`: Asynchronous trading operations and order management
- `cache.py`: Enhanced multi-layer cache system with Redis integration
- `candlestick_parser.py`: Candlestick pattern recognition and analysis
- `config_manager.py`: Centralized configuration management system
- `db_pool.py`: Database connection pooling and management
- `error_handling.py`: Standardized error handling and exception management
- `loggers.py`: Logging configuration and management
- `ml_optimization.py`: Machine learning model optimization utilities
- `parallel.py`: Parallel processing and concurrency utilities
- `secure_config.py`: Secure configuration handling and encryption
- `secure_credentials.py`: Secure credential storage and retrieval
- `standardized_loggers.py`: Enhanced logging with standardized formatting
- `utils.py`: General utility functions
- `validation.py`: Data validation and sanitization utilities
- `websocket_util.py`: WebSocket connection management and utilities

## Components

### 1. Manual Trading
Direct trade execution through the trading tab with real-time market data and risk management controls.

### 2. Machine Learning Trading
Automated trading based on ML models, with support for training, testing, and prediction.

### 3. Reinforcement Learning Trading
Advanced DQL-based trading strategies with hyperparameter optimization using Optuna.

### 4. Bot Management
Create and manage multiple trading bots with different strategies.

### 5. Charting
Visual market data analysis with real-time updates.

### 6. Exchange Integration
Connect to multiple cryptocurrency exchanges with sandbox mode support for testing.

### 7. Refactored Components
The application includes refactored versions of core components for improved maintainability:
- `model/refactored_models.py`: Enhanced model layer with better organization
- `refactored_presenters.py`: Simplified presenter layer with improved logic
- `view/refactored_views.py`: Enhanced view layer with better feedback and styling

## Dependencies

Key technologies used in this project:
- **Python**: Core programming language
- **ccxt**: Cryptocurrency exchange library
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **tensorflow/pytorch**: Deep learning frameworks
- **ttkbootstrap**: Enhanced tkinter widgets for UI
- **matplotlib**: Charting and visualization
- **psycopg2**: PostgreSQL database adapter
- **optuna**: Hyperparameter optimization
- **pandas-ta**: Technical analysis library

For a complete list of dependencies, see `pyproject.toml` or `requirements.txt`.