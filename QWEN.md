# QWEN.md - Trading Application Context

## Project Overview

This is a comprehensive trading application built with Python that provides multiple trading strategies including manual trading, machine learning-based trading, and reinforcement learning-based trading. The application features a graphical user interface built with Tkinter and ttkbootstrap for an enhanced user experience.

### Key Features

1. **Manual Trading**: Direct control over trades with real-time market data
2. **Machine Learning Trading**: Automated trading based on ML models
3. **Reinforcement Learning Trading**: Advanced trading strategies using deep Q-learning
4. **Multiple Exchange Support**: Connect to various cryptocurrency exchanges
5. **Real-time Data**: Live market data and price updates
6. **Technical Analysis**: Comprehensive set of trading indicators
7. **Risk Management**: Stop loss, take profit, and position sizing controls
8. **Charting**: Visual representation of market data
9. **Bot Management**: Create, start, stop, and manage trading bots

## Project Structure

```
tradingApp/
├── config.ini                 # Application configuration
├── main.py                    # Main entry point
├── init_app.py                # Initialization checker
├── install_deps.py            # Dependency installer
├── presenters.py              # Application presenters
├── hyper.py                   # Hyperparameter optimization
├── predict_tester.py          # ML model tester
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── README.md                 # Project overview
├── USAGE.md                  # Usage guide
├── QWEN.md                   # This file
├── VIEW_CLEANUP_SUMMARY.md   # View folder cleanup summary
├── MODEL_CLEANUP_SUMMARY.md  # Model folder cleanup summary
├── docs/                     # Documentation
│   ├── architecture.md       # Architecture details
│   └── setup.md              # Setup instructions
├── data/                     # Data storage (git-ignored)
├── model/                    # Business logic
│   ├── backup_unused/        # Backup of unused model files
├── view/                     # User interface
│   ├── backup_unused/        # Backup of unused view files
└── util/                     # Utility functions
```

## Architecture

The application follows the Model-View-Presenter (MVP) architectural pattern:

### Models
The model layer contains all business logic and data handling:
- `models.py`: Main model classes that coordinate different aspects of the application
- `bot_system.py`: Bot system implementation
- `features.py`: Trading features implementation
- `manualtrading/`: Implementation of manual trading functionality
- `ml_system/`: Machine learning system implementation
- `reinforcement/`: Implementation of reinforcement learning trading strategies
- `rl_system/`: Reinforcement learning system implementation

### Views
The view layer handles the user interface:
- `views.py`: Main application window and login view
- `tradetab.py`: Manual trading interface
- `bottab_optimized.py`: Bot management interface
- `charttab.py`: Charting interface
- `mltab.py`: Machine learning interface
- `rltab.py`: Reinforcement learning interface
- `trading_system_tab.py`: Advanced trading system interface
- `ml_system_tab.py`: Machine learning system interface
- `rl_system_tab.py`: Reinforcement learning system interface
- `advanced_rl_system_tab.py`: Advanced reinforcement learning system interface

### Presenters
The presenter layer acts as an intermediary between models and views:
- `presenters.py`: Contains all presenter classes that handle the logic between models and views

## Technology Stack

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

## Setup and Configuration

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

## Running the Application

1. **Install dependencies**:
   ```
   python install_deps.py
   ```
   Or manually:
   ```
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```
   cp .env.example .env
   ```
   Then edit `.env` and add your actual API keys and database credentials

3. **Initialize the application**:
   ```
   python init_app.py
   ```

4. **Run the application**:
   ```
   python main.py
   ```

## Key Components

### 1. Manual Trading
Direct trade execution through the trading tab with real-time market data and risk management controls.

### 2. Machine Learning Trading
Automated trading based on ML models, with support for training, testing, and prediction through the ML system tab.

### 3. Reinforcement Learning Trading
Advanced DQL-based trading strategies with hyperparameter optimization using Optuna through the RL system tabs.

### 4. Bot Management
Create and manage multiple trading bots with different strategies through the optimized bot tab.

### 5. Charting
Visual market data analysis with real-time updates through the chart tab.

### 6. Exchange Integration
Connect to multiple cryptocurrency exchanges with sandbox mode support for testing through exchange integration in models.

### 7. Advanced Trading Systems
Integrated trading systems including ML System, RL System, and Advanced RL System through dedicated tabs.

## Recent Cleanup Activities

### View Folder Cleanup
Unused view files have been moved to `view/backup_unused/` to maintain a cleaner codebase:
- `bottab.py` (replaced by `bottab_optimized.py`)
- `charttab_optimized.py` (replaced by `charttab.py`)
- `exchangetab.py` and `exchangetab_optimized.py` (not used anywhere)
- `mltab_optimized.py` (replaced by `mltab.py`)
- `rltab_optimized.py` (replaced by `rltab.py`)
- `tradetab_optimized.py` (replaced by `tradetab.py`)
- `views_optimized.py` and `views_original_backup.py` (older versions)

See `VIEW_CLEANUP_SUMMARY.md` for detailed information.

### Model Folder Cleanup
Unused model files have been moved to `model/backup_unused/`:
- `models_optimized.py` (replaced by `models.py`)
- `models_original_backup.py` (backup file)
- `exampletab.py` (example file not used in production)

See `MODEL_CLEANUP_SUMMARY.md` for detailed information.

## Development Guidelines

### Code Structure
- Follow the MVP pattern with clear separation of concerns
- Use logging for debugging and monitoring
- Handle exceptions properly with meaningful error messages
- Write modular, reusable code

### Testing
- Test with default credentials (username: test, password: t)
- Use sandbox mode for exchange connections during development
- Run the initialization checker (`python init_app.py`) to verify setup

### Logging
The application uses multiple loggers for different components:
- autobot, tradex, presenter, model, view, rl_trading, app, rl, env, agent, manual

## Common Tasks

### Adding a New Exchange
1. Add API keys to .env file
2. Update the exchange creation logic in models.py
3. Add UI elements in the exchange integration components

### Creating a New ML Model
1. Implement in the ml_system module
2. Add UI elements in the ML system tab
3. Connect through the presenter layer

### Implementing a New RL Strategy
1. Create new agent or modify existing agents in rl_system
2. Add hyperparameters to the optuna optimization
3. Update the training and testing logic

### Adding a New Trading Feature
1. Implement model logic in the appropriate module under models/
2. Create UI components in the view/ folder
3. Connect through presenters in presenters.py
4. Test with default credentials (username: test, password: t)

## Troubleshooting

1. Run `python init_app.py` to check your setup
2. Ensure all environment variables are set correctly
3. Check that your database is accessible
4. Verify API keys have necessary permissions
5. Check log files in `data/logs/` for detailed error information