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
├── docs/                     # Documentation
│   ├── architecture.md       # Architecture details
│   └── setup.md              # Setup instructions
├── data/                     # Data storage (git-ignored)
├── model/                    # Business logic
├── view/                     # User interface
└── util/                     # Utility functions
```

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
Automated trading based on ML models, with support for training, testing, and prediction.

### 3. Reinforcement Learning Trading
Advanced DQL-based trading strategies with hyperparameter optimization using Optuna.

### 4. Bot Management
Create and manage multiple trading bots with different strategies.

### 5. Charting
Visual market data analysis with real-time updates.

### 6. Exchange Integration
Connect to multiple cryptocurrency exchanges with sandbox mode support for testing.

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
3. Add UI elements in the exchange tab

### Creating a New ML Model
1. Implement in the machinelearning module
2. Add UI elements in the ML tab
3. Connect through the presenter layer

### Implementing a New RL Strategy
1. Create new agent or modify existing DQLAgent
2. Add hyperparameters to the optuna optimization
3. Update the training and testing logic

## Troubleshooting

1. Run `python init_app.py` to check your setup
2. Ensure all environment variables are set correctly
3. Check that your database is accessible
4. Verify API keys have necessary permissions
5. Check log files in `data/logs/` for detailed error information