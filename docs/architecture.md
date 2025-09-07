# Project Documentation

## Architecture Overview

This trading application follows the Model-View-Presenter (MVP) architectural pattern:

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

## Component Details

### Main Application (`main.py`)
The entry point of the application that initializes the model, view, and presenter components.

### Configuration (`config.ini`)
Contains application settings including:
- Window dimensions
- WebSocket parameters
- Data timeframes
- Trading parameters
- Model hyperparameters
- Environment settings

### Dependencies (`requirements.txt`)
Lists all Python packages required for the application to run.

## Data Flow

1. User interacts with the View (UI)
2. View sends user actions to the Presenter
3. Presenter processes the action and updates the Model
4. Model updates its state and notifies the Presenter
5. Presenter updates the View with new data
6. View displays updated information to the user

This pattern ensures a clean separation of concerns and makes the application easier to maintain and extend.