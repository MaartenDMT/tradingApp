
# GEMINI.md

## Project Overview

This is a comprehensive trading application built with Python. It provides multiple trading strategies, including manual trading, machine learning-based trading, and reinforcement learning-based trading. The application features a graphical user interface built with Tkinter and ttkbootstrap.

The project follows the Model-View-Presenter (MVP) architectural pattern.

**Key Technologies:**

*   **Core:** Python
*   **GUI:** Tkinter, ttkbootstrap
*   **Data Analysis:** pandas, numpy
*   **Machine Learning:** scikit-learn, tensorflow, pytorch
*   **Trading:** ccxt
*   **Database:** PostgreSQL
*   **Package Management:** uv

## Building and Running

### Quick Setup with UV

1.  **Install UV:**
    ```bash
    pip install uv
    ```

2.  **Create virtual environment and install dependencies:**
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

3.  **Activate the virtual environment:**
    ```bash
    # On Windows
    .venv\Scripts\activate
   
    # On Unix-like systems
    source .venv/bin/activate
    ```

4.  **Run the application:**
    ```bash
    python main.py
    ```

### Makefile Commands

*   **Run the application:**
    ```bash
    make run
    ```
*   **Run tests:**
    ```bash
    make test
    ```
*   **Sync dependencies:**
    ```bash
    make sync
    ```

## Development Conventions

*   **Formatting:** The project uses `black` for code formatting.
*   **Linting:** `flake8` and `pylint` are used for linting.
*   **Testing:** Tests are written with `pytest` and can be found in the `test/` directory.
