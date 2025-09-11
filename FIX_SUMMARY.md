# Fix for Tkinter Application Import Errors

## Problem
The trading application was failing to start with import errors for several dependencies:
- `textblob`
- `vaderSentiment` 
- `ttkbootstrap`

## Root Cause
The dependencies were not available in the current Python environment (Anaconda Python 3.10) even though they were listed in the project's `pyproject.toml` and `requirements.txt` files.

## Solution
Installed the missing dependencies directly for the Anaconda Python environment:

```bash
# Install textblob
D:\Users\Maart\anaconda3\python.exe -m pip install textblob

# Install vaderSentiment
D:\Users\Maart\anaconda3\python.exe -m pip install vaderSentiment

# Install ttkbootstrap
D:\Users\Maart\anaconda3\python.exe -m pip install ttkbootstrap
```

## Verification
After installing the dependencies, the application starts successfully:
- Main window displays correctly
- Login view is shown
- All tabs load without errors
- Application runs and closes gracefully

## Additional Notes
While the `install_deps.py` script was run, it appears that the dependencies were installed in a different Python environment than the one being used to run the application. This is a common issue when using multiple Python installations (system Python, Anaconda, etc.) on the same machine.

For future development, it would be recommended to:
1. Ensure all dependencies are installed in the correct Python environment
2. Use virtual environments to isolate project dependencies
3. Consider using conda instead of pip for dependency management when working with Anaconda