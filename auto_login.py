import tkinter as tk
from tkinter import messagebox
import threading
import time
import subprocess
import os

def auto_login():
    # Wait a moment for the app to start
    time.sleep(3)
    
    try:
        # Find the main window
        root = tk._default_root
        if root:
            # Find all children that might be our login window
            for child in root.winfo_children():
                # Look for the login window
                if str(child).startswith('.!loginview'):
                    # Get the username and password entries
                    # Based on the grid layout in the code:
                    # Row 0, column 1 should be username entry
                    # Row 1, column 1 should be password entry
                    
                    # Set the test credentials
                    root.nametowidget(child.winfo_children()[1]).delete(0, tk.END)
                    root.nametowidget(child.winfo_children()[1]).insert(0, "test")
                    
                    root.nametowidget(child.winfo_children()[3]).delete(0, tk.END)
                    root.nametowidget(child.winfo_children()[3]).insert(0, "t")
                    
                    # Click the login button
                    # Row 2 should contain the login button
                    root.nametowidget(child.winfo_children()[4]).invoke()
                    break
    except Exception as e:
        print(f"Auto-login error: {e}")

if __name__ == "__main__":
    # Start the auto-login in a separate thread
    login_thread = threading.Thread(target=auto_login)
    login_thread.daemon = True
    login_thread.start()
    
    # Run the main application
    os.system(".venv\\Scripts\\activate && python main.py")