"""
Dependency Installation Script

This script provides a robust dependency installation process for the trading application
using the pyproject.toml file as the source of truth.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_uv_installed():
    """Check if uv is installed and available in PATH."""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_with_uv():
    """Install dependencies using uv."""
    try:
        print("🔧 Installing dependencies with uv...")
        # Sync dependencies from pyproject.toml
        subprocess.run(["uv", "sync"], check=True)
        print("✅ Dependencies installed successfully with uv")
        
        # Install PyTorch with CUDA support as a separate step
        print("🔧 Installing PyTorch with CUDA 12.4 support...")
        subprocess.run([
            "uv", "pip", "install", 
            "torch==2.6.0+cu124", 
            "torchvision==0.21.0+cu124", 
            "torchaudio==2.6.0+cu124",
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ], check=True)
        print("✅ PyTorch with CUDA support installed successfully")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies with uv: {e}")
        return False

def install_with_pip():
    """Install dependencies using pip."""
    try:
        print("🔧 Installing dependencies with pip...")
        # Install from pyproject.toml using pip
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("✅ Dependencies installed successfully with pip")
        
        # Install PyTorch with CUDA support
        print("🔧 Installing PyTorch with CUDA 12.4 support...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch==2.6.0+cu124", 
            "torchvision==0.21.0+cu124", 
            "torchaudio==2.6.0+cu124",
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ], check=True)
        print("✅ PyTorch with CUDA support installed successfully")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies with pip: {e}")
        return False

def create_requirements_txt():
    """Create a requirements.txt file from pyproject.toml for compatibility."""
    try:
        print("🔧 Creating requirements.txt from pyproject.toml...")
        # Use uv to export dependencies to requirements.txt
        result = subprocess.run(["uv", "pip", "compile", "pyproject.toml", "--output-file", "requirements.txt"], 
                              capture_output=True, text=True, check=True)
        print("✅ requirements.txt created successfully")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Failed to create requirements.txt with uv. Creating manually...")
        # Fallback: manually create requirements.txt
        try:
            from pyproject_hooks import read_pyproject_toml
            import tomli
            
            # Read pyproject.toml
            with open("pyproject.toml", "rb") as f:
                pyproject_data = tomli.load(f)
            
            # Extract dependencies
            dependencies = pyproject_data.get("project", {}).get("dependencies", [])
            
            # Write to requirements.txt
            with open("requirements.txt", "w") as f:
                for dep in dependencies:
                    f.write(f"{dep}\n")
            
            print("✅ requirements.txt created manually")
            return True
        except Exception as e:
            print(f"❌ Failed to create requirements.txt manually: {e}")
            return False

def main():
    """Main installation function."""
    print("🚀 TRADING APPLICATION DEPENDENCY INSTALLER")
    print("=" * 50)
    
    # Check if we're in the correct directory
    if not Path("pyproject.toml").exists():
        print("❌ pyproject.toml not found. Please run this script from the project root directory.")
        return False
    
    # Create requirements.txt for compatibility
    create_requirements_txt()
    
    # Try to install with uv first (faster)
    if check_uv_installed():
        print("✅ Found uv package manager")
        if install_with_uv():
            print("\n🎉 ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
            return True
    else:
        print("⚠️  uv not found, installing with pip...")
        
    # Fallback to pip
    if install_with_pip():
        print("\n🎉 ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
        return True
    else:
        print("\n❌ FAILED TO INSTALL DEPENDENCIES!")
        print("Please install dependencies manually by running:")
        print("  pip install -e .")
        print("  pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)