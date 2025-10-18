#!/usr/bin/env python3
"""
Setup script for BetYard ML Backend
Installs dependencies and initializes the ML model
"""

import subprocess
import sys
import os

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def test_installation():
    """Test if all dependencies are installed correctly"""
    print("🧪 Testing installation...")
    try:
        import flask
        import pandas as pd
        import numpy as np
        import xgboost as xgb
        import sklearn
        print("✅ All dependencies installed successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def initialize_model():
    """Initialize the ML model"""
    print("🧠 Initializing XGBoost model...")
    
    # This will create and train the model on first run
    from app import ml_model
    print("✅ Model initialized successfully!")

def main():
    """Main setup function"""
    print("🏈 BetYard ML Backend Setup")
    print("=" * 40)
    
    try:
        install_requirements()
        
        if test_installation():
            initialize_model()
            print("\n🎉 Setup completed successfully!")
            print("\nTo start the ML backend:")
            print("  python app.py")
            print("\nAPI will be available at:")
            print("  http://localhost:5000")
        else:
            print("❌ Setup failed. Please check error messages above.")
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")

if __name__ == "__main__":
    main()