#!/usr/bin/env python3
import subprocess
import sys
import os

def deploy_nemo():
    print("🐟 NEMO Deployment Script")
    print("=" * 40)
    
    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Create model if it doesn't exist
    if not os.path.exists('models/predictor.pkl'):
        print("🤖 Creating ML model...")
        subprocess.run([sys.executable, "create_model.py"])
    
    # Create directories
    os.makedirs('frontend/static/audio', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    print("✅ NEMO is ready!")
    print("🚀 Run: python app.py")
    print("🌐 Visit: http://localhost:5000")

if __name__ == "__main__":
    deploy_nemo()
