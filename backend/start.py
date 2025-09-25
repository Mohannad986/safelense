#!/usr/bin/env python3
"""
TrustAI Backend Startup Script
"""
import subprocess
import sys
import os

def install_requirements():
    """Install Python requirements"""
    print("Installing Python dependencies...")
    try:
        # Upgrade pip first to ensure robust dependency installation
        print("Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements with more verbose output and timeout handling
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements.txt", 
            "--timeout", "300",
            "--retries", "3"
        ])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("💡 Try running: pip install -r backend/requirements.txt manually")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during installation: {e}")
        return False
    return True

def start_server():
    """Start the FastAPI server"""
    print("Starting TrustAI backend server...")
    print("🔗 Server will be available at: http://localhost:8000")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], stdout=sys.stdout, stderr=sys.stderr)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        print("💡 Make sure port 8000 is not already in use")
        print("💡 Try running: python backend/app.py directly")

if __name__ == "__main__":
    print("🚀 TrustAI Backend Starting...")
    
    if install_requirements():
        start_server()
    else:
        print("❌ Failed to start server due to dependency issues")
        sys.exit(1)