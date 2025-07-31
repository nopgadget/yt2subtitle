#!/usr/bin/env python3
"""
Setup script for YouTube to Subtitles Converter
Automates the installation process and environment setup.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
            return True
    except ImportError:
        print("⚠️  PyTorch not installed - will install during setup")
        return True

def install_requirements():
    """Install required packages."""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found!")
        return False
    
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def create_directories():
    """Create necessary directories."""
    directories = ["downloaded", "tools"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main setup function."""
    print("🚀 YouTube to Subtitles Converter - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed! Please check the error messages above.")
        sys.exit(1)
    
    # Check CUDA
    check_cuda()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate your environment: conda activate subtitles")
    print("2. Test the script: python youtube_to_subtitles.py --url \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\"")
    print("3. Check the README.md for detailed usage instructions")
    
    print("\n💡 Tips:")
    print("- The first run will download the AI model (~3GB)")
    print("- GPU acceleration requires NVIDIA GPU with CUDA support")
    print("- For help, check the troubleshooting section in README.md")

if __name__ == "__main__":
    main() 