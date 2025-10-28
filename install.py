#!/usr/bin/env python3
"""
Installation script for Video Clip Generator custom node
"""
import subprocess
import sys
import os
from pathlib import Path

def find_comfyui_python():
    """Find the correct Python executable for ComfyUI (embedded or system)"""
    script_dir = Path(__file__).parent.absolute()
    
    # Try to find embedded Python (standalone installation)
    possible_paths = [
        script_dir / ".." / ".." / "python_embeded" / "python.exe",  # Windows standalone
        script_dir / ".." / ".." / ".." / "python_embeded" / "python.exe",  # Alternative structure
        script_dir / ".." / ".." / "python_embeded" / "python",  # Linux/Mac
        script_dir / ".." / ".." / ".." / "python_embeded" / "python",  # Alternative Linux/Mac
    ]
    
    for python_path in possible_paths:
        if python_path.exists():
            print(f"[OK] Found ComfyUI embedded Python: {python_path}")
            return str(python_path.resolve())
    
    # Fallback to system Python
    print(f"[WARNING] Using system Python (embedded Python not found): {sys.executable}")
    return sys.executable

def install_requirements():
    """Install required packages from requirements.txt"""
    print("=" * 60)
    print("Installing Video Clip Generator Dependencies")
    print("=" * 60)
    print()
    
    script_dir = Path(__file__).parent.absolute()
    requirements_file = script_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"[ERROR] Error: requirements.txt not found at {requirements_file}")
        return False
    
    # Find correct Python executable
    python_exe = find_comfyui_python()
    
    print(f"[INSTALL] Installing packages from: {requirements_file}")
    print(f"[PYTHON] Using Python: {python_exe}")
    print()
    
    try:
        # Use the detected Python executable
        subprocess.check_call([
            python_exe,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_file)
        ])
        
        print("=" * 60)
        print("[OK] Installation completed successfully!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Restart ComfyUI to load the new nodes")
        print("2. Look for nodes in the 'VibeMusicEngine', 'video', 'animation' categories")
        print()
        print("Available nodes:")
        print("   - Vibe Music Engine (audio transcription & beat detection)")
        print("   - Video Merger (merge multiple videos)")
        print("   - Wiggle Effect (After Effects-style animation)")
        print("   - Moving Titles (animated text overlays)")
        print("   - Optical Compensation (lens distortion effects)")
        print("   - And more utility nodes!")
        print()
        
        return True
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("[ERROR] Installation failed!")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        print("Troubleshooting tips:")
        print("1. Make sure you have internet connection")
        print("2. Try running with administrator/sudo privileges")
        print("3. Check if pip is properly installed")
        print("4. Some packages may already be installed")
        print("   by ComfyUI - this is normal")
        print()
        return False
    
    except Exception as e:
        print()
        print("=" * 60)
        print("[ERROR] Unexpected error during installation!")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        return False

if __name__ == "__main__":
    print()
    print(f"Python executable (current): {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print()
    
    success = install_requirements()
    
    if success:
        print("[OK] Installation completed successfully!")
        print("Please restart ComfyUI to use the new nodes.")
        sys.exit(0)
    else:
        print("[ERROR] Installation failed! Please check the error messages above.")
        sys.exit(1)

