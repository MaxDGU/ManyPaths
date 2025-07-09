#!/usr/bin/env python3
"""
Quick script to check what camera-ready files exist on Della
"""

import os
import glob
from pathlib import Path

def check_files():
    print("🔍 Checking for camera-ready files on Della...")
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("❌ logs/ directory not found")
        return
    
    # Check different patterns
    patterns = [
        "camera_ready_array_*",
        "camera_ready_array_*_*.out", 
        "camera_ready_array_*_*.err",
        "*camera_ready*",
        "focused_camera_ready*",
        "*array*"
    ]
    
    for pattern in patterns:
        files = list(logs_dir.glob(pattern))
        print(f"\nPattern '{pattern}': {len(files)} files")
        if files:
            for file in files[:5]:  # Show first 5
                print(f"  - {file.name}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")
    
    # Also check with ls command patterns
    print(f"\n📁 Full logs directory listing:")
    try:
        result = os.popen("ls -la logs/ | head -20").read()
        print(result)
    except:
        print("Could not run ls command")

if __name__ == "__main__":
    check_files() 