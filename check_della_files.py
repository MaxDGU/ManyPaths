#!/usr/bin/env python3
"""
Quick script to check what camera-ready files exist on Della
"""

import os
import glob
from pathlib import Path

def check_files():
    print("🔍 Checking for camera-ready files on Della...")
    
    # Check different patterns in multiple locations
    patterns = [
        "camera_ready_array_*",
        "camera_ready_array_*_*.out", 
        "camera_ready_array_*_*.err",
        "*camera_ready*",
        "focused_camera_ready*",
        "*array*"
    ]
    
    # Check in main directory and logs directory
    search_dirs = [Path("."), Path("logs")]
    
    for search_dir in search_dirs:
        print(f"\n📁 Searching in: {search_dir.absolute()}")
        if not search_dir.exists():
            print(f"❌ {search_dir} directory not found")
            continue
            
        for pattern in patterns:
            files = list(search_dir.glob(pattern))
            if files:
                print(f"  Pattern '{pattern}': {len(files)} files")
                for file in files[:5]:  # Show first 5
                    print(f"    - {file.name}")
                if len(files) > 5:
                    print(f"    ... and {len(files) - 5} more")
    
    # Also check with ls command patterns in main directory
    print(f"\n📁 Main directory listing (looking for camera_ready):")
    try:
        result = os.popen("ls -la | grep -i camera").read()
        if result.strip():
            print(result)
        else:
            print("No files matching 'camera' in main directory")
    except:
        print("Could not run ls command")
        
    print(f"\n📁 Main directory listing (looking for array):")
    try:
        result = os.popen("ls -la | grep -i array").read()
        if result.strip():
            print(result)
        else:
            print("No files matching 'array' in main directory")
    except:
        print("Could not run ls command")

if __name__ == "__main__":
    check_files() 