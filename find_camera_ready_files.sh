#!/bin/bash

echo "🔍 Searching for camera-ready files..."
echo "Current directory: $(pwd)"
echo ""

echo "📁 Searching in main directory:"
find . -maxdepth 1 -name "camera_ready_array_*" -type f | head -10

echo ""
echo "📁 Searching in logs directory:"
find logs/ -name "camera_ready_array_*" -type f 2>/dev/null | head -10

echo ""
echo "📁 All files matching 'camera_ready' pattern:"
find . -name "*camera_ready*" -type f 2>/dev/null | head -10

echo ""
echo "📁 All files matching 'array' pattern in current directory:"
ls -la | grep array

echo ""
echo "📁 Files with job ID 65683885:"
find . -name "*65683885*" -type f 2>/dev/null | head -10 