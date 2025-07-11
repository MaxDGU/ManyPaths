#!/bin/bash
# Script to pull landscape trajectory results from della

echo "📥 Pulling landscape results from della..."

# Create local results directory if it doesn't exist
mkdir -p della_analysis_results

# Define the della path
DELLA_PATH="mg7411@della-gpu.princeton.edu:/scratch/gpfs/mg7411/ManyPaths/results/"

echo "🔍 Checking for landscape trajectory files on della..."

# Pull all landscape trajectory CSV files
echo "📁 Pulling landscape trajectory CSV files..."
scp "${DELLA_PATH}*landscape_trajectory.csv" della_analysis_results/ 2>/dev/null

# Pull any landscape log files that might contain useful info
echo "📁 Pulling landscape log files..."
scp "${DELLA_PATH}*landscape*.out" della_analysis_results/ 2>/dev/null

# Check what we got
echo ""
echo "✅ Files retrieved:"
ls -la della_analysis_results/*landscape* 2>/dev/null | head -20

echo ""
echo "📊 Summary of landscape files:"
echo "  Trajectory files: $(ls della_analysis_results/*landscape_trajectory.csv 2>/dev/null | wc -l)"
echo "  Log files: $(ls della_analysis_results/*landscape*.out 2>/dev/null | wc -l)"

echo ""
echo "🎯 Next steps:"
echo "1. python analyze_landscape_results.py  # Analyze all trajectory files"
echo "2. python create_concept_landscapes.py  # Generate loss landscapes by concept type" 