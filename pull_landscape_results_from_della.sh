#!/bin/bash
# Script to pull landscape trajectory results from della

echo "📥 Pulling landscape results from della..."

# Create local results directory if it doesn't exist
mkdir -p della_analysis_results

# Define the della paths
DELLA_RESULTS_PATH="mg7411@della-gpu.princeton.edu:/scratch/gpfs/mg7411/ManyPaths/results/"
DELLA_LOGS_PATH="mg7411@della-gpu.princeton.edu:/scratch/gpfs/mg7411/ManyPaths/logs/"

echo "🔍 Checking for landscape trajectory files on della..."

# Pull all landscape trajectory CSV files from results directory
echo "📁 Pulling landscape trajectory CSV files from results/..."
scp "${DELLA_RESULTS_PATH}*landscape_trajectory.csv" della_analysis_results/ 2>/dev/null

# Pull landscape log files from logs directory (correct location)
echo "📁 Pulling landscape log files from logs/..."
scp "${DELLA_LOGS_PATH}mp_landscape_overnight_*.out" della_analysis_results/ 2>/dev/null
scp "${DELLA_LOGS_PATH}mp_landscape_overnight_*.err" della_analysis_results/ 2>/dev/null

# Also check for any other landscape-related log files
echo "📁 Pulling any other landscape log files..."
scp "${DELLA_LOGS_PATH}*landscape*.out" della_analysis_results/ 2>/dev/null
scp "${DELLA_LOGS_PATH}*landscape*.err" della_analysis_results/ 2>/dev/null

# Check what we got
echo ""
echo "✅ Files retrieved:"
ls -la della_analysis_results/*landscape* 2>/dev/null | head -20

echo ""
echo "📊 Summary of landscape files:"
echo "  Trajectory files: $(ls della_analysis_results/*landscape_trajectory.csv 2>/dev/null | wc -l)"
echo "  Log files (.out): $(ls della_analysis_results/*landscape*.out 2>/dev/null | wc -l)"
echo "  Error files (.err): $(ls della_analysis_results/*landscape*.err 2>/dev/null | wc -l)"
echo "  Overnight log files: $(ls della_analysis_results/mp_landscape_overnight_*.out 2>/dev/null | wc -l)"

echo ""
echo "🎯 Next steps:"
echo "1. python create_concept_landscapes.py  # Generate loss landscapes by concept type"
echo "2. Check log files for any errors or completion status" 