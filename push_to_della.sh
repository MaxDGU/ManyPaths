#!/bin/bash
# Quick script to push all gradient alignment files to della

echo "🚀 Pushing gradient alignment files to della..."

# Define files to push
FILES=(
    "gradient_alignment_analysis.py"
    "k1_vs_k10_comparison.py" 
    "enhanced_data_efficiency_analysis.py"
    "quick_start_analysis.py"
    "della_test_gradient_alignment.py"
    "della_full_gradient_experiments.py"
    "run_gradient_alignment_experiments.slurm"
)

# Push files
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "📁 Pushing $file..."
        scp "$file" mg7411@della-gpu.princeton.edu:/scratch/gpfs/mg7411/ManyPaths/
    else
        echo "⚠️  Warning: $file not found"
    fi
done

echo "✅ Push complete!"
echo ""
echo "📋 Next steps:"
echo "1. ssh mg7411@della-gpu.princeton.edu"
echo "2. cd /scratch/gpfs/mg7411/ManyPaths"
echo "3. python della_test_gradient_alignment.py"
echo "4. sbatch run_gradient_alignment_experiments.slurm" 