#!/bin/bash
echo "🔄 Collecting SGD Results from Della via Git"
echo "============================================"

# Step 1: On Della, commit and push SGD results
echo "📋 Step 1: Run these commands on Della after SGD jobs complete:"
echo ""
echo "cd /scratch/gpfs/mg7411/ManyPaths"
echo "git add baseline_checkpoints/"
echo "git add results/baseline_sgd/"
echo "git add sgd_baseline_*.out"
echo "git add sgd_baseline_*.err"
echo "git commit -m 'SGD baseline results: checkpoints and trajectories'"
echo "git push origin master"
echo ""

# Step 2: On local machine, pull the results
echo "📋 Step 2: Run these commands locally after Della push:"
echo ""
echo "git pull origin master"
echo "ls -la baseline_checkpoints/  # Verify checkpoints"
echo "ls -la results/baseline_sgd/   # Verify trajectories"
echo ""

# Step 3: Generate comparison plots locally
echo "📋 Step 3: Generate comparison plots locally:"
echo ""
echo "python parameter_curvature_meta_vs_sgd.py"
echo "python plot_learning_curves_meta_vs_sgd.py"
echo "python loss_landscape_dashboard_meta_vs_sgd.py"
echo ""

echo "✅ Collection plan ready!" 