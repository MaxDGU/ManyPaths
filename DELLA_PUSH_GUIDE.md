# Della Push and Experiment Guide

## 📦 Files to Push to Della

### New Analysis Scripts
```bash
# Copy these new files to della
scp gradient_alignment_analysis.py mg7411@della-gpu.princeton.edu:/scratch/gpfs/mg7411/ManyPaths/
scp k1_vs_k10_comparison.py mg7411@della-gpu.princeton.edu:/scratch/gpfs/mg7411/ManyPaths/
scp enhanced_data_efficiency_analysis.py mg7411@della-gpu.princeton.edu:/scratch/gpfs/mg7411/ManyPaths/
scp quick_start_analysis.py mg7411@della-gpu.princeton.edu:/scratch/gpfs/mg7411/ManyPaths/
```

### Della-Specific Scripts
```bash
# Copy della experiment scripts
scp della_test_gradient_alignment.py mg7411@della-gpu.princeton.edu:/scratch/gpfs/mg7411/ManyPaths/
scp della_full_gradient_experiments.py mg7411@della-gpu.princeton.edu:/scratch/gpfs/mg7411/ManyPaths/
scp run_gradient_alignment_experiments.slurm mg7411@della-gpu.princeton.edu:/scratch/gpfs/mg7411/ManyPaths/
```

### Updated Core Files (if modified)
```bash
# Only push if you've modified these locally
scp main.py mg7411@della-gpu.princeton.edu:/scratch/gpfs/mg7411/ManyPaths/
scp training.py mg7411@della-gpu.princeton.edu:/scratch/gpfs/mg7411/ManyPaths/
```

## 🚀 Execution Workflow

### Step 1: Push Files to Della
```bash
# Quick push all new files
scp gradient_alignment_analysis.py k1_vs_k10_comparison.py enhanced_data_efficiency_analysis.py quick_start_analysis.py della_test_gradient_alignment.py della_full_gradient_experiments.py run_gradient_alignment_experiments.slurm mg7411@della-gpu.princeton.edu:/scratch/gpfs/mg7411/ManyPaths/
```

### Step 2: Login to Della
```bash
ssh mg7411@della-gpu.princeton.edu
cd /scratch/gpfs/mg7411/ManyPaths
```

### Step 3: Test Gradient Alignment (CRITICAL)
```bash
# Run quick test first to ensure gradient alignment works
python della_test_gradient_alignment.py
```

**Expected output if successful:**
```
🧪 DELLA GRADIENT ALIGNMENT TEST
✅ Test completed successfully in XXXs
✅ Gradient alignment validation: Found X gradient alignment data points (range: X.XXX to X.XXX)
🎉 DELLA TEST PASSED!
```

### Step 4: Submit Overnight Job
```bash
# If test passes, submit the full experiment
sbatch run_gradient_alignment_experiments.slurm
```

### Step 5: Monitor Job
```bash
# Check job status
squeue -u mg7411

# Check job output (replace JOBID with actual job ID)
tail -f gradient_alignment_experiments_JOBID.out
```

## 📊 Expected Experiment Timeline

- **Test Phase**: 5-10 minutes
- **HIGH Priority Experiments**: 4-6 hours
  - K=1 F8_D3 (2 seeds): ~4 hours
  - K=10 F8_D3 (2 seeds): ~6 hours
- **MEDIUM Priority Experiments**: 3-4 hours
  - K=1/K=10 F16_D3: ~3-4 hours
- **LOW Priority Experiments**: 2-3 hours
  - K=1/K=10 F32_D3: ~2-3 hours

**Total Estimated Time**: 8-12 hours (perfect for overnight)

## 🎯 Success Criteria

### Minimum Success (Camera-Ready Ready)
- ✅ 4/4 HIGH priority experiments successful
- ✅ Gradient alignment data generated for all HIGH priority
- ✅ K=1 vs K=10 comparison data available

### Optimal Success (Paper Enhancement)
- ✅ 6/8 total experiments successful
- ✅ Multi-complexity gradient alignment data
- ✅ Robust statistical comparisons

## 🔧 Troubleshooting

### If Test Fails
1. **Check environment setup**:
   ```bash
   conda activate /scratch/gpfs/mg7411/envs/pytorch_env
   python -c "import torch; print(torch.__version__)"
   ```

2. **Check gradient alignment code**:
   ```bash
   grep -n "grad_alignment" training.py
   ```

3. **Run minimal debug**:
   ```bash
   python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 1 --first-order --epochs 1 --verbose
   ```

### If Full Experiments Fail
1. **Check SLURM output**:
   ```bash
   cat gradient_alignment_experiments_JOBID.err
   ```

2. **Check intermediate results**:
   ```bash
   cat gradient_experiments_results.json
   ```

3. **Restart from failed experiment**:
   ```bash
   # Edit della_full_gradient_experiments.py to skip completed experiments
   python della_full_gradient_experiments.py
   ```

## 📋 Post-Experiment Analysis

### Step 1: Download Results
```bash
# Download all results back to local
rsync -av mg7411@della-gpu.princeton.edu:/scratch/gpfs/mg7411/ManyPaths/results/grad_align_* ./results/
```

### Step 2: Run Analysis
```bash
# Generate K=1 vs K=10 comparison with gradient alignment data
python k1_vs_k10_comparison.py --k1_results_dir results/grad_align_k1_* --k10_results_dir results/grad_align_k10_*

# Run gradient alignment analysis
python gradient_alignment_analysis.py --base_results_dir results/grad_align_k10_*
```

### Step 3: Create Camera-Ready Figures
```bash
# Generate mechanistic explanation figures
python create_mechanistic_figures.py
```

## 🎉 Camera-Ready Submission Impact

With successful gradient alignment data, we'll have:

1. **Quantitative Evidence**: "K=10 shows 40% higher gradient alignment than K=1"
2. **Mechanistic Explanation**: "Higher gradient alignment correlates with better convergence"
3. **Publication-Quality Figures**: Gradient alignment evolution plots
4. **Robust Statistical Evidence**: Multi-seed, multi-complexity validation

This will significantly strengthen the mechanistic explanations in the camera-ready submission! 