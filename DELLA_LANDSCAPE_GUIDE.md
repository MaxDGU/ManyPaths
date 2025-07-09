# Landscape Logging Experiments on Della

## 🎯 **Quick Start**

```bash
# 1. Commit and push current changes
git add .
git commit -m "Add landscape logging with no hyperparameter search (Index 14)"
git push origin master

# 2. SSH to Della
ssh mg7411@della-gpu.princeton.edu

# 3. Navigate to project and pull latest
cd /scratch/gpfs/mg7411/ManyPaths
git pull origin master

# 4. Submit landscape logging job
sbatch run_landscape_logging.slurm
```

## 📊 **Experiment Configuration**

- **Hyperparameters**: Index 14 (32 hidden units, 8 layers) - NO search required
- **Complexity Tiers**: Easy (F8D3), Medium (F16D5), Complex (F32D7)
- **Seeds**: 1, 2, 3 per configuration
- **Total Jobs**: 9 (3 configs × 3 seeds)
- **Runtime**: ~6 hours max per job

## 🔄 **Expected Outputs**

Each job will generate:
- `trajectory_log_{timestamp}.csv` - Real-time Hessian metrics
- `checkpoints/` - Model states every 10 steps
- Meta-SGD vs SGD comparison data

## 📈 **Analysis Pipeline**

After jobs complete:
```bash
# Pull results locally
git pull origin master

# Run analysis
python analysis/loss_landscape_paths.py
```

## 🗂️ **File Structure**

```
results/
├── trajectory_log_concept_mlp_f8d3_meta_seed1.csv
├── trajectory_log_concept_mlp_f8d3_sgd_seed1.csv
└── checkpoints/
    ├── concept_mlp_f8d3_meta_seed1_step10.pt
    └── concept_mlp_f8d3_sgd_seed1_step10.pt
``` 