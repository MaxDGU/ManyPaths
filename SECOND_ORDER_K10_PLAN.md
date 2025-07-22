# Second-Order Meta-SGD K=10 Implementation Plan

## 🎯 **Objective**
Add **second-order meta-SGD with K=10 adaptation steps** to complete the experimental matrix for ICML paper.

## 📊 **Current Experimental Matrix Status**
| Algorithm | K=1 | K=10 |
|-----------|-----|------|
| SGD Baseline | ✅ Complete | N/A |
| First-Order Meta-SGD | ✅ Complete | ✅ Complete |
| Second-Order Meta-SGD | ✅ Complete | ❌ **MISSING** |

## 🆕 **What This Adds**
- **Complete 2×2 meta-learning matrix**: First/Second Order × K=1/K=10
- **Computational trade-off analysis**: Second-order K=10 vs First-order K=10
- **Complete Figure 2 results**: All four meta-learning variants

## 📁 **New Files Created**

### 1. `run_concept_10step_second_order.slurm`
**Purpose**: SLURM script for second-order meta-SGD with K=10 adaptation steps
**Configuration**:
- **Experiments**: 9 total (3 features × 3 depths × 1 order)
- **Features**: 8, 16, 32
- **Depths**: 3, 5, 7  
- **Order**: Second-order only (no `--first-order` flag)
- **Adaptation Steps**: 10
- **Time allocation**: 12 hours (increased for second-order complexity)

**Array mapping**:
```bash
# Array index 0-8 covers all combinations:
# Index 0: F8_D3_2ndOrd_K10
# Index 1: F8_D5_2ndOrd_K10  
# Index 2: F8_D7_2ndOrd_K10
# Index 3: F16_D3_2ndOrd_K10
# Index 4: F16_D5_2ndOrd_K10
# Index 5: F16_D7_2ndOrd_K10
# Index 6: F32_D3_2ndOrd_K10
# Index 7: F32_D5_2ndOrd_K10
# Index 8: F32_D7_2ndOrd_K10
```

## 🚀 **Execution Plan**

### Phase 1: Local Testing (Optional)
```bash
# Test one configuration locally
python main.py \
    --experiment concept \
    --m mlp \
    --data-type bits \
    --num-concept-features 8 \
    --pcfg-max-depth 3 \
    --adaptation-steps 10 \
    --epochs 100 \
    --seed 0 \
    --save \
    --no_hyper_search \
    --hyper-index 14
# Note: No --first-order flag (defaults to second-order)
```

### Phase 2: Git Integration
```bash
# Add and commit new files
git add run_concept_10step_second_order.slurm SECOND_ORDER_K10_PLAN.md
git commit -m "Add second-order meta-SGD K=10 experiments

- Complete meta-learning experimental matrix
- Add run_concept_10step_second_order.slurm for systematic evaluation
- Enable comparison of all four meta-learning variants"
git push origin master
```

### Phase 3: Della Execution
```bash
# On della, pull latest changes
cd /scratch/gpfs/mg7411/ManyPaths
git pull origin master

# Submit second-order K=10 experiments
sbatch run_concept_10step_second_order.slurm

# Monitor progress
squeue -u mg7411 | grep k10_2nd
```

## 📈 **Expected Results**

### Filename Pattern
Results will follow the established pattern:
```
concept_mlp_<index>_bits_feats<F>_depth<D>_adapt10_2ndOrd_seed0_trajectory.csv
```

**Examples**:
- `concept_mlp_14_bits_feats8_depth3_adapt10_2ndOrd_seed0_trajectory.csv`
- `concept_mlp_14_bits_feats16_depth3_adapt10_2ndOrd_seed0_trajectory.csv`
- `concept_mlp_14_bits_feats32_depth3_adapt10_2ndOrd_seed0_trajectory.csv`

### Timeline Estimate
- **F8 experiments** (indices 0-2): ~3-4 hours each
- **F16 experiments** (indices 3-5): ~4-5 hours each  
- **F32 experiments** (indices 6-8): ~4-6 hours each
- **Total runtime**: ~35-45 hours (well within 12-hour time allocation per array task)

## 🔍 **Integration with Existing Analysis**

### 1. Aggregate Results Script
The existing `aggregate_seed_results.py` will automatically detect the new second-order K=10 results:
```bash
python aggregate_seed_results.py \
    --experiment_type meta_sgd \
    --features_list 8 16 32 \
    --depths_list 3 5 7 \
    --orders_list 0 1 \
    --adaptation_steps_list 1 10
```

### 2. Analysis Pipeline Integration
Update analysis scripts to include the new variant:
```python
# In analysis scripts, add second-order K=10
EXPERIMENT_VARIANTS = [
    "SGD_baseline",
    "MetaSGD_1stOrd_K1", 
    "MetaSGD_1stOrd_K10",
    "MetaSGD_2ndOrd_K1",
    "MetaSGD_2ndOrd_K10"  # NEW
]
```

### 3. Figure 2 Update
The new results will complete Figure 2 with all four meta-learning variants:
- **First-Order K=1** (existing)
- **First-Order K=10** (existing)  
- **Second-Order K=1** (existing)
- **Second-Order K=10** (new)

## ⚠️ **Important Notes**

### 1. Computational Considerations
- **Second-order K=10 is computationally intensive**: Uses full Hessian-vector products for 10 gradient steps
- **Memory requirements**: May need more GPU memory than first-order variants
- **Runtime**: Expected to be slower than first-order K=10

### 2. Parameter Consistency
All parameters match existing experiments:
- `--hyper-index 14` (same as other experiments)
- `--seed 0` (consistent with single-seed runs)
- `--epochs 1000` (same as other K=10 experiments)
- `--tasks_per_meta_batch 4`
- `--outer_lr 1e-3`, `--inner_lr 1e-2`

### 3. Error Handling
If experiments timeout or fail:
1. Check GPU memory usage in logs
2. Consider reducing batch size or epochs
3. Can run subset of configurations (e.g., F8 and F16 only)

## 📝 **Documentation Updates Needed**

1. **Update RESEARCH_SUMMARY.md** to include second-order K=10
2. **Update paper draft** to mention complete 2×2 matrix
3. **Update analysis README** with new experimental variant

## 🎯 **Success Criteria**
- [ ] All 9 second-order K=10 experiments complete successfully
- [ ] Results integrate cleanly with existing analysis pipeline
- [ ] Figure 2 updated with complete meta-learning comparison
- [ ] Paper strengthened with comprehensive experimental coverage

## 🤝 **Next Steps After Completion**
1. **Run comprehensive analysis** comparing all four meta-learning variants
2. **Update paper figures** with complete experimental matrix
3. **Statistical analysis** of computational trade-offs (second-order K=1 vs first-order K=10)
4. **Write discussion** of when to use each variant

---

**This implementation completes the experimental matrix and provides the missing piece for a comprehensive meta-learning analysis in your ICML paper.** 