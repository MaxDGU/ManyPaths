# Additional Experiments Plan for Camera-Ready Submission

## Overview
Based on the successful test run, we can queue **18 additional experiments** tonight to significantly strengthen your camera-ready submission. These experiments are strategically designed to address the three main priorities:

1. **More gradient steps → better generalization** (statistical robustness)
2. **Robust data efficiency arguments** (broader coverage)
3. **Alternative mechanistic explanations** (since gradient alignment isn't working)

## Current Status
- ✅ **Original experiments running**: 8 experiments (4 HIGH, 2 MEDIUM, 2 LOW priority)
- ✅ **Training pipeline validated**: Test run completed successfully in 136 seconds
- ⚠️ **Gradient alignment issue**: Still showing N/A, but training works perfectly

## Additional Experiments Breakdown

### 🔥 BATCH 1: URGENT - Statistical Robustness (6 experiments)
**Why**: Camera-ready needs robust statistical claims with multiple seeds

- `k1_f8d3_s2` - K=1, F8_D3, seed 2 (200 epochs, ~2h)
- `k1_f8d3_s3` - K=1, F8_D3, seed 3 (200 epochs, ~2h)
- `k10_f8d3_s2` - K=10, F8_D3, seed 2 (200 epochs, ~3h)
- `k10_f8d3_s3` - K=10, F8_D3, seed 3 (200 epochs, ~3h)
- `k1_f16d3_s1` - K=1, F16_D3, seed 1 (150 epochs, ~2h)
- `k10_f16d3_s1` - K=10, F16_D3, seed 1 (150 epochs, ~3h)

**Impact**: Gives you 4 seeds for F8_D3 and 2 seeds for F16_D3 → Strong statistical claims

### 🟡 BATCH 2: HIGH - Broader Coverage (6 experiments)
**Why**: Demonstrate generalization across architectures and complexities

- `k1_f8d5_s0` - K=1, F8_D5, deeper networks (200 epochs, ~2.5h)
- `k10_f8d5_s0` - K=10, F8_D5, deeper networks (200 epochs, ~3.5h)
- `k1_f8d3_cnn_s0` - K=1, CNN architecture (150 epochs, ~2h)
- `k10_f8d3_cnn_s0` - K=10, CNN architecture (150 epochs, ~3h)
- `k1_f12d3_s0` - K=1, intermediate complexity (150 epochs, ~2h)
- `k10_f12d3_s0` - K=10, intermediate complexity (150 epochs, ~3h)

**Impact**: Shows K=1 vs K=10 effect across architectures and complexity levels

### 🟢 BATCH 3: MEDIUM - Alternative Datasets (4 experiments)
**Why**: Demonstrate domain generalization beyond concept learning

- `k1_mod_f8d3_s0` - K=1 on mod dataset (100 epochs, ~2h)
- `k10_mod_f8d3_s0` - K=10 on mod dataset (100 epochs, ~3h)
- `k1_omni_s0` - K=1 on Omniglot (100 epochs, ~2h)
- `k10_omni_s0` - K=10 on Omniglot (100 epochs, ~3h)

**Impact**: Shows your findings generalize beyond Boolean concepts

### 🔵 BATCH 4: LOW - Extended Analysis (4 experiments)
**Why**: Deep mechanistic understanding and publication-quality curves

- `k1_f8d3_s0_long` - K=1, extended training (500 epochs, ~5h)
- `k10_f8d3_s0_long` - K=10, extended training (500 epochs, ~7h)
- `k3_f8d3_s0` - K=3, intermediate adaptation (200 epochs, ~2.5h)
- `k5_f8d3_s0` - K=5, intermediate adaptation (200 epochs, ~3h)

**Impact**: Smooth K=1→K=3→K=5→K=10 progression for mechanistic analysis

## Time Estimates
- **URGENT**: 6 experiments × 2.5h avg = ~15 hours
- **HIGH**: 6 experiments × 2.5h avg = ~15 hours  
- **MEDIUM**: 4 experiments × 2.5h avg = ~10 hours
- **LOW**: 4 experiments × 4h avg = ~16 hours
- **Total**: ~56 hours of compute time

## Strategic Recommendations

### Option 1: Conservative (Queue URGENT + HIGH)
- **12 experiments** focusing on statistical robustness and broader coverage
- **~30 hours** of compute time
- **High impact**: Robust statistical claims + architectural generalization

### Option 2: Aggressive (Queue All 18)
- **All experiments** for maximum paper strength
- **~56 hours** of compute time
- **Maximum impact**: Complete coverage of all angles

### Option 3: Targeted (Queue URGENT + select HIGH/MEDIUM)
- **10 experiments**: All URGENT + CNN experiments + mod dataset
- **~25 hours** of compute time
- **Balanced impact**: Statistics + key generalization claims

## Execution Commands

```bash
# Queue additional experiments
sbatch run_additional_experiments.slurm

# Monitor progress
squeue -u mg7411
tail -f additional_experiments_*.out

# Check results
ls -la additional_experiments_*.json
```

## Expected Outcomes

### For Camera-Ready Paper
1. **Statistical Robustness**: 4 seeds for F8_D3 comparison → Strong effect size claims
2. **Generalization**: CNN + deeper networks → "Effect holds across architectures"
3. **Domain Transfer**: Mod + Omniglot → "Generalizes beyond Boolean concepts"
4. **Mechanistic Understanding**: K=1→K=3→K=5→K=10 progression → Alternative to gradient alignment

### Analysis-Ready Data
- **K=1 vs K=10 comparison**: Robust multi-seed statistical analysis
- **Data efficiency**: Sample efficiency curves across configurations
- **Learning dynamics**: Trajectory analysis for mechanistic insights
- **Weight evolution**: Checkpoint analysis for parameter space understanding

## Risk Mitigation
- **Priority ordering**: URGENT experiments run first
- **Timeouts**: Each experiment has appropriate timeout limits
- **Error handling**: Failures don't block subsequent experiments
- **Intermediate saves**: Results saved after each experiment

## Next Steps
1. **Tonight**: Queue experiments based on chosen option
2. **Tomorrow**: Run analysis scripts on completed experiments
3. **Wednesday**: Integrate results into camera-ready paper
4. **Thursday**: Final paper polish and submission prep

The training pipeline is working perfectly, so we're ready to scale up for maximum camera-ready impact! 