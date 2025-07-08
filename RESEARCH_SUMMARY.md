# ManyPaths: Research Summary and Repository Guide

## Overview

This repository implements research on **gradient-based meta-learning for concept learning**, investigating how different meta-learning algorithms perform on Boolean concept learning tasks generated using Probabilistic Context-Free Grammars (PCFGs). The work was accepted at the ICML 2024 HilD Workshop on High-Dimensional Learning Dynamics.

**Research Question**: How do meta-learning strategies and standard supervised learning compare in acquiring PCFG-generated Boolean concepts as a function of concept complexity?

---

## Key Experiments

### 1. **Concept Complexity Sweep**
- **Purpose**: Systematic evaluation across feature dimensions and logical depths
- **Configuration**:
  - Features: 8, 16, 32, 64, 128 dimensions
  - PCFG Max Depth: 3, 5, 7 (logical complexity)
  - Algorithms: MetaSGD (1st/2nd order), Baseline SGD
  - Adaptation Steps: K=1, K=10
- **Scripts**: `run_concept_complexity_sweep.slurm`, `run_concept_complexity_sweep_multiseed.slurm`

### 2. **Multi-Seed Robustness**
- **Purpose**: Statistical validation across multiple random seeds
- **Configuration**:
  - Seeds: 0, 1, 2, 3, 4 (5 seeds total)
  - Same complexity parameters as above
  - Consistent hyperparameter settings
- **Scripts**: `run_concept_complexity_sweep_multiseed.slurm`

### 3. **Adaptation Steps Comparison (K=10)**
- **Purpose**: Evaluate impact of increased adaptation steps
- **Configuration**:
  - First-order MetaSGD with K=10 adaptation steps
  - Direct comparison with K=1 variants
- **Scripts**: `run_concept_10step.slurm`

### 4. **Baseline SGD Comparison**
- **Purpose**: Compare meta-learning against standard supervised learning
- **Configuration**:
  - SGD from scratch on same tasks
  - 100 steps per task, learning rate 1e-3
  - 200 tasks evaluated per configuration
- **Scripts**: `run_baseline_sgd_sweep.slurm`, `run_baseline_sgd_sweep_multiseed.slurm`

---

## File Organization

### Core Implementation
```
├── main.py                     # Meta-training pipeline (MetaSGD)
├── main_baseline_sgd.py       # Baseline SGD training
├── training.py                # Meta-training loops with gradient alignment
├── datasets.py                # Custom datasets (MetaBitConceptsDataset)
├── models.py                  # Neural network architectures
├── pcfg.py                    # PCFG concept generation (ATTACHED)
├── constants.py               # Hyperparameter configurations
├── initialization.py          # Model and dataset initialization
├── evaluation.py              # Model evaluation utilities
└── utils.py                   # Helper functions
```

### Analysis and Visualization
```
├── analyze_weights.py                    # Model weight analysis
├── analyze_weight_trajectory_pca.py      # PCA analysis of learning trajectories
├── analyze_checkpoint_drift.py          # Checkpoint analysis
├── compare_final_model_weights.py       # Weight comparison across models
├── compare_avg_model_weights_faceted.py  # Faceted weight analysis
├── aggregate_seed_results.py            # Multi-seed result aggregation
├── plot_learning_curves.py              # Learning curve visualization
├── visualize.py                         # General visualization utilities
└── summarize.py                         # Result summarization
```

### Experiment Scripts
```
├── bash/                      # Local execution scripts
│   ├── concept/              # Concept experiment scripts
│   ├── mod/                  # Modulo experiment scripts
│   └── omniglot/             # Omniglot experiment scripts
├── slurm/                    # SLURM cluster scripts
└── run_*.slurm               # Main experiment SLURM scripts
```

---

## Data Organization

### Generated Datasets
```
├── saved_datasets/           # Cached generated datasets
│   ├── concept_mlp_bits_feats{F}_depth{D}_seed{S}_train_dataset.pt
│   ├── concept_mlp_bits_feats{F}_depth{D}_seed{S}_val_dataset.pt
│   └── concept_mlp_bits_feats{F}_depth{D}_seed{S}_test_dataset.pt
```

### Model Checkpoints
```
├── saved_models/
│   ├── concept_multiseed/    # Multi-seed experiment models
│   ├── baseline_sgd_task_models/  # Baseline SGD models
│   └── checkpoints/          # Periodic training checkpoints
```

### Results and Trajectories
```
├── results/
│   ├── concept_multiseed/    # Multi-seed results
│   ├── baseline_sgd/         # Baseline SGD results
│   └── *_trajectory.csv      # Learning trajectory files
```

### Key Data Formats
- **Trajectory Files**: `{experiment}_{model}_{config}_trajectory.csv`
  - Columns: `log_step`, `val_loss`, `val_accuracy`, `grad_alignment`
- **Model Files**: `{experiment}_{model}_{config}_best_model_at_end_of_train.pt`
- **Checkpoint Files**: `{experiment}_{model}_{config}_epoch_{N}.pt`

---

## Core Modules

### 1. **PCFG Concept Generation (`pcfg.py`)**
- **Purpose**: Generate Boolean concepts of controllable complexity
- **Key Functions**:
  - `sample_concept_from_pcfg()`: Sample concept from grammar
  - `evaluate_pcfg_concept()`: Evaluate concept on input
- **Grammar Rules**:
  - OR: 25%, AND: 25%, NOT: 20%, LITERAL: 30%
- **Complexity Control**: Max depth parameter limits logical nesting

### 2. **Meta-Training (`main.py`, `training.py`)**
- **Purpose**: Implement MetaSGD training with gradient alignment tracking
- **Key Features**:
  - First-order and second-order MetaSGD
  - Configurable adaptation steps (K=1, K=10)
  - Gradient alignment computation
  - Periodic checkpointing
- **Output**: Learning trajectories, model checkpoints, gradient alignments

### 3. **Dataset Generation (`datasets.py`)**
- **Purpose**: Generate meta-learning datasets for concept learning
- **Key Classes**:
  - `MetaBitConceptsDataset`: PCFG-based concept learning tasks
  - `MetaModuloDataset`: Modulo arithmetic tasks
  - `Omniglot`: Few-shot character classification
- **Features**: Task caching, configurable support/query sets

### 4. **Baseline Comparison (`main_baseline_sgd.py`)**
- **Purpose**: Standard SGD baseline for comparison
- **Configuration**: 100 SGD steps per task, 200 tasks evaluated
- **Output**: Per-task learning trajectories for fair comparison

### 5. **Analysis Pipeline**
- **Weight Analysis**: Layer-wise weight norm analysis
- **Trajectory Analysis**: PCA of parameter trajectories
- **Result Aggregation**: Multi-seed statistical analysis
- **Visualization**: Learning curves, performance comparisons

---

## Key Experimental Configurations

### Meta-Learning Settings
```python
# Default MetaSGD Configuration
epochs = 1000                    # Meta-training episodes
tasks_per_meta_batch = 4        # Tasks per meta-batch
adaptation_steps = 1 or 10      # Inner-loop steps
outer_lr = 1e-3                 # Meta-optimizer learning rate
inner_lr = 1e-2                 # Task-specific learning rate
```

### Concept Complexity Parameters
```python
# PCFG Configuration
num_concept_features = [8, 16, 32, 64, 128]  # Feature dimensions
pcfg_max_depth = [3, 5, 7]                   # Logical depth
grammar_probs = {
    'OR': 0.25, 'AND': 0.25, 
    'NOT': 0.2, 'LITERAL': 0.3
}
```

### Model Architecture
```python
# MLP Configuration (from constants.py)
MLP_PARAMS = [
    (8, 2),   # (hidden_units, num_layers)
    (16, 2),  # Index 0-15 for hyperparameter search
    (32, 2),
    (64, 2),
    # ... up to (64, 8)
]
DEFAULT_INDEX = 7  # (64, 2) configuration
```

---

## Key Findings Summary

### 1. **Meta-Learning Superiority**
- **Result**: MetaSGD variants consistently outperform SGD baselines
- **Magnitude**: Orders of magnitude fewer samples needed to reach threshold performance
- **Consistency**: Advantage holds across all complexity levels tested

### 2. **First-Order vs. Second-Order Trade-offs**
- **Key Finding**: First-order MetaSGD with K=10 steps often matches/exceeds second-order with K=1
- **Implication**: Computational efficiency can be achieved without sacrificing performance
- **Practical Impact**: Enables scaling to larger problems

### 3. **Complexity Scaling Behavior**
- **Feature Dimension**: Performance degrades gracefully with increased dimensions
- **Logical Depth**: Deeper logical structures are harder but meta-learning maintains advantages
- **Sweet Spot**: 8-16 features with moderate depth (3-5) show optimal results

### 4. **Learning Dynamics Insights**
- **Gradient Alignment**: Meta-gradients show meaningful alignment with task-specific gradients
- **Trajectory Analysis**: Meta-learning finds fundamentally different optimization paths
- **Convergence**: Faster convergence to higher-quality solutions

### 5. **Data Efficiency**
- **Threshold Analysis**: Meta-learning reaches 60% accuracy with dramatically fewer samples
- **Sample Complexity**: Exponential improvements over baseline SGD
- **Practical Significance**: Enables few-shot learning in resource-constrained settings

---

## Usage Guide

### Running Core Experiments
```bash
# Single concept experiment
python main.py --experiment concept --m mlp --data-type bits \
    --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 1 \
    --first-order --save

# Baseline SGD comparison
python main_baseline_sgd.py --num-concept-features 8 --pcfg-max-depth 3 \
    --num-tasks-to-evaluate 200 --num-sgd-steps-per-task 100

# Multi-seed cluster experiment
sbatch run_concept_complexity_sweep_multiseed.slurm
```

### Analyzing Results
```bash
# Aggregate multi-seed results
python aggregate_seed_results.py --experiment_type metasgd \
    --features_list 8 16 32 --depths_list 3 5 7

# Weight trajectory analysis
python analyze_weight_trajectory_pca.py --checkpoint_dir saved_models/checkpoints/ \
    --run_identifier_prefix concept_mlp_14_bits_feats8_depth3_adapt1_1stOrd_seed0
```

---

## Research Impact

This work provides:
1. **Empirical evidence** for meta-learning's advantages in structured concept learning
2. **Practical insights** about first-order vs. second-order trade-offs
3. **Methodological contributions** using PCFGs for controllable concept generation
4. **Analysis tools** for understanding meta-learning dynamics

The research demonstrates that meta-learning can effectively acquire complex logical concepts from few examples, with significant implications for few-shot learning and concept discovery in AI systems. 