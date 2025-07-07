#!/bin/bash
#SBATCH --job-name=extension_low
#SBATCH --output=extension_low_%j.out
#SBATCH --error=extension_low_%j.err
#SBATCH --time=49:00:00
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu

echo "EXTENSION EXPERIMENTS - LOW PRIORITY"
echo "Host: $(hostname), Working Dir: $(pwd), Date: $(date)"
echo "Total experiments: 14"
echo "Estimated time: 49 hours"

# Load modules
module load anaconda3/2023.3
module load cudatoolkit/11.8

# Activate conda environment
conda activate tensorflow
export CUDA_VISIBLE_DEVICES=0

echo "============================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================"

# Run experiments

echo "🚀 EXPERIMENT 1/14: k1_f8_d7_s1"
echo "   Priority: LOW, Batch: complete_grid"
echo "   Justification: Max depth simple features"
timeout 9600 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 1/14 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 2/14: k10_f8_d7_s1"
echo "   Priority: LOW, Batch: complete_grid"
echo "   Justification: Max depth simple features"
timeout 14400 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 2/14 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 3/14: k1_f16_d7_s1"
echo "   Priority: LOW, Batch: complete_grid"
echo "   Justification: Max depth medium features"
timeout 9600 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 3/14 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 4/14: k10_f16_d7_s1"
echo "   Priority: LOW, Batch: complete_grid"
echo "   Justification: Max depth medium features"
timeout 14400 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 4/14 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 5/14: k1_f32_d7_s1"
echo "   Priority: LOW, Batch: complete_grid"
echo "   Justification: Max complexity"
timeout 9600 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 5/14 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 6/14: k10_f32_d7_s1"
echo "   Priority: LOW, Batch: complete_grid"
echo "   Justification: Max complexity"
timeout 14400 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 6/14 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 7/14: k1_mod_f8d3_s0"
echo "   Priority: LOW, Batch: alternative_domains"
echo "   Justification: Domain transfer"
timeout 7200 python main.py --experiment mod --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 100 --seed 0 --save --no_hyper_search --first-order
echo "   Completed experiment 7/14 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 8/14: k1_mod_f8d3_s1"
echo "   Priority: LOW, Batch: alternative_domains"
echo "   Justification: Domain transfer"
timeout 7200 python main.py --experiment mod --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 8/14 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 9/14: k10_mod_f8d3_s0"
echo "   Priority: LOW, Batch: alternative_domains"
echo "   Justification: Domain transfer"
timeout 10800 python main.py --experiment mod --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 100 --seed 0 --save --no_hyper_search --first-order
echo "   Completed experiment 9/14 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 10/14: k10_mod_f8d3_s1"
echo "   Priority: LOW, Batch: alternative_domains"
echo "   Justification: Domain transfer"
timeout 10800 python main.py --experiment mod --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 10/14 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 11/14: k1_omniglot_f8d3_s0"
echo "   Priority: LOW, Batch: alternative_domains"
echo "   Justification: Visual domain"
timeout 7200 python main.py --experiment omniglot --m cnn --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 100 --seed 0 --save --no_hyper_search --first-order
echo "   Completed experiment 11/14 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 12/14: k1_omniglot_f8d3_s1"
echo "   Priority: LOW, Batch: alternative_domains"
echo "   Justification: Visual domain"
timeout 7200 python main.py --experiment omniglot --m cnn --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 12/14 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 13/14: k10_omniglot_f8d3_s0"
echo "   Priority: LOW, Batch: alternative_domains"
echo "   Justification: Visual domain"
timeout 10800 python main.py --experiment omniglot --m cnn --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 100 --seed 0 --save --no_hyper_search --first-order
echo "   Completed experiment 13/14 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 14/14: k10_omniglot_f8d3_s1"
echo "   Priority: LOW, Batch: alternative_domains"
echo "   Justification: Visual domain"
timeout 10800 python main.py --experiment omniglot --m cnn --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 14/14 at $(date)"
echo "   ----------------------------------------"

echo "============================================"
echo "LOW PRIORITY EXPERIMENTS COMPLETED"
echo "Job finished at: $(date)"
echo "Total experiments run: 14"
echo "============================================"
