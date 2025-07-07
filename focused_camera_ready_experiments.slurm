#!/bin/bash
#SBATCH --job-name=camera_ready_focused
#SBATCH --output=camera_ready_focused_%j.out
#SBATCH --error=camera_ready_focused_%j.err
#SBATCH --time=96:00:00
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu

echo "🚀 CAMERA-READY FOCUSED EXPERIMENTS"
echo "Core complexity scaling: F8D3, F16D3, F32D3"
echo "K=1 vs K=10, 3 seeds each = 18 total experiments"
echo "Host: $(hostname), Date: $(date)"

module load anaconda3/2023.3
module load cudatoolkit/11.8
conda activate tensorflow
export CUDA_VISIBLE_DEVICES=0

echo "============================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID, Node: $(hostname)"
echo "============================================"

# F8D3 Experiments (6 total)
echo "🧠 STARTING F8D3 EXPERIMENTS"

echo "F8D3_K1_S1 - Est: 18h"
timeout 72000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 1 --save --no_hyper_search --first-order
echo "F8D3_K1_S1 completed at $(date)"

echo "F8D3_K1_S2 - Est: 18h"
timeout 72000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 2 --save --no_hyper_search --first-order
echo "F8D3_K1_S2 completed at $(date)"

echo "F8D3_K1_S3 - Est: 18h"
timeout 72000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 3 --save --no_hyper_search --first-order
echo "F8D3_K1_S3 completed at $(date)"

echo "F8D3_K10_S1 - Est: 28h"
timeout 108000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 1 --save --no_hyper_search --first-order
echo "F8D3_K10_S1 completed at $(date)"

echo "F8D3_K10_S2 - Est: 28h"
timeout 108000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 2 --save --no_hyper_search --first-order
echo "F8D3_K10_S2 completed at $(date)"

echo "F8D3_K10_S3 - Est: 28h"
timeout 108000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 3 --save --no_hyper_search --first-order
echo "F8D3_K10_S3 completed at $(date)"

echo "✅ F8D3 EXPERIMENTS COMPLETED"

# F16D3 Experiments (6 total)
echo "🧠 STARTING F16D3 EXPERIMENTS"

echo "F16D3_K1_S1 - Est: 36h"
timeout 144000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 1 --save --no_hyper_search --first-order
echo "F16D3_K1_S1 completed at $(date)"

echo "F16D3_K1_S2 - Est: 36h"
timeout 144000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 2 --save --no_hyper_search --first-order
echo "F16D3_K1_S2 completed at $(date)"

echo "F16D3_K1_S3 - Est: 36h"
timeout 144000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 3 --save --no_hyper_search --first-order
echo "F16D3_K1_S3 completed at $(date)"

echo "F16D3_K10_S1 - Est: 56h"
timeout 216000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 1 --save --no_hyper_search --first-order
echo "F16D3_K10_S1 completed at $(date)"

echo "F16D3_K10_S2 - Est: 56h"
timeout 216000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 2 --save --no_hyper_search --first-order
echo "F16D3_K10_S2 completed at $(date)"

echo "F16D3_K10_S3 - Est: 56h"
timeout 216000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 3 --save --no_hyper_search --first-order
echo "F16D3_K10_S3 completed at $(date)"

echo "✅ F16D3 EXPERIMENTS COMPLETED"

# F32D3 Experiments (6 total)
echo "🧠 STARTING F32D3 EXPERIMENTS"

echo "F32D3_K1_S1 - Est: 72h"
timeout 288000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 1 --save --no_hyper_search --first-order
echo "F32D3_K1_S1 completed at $(date)"

echo "F32D3_K1_S2 - Est: 72h"
timeout 288000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 2 --save --no_hyper_search --first-order
echo "F32D3_K1_S2 completed at $(date)"

echo "F32D3_K1_S3 - Est: 72h"
timeout 288000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 3 --save --no_hyper_search --first-order
echo "F32D3_K1_S3 completed at $(date)"

echo "F32D3_K10_S1 - Est: 112h (may timeout)"
timeout 288000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 1 --save --no_hyper_search --first-order
echo "F32D3_K10_S1 completed at $(date)"

echo "F32D3_K10_S2 - Est: 112h (may timeout)"
timeout 288000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 2 --save --no_hyper_search --first-order
echo "F32D3_K10_S2 completed at $(date)"

echo "F32D3_K10_S3 - Est: 112h (may timeout)"
timeout 288000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 3 --save --no_hyper_search --first-order
echo "F32D3_K10_S3 completed at $(date)"

echo "✅ F32D3 EXPERIMENTS COMPLETED"

echo "============================================"
echo "🎉 CAMERA-READY FOCUSED EXPERIMENTS COMPLETED"
echo "Completed at: $(date)"
echo "Total experiments: 18"
echo "============================================" 