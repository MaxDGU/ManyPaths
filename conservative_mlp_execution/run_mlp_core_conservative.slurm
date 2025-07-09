#!/bin/bash
#SBATCH --job-name=mlp_core_grid
#SBATCH --output=mlp_core_grid_%j.out
#SBATCH --error=mlp_core_grid_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu

echo "MLP GRID COMPLETION - P1_CORE"
echo "Conservative timing model (sublinear scaling)"
echo "Host: $(hostname), Date: $(date)"
echo "Experiments: 12, Allocated: 48h, Est: 537.2h"
echo "Configs: F16_D3, F32_D3"

module load anaconda3/2023.3
module load cudatoolkit/11.8
conda activate tensorflow
export CUDA_VISIBLE_DEVICES=0

echo "============================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID, Node: $(hostname)"
echo "============================================"


echo "Experiment 1/12: F16_D3_K1_S1"
echo "  Est: 27.3h, Timeout: 98250s"
echo "  Started: $(date)"

timeout 98250 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 27.3h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 2/12: F16_D3_K1_S2"
echo "  Est: 27.3h, Timeout: 98250s"
echo "  Started: $(date)"

timeout 98250 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 27.3h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 3/12: F16_D3_K1_S3"
echo "  Est: 27.3h, Timeout: 98250s"
echo "  Started: $(date)"

timeout 98250 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 27.3h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 4/12: F16_D3_K10_S1"
echo "  Est: 40.9h, Timeout: 147375s"
echo "  Started: $(date)"

timeout 147375 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 40.9h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 5/12: F16_D3_K10_S2"
echo "  Est: 40.9h, Timeout: 147375s"
echo "  Started: $(date)"

timeout 147375 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 40.9h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 6/12: F16_D3_K10_S3"
echo "  Est: 40.9h, Timeout: 147375s"
echo "  Started: $(date)"

timeout 147375 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 40.9h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 7/12: F32_D3_K1_S1"
echo "  Est: 44.3h, Timeout: 159607s"
echo "  Started: $(date)"

timeout 159607 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 44.3h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 8/12: F32_D3_K1_S2"
echo "  Est: 44.3h, Timeout: 159607s"
echo "  Started: $(date)"

timeout 159607 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 44.3h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 9/12: F32_D3_K1_S3"
echo "  Est: 44.3h, Timeout: 159607s"
echo "  Started: $(date)"

timeout 159607 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 44.3h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 10/12: F32_D3_K10_S1"
echo "  Est: 66.5h, Timeout: 239411s"
echo "  Started: $(date)"

timeout 239411 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 66.5h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 11/12: F32_D3_K10_S2"
echo "  Est: 66.5h, Timeout: 239411s"
echo "  Started: $(date)"

timeout 239411 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 66.5h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 12/12: F32_D3_K10_S3"
echo "  Est: 66.5h, Timeout: 239411s"
echo "  Started: $(date)"

timeout 239411 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 66.5h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "============================================"
echo "P1_CORE COMPLETED at $(date)"
echo "Time allocation: 48h, Estimated: 537.2h"
echo "============================================"
