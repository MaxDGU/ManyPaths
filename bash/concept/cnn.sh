#!/bin/bash

SEED=${1:-0}

# image
sbatch --export=SEED=$SEED,DATA_TYPE=image slurm/concept/cnn.slurm;