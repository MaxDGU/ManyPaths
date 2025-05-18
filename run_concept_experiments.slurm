#!/bin/bash

#SBATCH --job-name=concept_meta_learn
#SBATCH --output=slurm_logs/concept_meta_learn_%A_%a.out  # Save STDOUT, %A is Job ID, %a is Array Task ID
#SBATCH --error=slurm_logs/concept_meta_learn_%A_%a.err   # Save STDERR
#SBATCH --partition=fill_this_in          # Specify the partition (queue)
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks-per-node=1               # Number of tasks (usually 1 for non-MPI)
#SBATCH --cpus-per-task=4                 # Number of CPUs per task (adjust as needed)
#SBATCH --mem=8G                          # Memory per node (adjust as needed)
#SBATCH --time=0-04:00:00                 # Max runtime D-HH:MM:SS (e.g., 4 hours)
#SBATCH --array=0-1                       # Array job for two experiments (0 and 1)

# --- User Configuration ---
CONDA_ENV_NAME="tensorflow" # Replace with your conda environment name
PROJECT_DIR="/Users/maxgupta/Desktop/Desktop - Mac/Princeton/CoCoSci_lab/sae-stuff/ManyPaths" # Replace with your project directory on the cluster

# --- SLURM Setup ---
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Starting time: $(date)"
echo "Working directory: $(pwd)"
echo "SLURM jobs_dir: $SLURM_SUBMIT_DIR"

# Create logging directory for SLURM outputs if it doesn't exist
mkdir -p slurm_logs

# Load necessary modules (example, adjust for your cluster)
# module purge # Optional: clear any inherited modules
# module load Python/3.10.8 # Or your specific Python version
# module load CUDA/11.7     # If using GPU, though MLP example is CPU

# Activate Conda environment
# Check if conda is initialized
if [ -z "$CONDA_SHLVL" ]; then
    echo "Conda not initialized. Attempting to initialize..."
    # Replace with the correct path to your conda.sh if different
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        echo "conda.sh not found in anaconda3 or miniconda3. Please check your Conda installation path."
        exit 1
    fi
fi

source activate "$CONDA_ENV_NAME"
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment: $CONDA_ENV_NAME"
    exit 1
fi
echo "Conda environment activated: $CONDA_ENV_NAME"

# Navigate to the project directory
cd "$PROJECT_DIR"
if [ $? -ne 0 ]; then
    echo "Failed to navigate to project directory: $PROJECT_DIR"
    exit 1
fi
echo "Changed directory to $PROJECT_DIR"

# --- Experiment Parameters ---
COMMON_ARGS="--experiment concept \ 
  --m mlp \ 
  --data-type bits \ 
  --num-concept-features 8 \ 
  --adaptation-steps 1 \ 
  --epochs 20000 \ 
  --tasks_per_meta_batch 4 \ 
  --outer_lr 1e-3 \ 
  --inner_lr 1e-2 \ 
  --seed 0 \ 
  --no_hyper_search \ 
  --hyper-index 14 \ 
  --save \ 
  --plot"

# Array task ID will determine which experiment to run
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  EXPERIMENT_NAME="1stOrd_s0_e20k"
  ORDER_FLAG="--first-order"
  echo "Running Experiment: First-Order MetaSGD"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  EXPERIMENT_NAME="2ndOrd_s0_e20k"
  ORDER_FLAG="" # No flag for second-order (default)
  echo "Running Experiment: Second-Order MetaSGD"
else
  echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
  exit 1
fi

LOG_FILE="run_logs/concept_mlp_f8_adapt1_${EXPERIMENT_NAME}.log"
mkdir -p run_logs # Create directory for python script logs

# --- Run Experiment ---
echo "Command: python main.py $COMMON_ARGS $ORDER_FLAG"
echo "Logging to: $LOG_FILE"

python main.py $COMMON_ARGS $ORDER_FLAG > "$LOG_FILE" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "Experiment completed successfully."
else
  echo "Experiment failed with exit code $EXIT_CODE."
fi

echo "Ending time: $(date)"

# Deactivate conda environment (optional)
# conda deactivate

exit $EXIT_CODE 