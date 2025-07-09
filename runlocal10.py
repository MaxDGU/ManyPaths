import subprocess
import os

# --- Configuration ---
FEATURES_LIST = [32]
DEPTHS_LIST = [3, 5, 7]
# ORDERS_LIST = [1] # Forcing 1st Order for these K=10 runs

ADAPTATION_STEPS_K10 = 10
CURRENT_ORDER_VAL_1ST = 1 # 1 for 1st Order

SEED_VAL = 1 # Assuming you want to run for seed 0, adjust if running other/multiple seeds locally
HYPER_INDEX_VAL = 14
TASKS_PER_META_BATCH = 4
OUTER_LR = 1e-3
INNER_LR = 1e-2
EPOCHS_VAL = 100  # Same as your SLURM for consistency
CHECKPOINT_INTERVAL_VAL = 4
SAVE_FLAG_STR = "--save"
FIRST_ORDER_FLAG_STR = "--first-order" # Since we are doing 1st order

# --- Output Directories (ensure they match where main.py saves things) ---
# These are based on defaults in main.py. If you override them in main.py calls, adjust here too.
RESULTS_DIR = "results/concept_multiseed" # For trajectory CSVs
SAVED_MODELS_DIR = "saved_models"           # For final models
CHECKPOINT_SUBDIR = "checkpoints"           # Subdir within saved_models_dir for .pt checkpoints
SAVED_DATASETS_DIR = "saved_datasets"       # For generated .pt datasets

# Ensure these directories exist, similar to how main.py might create them
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVED_MODELS_DIR, CHECKPOINT_SUBDIR), exist_ok=True)
os.makedirs(SAVED_DATASETS_DIR, exist_ok=True)

# --- Iterate and Run ---
for features in FEATURES_LIST:
    for depth in DEPTHS_LIST:
        # For 1st Order with K=10
        order_str_for_filename = "1stOrd" # From main.py logic

        print(f"--- Starting Local Run: Features={features}, Depth={depth}, Order={order_str_for_filename}, K={ADAPTATION_STEPS_K10}, Seed={SEED_VAL} ---")

        command = [
            "python", "main.py",
            "--experiment", "concept",
            "--m", "mlp",
            "--data-type", "bits",
            "--num-concept-features", str(features),
            "--pcfg-max-depth", str(depth),
            "--adaptation-steps", str(ADAPTATION_STEPS_K10),
            FIRST_ORDER_FLAG_STR, # Adds --first-order
            "--epochs", str(EPOCHS_VAL),
            "--checkpoint-interval", str(CHECKPOINT_INTERVAL_VAL),
            "--tasks_per_meta_batch", str(TASKS_PER_META_BATCH),
            "--outer_lr", str(OUTER_LR),
            "--inner_lr", str(INNER_LR),
            SAVE_FLAG_STR, # Adds --save
            "--seed", str(SEED_VAL),
            "--no_hyper_search",
            "--hyper-index", str(HYPER_INDEX_VAL),
            # Pass the directory arguments explicitly to ensure consistency
            "--results_dir", RESULTS_DIR,
            "--saved_models_dir", SAVED_MODELS_DIR,
            "--checkpoint_subdir", CHECKPOINT_SUBDIR,
            "--saved_datasets_dir", SAVED_DATASETS_DIR
            # Note: --patience is high by default in main.py, effectively disabling it
        ]

        print(f"Executing: {' '.join(command)}")
        
        try:
            # Using shell=False is generally safer, command list is already prepared.
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Stream output
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            stderr_output = process.stderr.read()
            if stderr_output:
                print("--- STDERR ---")
                print(stderr_output.strip())
                print("--------------")

            process.wait() # Wait for the process to complete

            if process.returncode == 0:
                print(f"Successfully completed: Features={features}, Depth={depth}, K={ADAPTATION_STEPS_K10}")
            else:
                print(f"ERROR: Run failed with exit code {process.returncode} for Features={features}, Depth={depth}, K={ADAPTATION_STEPS_K10}")
        
        except FileNotFoundError:
            print(f"ERROR: main.py not found. Make sure you are in the ManyPaths directory and your environment is active.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
        print("-" * 50)

print("All local K=10 (1st Order) runs attempted.")
