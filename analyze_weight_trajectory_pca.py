import torch
import os
import re
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Attempt to import from constants, but provide fallbacks if not found or running standalone
try:
    from constants import CHECKPOINT_SUBDIR, PCFG_DEFAULT_MAX_DEPTH
except ImportError:
    print("Warning: Could not import from constants.py. Using fallback default values.")
    CHECKPOINT_SUBDIR = "checkpoints"
    PCFG_DEFAULT_MAX_DEPTH = 5

if 'CHECKPOINT_SUBDIR' not in globals():
    CHECKPOINT_SUBDIR = "checkpoints"
if 'PCFG_DEFAULT_MAX_DEPTH' not in globals():
    PCFG_DEFAULT_MAX_DEPTH = 5

# --- Model Definition (copied from analyze_weights.py for standalone use if needed) ---
# This is crucial for when we might need to load a model to verify architecture, 
# even if get_model_param_vec works directly on state_dict for PCA.
class MLP(torch.nn.Module):
    def __init__(
        self,
        n_input: int = 32 * 32, 
        n_output: int = 1,
        n_hidden: int = 64, 
        n_layers: int = 8,  
        n_input_channels: int = 1, 
    ):
        super().__init__()
        layers = []
        if n_input < 64:
            layers.extend([
                torch.nn.Linear(n_input, 32 * 32 * n_input_channels),
                torch.nn.BatchNorm1d(32 * 32 * n_input_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(32 * 32 * n_input_channels, n_hidden),
                torch.nn.BatchNorm1d(n_hidden),
            ])
        else:
            layers.extend([
                torch.nn.Linear(n_input, n_hidden),
                torch.nn.BatchNorm1d(n_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.BatchNorm1d(n_hidden),
            ])
        layers.append(torch.nn.ReLU())
        for _ in range(n_layers - 2):
            layers.extend([
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.BatchNorm1d(n_hidden),
                torch.nn.ReLU(),
            ])
        layers.append(torch.nn.Linear(n_hidden, n_output))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
# --- End Model Definition ---

def get_model_param_vec(model_state_dict):
    '''Extracts and flattens all parameters from a model\'s state_dict.'''
    param_tensors = []
    # print(f"  [get_model_param_vec] Processing state_dict with keys: {list(model_state_dict.keys())}") # Verbose
    extracted_keys = []
    # Prefer 'module.' prefix if present (typical for l2l.algorithms.MetaSGD)
    module_params_found = False
    for key, param in model_state_dict.items():
        if key.startswith("module."): 
            if torch.is_tensor(param):
                param_tensors.append(param.cpu().view(-1))
                extracted_keys.append(key)
                module_params_found = True
    
    if not module_params_found:
        # print("  [get_model_param_vec] No params found with 'module.' prefix. Extracting all tensors.") # Verbose
        for key, param in model_state_dict.items():
            if torch.is_tensor(param):
                 param_tensors.append(param.cpu().view(-1))
                 extracted_keys.append(key)
    
    if not param_tensors:
        print("  [get_model_param_vec] No parameters found in state_dict to create a vector.")
        raise ValueError("No parameters found in state_dict to create a vector.")
    
    # print(f"  [get_model_param_vec] Extracted parameters from keys: {extracted_keys}") # Verbose
    flat_params = torch.cat(param_tensors)
    # print(f"  [get_model_param_vec] Total elements in flat_params: {flat_params.nelement()}") # Verbose
    return flat_params

def analyze_pca(args):
    '''Analyzes parameter trajectories using PCA.'''
    file_prefix_parts = []
    if args.file_prefix:
        prefix_to_match = args.file_prefix
        print(f"Using provided file prefix: {prefix_to_match}")
    else:
        file_prefix_parts.extend([args.experiment, args.model_type])
        if args.index is None:
            print("Warning: --index not provided. Filename matching might be incomplete if index is part of it.")
            # Attempt to create a sensible default or allow wildcard later if needed
            # For now, let's assume a common index part if not specified for some experiments
            index_str = str(args.index_default) # Default index_str, e.g., "0" or "14"
        else:
            index_str = str(args.index)
        file_prefix_parts.append(index_str)
        file_prefix_parts.append(args.data_type)

        if args.experiment == "concept":
            file_prefix_parts.extend([
                f"feats{args.features}",
                f"depth{args.depth}",
                f"adapt{args.adapt_steps}",
                "1stOrd" if args.first_order else "2ndOrd"
            ])
        elif args.experiment == "mod":
            file_prefix_parts.append(str(args.skip))
        
        file_prefix_parts.append(f"seed{args.seed}")
        prefix_to_match = "_".join(filter(None, file_prefix_parts)) # filter(None,...) removes potential None if index was optional and not given
        print(f"Constructed file prefix for matching: {prefix_to_match}")

    checkpoints_dir = os.path.join(args.saved_models_root, CHECKPOINT_SUBDIR)
    if not os.path.isdir(checkpoints_dir):
        print(f"Error: Checkpoints directory not found: {checkpoints_dir}")
        return

    print(f"Searching for checkpoints in: {checkpoints_dir} with prefix pattern: {prefix_to_match}..._epoch_*.pt")
    
    checkpoint_files_info = []
    # Regex to capture epoch number: prefix_epoch_NUM.pt
    # Example: concept_mlp_0_bits_feats8_depth3_adapt1_2ndOrd_seed0_epoch_10.pt
    regex_pattern = re.compile(re.escape(prefix_to_match) + r"_epoch_(\d+)\.pt$")

    for f_name in os.listdir(checkpoints_dir):
        match = regex_pattern.match(f_name)
        if match:
            epoch_num = int(match.group(1))
            checkpoint_files_info.append({"epoch": epoch_num, "path": os.path.join(checkpoints_dir, f_name), "filename": f_name})
    
    if not checkpoint_files_info:
        print(f"No checkpoint files found matching pattern '{prefix_to_match}_epoch_*.pt' in {checkpoints_dir}")
        print(f"Please check your prefix construction and file naming convention.")
        return

    checkpoint_files_info.sort(key=lambda x: x["epoch"])
    print(f"Found {len(checkpoint_files_info)} checkpoints.")
    # for cp_info in checkpoint_files_info: print(f"  - {cp_info['filename']}") # Verbose listing

    param_vectors = []
    epochs_loaded = []
    first_vec_size = None

    for cp_info in checkpoint_files_info:
        try:
            # print(f"Loading checkpoint: {cp_info['path']} (epoch {cp_info['epoch']})") # Verbose
            state_dict = torch.load(cp_info["path"], map_location=torch.device('cpu'))
            param_vec = get_model_param_vec(state_dict)
            
            if first_vec_size is None:
                first_vec_size = param_vec.nelement()
            elif param_vec.nelement() != first_vec_size:
                print(f"Warning: Parameter vector size mismatch for {cp_info['filename']} ({param_vec.nelement()}) vs initial ({first_vec_size}). Skipping.")
                continue

            param_vectors.append(param_vec.numpy()) # PCA expects NumPy arrays
            epochs_loaded.append(cp_info["epoch"])
        except Exception as e:
            print(f"Error loading or processing checkpoint {cp_info['path']}: {e}. Skipping.")
            continue
            
    if len(param_vectors) < 2:
        print("Not enough valid parameter vectors (need at least 2) to perform PCA.")
        return
    
    X = np.array(param_vectors)
    print(f"Created data matrix X with shape: {X.shape}") # (num_checkpoints, num_parameters)

    # Standardize the data before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Determine actual model parameters for potential verification ---
    # This part is tricky because args for MLP are tied to filename conventions
    # which are now abstracted by --checkpoint_dir and --run_identifier_prefix.
    # For now, we rely on param_vectors having consistent size.
    # If strict architectural verification is needed here, we'd need to parse relevant
    # parts (feats, depth, etc.) from run_identifier_prefix or checkpoint_dir path.
    
    # Example: if run_identifier_prefix is concept_mlp_14_bits_feats32_depth7_adapt1_1stOrd_seed0
    # n_input = 32 (from feats32)
    # n_layers = 8 (depth7 implies 8 linear layers for this MLP structure)
    # n_hidden = 32 (deduced from previous analysis of similar files)
    # n_input_channels = 3 (deduced from previous analysis)
    # This is illustrative; we are NOT using it to load a model yet, just for context.
    # model_args_for_verification = {
    #     'n_input': 32, 'n_layers': 8, 'n_hidden': 32, 
    #     'n_input_channels': 3, 'n_output': 1
    # }
    # verify_model = MLP(**model_args_for_verification)
    # expected_param_count = sum(p.numel() for p in verify_model.parameters())
    # if first_vec_size != expected_param_count:
    #     print(f"CRITICAL WARNING: Loaded parameter vectors (size {first_vec_size}) "
    #           f"do not match expected parameter count ({expected_param_count}) for deduced architecture.")
    #     print("PCA results might be misleading if architecture is inconsistent across checkpoints.")
    # --- End verification idea ---

    # Perform PCA
    # Limit n_components to be min(n_samples, n_features) and also not too large for plotting
    max_pca_components = min(X_scaled.shape[0], X_scaled.shape[1], args.pca_components) 
    if max_pca_components < 2:
        print(f"Cannot perform PCA with less than 2 components. Max components possible: {max_pca_components}")
        return
        
    pca = PCA(n_components=max_pca_components)
    X_pca = pca.fit_transform(X_scaled)

    print(f"PCA performed. Explained variance ratio by component: {pca.explained_variance_ratio_}")
    print(f"Shape of data after PCA: {X_pca.shape}")

    # Plotting
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=epochs_loaded, cmap='viridis', alpha=0.7)
    plt.title(f"PCA of Weight Space Trajectory (First 2 Components)\nRun: {prefix_to_match}")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
    
    # Add lines connecting points in order of epoch
    for i in range(len(epochs_loaded) - 1):
        plt.plot([X_pca[i, 0], X_pca[i+1, 0]], [X_pca[i, 1], X_pca[i+1, 1]], 
                 color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    cbar = plt.colorbar(scatter, label='Epoch Number')
    plt.grid(True)
    
    # Save the plot
    plot_out_dir = os.path.join(args.results_basedir, f"{args.run_name}_pca_plots") 
    os.makedirs(plot_out_dir, exist_ok=True)
    plot_filename = os.path.join(plot_out_dir, f"pca_trajectory_{prefix_to_match}.png")
    plt.savefig(plot_filename)
    print(f"Saved PCA trajectory plot to: {plot_filename}")
    plt.show()

    if max_pca_components >= 3:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter_3d = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=epochs_loaded, cmap='viridis', alpha=0.7)
        ax.set_title(f"PCA of Weight Space Trajectory (First 3 Components)\nRun: {prefix_to_match}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.2f}%)")
        
        # Add lines connecting points in order of epoch for 3D plot
        for i in range(len(epochs_loaded) - 1):
            ax.plot([X_pca[i, 0], X_pca[i+1, 0]], [X_pca[i, 1], X_pca[i+1, 1]], [X_pca[i, 2], X_pca[i+1, 2]],
                    color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        cbar_3d = fig.colorbar(scatter_3d, label='Epoch Number', ax=ax, pad=0.1)
        plot_filename_3d = os.path.join(plot_out_dir, f"pca_trajectory_3D_{prefix_to_match}.png")
        plt.savefig(plot_filename_3d)
        print(f"Saved 3D PCA trajectory plot to: {plot_filename_3d}")
        plt.show()

def analyze_pca_v2(checkpoints_dir, run_identifier_prefix, run_name_for_output, results_basedir, pca_components_arg):
    print(f"Starting PCA analysis for run: {run_identifier_prefix}")
    print(f"Searching for checkpoints in: {checkpoints_dir}")
    print(f"Matching prefix: {run_identifier_prefix}_epoch_*.pt")
    
    if not os.path.isdir(checkpoints_dir):
        print(f"Error: Checkpoints directory not found: {checkpoints_dir}")
        return

    checkpoint_files_info = []
    regex_pattern = re.compile(re.escape(run_identifier_prefix) + r"_epoch_(\d+)\.pt$")

    for f_name in os.listdir(checkpoints_dir):
        match = regex_pattern.match(f_name)
        if match:
            epoch_num = int(match.group(1))
            checkpoint_files_info.append({"epoch": epoch_num, "path": os.path.join(checkpoints_dir, f_name), "filename": f_name})
    
    if not checkpoint_files_info:
        print(f"No checkpoint files found matching pattern '{run_identifier_prefix}_epoch_*.pt' in {checkpoints_dir}")
        return

    checkpoint_files_info.sort(key=lambda x: x["epoch"])
    print(f"Found {len(checkpoint_files_info)} checkpoints.")

    param_vectors = []
    epochs_loaded = []
    first_vec_size = None

    for cp_info in checkpoint_files_info:
        try:
            state_dict = torch.load(cp_info["path"], map_location=torch.device('cpu'))
            param_vec = get_model_param_vec(state_dict) # Uses existing function
            
            if first_vec_size is None:
                first_vec_size = param_vec.nelement()
            elif param_vec.nelement() != first_vec_size:
                print(f"Warning: Parameter vector size mismatch for {cp_info['filename']} ({param_vec.nelement()}) vs initial ({first_vec_size}). Skipping.")
                continue

            param_vectors.append(param_vec.numpy())
            epochs_loaded.append(cp_info["epoch"])
        except Exception as e:
            print(f"Error loading or processing checkpoint {cp_info['path']}: {e}. Skipping.")
            continue
            
    if len(param_vectors) < 2:
        print("Not enough valid parameter vectors (need at least 2) to perform PCA.")
        return
    
    X = np.array(param_vectors)
    print(f"Created data matrix X with shape: {X.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA (reuse existing logic)
    max_pca_components = min(X_scaled.shape[0], X_scaled.shape[1], pca_components_arg)
    if max_pca_components < 2:
        print(f"Cannot perform PCA with less than 2 components. Max components possible: {max_pca_components}")
        return
        
    pca = PCA(n_components=max_pca_components)
    X_pca = pca.fit_transform(X_scaled)

    print(f"PCA performed. Explained variance ratio by component: {pca.explained_variance_ratio_}")
    print(f"Shape of data after PCA: {X_pca.shape}")

    # Plotting (reuse existing logic, adapt titles/paths)
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=epochs_loaded, cmap='viridis', alpha=0.7)
    plt.title(f"PCA of Weight Space Trajectory (First 2 Components)\nRun ID: {run_identifier_prefix}")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
    
    if X_pca.shape[0] > 1: # Ensure there is more than one point to draw lines
        for i in range(len(epochs_loaded) - 1):
            plt.plot([X_pca[i, 0], X_pca[i+1, 0]], [X_pca[i, 1], X_pca[i+1, 1]], 
                     color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.colorbar(scatter, label='Epoch Number')
    plt.grid(True)
    
    plot_out_dir = os.path.join(results_basedir, f"{run_name_for_output}_plots")
    os.makedirs(plot_out_dir, exist_ok=True)
    plot_filename = os.path.join(plot_out_dir, f"pca_trajectory_2D_{run_identifier_prefix}.png")
    plt.savefig(plot_filename)
    print(f"Saved 2D PCA trajectory plot to: {plot_filename}")
    plt.show()

    if max_pca_components >= 3:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter_3d = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=epochs_loaded, cmap='viridis', alpha=0.7)
        ax.set_title(f"PCA of Weight Space Trajectory (First 3 Components)\nRun ID: {run_identifier_prefix}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.2f}%)")
        
        if X_pca.shape[0] > 1: # Ensure there is more than one point to draw lines
            for i in range(len(epochs_loaded) - 1):
                ax.plot([X_pca[i, 0], X_pca[i+1, 0]], [X_pca[i, 1], X_pca[i+1, 1]], [X_pca[i, 2], X_pca[i+1, 2]],
                        color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        cbar_3d = fig.colorbar(scatter_3d, label='Epoch Number', ax=ax, pad=0.1)
        plot_filename_3d = os.path.join(plot_out_dir, f"pca_trajectory_3D_{run_identifier_prefix}.png")
        plt.savefig(plot_filename_3d)
        print(f"Saved 3D PCA trajectory plot to: {plot_filename_3d}")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze parameter trajectory using PCA from model checkpoints.")
    
    # Simplified arguments: point to a directory of checkpoints for a single run/seed
    parser.add_argument("--checkpoint_dir", type=str, required=True, 
                        help="Directory containing checkpoint files (e.g., .../featsX_depthY/seedZ/).")
    parser.add_argument("--run_identifier_prefix", type=str, required=True,
                        help="The common prefix for checkpoint files within the directory, up to _epoch_ (e.g., 'concept_mlp_14_bits_feats32_depth7_adapt1_1stOrd_seed0').")
    parser.add_argument("--run-name", type=str, default="pca_analysis", help="Name for the run to create specific plot subdirectories (often derived from checkpoint_dir path). You can override for custom naming.")
    parser.add_argument("--results-basedir", type=str, default="results", help="Base directory where plot subdirectories will be created.")
    parser.add_argument("--pca-components", type=int, default=3, help="Number of principal components to compute (at least 2 for 2D plot, 3 for 3D plot).")
    # parser.add_argument("--saved_models_root", type=str, default="saved_models", help="Root directory where models and checkpoints are saved.")
    # parser.add_argument("--file-prefix", type=str, default=None, help="Exact file prefix to match checkpoints (e.g., concept_mlp_0_bits_feats8_depth3_adapt1_2ndOrd_seed0). Overrides other specific args if provided.")
    # parser.add_argument("--experiment", type=str, default="concept", choices=["mod", "concept"], help="Experiment type.")
    # parser.add_argument("--model-type", "--m", type=str, default="mlp", dest="model_type", help="Model type (e.g., mlp).")
    # parser.add_argument("--data-type", type=str, default="bits", help="Data type (e.g., bits).") # Added for filename construction
    # parser.add_argument("--seed", type=int, default=0, help="Random seed used for the run.")
    # parser.add_argument("--index", type=int, default=None, help="Hyperparameter index used in the filename (e.g., the '0' in concept_mlp_0_bits...). If None, uses --index-default.")
    # parser.add_argument("--index-default", type=int, default=0, help="Default hyperparameter index if --index is not provided.")
    # parser.add_argument("--features", type=int, default=8, dest="num_concept_features", help="Number of features (for concept).")
    # parser.add_argument("--depth", type=int, default=PCFG_DEFAULT_MAX_DEPTH, dest="pcfg_max_depth", help="Max depth (for concept).")
    # parser.add_argument("--adapt-steps", type=int, default=1, help="Adaptation steps (for concept).")
    # parser.add_argument("--first-order", action="store_true", help="If the run used first-order MAML.")
    # parser.add_argument("--skip", type=int, default=1, help="Skip value (for mod experiment).")


    args = parser.parse_args()

    # --- Construct prefix_to_match (now called full_prefix_pattern) and adapt analyze_pca --- 
    # The old analyze_pca took a constructed prefix. Now it will take checkpoint_dir and run_identifier_prefix.
    
    # Modify analyze_pca to accept these new args or derive them.
    # For simplicity, let's pass them directly. We also need to adjust plot title and save paths.

    # If run_name is default, try to make it more specific from checkpoint_dir
    if args.run_name == "pca_analysis":
        try:
            # Assumes checkpoint_dir is like .../primary_folder/seed_folder/
            path_parts = Path(args.checkpoint_dir).parts
            if len(path_parts) >= 2:
                args.run_name = f"{path_parts[-2]}_{path_parts[-1]}_pca"
            elif len(path_parts) == 1:
                 args.run_name = f"{path_parts[-1]}_pca"
        except Exception:
            pass # Keep default if parsing fails

    # Call the main analysis function
    analyze_pca_v2(args.checkpoint_dir, args.run_identifier_prefix, args.run_name, args.results_basedir, args.pca_components)



# Old main and analyze_pca function are now replaced by the __main__ block above
# and analyze_pca_v2. The old command line parsing for individual filename components
# is removed in favor of the simpler --checkpoint_dir and --run_identifier_prefix.

# Example Usage (after organizing checkpoints):
# python analyze_weight_trajectory_pca.py \
#   --checkpoint_dir "/scratch/gpfs/mg7411/ManyPaths/organized_checkpoints/feats32_depth7_adapt1_1stOrd/seed0" \
#   --run_identifier_prefix "concept_mlp_14_bits_feats32_depth7_adapt1_1stOrd_seed0" \
#   --results_basedir "/scratch/gpfs/mg7411/ManyPaths/pca_results"
#   --run-name feats32_depth7_1stOrd_seed0_pca # Optional: custom name for output subfolder

# Make sure you have scikit-learn and matplotlib installed:
# pip install scikit-learn matplotlib 