import torch
import os
import re
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from initialization import init_model # Assuming this can init the model structure

# Attempt to import from constants, but provide fallbacks if not found or running standalone
try:
    from constants import CHECKPOINT_SUBDIR, PCFG_DEFAULT_MAX_DEPTH
except ImportError:
    print("Warning: Could not import from constants.py. Using fallback default values.")
    CHECKPOINT_SUBDIR = "checkpoints"
    PCFG_DEFAULT_MAX_DEPTH = 5 # A common default, adjust if needed

# Ensure CHECKPOINT_SUBDIR is defined one way or another.
if 'CHECKPOINT_SUBDIR' not in globals():
    CHECKPOINT_SUBDIR = "checkpoints" 
if 'PCFG_DEFAULT_MAX_DEPTH' not in globals():
    PCFG_DEFAULT_MAX_DEPTH = 5 # Fallback for arg default

def get_model_param_vec(model_state_dict):
    '''Extracts and flattens all parameters from a model\'s state_dict.'''
    param_tensors = []
    print(f"  [get_model_param_vec] Processing state_dict with keys: {list(model_state_dict.keys())}")
    extracted_keys = []
    for key, param in model_state_dict.items():
        if key.startswith("module."): 
            param_tensors.append(param.cpu().view(-1))
            extracted_keys.append(key)
    if not param_tensors:
        print("  [get_model_param_vec] No params found with 'module.' prefix. Falling back to all tensors.")
        for key, param in model_state_dict.items():
            if torch.is_tensor(param):
                 param_tensors.append(param.cpu().view(-1))
                 extracted_keys.append(key)
    
    if not param_tensors:
        print("  [get_model_param_vec] No parameters found in state_dict to create a vector.")
        raise ValueError("No parameters found in state_dict to create a vector.")
    
    print(f"  [get_model_param_vec] Extracted parameters from keys: {extracted_keys}")
    flat_params = torch.cat(param_tensors)
    print(f"  [get_model_param_vec] Total elements in flat_params: {flat_params.nelement()}")
    return flat_params

def analyze_drift(args):
    '''Analyzes parameter drift for a given experiment configuration.'''
    # Construct the base prefix for checkpoint files
    file_prefix_parts = [args.experiment, args.model_type]
    
    # Determine hyper_index (assuming 'index' in main.py filename corresponds to this)
    # This is a simplification. If hyper_index isn't part of the filename in a predictable way,
    # this part needs adjustment or the user needs to supply the full prefix.
    # For now, let's assume a DEFAULT_INDEX if not specified, or use a placeholder if needed.
    # We will rely on the user providing the most specific prefix possible via --file-prefix if auto-construction is hard.

    if args.file_prefix:
        prefix_to_match = args.file_prefix
        print(f"Using provided file prefix: {prefix_to_match}")
    else: # Construct from args if no specific prefix given
        # This construction logic needs to exactly match main.py's logic for full auto-matching
        # Example: concept_mlp_feats8_depth3_adapt1_2ndOrd_seed0
        # For concept learning (most complex case from user's setup)
        if args.experiment == "concept":
            # We need index for init_model, but it's not used for drift if we load state_dicts
            # The filename structure needs to be matched:
            # concept_mlp_<INDEX>_bits_feats<F>_depth<D>_adapt<A>_<ORDER>Ord_seed<S>
            # The <INDEX> part is tricky if it's not fixed or easily derivable.
            # Let's assume for now the user runs this for a *known* single index run,
            # or we'll need a more robust way to get the index or the full filename prefix.
            # The script that generated the files (main.py) used args.hyper_index or DEFAULT_INDEX.
            # For simplicity, let's require a fairly complete set of args or a direct file_prefix.

            if args.index is None:
                print("Warning: --index not provided for concept experiment. Filename matching might be incomplete.")
                # Fallback to a generic part or expect user to use --file-prefix
                index_str = "default_idx" 
            else:
                index_str = str(args.index)

            file_prefix_parts.extend([
                index_str, # This 'index' corresponds to hyper_index in main.py
                "bits", # Assuming data_type is 'bits' for concept
                f"feats{args.features}",
                f"depth{args.depth}",
                f"adapt{args.adapt_steps}",
                "1stOrd" if args.first_order else "2ndOrd"
            ])
        elif args.experiment == "mod":
             # Simpler example for 'mod'
            if args.index is None: index_str = "default_idx"
            else: index_str = str(args.index)
            file_prefix_parts.extend([index_str, "bits", str(args.skip)]) # Add relevant parts for 'mod'
        else:
            print(f"Filename prefix auto-construction for experiment '{args.experiment}' is not fully implemented. Please use --file-prefix.")
            return

        file_prefix_parts.append(f"seed{args.seed}")
        prefix_to_match = "_".join(file_prefix_parts)
        print(f"Constructed file prefix for matching: {prefix_to_match}")

    checkpoints_dir = os.path.join(args.saved_models_root, CHECKPOINT_SUBDIR)
    if not os.path.isdir(checkpoints_dir):
        print(f"Error: Checkpoints directory not found: {checkpoints_dir}")
        return

    print(f"Searching for checkpoints in: {checkpoints_dir} with prefix: {prefix_to_match}")
    
    checkpoint_files = []
    for f_name in os.listdir(checkpoints_dir):
        if f_name.startswith(prefix_to_match) and f_name.endswith(".pt") and "_episode_" in f_name:
            match = re.search(r"_episode_(\d+)\.pt$", f_name)
            if match:
                episode_num = int(match.group(1))
                checkpoint_files.append({"episode": episode_num, "path": os.path.join(checkpoints_dir, f_name)})
    
    if not checkpoint_files:
        print(f"No checkpoint files found for prefix '{prefix_to_match}' in {checkpoints_dir}")
        return

    checkpoint_files.sort(key=lambda x: x["episode"])
    print(f"Found {len(checkpoint_files)} checkpoints.")

    # Load initial parameters
    try:
        initial_cp_path = checkpoint_files[0]["path"]
        initial_episode = checkpoint_files[0]["episode"]
        print(f"Loading initial checkpoint: {initial_cp_path} (episode {initial_episode})")
        initial_state_dict = torch.load(initial_cp_path, map_location=torch.device('cpu'))
        # If state_dict is from MetaSGD, it might contain the model in 'module'
        # or it could be the model's state_dict directly if saved differently.
        # get_model_param_vec tries to handle this.
        initial_params_vec = get_model_param_vec(initial_state_dict)
        print(f"Initial parameter vector size: {initial_params_vec.nelement()}, Initial param norm: {torch.norm(initial_params_vec).item()}")
        print(f"Initial params (first 5 elements): {initial_params_vec[:5]}")
    except Exception as e:
        print(f"Error loading or processing initial checkpoint {initial_cp_path}: {e}")
        return

    drifts = [{"episode": initial_episode, "l2_drift": 0.0}]
    
    for i in range(1, len(checkpoint_files)):
        cp_info = checkpoint_files[i]
        try:
            print(f"\nProcessing checkpoint: {cp_info['path']} (episode {cp_info['episode']})")
            current_state_dict = torch.load(cp_info["path"], map_location=torch.device('cpu'))
            current_params_vec = get_model_param_vec(current_state_dict)

            print(f"Current param vector size: {current_params_vec.nelement()}, Current param norm: {torch.norm(current_params_vec).item()}")
            print(f"Current params (first 5 elements): {current_params_vec[:5]}")

            if current_params_vec.nelement() != initial_params_vec.nelement():
                print(f"Warning: Parameter count mismatch! Skipping episode {cp_info['episode']}.")
                continue
            
            diff_vec = current_params_vec - initial_params_vec
            l2_drift = torch.norm(diff_vec).item()
            print(f"Difference vector (first 5 elements): {diff_vec[:5]}")
            print(f"L2 Drift for episode {cp_info['episode']}: {l2_drift:.8f}")
            drifts.append({"episode": cp_info["episode"], "l2_drift": l2_drift})
        except Exception as e:
            print(f"Error loading or processing checkpoint {cp_info['path']}: {e}")
            continue
            
    if len(drifts) < 2:
        print("Not enough data points to plot drift.")
        return

    # Plotting
    df = pd.DataFrame(drifts)
    plt.figure(figsize=(10, 6))
    plt.plot(df["episode"], df["l2_drift"], marker='o', linestyle='-')
    plt.title(f"Parameter L2 Drift from Initial State\\nRun: {prefix_to_match}")
    plt.xlabel("Training Episode")
    plt.ylabel("L2 Norm of (Current Params - Initial Params)")
    plt.grid(True)
    
    # Save the plot - Parameterized output directory
    plot_out_dir = os.path.join(args.results_basedir, f"{args.run_name}_plots") 
    os.makedirs(plot_out_dir, exist_ok=True)
    plot_filename = os.path.join(plot_out_dir, f"drift_{prefix_to_match}.png")
    plt.savefig(plot_filename)
    print(f"Saved drift plot to: {plot_filename}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze parameter drift from model checkpoints.")
    parser.add_argument("--saved_models_root", type=str, default="saved_models", help="Root directory where models and checkpoints are saved.")
    
    # Args to help construct the filename prefix (similar to main.py)
    # Option 1: Provide the exact file prefix if known
    parser.add_argument("--file-prefix", type=str, default=None, help="Exact file prefix to match checkpoints (e.g., concept_mlp_0_bits_feats8_depth3_adapt1_2ndOrd_seed0). Overrides other specific args if provided.")

    # Option 2: Provide individual components to construct the prefix
    parser.add_argument("--experiment", type=str, default="concept", choices=["mod", "concept"], help="Experiment type.")
    parser.add_argument("--model-type", "--m", type=str, default="mlp", dest="model_type", help="Model type (e.g., mlp).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for the run.")
    
    # Concept-specific args (used if experiment is 'concept' and --file-prefix is not given)
    parser.add_argument("--features", type=int, default=8, dest="num_concept_features", help="Number of features (for concept).") # Renamed to match main.py for clarity
    parser.add_argument("--depth", type=int, default=PCFG_DEFAULT_MAX_DEPTH, dest="pcfg_max_depth", help="Max depth (for concept).") # Renamed
    parser.add_argument("--adapt-steps", type=int, default=1, help="Adaptation steps (for concept).")
    parser.add_argument("--first-order", action="store_true", help="If the run used first-order MAML.")
    parser.add_argument("--index", type=int, default=0, help="Hyperparameter index used in the filename (e.g., the '0' in concept_mlp_0_bits...).") # Default to 0 as an example

    # Mod-specific args (used if experiment is 'mod' and --file-prefix is not given)
    parser.add_argument("--skip", type=int, default=1, help="Skip value (for mod experiment).")

    # New arguments for output directory parameterization
    parser.add_argument("--run-name", type=str, default="run1", help="Name of the run (e.g., run1, run2) to create specific plot subdirectories.")
    parser.add_argument("--results-basedir", type=str, default="results", help="Base directory where plot subdirectories will be created (e.g., results/<run_name>_plots/).")

    args = parser.parse_args()

    # Ensure args.features and args.depth exist for internal logic, aliasing from actual arg names
    if hasattr(args, 'num_concept_features'): 
        args.features = args.num_concept_features
    elif not hasattr(args, 'features'):
        # If --features was not defined (e.g. if constants import failed and it was removed from argparse)
        # provide a default if experiment is concept. This is defensive.
        if args.experiment == "concept":
            print("Warning: --features not explicitly set, defaulting to 8 for concept experiment prefix construction.")
            args.features = 8 
            
    if hasattr(args, 'pcfg_max_depth'): 
        args.depth = args.pcfg_max_depth
    elif not hasattr(args, 'depth'):
        if args.experiment == "concept":
            print(f"Warning: --depth not explicitly set, defaulting to {PCFG_DEFAULT_MAX_DEPTH} for concept experiment prefix construction.")
            args.depth = PCFG_DEFAULT_MAX_DEPTH

    analyze_drift(args)

# Example usage:
# python analyze_checkpoint_drift.py --experiment concept --model-type mlp --features 8 --depth 3 --adapt-steps 1 --seed 0 
# (if it was a 2nd order run, don't add --first-order)
#
# Or, if you know the exact prefix:
# python analyze_checkpoint_drift.py --file-prefix concept_mlp_0_bits_feats8_depth3_adapt1_2ndOrd_seed0

# Note: The 'index' (hyperparameter index) part of the filename might require careful handling.
# The current script uses args.index (default 0) if not constructing from full --file-prefix.
# Ensure this matches the 'index' used in your filenames.
# The init_model function is imported but NOT USED to build a model;
# we only work with state_dicts. It's a placeholder if we needed to know model structure,
# but get_model_param_vec works directly on state_dict. 