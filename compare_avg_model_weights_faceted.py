import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import pandas as pd # For collecting and managing norm data
import re

# --- MLP Model Definition (consistent with previous findings) ---
class MLP(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int = 1,
        n_hidden: int = 32,
        n_layers: int = 8,  # Fixed based on our successful loading (9 linear layers total)
        n_input_channels: int = 3,
    ):
        super().__init__()
        layers = []
        # This specific structure for n_input < 64 was key
        if n_input < 64: # All our features (8, 16, 32) are < 64
            layers.extend([
                nn.Linear(n_input, 32 * 32 * n_input_channels),
                nn.BatchNorm1d(32 * 32 * n_input_channels),
                nn.ReLU(),
                nn.Linear(32 * 32 * n_input_channels, n_hidden),
                nn.BatchNorm1d(n_hidden),
            ])
        else: # Fallback, not expected to be used for feats 8,16,32
            layers.extend([
                nn.Linear(n_input, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
            ])
        layers.append(nn.ReLU())
        # n_layers (e.g., 8) determines the number of these blocks via n_layers-2 loop
        # For n_layers=8, loop is 6 times. Total linear layers = 2 (initial) + 6 (loop) + 1 (output) = 9
        for _ in range(n_layers - 2):
            layers.extend([
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
            ])
        layers.append(nn.Linear(n_hidden, n_output))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
# --- End MLP Model Definition ---

# --- Model Loading Function (from compare_final_model_weights.py) ---
def load_model_weights(model_path, model_class, **model_args):
    # print(f"  Instantiating MLP with: {model_args}") # Debug
    model = model_class(**model_args)
    try:
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
        
        saved_object = torch.load(model_path, map_location=torch.device('cpu'))

        if isinstance(saved_object, dict):
            possible_keys = ['model_state_dict', 'model', 'net', 'state_dict']
            state_dict = None
            for key in possible_keys:
                if key in saved_object:
                    state_dict = saved_object[key]
                    # print(f"Extracted state_dict from checkpoint using key: '{key}' for {model_path}")
                    break
            if state_dict is None:
                # print(f"Warning: Could not find common state_dict key. Using loaded dict as state_dict for {model_path}.")
                state_dict = saved_object
        else:
            state_dict = saved_object
            # print(f"Loaded object is not a dict, assuming state_dict directly for {model_path}.")

        if not isinstance(state_dict, dict):
            print(f"Error: Extracted state_dict is not a dictionary. Type: {type(state_dict)} for {model_path}")
            return None
        
        processed_state_dict = {}
        # has_module_prefix_stripped = False # Not needed for print here
        for k, v in state_dict.items():
            if k.startswith('module.'):
                processed_state_dict[k[len('module.'):]] = v
                # has_module_prefix_stripped = True
            else:
                processed_state_dict[k] = v
        
        # if has_module_prefix_stripped:
            # print(f"Processed keys by stripping 'module.' prefix for {model_path}.")
        
        current_model_keys = model.state_dict().keys()
        filtered_state_dict = {k: v for k, v in processed_state_dict.items() if k in current_model_keys}

        missing_keys = [k for k in current_model_keys if k not in filtered_state_dict]
        if missing_keys:
            # This is now a more critical error if our fixed architecture assumption is correct
            print(f"ERROR: Missing keys for {model_path} with fixed architecture: {missing_keys}")
            # print(f"  Expected keys by model: {list(current_model_keys)}")
            # print(f"  Keys found in file (after module strip & filter): {list(filtered_state_dict.keys())}")
            return None # Fail hard if fixed architecture doesn't match
        
        model.load_state_dict(filtered_state_dict, strict=True)
        model.eval()
        # print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None
# --- End Model Loading Function ---

# --- Norm Calculation Function (from compare_final_model_weights.py) ---
def get_layer_weight_norms(model):
    norms = {}
    layer_idx = 0
    if hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
        for i, module_in_seq in enumerate(model.model):
            if isinstance(module_in_seq, nn.Linear):
                layer_name = f"layer_{layer_idx}_model.{i}"
                weight_norm = torch.linalg.norm(module_in_seq.weight.data).item()
                norms[layer_name] = weight_norm
                layer_idx += 1
    else:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_name = f"layer_{layer_idx}_{name.replace('.', '_')}"
                weight_norm = torch.linalg.norm(module.weight.data).item()
                norms[layer_name] = weight_norm
                layer_idx += 1
    return norms
# --- End Norm Calculation ---

def main():
    parser = argparse.ArgumentParser(description="Compare averaged model weight norms across seeds for different configurations.")
    parser.add_argument("--organized_checkpoint_dir", type=str, required=True,
                        help="Base directory of organized checkpoints (e.g., .../organized_checkpoints/).")
    parser.add_argument("--output_dir", type=str, default="results/norm_comparisons_faceted",
                        help="Directory to save plots.")
    parser.add_argument("--run_name", type=str, default="faceted_norm_comparison",
                        help="A name for this comparison run, used in plot filenames.")
    
    # Fixed MLP architecture parameters (except n_input which varies with features)
    parser.add_argument("--mlp_n_layers", type=int, default=8, help="Fixed n_layers arg for MLP constructor (e.g., 8 for 9 linear layers).")
    parser.add_argument("--mlp_n_hidden", type=int, default=32, help="Fixed n_hidden arg for MLP.")
    parser.add_argument("--mlp_n_input_channels", type=int, default=3, help="Fixed n_input_channels for MLP.")

    parser.add_argument("--features_list", type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument("--concept_depths_list", type=int, nargs='+', default=[3, 5, 7], help="List of concept recursion depths (from filenames).")
    parser.add_argument("--seeds_list", type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument("--hyper_param_idx_in_filename", type=str, default="14", help="The hyperparameter index part of the filename (e.g., '14').")
    parser.add_argument("--adapt_steps_in_filename", type=int, default=1, help="Adaptation steps part of the filename.")
    parser.add_argument("--epoch_to_load", type=str, default="100", help="Epoch number string for the checkpoint file (e.g., '100').")


    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_avg_norms_data = [] # List to store dicts for DataFrame

    # Define the fixed part of the MLP configuration
    base_model_config = {
        'n_output': 1,
        'n_hidden': args.mlp_n_hidden,
        'n_layers': args.mlp_n_layers,
        'n_input_channels': args.mlp_n_input_channels
    }

    for features_val in args.features_list:
        current_model_config = base_model_config.copy()
        current_model_config['n_input'] = features_val
        print(f"\n--- Processing Feature Set: {features_val} --- MLP Config: {current_model_config}")

        for concept_depth_val in args.concept_depths_list:
            print(f"  -- Concept Recursion Depth (filename): {concept_depth_val} --")
            for order_str_short, order_str_long in [("1stOrd", "1st Order"), ("2ndOrd", "2nd Order")]:
                seed_norms_for_config = [] # List of norm dicts for current (feat, concept_depth, order)
                
                primary_folder_name = f"feats{features_val}_depth{concept_depth_val}_adapt{args.adapt_steps_in_filename}_{order_str_short}"
                
                for seed_val in args.seeds_list:
                    seed_folder_name = f"seed{seed_val}"
                    checkpoint_dir_for_seed = Path(args.organized_checkpoint_dir) / primary_folder_name / seed_folder_name
                    
                    # Construct filename: concept_mlp_<IDX>_bits_feats<F>_depth<D>_adapt<A>_<ORDER>_seed<S>_epoch_<E>.pt
                    # Default path construction
                    default_filename = f"concept_mlp_{args.hyper_param_idx_in_filename}_bits_feats{features_val}_depth{concept_depth_val}_adapt{args.adapt_steps_in_filename}_{order_str_short}_seed{seed_val}_epoch_{args.epoch_to_load}.pt"
                    model_path = checkpoint_dir_for_seed / default_filename

                    # Override for special cases based on user-provided paths
                    # These paths are used only if --adapt_steps_in_filename is 1, matching the 'adapt1' in the special filenames.
                    if features_val == 16 and concept_depth_val == 7 and order_str_short == "1stOrd" and seed_val == 3 and args.adapt_steps_in_filename == 1:
                        special_path_str = "/scratch/gpfs/mg7411/ManyPaths/saved_models/concept_multiseed/concept_mlp_14_bits_feats16_depth7_adapt1_1stOrd_seed3_best_model_at_end_of_train.pt"
                        model_path = Path(special_path_str)
                        print(f"    INFO: Overriding path for F=16, D=7, 1stOrd, Seed=3 (adapt_steps_in_filename={args.adapt_steps_in_filename}): {model_path}")
                    elif features_val == 16 and concept_depth_val == 3 and order_str_short == "1stOrd" and seed_val == 3 and args.adapt_steps_in_filename == 1:
                        special_path_str = "/scratch/gpfs/mg7411/ManyPaths/saved_models/concept_multiseed/concept_mlp_14_bits_feats16_depth3_adapt1_1stOrd_seed3_best_model_at_end_of_train.pt"
                        model_path = Path(special_path_str)
                        print(f"    INFO: Overriding path for F=16, D=3, 1stOrd, Seed=3 (adapt_steps_in_filename={args.adapt_steps_in_filename}): {model_path}")
                    
                    if model_path.exists():
                        # print(f"    Loading model: {model_path}")
                        model = load_model_weights(str(model_path), MLP, **current_model_config)
                        if model:
                            norms = get_layer_weight_norms(model)
                            seed_norms_for_config.append(norms)
                        else:
                            print(f"    Failed to load model: {model_path}")
                    # else:
                        # print(f"    Model path not found: {model_path}") # Can be verbose
                
                if not seed_norms_for_config:
                    print(f"    No models loaded for Feats: {features_val}, ConceptDepth: {concept_depth_val}, Order: {order_str_long}. Skipping averaging.")
                    continue

                # Average norms across seeds for this specific configuration
                if not seed_norms_for_config[0]: # Check if the first norm dict is empty
                    print(f"    First norm dict is empty for Feats: {features_val}, ConceptDepth: {concept_depth_val}, Order: {order_str_long}. Skipping averaging.")
                    continue
                layer_names = list(seed_norms_for_config[0].keys())
                averaged_norms = {name: [] for name in layer_names}
                for p_norms_dict in seed_norms_for_config:
                    if p_norms_dict: # Ensure dict is not empty
                        for name in layer_names:
                            averaged_norms[name].append(p_norms_dict.get(name, np.nan)) # Use NaN for missing layers if any
                
                final_avg_norms = {name: np.nanmean(vals) for name, vals in averaged_norms.items()}
                num_seeds_found = len(seed_norms_for_config)
                print(f"    Averaged norms for Feats: {features_val}, ConceptDepth (file): {concept_depth_val}, Order: {order_str_long} (from {num_seeds_found} seeds): {final_avg_norms}")

                # Store for DataFrame
                for layer_name, avg_norm_val in final_avg_norms.items():
                    all_avg_norms_data.append({
                        'features': features_val,
                        'concept_depth_file': concept_depth_val,
                        'order': order_str_long,
                        'layer': layer_name,
                        'avg_norm': avg_norm_val,
                        'num_seeds': num_seeds_found
                    })
    
    if not all_avg_norms_data:
        print("\nNo data collected for plotting. Exiting.")
        return

    df_norms = pd.DataFrame(all_avg_norms_data)
    df_norms.to_csv(Path(args.output_dir) / f"{args.run_name}_avg_norms_data.csv", index=False)
    print(f"\nSaved aggregated norm data to {Path(args.output_dir) / f'{args.run_name}_avg_norms_data.csv'}")

    # Faceted Plotting
    features_cat = pd.CategoricalDtype(args.features_list, ordered=True)
    concept_depths_cat = pd.CategoricalDtype(args.concept_depths_list, ordered=True)
    df_norms['features'] = df_norms['features'].astype(features_cat)
    df_norms['concept_depth_file'] = df_norms['concept_depth_file'].astype(concept_depths_cat)

    if not df_norms.empty and 'layer' in df_norms.columns:
        def get_layer_sort_key(layer_name_str):
            match = re.search(r"model\.(\d+)", layer_name_str)
            return int(match.group(1)) if match else -1
        unique_layers = sorted(df_norms['layer'].unique(), key=get_layer_sort_key)
    else:
        unique_layers = []
        print("Warning: No layer data found for plotting.")

    if not unique_layers:
        print ("No unique layers to plot. Exiting plotting.")
        return
        
    num_unique_layers = len(unique_layers)

    # Create a 3x3 grid of subplots (Features x Concept Recursion Depth)
    # Adjust figure size based on number of unique layers for better x-axis label readability
    fig_width = max(5 * len(args.concept_depths_list), num_unique_layers * 0.8 * len(args.concept_depths_list) * 0.5) # Heuristic
    fig_height = 4 * len(args.features_list)

    fig, axes = plt.subplots(len(args.features_list), len(args.concept_depths_list),
                             figsize=(fig_width, fig_height),
                             sharey=True, sharex=True) # ShareX might be tricky if layer names differ a lot, but generally good
    
    # Handle cases where subplots return a 1D array or a single Axes object
    if len(args.features_list) == 1 and len(args.concept_depths_list) == 1:
        axes = np.array([[axes]]) # Make it 2D for consistent indexing
    elif len(args.features_list) == 1:
        axes = np.array([axes]) # Make it 2D row vector
    elif len(args.concept_depths_list) == 1:
        axes = np.array([[ax] for ax in axes]) # Make it 2D column vector

    fig.suptitle(f"Comparative Avg. Weight Norms ({args.run_name})\nMLP Layers: {args.mlp_n_layers}, Hidden: {args.mlp_n_hidden}, InChannels: {args.mlp_n_input_channels}", fontsize=16, y=1.03)

    for i, features_val in enumerate(args.features_list):
        for j, concept_depth_val in enumerate(args.concept_depths_list):
            ax = axes[i, j]
            
            subset_df = df_norms[(df_norms['features'] == features_val) &
                                 (df_norms['concept_depth_file'] == concept_depth_val)]
            
            if subset_df.empty:
                ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            else:
                pivot_df = subset_df.pivot(index='layer', columns='order', values='avg_norm').reindex(unique_layers)
                pivot_df.plot(kind='bar', ax=ax, width=0.8, legend=(i==0 and j==len(args.concept_depths_list)-1))
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)

            ax.set_xticks(range(len(unique_layers)))
            ax.set_xticklabels(unique_layers, rotation=60, ha='right')
            ax.set_xlabel("Layer")

            if j == 0:
                ax.set_ylabel(f"Features: {features_val}\nL2 Norm")
            else:
                ax.set_ylabel("")
            
            if i == 0:
                ax.set_title(f"ConceptDepth (file): {concept_depth_val}")
            else:
                ax.set_title("")
                
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_savename = f"faceted_norm_comparison_{args.run_name}.png"
    plot_save_path = Path(args.output_dir) / plot_savename
    fig.savefig(plot_save_path, dpi=150)
    print(f"\nFaceted comparative plot saved to {plot_save_path}")
    plt.close(fig) # Close the figure to free memory

if __name__ == "__main__":
    main()

# Example Usage:
# python compare_avg_model_weights_faceted.py \
#   --organized_checkpoint_dir "/path/to/your/organized_checkpoints" \
#   --output_dir "results/norm_comparisons_faceted" \
#   --run_name "all_configs_avg_norms" \
#   --mlp_n_layers 8 \
#   --mlp_n_hidden 32 \
#   --mlp_n_input_channels 3 \
#   --features_list 8 16 32 \
#   --concept_depths_list 3 5 7 \
#   --seeds_list 0 1 2 3 4 \
#   --hyper_param_idx_in_filename "14" \
#   --adapt_steps_in_filename 1 \
#   --epoch_to_load "100" 