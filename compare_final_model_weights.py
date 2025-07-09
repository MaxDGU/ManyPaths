import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

# --- MLP Model Definition (copied from analyze_weights.py) ---
class MLP(nn.Module):
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
                nn.Linear(n_input, 32 * 32 * n_input_channels),
                nn.BatchNorm1d(32 * 32 * n_input_channels),
                nn.ReLU(),
                nn.Linear(32 * 32 * n_input_channels, n_hidden),
                nn.BatchNorm1d(n_hidden),
            ])
        else:
            layers.extend([
                nn.Linear(n_input, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
            ])
        layers.append(nn.ReLU())
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

# --- Model Loading Function (copied from analyze_weights.py) ---
def load_model_weights(model_path, model_class, **model_args):
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
                    print(f"Extracted state_dict from checkpoint using key: '{key}' for {model_path}")
                    break
            if state_dict is None:
                print(f"Warning: Could not find common state_dict key. Using loaded dict as state_dict for {model_path}.")
                state_dict = saved_object 
        else:
            state_dict = saved_object
            print(f"Loaded object is not a dict, assuming state_dict directly for {model_path}.")

        if not isinstance(state_dict, dict):
            print(f"Error: Extracted state_dict is not a dictionary. Type: {type(state_dict)} for {model_path}")
            return None
        
        processed_state_dict = {}
        has_module_prefix_stripped = False
        for k, v in state_dict.items():
            if k.startswith('module.'):
                processed_state_dict[k[len('module.'):]] = v
                has_module_prefix_stripped = True
            else:
                processed_state_dict[k] = v
        
        if has_module_prefix_stripped:
            print(f"Processed keys by stripping 'module.' prefix for {model_path}.")
        
        current_model_keys = model.state_dict().keys()
        filtered_state_dict = {k: v for k, v in processed_state_dict.items() if k in current_model_keys}

        missing_keys = [k for k in current_model_keys if k not in filtered_state_dict]
        if missing_keys:
            print(f"Warning: Missing keys for {model_path}: {missing_keys}")
        
        unexpected_keys = [k for k in filtered_state_dict if k not in current_model_keys]
        if unexpected_keys:
             print(f"Warning: Unexpected keys after filtering for {model_path}: {unexpected_keys}")

        model.load_state_dict(filtered_state_dict, strict=True)
        model.eval() 
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None
# --- End Model Loading Function ---

# --- Norm Calculation Function (copied from analyze_weights.py) ---
def get_layer_weight_norms(model):
    norms = {}
    layer_idx = 0
    # Iterate through the modules in self.model (the nn.Sequential)
    # This ensures we only get layers from our defined sequence and correct naming
    if hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
        for i, module_in_seq in enumerate(model.model):
            if isinstance(module_in_seq, nn.Linear):
                # Use a consistent naming scheme: layer_<sequential_index>_model.<original_sequential_index>
                # e.g. layer_0_model.0, layer_1_model.3 (if layer 1 is model.3)
                layer_name = f"layer_{layer_idx}_model.{i}" 
                weight_norm = torch.linalg.norm(module_in_seq.weight.data).item()
                norms[layer_name] = weight_norm
                layer_idx += 1
    else: # Fallback if model structure is not nn.Sequential named 'model'
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_name = f"layer_{layer_idx}_{name.replace('.', '_')}" 
                weight_norm = torch.linalg.norm(module.weight.data).item()
                norms[layer_name] = weight_norm
                layer_idx += 1
    return norms
# --- End Norm Calculation ---

def plot_comparative_weight_norms(models_norms, model_labels, title="Comparative Layer Weight Norms", save_path=None):
    if not models_norms or not all(models_norms):
        print("No valid norms data to plot.")
        return

    num_models = len(models_norms)
    if num_models == 0:
        print("No models to compare.")
        return

    # Assume all models have the same layers in the same order for comparison
    # Use the layer names from the first model as the reference
    layer_names = list(models_norms[0].keys())
    num_layers = len(layer_names)

    x = np.arange(num_layers)  # the label locations
    width = 0.8 / num_models  # the width of the bars, adjusted for number of models

    fig, ax = plt.subplots(figsize=(max(12, num_layers * num_models * 0.5), 7))

    for i, (norms_dict, label) in enumerate(zip(models_norms, model_labels)):
        # Ensure this model's norms_dict has the same layers or handle mismatch
        values = [norms_dict.get(name, 0) for name in layer_names] # Use .get for safety
        offset = (i - (num_models - 1) / 2) * width
        ax.bar(x + offset, values, width, label=label)

    ax.set_ylabel("L2 Norm of Weights")
    ax.set_xlabel("Layer")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=60, ha='right')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Comparative plot saved to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare final model weight norms.")
    parser.add_argument("--model_paths", type=str, nargs='+', required=True,
                        help="List of full paths to model .pt files.")
    parser.add_argument("--model_labels", type=str, nargs='+', required=True,
                        help="List of labels for the models (must match number of paths).")
    parser.add_argument("--n_input", type=int, required=True, help="Number of input features (e.g., 8 for feats8).")
    parser.add_argument("--depth_in_filename", type=int, required=True, 
                        help="Depth number from filename (e.g., 3 for depth3). Script calculates MLP n_layers as depth+1.")
    parser.add_argument("--n_hidden", type=int, default=32, help="Hidden layer size (default: 32).")
    parser.add_argument("--n_input_channels", type=int, default=3, help="Input channels for MLP's first layer (default: 3).")
    parser.add_argument("--n_output", type=int, default=1, help="Output size (default: 1).")
    parser.add_argument("--output_dir", type=str, default="results/norm_comparisons",
                        help="Directory to save plots.")
    parser.add_argument("--run_name", type=str, default="comparison", 
                        help="A name for this comparison run, used in plot filenames.")

    args = parser.parse_args()

    if len(args.model_paths) != len(args.model_labels):
        print("Error: Number of model paths must match number of model labels.")
        return

    # Calculate n_layers argument for MLP class
    # Based on previous findings: MLP_n_layers_arg = filename_depth + 1
    # This results in (filename_depth + 1) + 1 = filename_depth + 2 total *Linear* layers for n_input < 64
    # Re-checking: For depth7 file, we used MLP(n_layers=8). This means n_layers_arg_for_MLP = depth_in_filename + 1.
    # This MLP(n_layers=L) results in L+1 linear layers. So if depth_in_filename=7, MLP(n_layers=8) -> 9 linear layers. Correct.
    mlp_n_layers_arg = args.depth_in_filename + 1

    model_config = {
        'n_input': args.n_input,
        'n_output': args.n_output,
        'n_hidden': args.n_hidden,
        'n_layers': mlp_n_layers_arg, # This is the L in MLP(n_layers=L)
        'n_input_channels': args.n_input_channels
    }
    
    print(f"Using common MLP configuration for all models: {model_config}")

    all_models_norms = []
    valid_labels = []
    for model_path, model_label in zip(args.model_paths, args.model_labels):
        print(f"\nProcessing model: {model_label} from {model_path}")
        model = load_model_weights(model_path, MLP, **model_config)
        if model:
            norms = get_layer_weight_norms(model)
            print(f"  Norms for {model_label}: {norms}")
            all_models_norms.append(norms)
            valid_labels.append(model_label)
        else:
            print(f"Could not load model {model_label}, skipping.")

    if not all_models_norms:
        print("No models were successfully loaded and processed.")
        return

    plot_title = f"Comparative Weight Norms ({args.run_name})\nFeatures: {args.n_input}, Depth(file): {args.depth_in_filename}, Hidden: {args.n_hidden}, InChannels: {args.n_input_channels}"
    plot_savename = f"norm_comparison_{args.run_name}_feats{args.n_input}_depth{args.depth_in_filename}.png"
    plot_save_path = os.path.join(args.output_dir, plot_savename)

    plot_comparative_weight_norms(all_models_norms, valid_labels, title=plot_title, save_path=plot_save_path)

if __name__ == "__main__":
    main()

# Example usage:
# python compare_final_model_weights.py \
#   --model_paths /path/to/organized_checkpoints/feats8_depth3_adapt1_1stOrd/seed0/concept_mlp_X_bits_feats8_depth3_adapt1_1stOrd_seed0_epoch_100.pt \
#                 /path/to/organized_checkpoints/feats8_depth3_adapt1_2ndOrd/seed0/concept_mlp_Y_bits_feats8_depth3_adapt1_2ndOrd_seed0_epoch_100.pt \
#   --model_labels "1stOrder_feats8_depth3_seed0" "2ndOrder_feats8_depth3_seed0" \
#   --n_input 8 \
#   --depth_in_filename 3 \
#   --n_hidden 32 \
#   --n_input_channels 3 \
#   --output_dir results/norm_comparisons_final \
#   --run_name feats8_depth3_s0_comparison 