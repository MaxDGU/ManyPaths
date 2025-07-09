import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np

# --- BEGIN ACTUAL MODEL DEFINITION ---
class MLP(nn.Module):
    def __init__(
        self,
        n_input: int = 32 * 32, # Default, will be overridden by featsX
        n_output: int = 1,
        n_hidden: int = 64, # Default, might need to be passed if experiments used a different value
        n_layers: int = 8,  # Default, will be overridden by depthD
        n_input_channels: int = 1, # Default, likely fine
    ):
        super().__init__()
        layers = []
        # This logic matches the provided models.py
        if n_input < 64:
            layers.extend(
                [
                    nn.Linear(n_input, 32 * 32 * n_input_channels),
                    nn.BatchNorm1d(32 * 32 * n_input_channels),
                    nn.ReLU(),
                    nn.Linear(32 * 32 * n_input_channels, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                ]
            )
        else:
            layers.extend(
                [
                    nn.Linear(n_input, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                ]
            )
        layers.append(nn.ReLU()) # This ReLU is after the initial block in both cases
        
        # n_layers in the provided MLP seems to be the total number of Linear layers
        # The initial block has 2 Linear layers. So, n_layers - 2 more hidden linear layers are added.
        for _ in range(n_layers - 2): 
            layers.extend(
                [
                    nn.Linear(n_hidden, n_hidden),
                    nn.BatchNorm1d(n_hidden),
                    nn.ReLU(),
                ]
            )
        layers.append(nn.Linear(n_hidden, n_output)) # Final output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
# --- END ACTUAL MODEL DEFINITION ---


def load_model_weights(model_path, model_class, **model_args):
    """Loads a PyTorch model state_dict from a .pt file."""
    
    # Ensure all required model_args are present
    # For MLP: n_input, n_output, n_hidden, n_layers, n_input_channels
    required_args = ['n_input', 'n_output', 'n_hidden', 'n_layers', 'n_input_channels']
    for arg_name in required_args:
        if arg_name not in model_args:
            # Provide defaults if compatible with MLP or raise error
            if arg_name == 'n_output': model_args[arg_name] = 1
            elif arg_name == 'n_hidden': model_args[arg_name] = 64 # Default from MLP
            elif arg_name == 'n_input_channels': model_args[arg_name] = 1 # Default from MLP
            else:
                print(f"Error: Missing required model argument '{arg_name}' for {model_class.__name__}")
                return None
                
    model = model_class(**model_args)
    try:
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
        
        # Load the entire saved object
        saved_object = torch.load(model_path, map_location=torch.device('cpu')) 

        # Check if the loaded object is a dictionary (typical for checkpoints)
        # and try to extract the actual state_dict
        if isinstance(saved_object, dict):
            # Common keys for model state_dict in a checkpoint
            possible_keys = ['model_state_dict', 'model', 'net', 'state_dict']
            state_dict = None
            for key in possible_keys:
                if key in saved_object:
                    state_dict = saved_object[key]
                    print(f"Extracted state_dict from checkpoint using key: '{key}'")
                    break
            if state_dict is None:
                # If none of the common keys are found, assume the dict itself might be the state_dict
                # (less common for .pt files that also contain optimizer state etc.)
                # Or, it might be that the top-level saved_object IS the state_dict (if not saved as a checkpoint dict)
                print("Warning: Could not find a common state_dict key in the checkpoint. Attempting to use the loaded dictionary directly as state_dict, or it might be a raw state_dict.")
                state_dict = saved_object # Fallback, might be an issue if it contains non-state_dict items like 'lrs'
        else:
            # If not a dict, assume it's the state_dict itself
            state_dict = saved_object
            print("Loaded object is not a dictionary, assuming it is the state_dict directly.")

        if not isinstance(state_dict, dict):
            print(f"Error: Extracted state_dict is not a dictionary. Type: {type(state_dict)}")
            return None
        
        # New logic to handle 'module.' prefix more robustly,
        # especially when state_dict contains mixed key types (e.g., model params and lrs params)
        processed_state_dict = {}
        has_module_prefix_stripped = False
        for k, v in state_dict.items(): # state_dict is the raw dict from file
            if k.startswith('module.'):
                processed_state_dict[k[len('module.'):]] = v
                has_module_prefix_stripped = True
            else:
                processed_state_dict[k] = v
        
        if has_module_prefix_stripped:
            print("Processed keys by stripping 'module.' prefix where found.")
        
        # Filter state_dict to only include keys present in the current model architecture
        # This uses processed_state_dict which has prefixes handled.
        current_model_keys = model.state_dict().keys()
        filtered_state_dict = {k: v for k, v in processed_state_dict.items() if k in current_model_keys}

        missing_keys_after_filter = [k for k in current_model_keys if k not in filtered_state_dict]
        if missing_keys_after_filter:
            print(f"Warning: After filtering, there are still missing keys expected by the model: {missing_keys_after_filter}")
        
        unexpected_keys_after_filter = [k for k in filtered_state_dict if k not in current_model_keys]
        if unexpected_keys_after_filter: # Should be empty if filter logic is k in current_model_keys
             print(f"Warning: After filtering, there are still unexpected keys: {unexpected_keys_after_filter}") # Should not happen

        # Load with strict=False to be more robust to minor mismatches if any, 
        # though filtering should ideally handle major ones.
        # strict=True is the default and more advisable if perfect matches are expected.
        # Given the 'lrs' keys, strict=False was implicitly needed or explicit filtering.
        model.load_state_dict(filtered_state_dict, strict=True) # Let's try strict=True first with filtering
        model.eval() 
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def get_layer_weight_norms(model):
    """Calculates L2 norm of weights for each Linear layer in the model."""
    norms = {}
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_name = f"layer_{layer_idx}_{name.replace('.', '_')}" 
            weight_norm = torch.linalg.norm(module.weight.data).item()
            norms[layer_name] = weight_norm
            layer_idx += 1
    return norms

def plot_weight_norms(norms_dict, title="Layer Weight Norms", save_path=None):
    """Plots a bar chart of layer weight norms."""
    if not norms_dict:
        print("No norms to plot.")
        return

    names = list(norms_dict.keys())
    values = list(norms_dict.values())

    plt.figure(figsize=(12, 7)) # Adjusted figure size for potentially more layers
    plt.bar(names, values, color='skyblue')
    plt.xlabel("Layer")
    plt.ylabel("L2 Norm of Weights")
    plt.title(title)
    plt.xticks(rotation=60, ha='right') # Rotated more for longer layer names
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

def main_analyze_single_metasgd_model():
    print("Analyzing a single MetaSGD model...")
    
    # Parameters derived from the filename:
    # concept_mlp_14_bits_feats16_depth7_adapt1_1stOrd_seed4_best_model_at_end_of_train.pt
    model_n_input = 16  # from feats16
    model_n_layers = 8 # from depth7 (but adjusted based on key inspection)
    
    # Parameters that are likely fixed for this set of experiments or have defaults in MLP
    model_n_output = 1
    model_n_hidden = 32 # Default in MLP class. Adjust if your experiments used a different value (e.g., 128)
    model_n_input_channels = 3 # Default in MLP class
    
    sample_model_path = "concept_mlp_14_bits_feats16_depth7_adapt1_1stOrd_seed4_best_model_at_end_of_train.pt" 
    
    print(f"Attempting to load: {sample_model_path}")
    
    model_args = {
        'n_input': model_n_input,
        'n_output': model_n_output,
        'n_hidden': model_n_hidden,
        'n_layers': model_n_layers,
        'n_input_channels': model_n_input_channels
    }

    loaded_model = load_model_weights(
        sample_model_path, 
        MLP,  # Use the actual MLP class
        **model_args
    )

    if loaded_model:
        print("\nModel Structure:")
        print(loaded_model)
        norms = get_layer_weight_norms(loaded_model)
        print("\nWeight Norms:")
        for layer, norm_val in norms.items():
            print(f"  {layer}: {norm_val:.4f}")
        
        plot_title = f"Weight Norms for\n{os.path.basename(sample_model_path)}" # Added newline for long titles
        plot_save_path = f"weight_norms_{os.path.basename(sample_model_path).replace('.pt', '.png')}"
        plot_weight_norms(norms, title=plot_title, save_path=plot_save_path)

# --- Utility to inspect keys in a .pt file ---
def inspect_saved_model_keys(model_path):
    print(f"\n--- Inspecting keys in: {model_path} ---")
    if not os.path.exists(model_path):
        print(f"Error: File not found at {model_path}")
        return

    saved_object = torch.load(model_path, map_location=torch.device('cpu'))
    
    state_dict_to_inspect = None
    actual_keys_to_print = None

    if isinstance(saved_object, dict):
        print("Saved object is a dictionary. Top-level keys:")
        for k in saved_object.keys():
            print(f"  - {k}")

        possible_keys = ['model_state_dict', 'model', 'net', 'state_dict']
        found_key = False
        for pk in possible_keys:
            if pk in saved_object and isinstance(saved_object[pk], dict):
                state_dict_to_inspect = saved_object[pk]
                print(f"Identified potential state_dict under key: '{pk}'")
                found_key = True
                break
        if not found_key:
            print("Could not find a common state_dict key. Assuming top-level dict might be the state_dict or contain it directly.")
            # This is tricky. If 'lrs' and model keys are mixed, this is the one to inspect.
            state_dict_to_inspect = saved_object 
    else:
        print("Saved object is not a dictionary. Assuming it is the state_dict directly.")
        state_dict_to_inspect = saved_object

    if isinstance(state_dict_to_inspect, dict):
        print("\nKeys found in the identified (or assumed) state_dict:")
        # Handle potential 'module.' prefix before printing keys for clarity
        temp_keys = list(state_dict_to_inspect.keys())
        if all(key.startswith('module.') for key in temp_keys):
            print("(Note: 'module.' prefix detected and will be conceptually stripped for listing)")
            actual_keys_to_print = sorted([k[len('module.'):] for k in temp_keys])
        else:
            actual_keys_to_print = sorted(temp_keys)
        
        for k_idx, k_val in enumerate(actual_keys_to_print):
            print(f"  {k_idx:02d}: {k_val}")
    else:
        print("Could not identify a dictionary to inspect for state_dict keys.")

if __name__ == "__main__":
    # sample_path_for_inspection = "concept_mlp_14_bits_feats16_depth7_adapt1_1stOrd_seed4_best_model_at_end_of_train.pt"
    # inspect_saved_model_keys(sample_path_for_inspection)
    
    # Comment out inspection and uncomment main analysis when ready
    main_analyze_single_metasgd_model()

    # Later, we will add functions to analyze baseline SGD model trajectories
    # and to perform comparative analyses. 