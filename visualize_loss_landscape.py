import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import loss_landscapes
import loss_landscapes.metrics as metrics
import random

# --- MLP Model Definition (consistent with other analysis scripts) ---
class MLP(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int = 1,
        n_hidden: int = 32,
        n_layers: int = 8,
        n_input_channels: int = 3,
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

# --- Model Loading Function (adapted from compare_avg_model_weights_faceted.py) ---
def load_model_for_landscape(model_path_str, model_class, **model_args):
    model = model_class(**model_args)
    try:
        if not os.path.exists(model_path_str):
            print(f"Error: Model file not found at {model_path_str}")
            return None
        
        # Try with weights_only=True first for security and to avoid loading optimizer states etc.
        try:
            # print(f"Attempting torch.load with weights_only=True for {model_path_str}")
            saved_object = torch.load(model_path_str, map_location=torch.device('cpu'), weights_only=True)
        except Exception as e_wo_true:
            # print(f"torch.load with weights_only=True failed for {model_path_str}: {e_wo_true}. Falling back to weights_only=False.")
            try:
                saved_object = torch.load(model_path_str, map_location=torch.device('cpu'), weights_only=False)
                # If this works, it means the file likely contains more than just weights (e.g. pickled objects or full MetaSGD state)
                # The FutureWarning will be emitted by PyTorch in this case.
            except Exception as e_wo_false:
                print(f"Error: torch.load failed with both weights_only=True and weights_only=False for {model_path_str}: {e_wo_false}")
                return None

        actual_model_state_dict = None

        if isinstance(saved_object, dict):
            # Scenario 1: Checkpoint from l2l.algorithms.MetaSGD.save_checkpoint()
            # This saves a dict like {'model_state_dict': ..., 'optimizer_state_dict': ...}
            if 'model_state_dict' in saved_object and isinstance(saved_object['model_state_dict'], dict):
                actual_model_state_dict = saved_object['model_state_dict']
                # print(f"  Extracted 'model_state_dict' from saved_object for {model_path_str}")
            # Scenario 2: Checkpoint is the state_dict of the l2l.algorithms.MetaSGD object itself
            # This happens if torch.save(meta.state_dict(), path) was used.
            # The MetaSGD state_dict contains 'module' (the actual model), 'lrs', etc.
            elif 'module' in saved_object and isinstance(saved_object['module'], dict):
                actual_model_state_dict = saved_object['module']
                # print(f"  Extracted 'module' (as model's state_dict) from saved_object for {model_path_str}")
            # Scenario 3: Checkpoint is the state_dict of the raw model (no MetaSGD wrapper)
            elif all(isinstance(v, torch.Tensor) for k, v in saved_object.items() if k != 'adapt_steps'): # Heuristic: looks like a state_dict
                # print(f"  Assuming saved_object itself is the model's state_dict for {model_path_str}")
                actual_model_state_dict = saved_object
            else: # Fallback if it's a dict but not matching known patterns
                # print(f"  Warning: saved_object is a dict but structure unclear for {model_path_str}. Trying to use it directly.")
                actual_model_state_dict = saved_object

        elif isinstance(saved_object, nn.Module): # Scenario 4: Full model object saved
            # print(f"  Saved_object is an nn.Module instance; extracting its state_dict for {model_path_str}")
            actual_model_state_dict = saved_object.state_dict()
        
        else: # Scenario 5: Raw state_dict saved (not as a dict, but directly as the object - less common)
            if hasattr(saved_object, 'keys') and callable(saved_object.keys): # Check if it acts like a dict
                 # print(f"  Saved_object is not a dict or nn.Module, but has keys. Assuming it's a state_dict for {model_path_str}.")
                 actual_model_state_dict = saved_object
            else:
                print(f"Error: Unhandled type for saved_object: {type(saved_object)} for {model_path_str}")
                return None


        if actual_model_state_dict is None:
            print(f"Error: Could not extract the actual model's state_dict from {model_path_str}")
            return None

        if not isinstance(actual_model_state_dict, dict):
            print(f"Error: Extracted actual_model_state_dict is not a dictionary. Type: {type(actual_model_state_dict)} for {model_path_str}")
            return None
        
        # Strip 'module.' prefix if it exists from keys in actual_model_state_dict
        # This is common if the model within MetaSGD was DataParallel or if MetaSGD itself adds it.
        processed_model_keys = {}
        for k, v in actual_model_state_dict.items():
            if k.startswith('module.'):
                processed_model_keys[k[len('module.'):]] = v
            else:
                processed_model_keys[k] = v
        
        # Filter to only keys that the current MLP instance expects
        model_prototype_keys = model.state_dict().keys()
        filtered_for_mlp = {k: v for k, v in processed_model_keys.items() if k in model_prototype_keys}
        
        missing_keys = [k for k in model_prototype_keys if k not in filtered_for_mlp]
        unexpected_keys = [k for k in filtered_for_mlp if k not in model_prototype_keys] # Should be empty due to filter
        extra_keys_before_filter = [k for k in processed_model_keys if k not in model_prototype_keys]


        if missing_keys:
            print(f"ERROR: Missing keys when loading state_dict into MLP for {model_path_str}: {missing_keys}")
            # print(f"  MLP expects: {list(model_prototype_keys)}")
            # print(f"  Processed keys from file (after 'module.' strip): {list(processed_model_keys.keys())}")
            # print(f"  Extra keys found before filtering: {extra_keys_before_filter}")
            return None # Strict about missing keys for the model itself
        
        if extra_keys_before_filter:
             print(f"Warning: Ignored unexpected keys from checkpoint for {model_path_str}: {extra_keys_before_filter}")
             # These are keys like 'lrs.X' which we don't want to load into the MLP model itself.

        model.load_state_dict(filtered_for_mlp, strict=True) # Should be strict now.
        
        # print(f"Successfully loaded model from {model_path_str}")
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model from {model_path_str}: {e}")
        return None
# --- End Model Loading Function ---

# --- Data Generation ---
# Attempt to import necessary components for data generation
try:
    from pcfg import sample_concept_from_pcfg, evaluate_pcfg_concept, DEFAULT_MAX_DEPTH as PCFG_DEFAULT_MAX_DEPTH_FROM_PCFG_FILE
    DATA_GEN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import data generation components from pcfg.py: {e}. Data generation will not be available.")
    DATA_GEN_AVAILABLE = False
    sample_concept_from_pcfg, evaluate_pcfg_concept, PCFG_DEFAULT_MAX_DEPTH_FROM_PCFG_FILE = None, None, 5 # Fallback

def get_concept_data(features, concept_depth_from_filename, num_samples, pcfg_max_depth_for_sampling, seed=42):
    if not DATA_GEN_AVAILABLE:
        print("Error: Data generation components from pcfg.py not available. Cannot generate concept data.")
        return None, None

    # Set a consistent seed for reproducibility of the sampled task/concept for this landscape
    # This seed will determine THE concept chosen for this (F,D) pair.
    random.seed(seed) # pcfg.py uses random.random()
    np.random.seed(seed) 
    torch.manual_seed(seed)

    # 1. Sample a single, fixed concept for this (F, D) pair.
    # The concept_depth_from_filename arg in this script refers to the 'depthX' in filenames,
    # which corresponds to max_depth in pcfg.sample_concept_from_pcfg.
    # We use pcfg_max_depth_for_sampling, which should be args.concept_depth_val.
    print(f"  Sampling concept with num_features={features}, max_depth={pcfg_max_depth_for_sampling} (using seed {seed} for concept selection)")
    fixed_concept_expr, literals, expr_depth = sample_concept_from_pcfg(
        num_features=features,
        max_depth=pcfg_max_depth_for_sampling
    )
    print(f"  Sampled fixed concept for landscape: {fixed_concept_expr} (Literals: {literals}, Depth: {expr_depth})")

    # 2. Generate random binary input vectors.
    # The seed for np.random for *data point generation* should be different from concept selection if we want varied data for a fixed concept.
    # However, for landscape stability, using the same seed for both concept and data points is fine too.
    # Let's stick to the single seed for now for simplicity and exact reproducibility of the dataset.
    
    # Generate num_samples x features binary vectors
    # Each row is an input vector, values are 0 or 1.
    input_vectors_np = np.random.randint(0, 2, size=(num_samples, features))

    # 3. Label these input vectors using the fixed concept.
    labels_np = np.array([evaluate_pcfg_concept(fixed_concept_expr, vec) for vec in input_vectors_np], dtype=np.float32)
    labels_np = labels_np.reshape(-1, 1) # Reshape to (num_samples, 1)

    inputs_tensor = torch.tensor(input_vectors_np, dtype=torch.float32)
    targets_tensor = torch.tensor(labels_np, dtype=torch.float32)
    
    print(f"Generated data for F={features}, D(file)={concept_depth_from_filename}: inputs shape {inputs_tensor.shape}, targets shape {targets_tensor.shape}")
    return inputs_tensor, targets_tensor
# --- End Data Generation ---


def main():
    parser = argparse.ArgumentParser(description="Visualize loss landscapes for concept learning models.")
    parser.add_argument("--organized_checkpoint_dir", type=str, required=True,
                        help="Base directory of organized checkpoints (e.g., .../organized_checkpoints/).")
    parser.add_argument("--output_dir", type=str, default="results/loss_landscapes",
                        help="Directory to save plots.")
    parser.add_argument("--run_name_prefix", type=str, default="landscape", 
                        help="Prefix for plot filenames.")

    # Model and Task Configuration
    parser.add_argument("--features_list", type=int, nargs='+', default=[8, 16, 32],
                        help="List of feature sizes to analyze.")
    parser.add_argument("--concept_depth_val", type=int, default=3,
                        help="Concept recursion depth (from filename) - fixed for these plots.")
    parser.add_argument("--seed_to_load", type=int, default=0, help="Seed number for the models to load.")
    parser.add_argument("--epoch_to_load", type=str, default="100", help="Epoch string for checkpoint.")
    parser.add_argument("--hyper_param_idx_in_filename", type=str, default="14")
    parser.add_argument("--adapt_steps_in_filename", type=int, default=1)

    # MLP Architecture (fixed)
    parser.add_argument("--mlp_n_layers", type=int, default=8)
    parser.add_argument("--mlp_n_hidden", type=int, default=32)
    parser.add_argument("--mlp_n_input_channels", type=int, default=3)

    # Landscape parameters
    parser.add_argument("--num_val_samples", type=int, default=1000,
                        help="Number of validation samples for loss evaluation.")
    parser.add_argument("--interpolation_steps", type=int, default=50)
    parser.add_argument("--contour_steps", type=int, default=30)
    parser.add_argument("--data_gen_seed", type=int, default=42, help="Seed for generating the fixed validation dataset for landscape.")


    args = parser.parse_args()
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Loss Function ---
    # Assuming Binary Cross Entropy with Logits, common for classification
    criterion = nn.BCEWithLogitsLoss()

    # --- Fixed MLP architecture part (from args) ---
    base_model_arch_args = {
        'n_output': 1,
        'n_hidden': args.mlp_n_hidden,
        'n_layers': args.mlp_n_layers,
        'n_input_channels': args.mlp_n_input_channels
    }

    for features_val in args.features_list:
        print(f"\\n--- Processing Landscape for Features: {features_val}, Concept Depth (file): {args.concept_depth_val} ---")
        
        current_output_dir = base_output_dir / f"feats{features_val}_depth{args.concept_depth_val}"
        current_output_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Generate/Load Validation Data for this specific (F, D) task ---
        print(f"  Generating validation data (seed {args.data_gen_seed})...")
        inputs, targets = get_concept_data(features_val, args.concept_depth_val, args.num_val_samples, 
                                           pcfg_max_depth_for_sampling=args.concept_depth_val, 
                                           seed=args.data_gen_seed)
        if inputs is None or targets is None:
            print(f"  Skipping F={features_val}, D={args.concept_depth_val} due to data generation failure.")
            continue
        
        # --- 2. Load Models (1st and 2nd Order for the current F, D, Seed) ---
        model_arch_args = {**base_model_arch_args, 'n_input': features_val}
        model1_1st = None
        model2_2nd = None

        for order_short, order_long_name in [("1stOrd", "1stOrder"), ("2ndOrd", "2ndOrder")]:
            primary_folder = f"feats{features_val}_depth{args.concept_depth_val}_adapt{args.adapt_steps_in_filename}_{order_short}"
            seed_folder = f"seed{args.seed_to_load}"
            filename = f"concept_mlp_{args.hyper_param_idx_in_filename}_bits_feats{features_val}_depth{args.concept_depth_val}_adapt{args.adapt_steps_in_filename}_{order_short}_seed{args.seed_to_load}_epoch_{args.epoch_to_load}.pt"
            model_path = Path(args.organized_checkpoint_dir) / primary_folder / seed_folder / filename
            
            print(f"  Attempting to load {order_long_name} model from: {model_path}")
            current_model = load_model_for_landscape(str(model_path), MLP, **model_arch_args)

            if current_model:
                print(f"    Successfully loaded {order_long_name} model.")
                if order_short == "1stOrd":
                    model1_1st = current_model
                else:
                    model2_2nd = current_model
            else:
                print(f"    Failed to load {order_long_name} model.")

        # --- 3. Perform Landscape Visualizations ---
        
        # 3a. 1D Linear Interpolation (1st Order <-> 2nd Order)
        if model1_1st and model2_2nd:
            print("  Starting 1D Linear Interpolation (1st <-> 2nd Order)...")
            try:
                metric = metrics.Loss(criterion, inputs, targets)
                loss_data_interp = loss_landscapes.linear_interpolation(
                    model1_1st, model2_2nd, metric, args.interpolation_steps, deepcopy_model=True
                )

                plt.figure()
                plot_coords = []
                plot_losses = []
                print(f"    DEBUG 1D: loss_data_interp type: {type(loss_data_interp)}")
                
                if isinstance(loss_data_interp, np.ndarray):
                    print(f"    DEBUG 1D: loss_data_interp is a NumPy array with shape {loss_data_interp.shape}. Assuming it contains only loss values.")
                    # Ensure it's 1D
                    if loss_data_interp.ndim == 1:
                        plot_losses = [float(val) for val in loss_data_interp] # Convert numpy floats to python floats
                        plot_coords = np.linspace(0, 1, num=len(plot_losses)).tolist()
                        print(f"      DEBUG 1D: Generated {len(plot_coords)} coords for {len(plot_losses)} losses.")
                    else:
                        print(f"    Error 1D: loss_data_interp is a NumPy array, but not 1D. Shape: {loss_data_interp.shape}. Cannot process for 1D plot.")

                elif isinstance(loss_data_interp, list):
                    print(f"    DEBUG 1D: loss_data_interp length: {len(loss_data_interp)}")
                    if len(loss_data_interp) > 0:
                        print(f"    DEBUG 1D: First item in loss_data_interp: {loss_data_interp[0]}, type: {type(loss_data_interp[0])}")

                        for i, item in enumerate(loss_data_interp):
                            print(f"      DEBUG 1D: Processing item {i}: {item}, type: {type(item)}")
                            if isinstance(item, (list, tuple)) and len(item) == 2:
                                coord, val = item
                                print(f"        DEBUG 1D: Unpacked coord: {coord}, val: {val} (type: {type(val)})")
                                plot_coords.append(coord)
                                current_loss = np.nan # Default to NaN
                                if torch.is_tensor(val):
                                    current_loss = val.item()
                                elif hasattr(val, 'item') and callable(getattr(val, 'item')): # For numpy types like np.float64
                                    try:
                                        current_loss = float(val.item())
                                    except TypeError as te:
                                        print(f"        DEBUG 1D: val.item() failed for numpy type: {te}. val was {val}. Using float(val).")
                                        current_loss = float(val) # Fallback
                                elif isinstance(val, (float, int, np.number)): # Handles Python floats/ints and numpy numbers
                                    current_loss = float(val)
                                else:
                                    print(f"        Warning 1D: val for coord {coord} is of unhandled type {type(val)}: {val}. Using NaN.")
                                plot_losses.append(current_loss)
                            elif isinstance(item, (float, int, np.number)): # If item itself is a number
                                 print(f"      Warning 1D: Item {i} in loss_data_interp is a raw number: {item}. Assuming it's a loss value, but coordinate is missing. Skipping.")
                            else:
                                print(f"      Warning 1D: Unexpected item format in loss_data_interp at index {i}: {item} (Type: {type(item)}). Skipping this point.")
                else: # If loss_data_interp is not a list or NumPy array
                    print(f"    Error 1D: loss_data_interp is not a list or NumPy array as expected. Value: {loss_data_interp}")


                if plot_coords and plot_losses:
                    plt.plot(plot_coords, plot_losses)
                else:
                    print("    Error: No valid data points to plot for 1D interpolation.")
                
                plt.title(f"Loss Landscape (1D Interpolation)\\nFeats: {features_val}, Depth: {args.concept_depth_val} (Seed {args.seed_to_load})\\n1st Order (0.0) vs 2nd Order (1.0)")
                plt.xlabel("Interpolation Path (0 = 1st Order, 1 = 2nd Order)")
                plt.ylabel("Loss")
                plt.grid(True)
                plot_path = current_output_dir / f"{args.run_name_prefix}_F{features_val}D{args.concept_depth_val}_S{args.seed_to_load}_1D_1st_vs_2nd.png"
                plt.savefig(plot_path)
                plt.close()
                print(f"    Saved 1D interpolation plot to {plot_path}")
            except Exception as e:
                print(f"    Error during 1D interpolation for F{features_val}D{args.concept_depth_val}: {e}")
        else:
            print("  Skipping 1D interpolation (one or both models not loaded).")

        # 3b. 2D Contour Plot (around each model solution)
        for model_to_plot, model_name_suffix in [(model1_1st, "1stOrder"), (model2_2nd, "2ndOrder")]:
            if model_to_plot:
                print(f"  Starting 2D Contour Plot for {model_name_suffix} solution...")
                try:
                    metric = metrics.Loss(criterion, inputs, targets)
                    loss_data_plane = loss_landscapes.random_plane(
                        model_to_plot, metric, distance=1.0, steps=args.contour_steps, normalization='filter', deepcopy_model=True
                    ) # distance=1.0 is an example, might need tuning
                    
                    print(f"    DEBUG 2D: type(loss_data_plane): {type(loss_data_plane)}")
                    if isinstance(loss_data_plane, (list, tuple)):
                        print(f"    DEBUG 2D: len(loss_data_plane): {len(loss_data_plane)}")
                        for idx, elem in enumerate(loss_data_plane):
                            print(f"    DEBUG 2D: Element {idx} type: {type(elem)}, Shape (if np.ndarray): {getattr(elem, 'shape', 'N/A')}")
                    elif isinstance(loss_data_plane, np.ndarray):
                         print(f"    DEBUG 2D: loss_data_plane is ndarray with shape: {loss_data_plane.shape}")
                    else:
                        print(f"    DEBUG 2D: loss_data_plane is some other type. Value: {loss_data_plane}")

                    # Expected: x_coords (2D np.array), y_coords (2D np.array), losses_2d (2D np.array)
                    if isinstance(loss_data_plane, tuple) and len(loss_data_plane) == 3:
                        x_coords, y_coords, losses_2d = loss_data_plane
                        
                        # Ensure x_coords, y_coords, losses_2d are numpy arrays for contourf
                        if not (isinstance(x_coords, np.ndarray) and isinstance(y_coords, np.ndarray) and isinstance(losses_2d, np.ndarray)):
                            print(f"    Warning 2D (Tuple Path): x_coords, y_coords, or losses_2d are not all numpy arrays after unpacking.")
                            print(f"      x_coords type: {type(x_coords)}, y_coords type: {type(y_coords)}, losses_2d type: {type(losses_2d)}")
                        
                        # Ensure they are 2D for contourf (loss_landscapes should provide this)
                        if not (x_coords.ndim == 2 and y_coords.ndim == 2 and losses_2d.ndim == 2):
                             print(f"    Warning 2D (Tuple Path): x_coords, y_coords, or losses_2d are not all 2D arrays.")
                             print(f"      x_coords dims: {x_coords.ndim}, y_coords dims: {y_coords.ndim}, losses_2d dims: {losses_2d.ndim}")
                        
                        ready_to_plot = True

                    elif isinstance(loss_data_plane, np.ndarray) and loss_data_plane.ndim == 2:
                        print(f"    DEBUG 2D (Array Path): loss_data_plane is a 2D NumPy array with shape {loss_data_plane.shape}. Assuming it contains only loss values.")
                        losses_2d = loss_data_plane
                        # Generate coordinates. Assuming normalized directions, spanning -0.5 to 0.5 based on distance=1.0 used in random_plane call.
                        # The number of points should match the dimensions of losses_2d.
                        x_coords_1d = np.linspace(-0.5, 0.5, num=losses_2d.shape[1])
                        y_coords_1d = np.linspace(-0.5, 0.5, num=losses_2d.shape[0])
                        x_coords, y_coords = np.meshgrid(x_coords_1d, y_coords_1d)
                        print(f"      DEBUG 2D (Array Path): Generated x_coords (shape {x_coords.shape}) and y_coords (shape {y_coords.shape}) for losses (shape {losses_2d.shape}).")
                        ready_to_plot = True
                    
                    else:
                        print(f"    Error 2D: loss_data_plane was not a 3-tuple or a 2D NumPy array as expected. Cannot unpack/use for contour plot. Skipping this plot.")
                        ready_to_plot = False

                    if ready_to_plot:
                        plt.figure(figsize=(8,6))
                        plt.contourf(x_coords, y_coords, losses_2d, levels=50)
                        plt.colorbar(label="Loss")
                        plt.title(f"Loss Landscape (2D Contour around {model_name_suffix})\\nFeats: {features_val}, Depth: {args.concept_depth_val} (Seed {args.seed_to_load})")
                        plt.xlabel("X (Normalized Random Direction)")
                        plt.ylabel("Y (Normalized Random Direction)")
                        plot_path = current_output_dir / f"{args.run_name_prefix}_F{features_val}D{args.concept_depth_val}_S{args.seed_to_load}_2D_{model_name_suffix}.png"
                        plt.savefig(plot_path)
                        plt.close()
                        print(f"    Saved 2D contour plot for {model_name_suffix} to {plot_path}")

                except Exception as e:
                    print(f"    Error during 2D contour plot for {model_name_suffix}, F{features_val}D{args.concept_depth_val}: {e}")
            else:
                print(f"  Skipping 2D contour plot for {model_name_suffix} (model not loaded).")
    
    print("\\nLoss landscape visualization script finished.")

if __name__ == "__main__":
    # This check is important for users to ensure the library is there.
    try:
        import loss_landscapes
    except ImportError:
        print("ERROR: The 'loss-landscapes' library is not installed.")
        print("Please install it by running: pip install loss-landscapes")
        exit(1)
        
    # Check for data generation components early too
    if not DATA_GEN_AVAILABLE:
        print("ERROR: Critical components for data generation (sample_concept_from_pcfg, etc. from pcfg.py) could not be imported.")
        print("Please ensure pcfg.py is in the same directory or accessible in the Python path.")
        # We might not exit here if user wants to run with pre-generated data later,
        # but for now, data generation is integral.
        # exit(1) # Optionally exit if data gen is strictly required now.

    main()

# Example Usage (adjust paths and details as needed):
# python visualize_loss_landscape.py \
#   --organized_checkpoint_dir "/scratch/gpfs/mg7411/ManyPaths/organized_checkpoints" \
#   --output_dir "results/loss_landscapes_depth3" \
#   --run_name_prefix "concept_landscape_d3" \
#   --features_list 8 16 32 \
#   --concept_depth_val 3 \
#   --seed_to_load 0 \
#   --epoch_to_load "100" \
#   --hyper_param_idx_in_filename "14" \
#   --adapt_steps_in_filename 1 \
#   --mlp_n_layers 8 \
#   --mlp_n_hidden 32 \
#   --mlp_n_input_channels 3 \
#   --num_val_samples 1000 \
#   --interpolation_steps 50 \
#   --contour_steps 30 \
#   --data_gen_seed 123 