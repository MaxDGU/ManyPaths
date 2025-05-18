import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import pandas as pd # For saving trajectory to CSV

# print(f"[DEBUG main.py] Initial sys.path before any local changes: {list(sys.path)}") # Optional: if you want to see it very early

# Keep these for context if needed, but they won't be used for sys.path manipulation now.
project_root = os.path.abspath(os.path.dirname(__file__))
learn2learn_repo_root = os.path.join(project_root, 'learn2learn')

print(f"[DEBUG main.py] Project root (not modifying sys.path with this): {project_root}")
print(f"[DEBUG main.py] Potential learn2learn repo root (not modifying sys.path with this): {learn2learn_repo_root}")

print("[DEBUG main.py] sys.path before importing learn2learn (no local script modifications to sys.path):")
for i, p in enumerate(sys.path):
    print(f"  sys.path[{i}]: {p}")

import learn2learn as l2l
print(f"[DEBUG main.py] learn2learn imported in main.py. Path: {getattr(l2l, '__file__', 'N/A')}")
print(f"[DEBUG main.py] learn2learn version in main.py: {getattr(l2l, '__version__', 'N/A')}")

import matplotlib.pyplot as plt

from training import meta_train, hyper_search
from evaluation import evaluate
from initialization import init_dataset, init_model, init_misc
from utils import set_random_seeds, save_model, get_collate
from visualize import plot_loss, plot_meta_test_results
from constants import *
from generate_concepts import PCFG_DEFAULT_MAX_DEPTH # Import for default value

CHECKPOINT_SUBDIR = "checkpoints" # Subdirectory within saved_models for periodic checkpoints
SAVED_DATASETS_DIR = "saved_datasets" # Directory for saving generated datasets

def main(
    seed: int = 0,
    experiment: str = "mod",  # ["mod", "concept", "omniglot"]
    m: str = "mlp",  # ['mlp', 'cnn', 'lstm', 'transformer']
    data_type: str = "bits",  # Default to bits as per user focus for concept learning
    a: str = "asian", # ['ancient', 'asian', 'all'] ... only for omniglot
    epochs: int = 1000,  # until convergence
    tasks_per_meta_batch: int = 4,
    adaptation_steps: int = 1, # This is 'k'
    outer_lr: float = 1e-3,
    inner_lr: float = 1e-2, # Added for MetaSGD's own lr (fast_lr)
    skip: int = 1,
    no_hyper_search: bool = False,
    plot: bool = False,
    save: bool = False,
    # Corrected default values
    num_concept_features: int = 8,
    pcfg_max_depth: int = PCFG_DEFAULT_MAX_DEPTH, # Added pcfg_max_depth
    first_order_meta_sgd: bool = False,
    hyper_index: int | None = None,  # New parameter for specific hyperparameter index
    patience: int = 10000 # New parameter for patience
):
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else ("cpu" if m in ["mlp", "lstm"] else "mps")
    )
    print(f"Device: {device}")

    set_random_seeds(seed)
    
    # init dataset related misc values
    # Pass num_concept_features to init_misc to determine 'bits' for model input size
    alphabet, bits_for_model, channels, n_output = init_misc(experiment, a, num_concept_features_override=num_concept_features if experiment == "concept" and data_type == "bits" else None)
    
    collate_fn = get_collate(experiment, device)
    
    # --- Construct dataset save paths ---
    os.makedirs(SAVED_DATASETS_DIR, exist_ok=True) # Ensure base dataset directory exists
    
    dataset_prefix_parts = [experiment, m, data_type]
    if experiment == "concept" and data_type == "bits": # Specific to concept learning datasets
        dataset_prefix_parts.append(f"feats{num_concept_features}")
        dataset_prefix_parts.append(f"depth{pcfg_max_depth}")
    elif experiment == "mod":
        dataset_prefix_parts.append(f"skip{skip}") # skip is an arg to main
    elif experiment == "omniglot":
        dataset_prefix_parts.append(a) # a is alphabet choice, an arg to main
    dataset_prefix_parts.append(f"seed{seed}")
    dataset_file_prefix = "_".join(dataset_prefix_parts)

    ds_train_path = os.path.join(SAVED_DATASETS_DIR, f"{dataset_file_prefix}_train_dataset.pt")
    ds_val_path = os.path.join(SAVED_DATASETS_DIR, f"{dataset_file_prefix}_val_dataset.pt")
    ds_test_path = os.path.join(SAVED_DATASETS_DIR, f"{dataset_file_prefix}_test_dataset.pt")

    print(f"Attempting to save train dataset to: {ds_train_path}")
    print(f"Attempting to save validation dataset to: {ds_val_path}")
    print(f"Attempting to save test dataset to: {ds_test_path}")

    # init dataset
    # Pass num_concept_features to init_dataset for MetaBitConceptsDataset
    train_dataset, test_dataset, val_dataset = init_dataset(
        experiment, 
        model_arch=m, # Renamed: m is the model architecture string
        data_type=data_type, 
        skip_param=skip, # Renamed: skip is the skip parameter
        alphabet=alphabet, 
        num_concept_features=num_concept_features if experiment == "concept" and data_type == "bits" else 4, 
        pcfg_max_depth=pcfg_max_depth if experiment == "concept" and data_type == "bits" else PCFG_DEFAULT_MAX_DEPTH,
        # Provide paths for saving datasets
        save_train_path=ds_train_path,
        save_val_path=ds_val_path,
        save_test_path=ds_test_path
        # load_X_path are None by default, so datasets will be generated and then saved.
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=tasks_per_meta_batch,
        shuffle=True,
        drop_last=True,
        pin_memory=(device == "cuda:0"),
        collate_fn=collate_fn,
    )

    if no_hyper_search:
        if hyper_index is not None:
            index = hyper_index
            print(f"Using specified hyperparameter index: {index}")
        else:
            index = DEFAULT_INDEX
            print(f"Using default hyperparameter index: {index}")
    else:
        # Note: hyper_search also creates its own MetaSGD instance. 
        # This might need adjustment if we want --first-order to apply during hyper_search too.
        # For now, focusing on the main training loop.
        # The hyper_search call itself doesn't currently take first_order_meta_sgd or inner_lr.
        print("Running hyperparameter search...")
        index = hyper_search(
            experiment,
            m,
            data_type,
            outer_lr, # Outer LR for AdamW in hyper_search
            # inner_lr is not directly passed to hyper_search's MetaSGD, it uses fixed 1e-3
            train_loader,
            val_dataset,
            test_dataset,
            device,
            channels=channels,
            bits=bits_for_model, # n_input for model
            n_output=n_output,
            # Pass tasks_per_meta_batch and adaptation_steps to hyper_search as well
            tasks_per_meta_batch=tasks_per_meta_batch,
            adaptation_steps=adaptation_steps,
            epochs=10 # Reduced epochs for hyper_search for speed, can be configured
        )
        print(f"Hyperparameter search concluded. Best index: {index}")

    # init meta-learner, loss, and meta-optimizer
    model = init_model(
        m, data_type, index=index, verbose=True, channels=channels, bits=bits_for_model, n_output=n_output
    ).to(device)
    
    # Use the new first_order_meta_sgd flag and inner_lr here
    meta = l2l.algorithms.MetaSGD(model, lr=inner_lr, first_order=first_order_meta_sgd).to(device)
    
    criterion = nn.MSELoss() if experiment == "mod" else nn.BCEWithLogitsLoss() if experiment == "concept" else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(meta.parameters(), lr=outer_lr)

    # meta-training
    # meta_train now returns 5 values
    val_losses, val_accuracies, grad_alignments, best_model_state_at_end, periodic_checkpoints = meta_train(
        meta,
        train_loader,
        val_dataset,
        criterion,
        optimizer,
        device,
        epochs,
        tasks_per_meta_batch,
        adaptation_steps,
        verbose=True,
        patience=patience 
    )

    # Construct filename prefix for saving outputs
    file_prefix_parts = [experiment, m, str(index), data_type]
    if experiment == "mod":
        file_prefix_parts.append(str(skip))
    elif experiment == "concept":
        file_prefix_parts.append(f"feats{num_concept_features}")
        file_prefix_parts.append(f"depth{pcfg_max_depth}") # Add depth to filename
        file_prefix_parts.append(f"adapt{adaptation_steps}")
        file_prefix_parts.append("1stOrd" if first_order_meta_sgd else "2ndOrd")
    elif experiment == "omniglot":
        file_prefix_parts.append(a)
    file_prefix_parts.append(f"seed{seed}") # Added seed to prefix
    file_prefix = "_".join(file_prefix_parts)

    results_dir = "results"
    saved_models_dir = "saved_models"
    checkpoints_fulldir = os.path.join(saved_models_dir, CHECKPOINT_SUBDIR)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(checkpoints_fulldir, exist_ok=True)

    # Save learning trajectory (validation losses, accuracies, and gradient alignments)
    # Ensure all lists are of the same length, use np.nan for missing grad_alignments if any edge cases
    num_logs = len(val_losses)
    processed_grad_alignments = grad_alignments[:num_logs] if grad_alignments else [np.nan] * num_logs
    if len(processed_grad_alignments) < num_logs:
        processed_grad_alignments.extend([np.nan] * (num_logs - len(processed_grad_alignments)))

    trajectory_data = {
        'log_step': list(range(1, num_logs + 1)), 
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'grad_alignment': processed_grad_alignments # Added grad_alignment
    }
    trajectory_df = pd.DataFrame(trajectory_data)
    # Define trajectory_filename based on a new run_name argument or a default
    # For now, let's assume a way to specify output subdirectories for different runs
    # This will be handled when parameterizing analysis scripts.
    # Example: results_run_subdir = args.run_name if args.run_name else "run_default"
    # trajectory_filename = os.path.join(results_dir, results_run_subdir, f"{file_prefix}_trajectory.csv")
    # For now, keep existing save path structure, will adjust when parameterizing analysis
    trajectory_filename = os.path.join(results_dir, f"{file_prefix}_trajectory.csv")
    os.makedirs(os.path.dirname(trajectory_filename), exist_ok=True) # Ensure dir exists if using subdirs

    trajectory_df.to_csv(trajectory_filename, index=False)
    print(f"Saved learning trajectory to {trajectory_filename}")

    if plot:
        # Pass actual validation losses and accuracies to plot_loss
        # plot_loss might need to be adapted to handle this data (e.g. two subplots)
        plot_loss([val_losses, val_accuracies], ["Validation Loss", "Validation Accuracy"]) 

    # Save the final best model
    if save and best_model_state_at_end:
        final_model_filename = os.path.join(saved_models_dir, f"{file_prefix}_best_model.pt")
        torch.save(best_model_state_at_end, final_model_filename)
        print(f"Saved best model (from end of training) to {final_model_filename}")
    elif save:
        print(f"Warning: No best model state found from meta-training. Final best model not saved.")

    # Save periodic checkpoints
    if save and periodic_checkpoints:
        print(f"Saving {len(periodic_checkpoints)} periodic checkpoints...")
        for ep_num, state_dict in periodic_checkpoints:
            chkpt_filename = os.path.join(checkpoints_fulldir, f"{file_prefix}_episode_{ep_num}.pt")
            torch.save(state_dict, chkpt_filename)
        print(f"Periodic checkpoints saved to {checkpoints_fulldir}")
    elif save:
        print("No periodic checkpoints to save.")

    # Load the best model for final evaluation if it was saved
    if best_model_state_at_end:
        meta.load_state_dict(best_model_state_at_end)
        print("Loaded best model state for final evaluation.")
    else:
        print("Warning: No best model state available for final evaluation. Evaluating current model state.")

    if experiment == "mod" and plot:
        _, results = evaluate(
            meta,
            val_dataset,
            criterion,
            device,
            [0, adaptation_steps],
            return_results=True,
        )
        plot_meta_test_results(results)
        _, results = evaluate(
            meta,
            test_dataset,
            criterion,
            device,
            [0, adaptation_steps],
            return_results=True,
        )
        plot_meta_test_results(results)
        plt.show()
    elif experiment == "concept" and plot:
        # Basic evaluation for concept learning after training
        print("Evaluating final model on a sample of validation tasks...")
        val_loss, val_acc = evaluate(meta, val_dataset, criterion, device, [0, adaptation_steps])
        print(f"Final Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        print("Evaluating final model on a sample of test tasks...")
        test_loss, test_acc = evaluate(meta, test_dataset, criterion, device, [0, adaptation_steps])
        print(f"Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Plotting for concept can be added if specific visualizations are needed
        # For now, just printing metrics.
        # Example: plot_meta_test_results could be adapted if results format matches

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main script")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--experiment", type=str, default="mod", choices=["mod", "concept", "omniglot"], help="Experiment type")
    parser.add_argument("--m", type=str, default="mlp", choices=["mlp", "cnn", "lstm", "transformer"], help="Model type")
    parser.add_argument("--data-type", type=str, default="bits", choices=["bits", "concept", "omniglot"], help="Data type")
    parser.add_argument("--a", type=str, default="asian", choices=["ancient", "asian", "all"], help="Additional data type")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--tasks_per_meta_batch", type=int, default=4, help="Tasks per meta batch")
    parser.add_argument("--adaptation-steps", type=int, default=1, help="Adaptation steps")
    parser.add_argument("--outer_lr", type=float, default=1e-3, help="Outer learning rate")
    parser.add_argument("--inner_lr", type=float, default=1e-2, help="Inner learning rate")
    parser.add_argument("--skip", type=int, default=1, help="Skip parameter")
    parser.add_argument("--no_hyper_search", action="store_true", help="Skip hyperparameter search")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    parser.add_argument("--save", action="store_true", help="Save model and checkpoints")
    parser.add_argument("--num-concept-features", type=int, default=8, dest="num_concept_features", help="Number of features for the binary concept learning task when experiment is 'concept' and data_type is 'bits'")
    parser.add_argument("--pcfg-max-depth", type=int, default=PCFG_DEFAULT_MAX_DEPTH, dest="pcfg_max_depth", help="Maximum depth for PCFG generated concepts when experiment is 'concept' and data_type is 'bits'")
    parser.add_argument("--first-order", action="store_true", dest="first_order_meta_sgd", help="Use first-order MetaSGD instead of second-order (MAML).")
    parser.add_argument("--hyper-index", type=int, default=None, help="Specify a hyperparameter index to use when --no-hyper-search is active. Overrides DEFAULT_INDEX.")
    parser.add_argument("--patience", type=int, default=10000, help="Patience for early stopping in meta_train (number of validation checks).") # New argument
    # Potentially add --run-name here later for organizing output files for different runs
    # parser.add_argument("--run-name", type=str, default=None, help="Subdirectory name for this run's results and models.")
    args = parser.parse_args()
    main(**vars(args))
