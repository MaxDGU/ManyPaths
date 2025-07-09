import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import pandas as pd
import time

# Assuming these can be imported directly and sys.path is okay from your cluster setup
from initialization import init_dataset, init_model, init_misc
from utils import (set_random_seeds, get_collate, initialize_trajectory_log, 
                   log_landscape_checkpoint, save_checkpoint_dict, extract_model_parameters)
from generate_concepts import PCFG_DEFAULT_MAX_DEPTH # For default value
from constants import DEFAULT_INDEX # Assuming this is relevant for model init if not doing hyper_search

# Directory for saving task-specific baseline models
BASELINE_MODELS_SAVE_DIR = "saved_models/baseline_sgd_task_models"
BASELINE_RESULTS_DIR = "results/baseline_sgd"
SAVED_DATASETS_DIR = "saved_datasets" # Base directory where MAML datasets are saved

def train_baseline_on_task(model, support_X, support_y, criterion, optimizer, device, num_train_steps, 
                          log_landscape=False, landscape_log_path=None, theta_start=None, task_idx=0):
    model.train()
    losses = []
    for step in range(num_train_steps):
        optimizer.zero_grad()
        pred = model(support_X)
        loss = criterion(pred, support_y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        # NEW: Landscape logging every 10 steps
        if log_landscape and landscape_log_path and step % 10 == 0:
            try:
                # Calculate accuracy
                with torch.no_grad():
                    pred_acc = ((torch.sigmoid(pred) > 0.5) == support_y.bool()).float().mean().item()
                
                log_landscape_checkpoint(
                    landscape_log_path, step + task_idx * num_train_steps, model,
                    loss.item(), pred_acc, support_X, support_y, criterion, theta_start
                )
            except Exception as e:
                print(f"Warning: Landscape logging failed at task {task_idx}, step {step}: {e}")
                
    return losses

def main(
    seed: int = 0,
    experiment: str = "concept",
    m: str = "mlp",
    data_type: str = "bits",
    num_tasks_to_evaluate: int = 100,
    num_sgd_steps_per_task: int = 100,
    lr: float = 1e-3,
    num_concept_features: int = 8,
    pcfg_max_depth: int = PCFG_DEFAULT_MAX_DEPTH,
    save_each_task_model: bool = True,
    save_checkpoints: bool = False,
    cache: str = None,
    epochs: int = None,
    momentum: float = 0.9,
    seeds: list = None,
    hyper_index: int = DEFAULT_INDEX,
    verbose: bool = False,
    run_name: str = "run_baseline",
    use_fixed_eval_set: str = None,
    maml_experiment_source: str = "concept",
    maml_model_arch_source: str = "mlp",
    maml_data_type_source: str = "bits",
    maml_num_concept_features_source: int = 8,
    maml_pcfg_max_depth_source: int = PCFG_DEFAULT_MAX_DEPTH,
    maml_seed_source: int = 0,
    maml_skip_param_source: int = 1,
    maml_alphabet_source: str = "asian",
    log_landscape: bool = False,  # NEW: Enable landscape logging
    checkpoint_every: int = 10    # NEW: Checkpoint every N steps for landscape analysis
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_random_seeds(seed)

    # Handle epochs alias
    if epochs is not None:
        num_sgd_steps_per_task = epochs

    # --- Output Directory Setup ---
    current_run_results_dir = os.path.join(BASELINE_RESULTS_DIR, run_name)
    current_run_models_dir = os.path.join(BASELINE_MODELS_SAVE_DIR, run_name)
    os.makedirs(current_run_results_dir, exist_ok=True)
    if save_each_task_model or save_checkpoints:
        os.makedirs(current_run_models_dir, exist_ok=True)

    # --- Handle Cache Loading ---
    if cache:
        print(f"📂 Loading tasks from cache: {cache}")
        try:
            cached_data = torch.load(cache, map_location='cpu', weights_only=False)
            if isinstance(cached_data, (tuple, list)) and len(cached_data) == 2:
                cached_tasks, meta_info = cached_data
            else:
                cached_tasks = cached_data
            print(f"✅ Loaded {len(cached_tasks)} tasks from cache")
            
            # Override num_tasks_to_evaluate with cache size if not specified
            if num_tasks_to_evaluate == 100:  # default value
                actual_num_tasks_to_evaluate = len(cached_tasks)
            else:
                actual_num_tasks_to_evaluate = min(num_tasks_to_evaluate, len(cached_tasks))
                
        except Exception as e:
            print(f"❌ Error loading cache {cache}: {e}")
            print("Falling back to on-the-fly generation")
            cached_tasks = None
            actual_num_tasks_to_evaluate = num_tasks_to_evaluate
    else:
        cached_tasks = None
        actual_num_tasks_to_evaluate = num_tasks_to_evaluate

    # NEW: Landscape logging setup for SGD baseline
    landscape_log_path = None
    theta_start = None
    checkpoint_dir = None
    
    if log_landscape:
        # Create file prefix for this run
        file_prefix_parts = [
            experiment, m, data_type, f"feats{num_concept_features}", 
            f"depth{pcfg_max_depth}", f"sgd_baseline", f"seed{seed}"
        ]
        file_prefix = "_".join(file_prefix_parts)
        
        # Initialize landscape logging
        landscape_log_path = os.path.join(current_run_results_dir, f"{file_prefix}_landscape_trajectory.csv")
        initialize_trajectory_log(landscape_log_path, "SGD")
        
        # Create checkpoint directory for landscape analysis
        checkpoint_dir = os.path.join(current_run_results_dir, "checkpoints", "sgd", 
                                     f"feats{num_concept_features}_depth{pcfg_max_depth}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize model to get starting parameters
        init_model_for_theta = init_model(
            m, data_type, index=hyper_index, verbose=False, 
            channels=channels, bits=bits_for_model, n_output=n_output
        ).to(device)
        theta_start = extract_model_parameters(init_model_for_theta).detach().clone()
        
        if verbose:
            print(f"Landscape logging enabled for SGD baseline")
            print(f"Log path: {landscape_log_path}")
            print(f"Checkpoints will be saved to: {checkpoint_dir}")

    # --- Dataset and Model Misc Init ---
    alphabet, bits_for_model, channels, n_output = init_misc(
        experiment, None, num_concept_features_override=num_concept_features
    )
    collate_fn = get_collate(experiment, device) # For concept learning

    eval_dataset = None
    dataset_source_info = ""
    dataset_iterator = None
    
    # Use cached tasks if available
    if cached_tasks:
        print(f"🧠 Using cached tasks, processing {actual_num_tasks_to_evaluate} tasks")
        dataset_source_info = f"cached tasks from {cache}"
        dataset_iterator = iter(cached_tasks[:actual_num_tasks_to_evaluate])
    elif use_fixed_eval_set and maml_experiment_source:
        print(f"Attempting to load fixed '{use_fixed_eval_set}' dataset generated by MAML.")
        print(f"  MAML source params: exp={maml_experiment_source}, model={maml_model_arch_source}, data={maml_data_type_source}, feats={maml_num_concept_features_source}, depth={maml_pcfg_max_depth_source}, seed={maml_seed_source}")

        dataset_prefix_parts = [maml_experiment_source, maml_model_arch_source, maml_data_type_source]
        if maml_experiment_source == "concept" and maml_data_type_source == "bits":
            dataset_prefix_parts.append(f"feats{maml_num_concept_features_source}")
            dataset_prefix_parts.append(f"depth{maml_pcfg_max_depth_source}")
        elif maml_experiment_source == "mod":
            dataset_prefix_parts.append(f"skip{maml_skip_param_source}")
        elif maml_experiment_source == "omniglot":
            dataset_prefix_parts.append(maml_alphabet_source)
        dataset_prefix_parts.append(f"seed{maml_seed_source}")
        dataset_file_prefix = "_".join(dataset_prefix_parts)

        load_path = None
        if use_fixed_eval_set == "val":
            load_path = os.path.join(SAVED_DATASETS_DIR, f"{dataset_file_prefix}_val_dataset.pt")
        elif use_fixed_eval_set == "test":
            load_path = os.path.join(SAVED_DATASETS_DIR, f"{dataset_file_prefix}_test_dataset.pt")
        else:
            print(f"Warning: Invalid value for --use-fixed-eval-set: {use_fixed_eval_set}. Must be 'val' or 'test'. Falling back.")
            use_fixed_eval_set = None # Trigger fallback

        if load_path:
            try:
                # Call init_dataset to load the specific set.
                # Pass MAML source parameters for context, though loading is by path.
                # n_support=None implies loading the full dataset as saved by main.py.
                # init_dataset returns (train, test, val) if n_support is None.
                _, loaded_test_s, loaded_val_s = init_dataset(
                    experiment=maml_experiment_source, 
                    model_arch=maml_model_arch_source, 
                    data_type=maml_data_type_source, 
                    skip_param=maml_skip_param_source, 
                    alphabet=maml_alphabet_source, # This needs to be the MAML source alphabet
                    num_concept_features=maml_num_concept_features_source,
                    pcfg_max_depth=maml_pcfg_max_depth_source,
                    n_support=None, # Critical: load full datasets as saved by main.py
                    load_train_path=None, # Not loading train set for this baseline script
                    load_val_path=load_path if use_fixed_eval_set == "val" else None,
                    load_test_path=load_path if use_fixed_eval_set == "test" else None
                )
                
                if use_fixed_eval_set == "val":
                    eval_dataset = loaded_val_s
                    dataset_source_info = f"loaded MAML val_dataset ({dataset_file_prefix})"
                elif use_fixed_eval_set == "test":
                    eval_dataset = loaded_test_s
                    dataset_source_info = f"loaded MAML test_dataset ({dataset_file_prefix})"
                
                if eval_dataset is None:
                    raise FileNotFoundError(f"Dataset object for '{use_fixed_eval_set}' from {load_path} was not returned correctly by init_dataset.")

                if not hasattr(eval_dataset, '__getitem__') or not hasattr(eval_dataset, '__len__'):
                    raise TypeError("Loaded eval_dataset is not a standard indexable/iterable PyTorch Dataset.")
                
                print(f"Successfully loaded: {dataset_source_info}, contains {len(eval_dataset)} tasks.")
                actual_num_tasks_to_evaluate = len(eval_dataset)
                fixed_task_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
                dataset_iterator = iter(fixed_task_loader)
                
            except FileNotFoundError as e:
                print(f"ERROR: Could not find the MAML dataset at '{load_path}'. Error: {e}")
                print("Please ensure the MAML parameters and seed match a saved dataset. Falling back to on-the-fly generation.")
                use_fixed_eval_set = None # Fallback
            except Exception as e:
                print(f"ERROR: An unexpected error occurred while loading/setting up fixed dataset. Error: {e}")
                print("Falling back to on-the-fly task generation.")
                use_fixed_eval_set = None # Fallback

    if not use_fixed_eval_set and not cached_tasks: # Fallback or default behavior
        print(f"Using on-the-fly task generation for {actual_num_tasks_to_evaluate} tasks (baseline exp: {experiment}, model: {m}).")
        # For on-the-fly generation, use the baseline's own parameters.
        # init_dataset returns (train, test, val) when n_support is None.
        # We typically use the train_dataset for generating tasks for baselines.
        # The skip_param for 'mod' should be what baseline intends (e.g. 0 or 1).
        # If baseline experiment is 'mod', its own 'skip' should be an arg. For now, hardcode or use a default.
        # For concept, skip_param is not critical. Let's use 0 as a general default if not 'mod'.
        baseline_skip_param = 0 # Default for on-the-fly if not 'mod'
        if experiment == "mod":
            # TODO: Add a `--baseline-skip-param` if necessary for on-the-fly mod generation.
            # For now, let's assume 0 or 1, matching MAML's default if it makes sense.
            baseline_skip_param = 1 # Matching MAML default for skip if baseline is 'mod'
            print(f"  (Using skip_param={baseline_skip_param} for on-the-fly 'mod' generation)")

        on_the_fly_train_ds, _, _ = init_dataset( 
            experiment=experiment, # Baseline's experiment type
            model_arch=m,        # Baseline's model arch
            data_type=data_type,   # Baseline's data type
            skip_param=baseline_skip_param, 
            alphabet=alphabet, # From baseline's init_misc
            num_concept_features=num_concept_features, # Baseline's num_features
            pcfg_max_depth=pcfg_max_depth,           # Baseline's pcfg_max_depth
            n_support=None # Generate a full dataset to sample from
        )
        if on_the_fly_train_ds is None:
            print("ERROR: Failed to generate on-the-fly training dataset. Exiting.")
            return
            
        on_the_fly_loader = DataLoader(
            on_the_fly_train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn, drop_last=True
        )
        dataset_iterator = iter(on_the_fly_loader)
        dataset_source_info = f"on-the-fly generated (exp={experiment}, model={m}) for {actual_num_tasks_to_evaluate} tasks"

    if dataset_iterator is None:
        print("ERROR: Dataset iterator was not initialized. Cannot proceed.")
        return

    if verbose: print(f"Dataset source: {dataset_source_info}")
    all_task_results = []
    
    overall_start_time = time.time()

    for task_idx in range(actual_num_tasks_to_evaluate):
        task_start_time = time.time()
        
        try:
            if cached_tasks:
                # Handle cached task format
                task_data = next(dataset_iterator)
                
                # Debug: Print task data structure for first task
                if task_idx == 0 and verbose:
                    print(f"   Task data type: {type(task_data)}")
                    if isinstance(task_data, (tuple, list)):
                        print(f"   Task data length: {len(task_data)}")
                        for i, item in enumerate(task_data):
                            print(f"   Item {i}: {type(item)} shape: {getattr(item, 'shape', 'N/A')}")
                    elif isinstance(task_data, dict):
                        print(f"   Task data keys: {task_data.keys()}")
                
                if isinstance(task_data, dict):
                    # Dictionary format
                    X_s = task_data.get('support_x', task_data.get('support_X'))
                    y_s = task_data.get('support_y', task_data.get('support_Y'))
                    X_q = task_data.get('query_x', task_data.get('query_X'))
                    y_q = task_data.get('query_y', task_data.get('query_Y'))
                    
                    if X_s is None or y_s is None or X_q is None or y_q is None:
                        print(f"ERROR: Missing required keys in task dict at task {task_idx}")
                        print(f"Available keys: {task_data.keys()}")
                        actual_num_tasks_to_evaluate = task_idx
                        break
                        
                elif isinstance(task_data, (tuple, list)):
                    # Tuple/list format - handle different lengths
                    if len(task_data) == 4:
                        X_s, y_s, X_q, y_q = task_data
                    elif len(task_data) == 5:
                        # Sometimes there's an extra metadata element
                        X_s, y_s, X_q, y_q, _ = task_data
                    elif len(task_data) == 6:
                        # Sometimes format is (X_s, y_s, X_q, y_q, task_info, concept_info)
                        X_s, y_s, X_q, y_q, _, _ = task_data
                    elif len(task_data) == 9:
                        # Format: (support_X_pos, support_X_neg, support_Y_pos, query_X_pos, query_X_neg, query_Y_pos, n_support, complexity, concept_id)
                        support_X_pos, support_X_neg, support_Y_pos, query_X_pos, query_X_neg, query_Y_pos, n_support, complexity, concept_id = task_data
                        
                        # Combine positive and negative support examples
                        X_s = torch.cat([support_X_pos, support_X_neg], dim=0)
                        # Create labels: positive examples get label 1, negative examples get label 0
                        y_s_pos = torch.ones(support_X_pos.shape[0], 1)
                        y_s_neg = torch.zeros(support_X_neg.shape[0], 1)
                        y_s = torch.cat([y_s_pos, y_s_neg], dim=0)
                        
                        # Combine positive and negative query examples
                        X_q = torch.cat([query_X_pos, query_X_neg], dim=0)
                        # Create labels: positive examples get label 1, negative examples get label 0
                        y_q_pos = torch.ones(query_X_pos.shape[0], 1)
                        y_q_neg = torch.zeros(query_X_neg.shape[0], 1)
                        y_q = torch.cat([y_q_pos, y_q_neg], dim=0)
                        
                        if verbose:
                            print(f"    Task {task_idx+1}: Processed 9-element format")
                            print(f"      Support set: {X_s.shape}, Query set: {X_q.shape}")
                            print(f"      Complexity: {complexity}, Concept ID: {concept_id}")
                    elif len(task_data) == 2:
                        # Format might be ((X_s, y_s), (X_q, y_q))
                        (X_s, y_s), (X_q, y_q) = task_data
                    else:
                        print(f"ERROR: Unexpected task data length {len(task_data)} at task {task_idx}")
                        print(f"Task data: {[type(x) for x in task_data]}")
                        actual_num_tasks_to_evaluate = task_idx
                        break
                else:
                    print(f"ERROR: Unexpected task data type {type(task_data)} at task {task_idx}")
                    actual_num_tasks_to_evaluate = task_idx
                    break
                
                # Ensure proper tensor format and device
                X_s = torch.as_tensor(X_s, dtype=torch.float32).to(device)
                y_s = torch.as_tensor(y_s, dtype=torch.float32).to(device)
                X_q = torch.as_tensor(X_q, dtype=torch.float32).to(device)
                y_q = torch.as_tensor(y_q, dtype=torch.float32).to(device)
                
                # Ensure proper dimensions
                if X_s.dim() == 1:
                    X_s = X_s.unsqueeze(0)
                if y_s.dim() == 1:
                    y_s = y_s.unsqueeze(0)
                if X_q.dim() == 1:
                    X_q = X_q.unsqueeze(0)
                if y_q.dim() == 1:
                    y_q = y_q.unsqueeze(0)
                    
            else:
                # Handle regular dataset format
                X_s, y_s, X_q, y_q = next(dataset_iterator)
                X_s, y_s, X_q, y_q = X_s.squeeze(0), y_s.squeeze(0), X_q.squeeze(0), y_q.squeeze(0)
                
        except StopIteration:
            print(f"Warning: Dataset iterator exhausted at task_idx {task_idx}. Processed {task_idx} tasks.")
            actual_num_tasks_to_evaluate = task_idx # Update to actual number processed
            break
        except Exception as e:
            print(f"ERROR: Failed to get next task data at task_idx {task_idx}. Error: {e}")
            if task_idx == 0:
                print(f"   This might be a cache format issue. Try running with --verbose for more details.")
            actual_num_tasks_to_evaluate = task_idx
            break

        # --- Initialize a fresh model for this task (using baseline's parameters) ---
        model = init_model(
            m, data_type, index=hyper_index, verbose=False, 
            channels=channels, bits=bits_for_model, n_output=n_output
        ).to(device)

        criterion = nn.BCEWithLogitsLoss() # Assuming concept learning
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        # --- Train on Support Set ---
        if verbose: print(f"  Task {task_idx+1}/{actual_num_tasks_to_evaluate} from {dataset_source_info.split()[0]}: Training... ({num_sgd_steps_per_task} steps)")
        support_losses = train_baseline_on_task(
            model, X_s, y_s, criterion, optimizer, device, num_sgd_steps_per_task,
            log_landscape=log_landscape, landscape_log_path=landscape_log_path, theta_start=theta_start, task_idx=task_idx
        )
        final_support_loss = support_losses[-1] if support_losses else float('nan')

        # --- Evaluate on Query Set ---
        model.eval()
        with torch.no_grad():
            query_pred = model(X_q)
            query_loss = criterion(query_pred, y_q)
            
            # Calculate accuracy for concept learning (binary classification)
            query_acc = ((torch.sigmoid(query_pred) > 0.5) == y_q.bool()).float().mean().item()

        task_duration = time.time() - task_start_time
        if verbose:
            print(f"    Task {task_idx+1}: QLoss:{query_loss.item():.4f}, QAcc:{query_acc:.4f}, SLoss:{final_support_loss:.4f} ({task_duration:.2f}s)")

        all_task_results.append({
            "task_idx": task_idx,
            "query_loss": query_loss.item(),
            "query_accuracy": query_acc,
            "final_support_loss": final_support_loss,
            "num_sgd_steps": num_sgd_steps_per_task,
            "lr": lr
        })

        # --- Save the task-specific trained model ---
        if save_each_task_model or save_checkpoints:
            # Create baseline_checkpoints directory structure
            checkpoint_dir = "baseline_checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Use task ID for filename
            checkpoint_filename = f"task_{task_idx}.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
            
            # Save both model state and task metadata
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'task_idx': task_idx,
                'seed': seed,
                'experiment': experiment,
                'model_arch': m,
                'num_concept_features': num_concept_features,
                'pcfg_max_depth': pcfg_max_depth,
                'lr': lr,
                'momentum': momentum,
                'num_sgd_steps': num_sgd_steps_per_task,
                'query_accuracy': query_acc,
                'query_loss': query_loss.item(),
                'final_support_loss': final_support_loss
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            if verbose: print(f"      Saved task checkpoint: {checkpoint_path}")
            
            # Also save in the original location for backward compatibility
            model_filename_parts = [
                experiment, m, str(hyper_index), data_type,
                f"feats{num_concept_features}", f"depth{pcfg_max_depth}",
                f"task{task_idx}", f"run{run_name}", f"seed{seed}"
            ]
            if use_fixed_eval_set: # Add MAML source info to filename if used
                model_filename_parts.append(f"fixed_{use_fixed_eval_set}_mamlseed{maml_seed_source}")
            model_filename = "_".join(model_filename_parts) + ".pt"
            model_save_path = os.path.join(current_run_models_dir, model_filename)
            torch.save(model.state_dict(), model_save_path)
            if verbose: print(f"      Saved task model: {model_save_path}")

    # --- Aggregate and Save Results ---
    avg_query_acc = np.mean([res["query_accuracy"] for res in all_task_results]) if all_task_results else float('nan')
    avg_query_loss = np.mean([res["query_loss"] for res in all_task_results]) if all_task_results else float('nan')
    
    print(f"--- Baseline SGD Summary ({run_name}) ---")
    print(f"Source of tasks: {dataset_source_info}")
    print(f"Experiment: {experiment}, Model: {m}, Features: {num_concept_features}, Depth: {pcfg_max_depth}, Seed: {seed}")
    print(f"Num SGD steps per task: {num_sgd_steps_per_task}, LR: {lr}")
    print(f"Average Query Accuracy over {len(all_task_results)} tasks: {avg_query_acc:.4f}")
    print(f"Average Query Loss over {len(all_task_results)} tasks: {avg_query_loss:.4f}")
    print(f"Total time: {(time.time() - overall_start_time):.2f}s")

    # Save detailed per-task results to CSV
    results_df = pd.DataFrame(all_task_results)
    summary_filename_parts = [
        experiment, m, str(hyper_index), data_type,
        f"feats{num_concept_features}", f"depth{pcfg_max_depth}",
        f"sgdsteps{num_sgd_steps_per_task}", f"lr{lr}", f"run{run_name}", f"seed{seed}"
    ]
    if use_fixed_eval_set: # Add MAML source info to filename if used
        summary_filename_parts.append(f"fixed_{use_fixed_eval_set}_mamlseed{maml_seed_source}")
        summary_filename_parts.append(f"mamlExp{maml_experiment_source}") # Potentially add more maml source details
    summary_filename = "_".join(summary_filename_parts) + "_baselinetrajectory.csv"
    summary_save_path = os.path.join(current_run_results_dir, summary_filename)
    results_df.to_csv(summary_save_path, index=False)
    print(f"Saved baseline SGD trajectory results to {summary_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline SGD learning for concept tasks.")
    parser.add_argument("--cache", type=str, default=None, help="Path to task cache file (e.g., data/concept_cache/pcfg_tasks_f8_d3_s2p3n_q5p5n_t10000.pt)")
    parser.add_argument("--epochs", type=int, default=32, help="Number of SGD training steps per task (alias for --num-sgd-steps-per-task)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for SGD optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer (not used with AdamW)")
    parser.add_argument("--seeds", type=int, nargs='+', default=[0], help="List of seeds to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--experiment", type=str, default="concept", choices=["concept"], help="Experiment type (fixed to concept for now).")
    parser.add_argument("--m", type=str, default="mlp", choices=["mlp"], help="Baseline model architecture (fixed to mlp for now).")
    parser.add_argument("--data-type", type=str, default="bits", choices=["bits"], help="Baseline data type (fixed to bits for now).")
    
    parser.add_argument("--num-tasks-to-evaluate", type=int, default=100, help="Number of distinct tasks to process (used if not loading fixed set, or as max if loader is shorter).")
    parser.add_argument("--num-sgd-steps-per-task", type=int, default=100, help="Number of SGD steps on the support set of each task.")
    
    parser.add_argument("--num-concept-features", type=int, default=8, help="Number of features for concept learning.")
    parser.add_argument("--pcfg-max-depth", type=int, default=PCFG_DEFAULT_MAX_DEPTH, help="Max depth for PCFG concept generation.")
    
    parser.add_argument("--no-save-task-models", action="store_false", dest="save_each_task_model", help="Do not save individual task-specific models.")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save checkpoints for each task (same as default behavior, provided for explicit control).")
    parser.add_argument("--hyper-index", type=int, default=DEFAULT_INDEX, help="Hyperparameter index for model initialization.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--run-name", type=str, default="baseline_run1", help="Name for this baseline run (for folder organization).")

    # New arguments for using fixed evaluation sets
    parser.add_argument("--use-fixed-eval-set", type=str, default=None, choices=["val", "test"], help="Specify to use a fixed 'val' or 'test' dataset from a MAML run.")
    parser.add_argument("--maml-experiment-source", type=str, default="concept", help="MAML run's experiment type (e.g., concept)")
    parser.add_argument("--maml-model-arch-source", type=str, default="mlp", help="MAML run's model architecture (e.g., mlp)")
    parser.add_argument("--maml-data-type-source", type=str, default="bits", help="MAML run's data type (e.g., bits)")
    parser.add_argument("--maml-num-concept-features-source", type=int, default=8, help="MAML run's num_concept_features")
    parser.add_argument("--maml-pcfg-max-depth-source", type=int, default=PCFG_DEFAULT_MAX_DEPTH, help="MAML run's pcfg_max_depth")
    parser.add_argument("--maml-seed-source", type=int, default=0, help="MAML run's seed (used for dataset identification)")
    parser.add_argument("--maml-skip-param-source", type=int, default=1, help="MAML run's skip_param (for mod experiment)")
    parser.add_argument("--maml-alphabet-source", type=str, default="asian", help="MAML run's alphabet (for omniglot experiment)")

    # New arguments for landscape logging
    parser.add_argument("--log-landscape", action="store_true", help="Enable landscape logging during training.")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Checkpoint every N steps for landscape analysis.")

    args = parser.parse_args()
    
    # Handle epochs alias
    if hasattr(args, 'epochs'):
        args.num_sgd_steps_per_task = args.epochs
    
    # Handle multiple seeds
    if hasattr(args, 'seeds') and len(args.seeds) > 1:
        for seed in args.seeds:
            print(f"\n🌱 Running with seed {seed}")
            args.seed = seed
            main(**vars(args))
    else:
        main(**vars(args)) 