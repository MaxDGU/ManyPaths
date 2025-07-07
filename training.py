import sys # Added for debugging
import time
import torch
from torch.nn.utils import clip_grad_norm_ as clip
import torch.nn as nn
import torch.nn.functional as F # Import F for cosine_similarity
import numpy as np
import learn2learn as l2l
import copy # Import the copy module
import pandas as pd # Added for saving intermediate trajectories
import os           # Added for path joining for intermediate trajectories

# Debug prints for learn2learn
print(f"[DEBUG] learn2learn imported. Path: {l2l.__file__}")
try:
    print(f"[DEBUG] learn2learn version: {l2l.__version__}")
except AttributeError:
    print("[DEBUG] learn2learn __version__ attribute not found.")
# print(f"[DEBUG] sys.path: {sys.path}") # Keep this commented for now to reduce output

from evaluation import evaluate
from initialization import init_model
from constants import *

# How often to save a periodic checkpoint (in terms of episodes_seen)
# This should align with or be a multiple of the logging frequency (1000)
PERIODIC_CHECKPOINT_INTERVAL = 5000 
LOG_INTERVAL = 1000

def meta_train(
    meta,
    train_loader,
    val_dataset,
    criterion,
    optimizer,
    device,
    epochs: int = 1,
    tasks_per_meta_batch: int = 4,
    adaptation_steps: int = 1,
    patience=20,  # Number of epochs to wait for improvement
    verbose: bool = False,
    checkpoint_interval: int = 40, # NEW: from main.py
    results_dir: str = "results",       # NEW: for saving intermediate trajectories
    file_prefix: str = "default_run"   # NEW: for saving intermediate trajectories
):
    if verbose:
        print("--- Meta-Training ---", flush=True)
    meta.train()
    val_losses_log = [] # Renamed for clarity
    val_accuracies_log = [] # Renamed for clarity
    grad_alignments_log = []

    best_model_info = {"epoch": -1, "state": None, "loss": float("inf")}
    # periodic_checkpoints will now store checkpoints taken at 'checkpoint_interval' or if it's a new best model.
    saved_checkpoints_info = [] # List of (epoch_num, state_dict) for checkpoints saved based on interval or best model
    
    no_improve_for_epochs = 0 # Renamed for clarity
    episodes_seen_total = 0 # Renamed for clarity
    stop_early_flag = False # Renamed for clarity
    
    # Ensure the directory for intermediate trajectories exists
    os.makedirs(results_dir, exist_ok=True)

    for epoch_num in range(epochs):
        epoch_start_time = time.time() # Timer for the epoch
        if verbose:
            print(f"Epoch {epoch_num + 1}/{epochs} starting...", flush=True)

        for i, (X_s, y_s, X_q, y_q) in enumerate(train_loader):
            optimizer.zero_grad()
            meta_loss_agg = 0.0
            task_query_grad_vecs_for_batch = []

            for t in range(tasks_per_meta_batch):
                learner = meta.clone()
                for _ in range(adaptation_steps):
                    support_pred = learner(X_s[t])
                    support_loss = criterion(support_pred, y_s[t])
                    learner.adapt(support_loss)
                
                query_pred = learner(X_q[t])
                task_query_loss = criterion(query_pred, y_q[t])
                meta_loss_agg += task_query_loss

                # Gradient alignment: calculated if it's a LOG_INTERVAL step for episodes_seen_total
                if episodes_seen_total % LOG_INTERVAL == 0: 
                    # Use base model parameters only (exclude learning rate parameters) to ensure dimensional compatibility
                    # MetaSGD stores both base model params and lr params, we need only base model params
                    task_base_params = [p for name, p in learner.named_parameters() if not name.startswith('lr')]
                    task_grads_raw = torch.autograd.grad(task_query_loss, task_base_params, retain_graph=True, allow_unused=True)
                    task_grad_vec = torch.cat([g.view(-1) for g in task_grads_raw if g is not None])
                    if task_grad_vec.nelement() > 0:
                        task_query_grad_vecs_for_batch.append(task_grad_vec)
            
            meta_loss_agg /= tasks_per_meta_batch
            meta_loss_agg.backward()
            
            current_alignment_for_log_step = np.nan # Default to NaN
            if episodes_seen_total % LOG_INTERVAL == 0 and task_query_grad_vecs_for_batch:
                # Use base model parameters only (exclude learning rate parameters) to ensure dimensional compatibility with task gradients
                # MetaSGD stores both base model params and lr params, we need only base model params
                meta_base_params = [p for name, p in meta.named_parameters() if not name.startswith('lr')]
                meta_grad_list = [p.grad.detach().view(-1) for p in meta_base_params if p.grad is not None]
                if meta_grad_list:
                    meta_grad_vec = torch.cat(meta_grad_list)
                    if meta_grad_vec.nelement() > 0:
                        alignments_for_tasks_in_batch = []
                        for task_grad_vec in task_query_grad_vecs_for_batch:
                            if task_grad_vec.nelement() == meta_grad_vec.nelement():
                                alignment = F.cosine_similarity(meta_grad_vec, task_grad_vec, dim=0)
                                alignments_for_tasks_in_batch.append(alignment.item())
                        if alignments_for_tasks_in_batch:
                            current_alignment_for_log_step = np.mean(alignments_for_tasks_in_batch)
            
            clip(meta.parameters(), 1.0)
            optimizer.step()
            episodes_seen_total += tasks_per_meta_batch # Assuming batch_size is tasks_per_meta_batch

            # Logging based on LOG_INTERVAL (default 1000 episodes from constants.py)
            if episodes_seen_total % LOG_INTERVAL == 0:
                meta_val_loss, meta_val_acc = evaluate(
                    meta, val_dataset, criterion, device, [0, adaptation_steps]
                )
                val_losses_log.append(meta_val_loss)
                val_accuracies_log.append(meta_val_acc)
                grad_alignments_log.append(current_alignment_for_log_step) # Appends mean or NaN
                no_improve_for_epochs += 1 # This counter is per LOG_INTERVAL check

                is_new_best_by_loss = False
                if meta_val_loss < best_model_info["loss"]:
                    best_model_info["loss"] = meta_val_loss
                    best_model_info["state"] = copy.deepcopy(meta.state_dict())
                    best_model_info["epoch"] = epoch_num + 1 # Store epoch number (1-indexed)
                    no_improve_for_epochs = 0
                    is_new_best_by_loss = True
                
                if verbose:
                    alignment_str = f"{current_alignment_for_log_step:.4f}" if not np.isnan(current_alignment_for_log_step) else "N/A"
                    print(
                        f"  Epoch {epoch_num + 1}, Batch {i+1}, Episodes Seen: {episodes_seen_total}, GradAlign: {alignment_str}, "
                        f"MetaTrainLoss: {meta_loss_agg.item():.4f}, MetaValLoss: {meta_val_loss:.4f}, "
                        f"MetaValAcc: {meta_val_acc:.4f} {'*' if is_new_best_by_loss else ''}", flush=True
                    )
                
                if no_improve_for_epochs * LOG_INTERVAL >= patience * LOG_INTERVAL: # Compare total episodes vs patience in episodes
                    if verbose: print(f"No validation improvement based on patience ({patience} LOG_INTERVAL checks). Stopping early...", flush=True)
                    stop_early_flag = True; break
                if np.isnan(meta_val_loss):
                    if verbose: print(f"Meta-training diverged (NaN val loss). Stopping early...", flush=True)
                    stop_early_flag = True; break
        # End of batches for one epoch

        # Checkpoint saving logic (per epoch, based on checkpoint_interval)
        # Also save if it's the best model identified in this epoch from any LOG_INTERVAL step
        is_best_model_this_epoch = best_model_info["epoch"] == (epoch_num + 1)

        if (epoch_num + 1) % checkpoint_interval == 0 or (epoch_num + 1) == epochs or is_best_model_this_epoch:
            # Save model checkpoint
            checkpoint_state = best_model_info["state"] if is_best_model_this_epoch else copy.deepcopy(meta.state_dict())
            saved_checkpoints_info.append((epoch_num + 1, checkpoint_state))
            if verbose:
                reason = "interval" if (epoch_num + 1) % checkpoint_interval == 0 else "final epoch" if (epoch_num + 1) == epochs else "new best model"
                print(f"  Saved checkpoint at Epoch {epoch_num + 1} (Reason: {reason}). Total checkpoints: {len(saved_checkpoints_info)}", flush=True)

            # Save INTERMEDIATE trajectory up to this epoch
            # Use log_step relative to LOG_INTERVAL counts
            num_logs_so_far = len(val_losses_log)
            if num_logs_so_far > 0:
                # Ensure grad_alignments_log is padded to the same length if it has fewer entries
                current_grad_alignments = grad_alignments_log[:num_logs_so_far]
                if len(current_grad_alignments) < num_logs_so_far:
                    current_grad_alignments.extend([np.nan] * (num_logs_so_far - len(current_grad_alignments)))

                intermediate_trajectory_data = {
                    'log_step': list(range(1, num_logs_so_far + 1)), # This is number of LOG_INTERVAL steps
                    'val_loss': val_losses_log[:num_logs_so_far],
                    'val_accuracy': val_accuracies_log[:num_logs_so_far],
                    'grad_alignment': current_grad_alignments
                }
                intermediate_df = pd.DataFrame(intermediate_trajectory_data)
                intermediate_traj_filename = os.path.join(results_dir, f"{file_prefix}_epoch_{(epoch_num + 1)}_trajectory.csv")
                intermediate_df.to_csv(intermediate_traj_filename, index=False)
                if verbose:
                    print(f"  Saved intermediate trajectory to {intermediate_traj_filename}", flush=True)
        
        if verbose:
            print(f"Epoch {epoch_num + 1}/{epochs} completed in {time.time() - epoch_start_time:.2f}s", flush=True)

        if stop_early_flag:
            if verbose: print("Early stopping triggered from within batch loop.", flush=True)
            break # Break from epoch loop
            
    if verbose:
        print("--- Meta-Training Finished ---", flush=True)
    return val_losses_log, val_accuracies_log, grad_alignments_log, best_model_info["state"], saved_checkpoints_info


def hyper_search(
    experiment,
    m,
    data_type,
    outer_lr,
    train_loader,
    val_dataset,
    test_dataset,
    device,
    epochs: int = 1, # This is epochs for each hyperparameter configuration test
    tasks_per_meta_batch: int = 4,
    adaptation_steps: int = 1,
    channels: int = 1,
    bits: int = 8,
    n_output: int = 1,
    results_dir: str = "results/hyper_search_temp", # Added results_dir parameter
    # Add checkpoint_interval, results_dir, file_prefix to signature if meta_train needs them for hyper_search calls
    # For now, assuming hyper_search's meta_train doesn't need to save these things.
):
    print("--- Hyperparameter Search ---", flush=True)
    best_index, best_val = 0, np.inf
    search_start_time = time.time()
    for index_loop_val in INDICES:
        model_init_start_time = time.time()
        model = init_model(m, data_type, index_loop_val, channels=channels, bits=bits, n_output=n_output)
        model = model.to(device)
        meta = l2l.algorithms.MetaSGD(model, lr=1e-3, first_order=False).to(device) 
        criterion = nn.MSELoss() if experiment == "mod" else nn.BCEWithLogitsLoss() if experiment == "concept" else nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(meta.parameters(), lr=outer_lr)
        
        # Call meta_train with appropriate args for hyper_search context
        # Pass dummy/default values for checkpoint_interval, results_dir, file_prefix
        # as these are not typically saved during hyperparameter search iterations.
        val_losses_hyper, _, _, _, _ = meta_train(
            meta, train_loader, val_dataset, criterion, optimizer, device,
            epochs=epochs, # Use epochs specific to hyper_search iteration
            tasks_per_meta_batch=tasks_per_meta_batch,
            adaptation_steps=adaptation_steps,
            patience=5, # Shorter patience for hyper_search
            verbose=False, # Usually less verbose for hyper_search
            checkpoint_interval=epochs + 1, # Effectively disable interval checkpoints for hyper_search meta_train
            results_dir=results_dir, # Use the provided results_dir parameter
            file_prefix=f"hyper_search_idx{index_loop_val}" # Dummy prefix
        )
        current_min_loss = min(val_losses_hyper) if val_losses_hyper else float('inf')
        print(
            f"Results: Model={m}, Parameter Index={index_loop_val}, Val Loss={current_min_loss:.3f} ({time.time() - model_init_start_time:.2f}s)", flush=True
        )
        if val_losses_hyper and current_min_loss < best_val:
            best_index = index_loop_val
            best_val = current_min_loss
    print(f"Hyperparameter search finished in {time.time() - search_start_time:.2f}s. Best index: {best_index}, Best val loss: {best_val:.3f}", flush=True)
    return best_index
