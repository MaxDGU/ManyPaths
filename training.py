import sys # Added for debugging
import time
import torch
from torch.nn.utils import clip_grad_norm_ as clip
import torch.nn as nn
import torch.nn.functional as F # Import F for cosine_similarity
import numpy as np
import learn2learn as l2l
import copy # Import the copy module

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
):
    if verbose:
        print("--- Meta-Training ---")
    meta.train()
    val_losses = []
    val_accuracies = []
    grad_alignments_log = [] # New: For logging gradient alignments

    # Store info about the best model found so far
    best_model_info = {"episode": -1, "state": None, "loss": float("inf")}
    periodic_checkpoints = []
    last_periodic_checkpoint_episode = 0
    no_improve_epochs = 0
    episodes_seen = 0
    stop_early = False
    start_time = time.time() # Renamed from 'start' to avoid conflict with any 'start' variables if they exist

    for epoch_num in range(epochs): # Changed _ to epoch_num for clarity if needed
        for i, (X_s, y_s, X_q, y_q) in enumerate(train_loader):
            optimizer.zero_grad()
            
            meta_loss_agg = 0.0
            # Store task-specific query gradients w.r.t. adapted learner params
            task_query_grad_vecs_for_batch = []

            for t in range(tasks_per_meta_batch):
                learner = meta.clone()
                # Inner loop adaptation
                for _ in range(adaptation_steps):
                    support_pred = learner(X_s[t])
                    support_loss = criterion(support_pred, y_s[t])
                    learner.adapt(support_loss)
                
                # Evaluate on the query set for this task
                query_pred = learner(X_q[t])
                task_query_loss = criterion(query_pred, y_q[t])
                meta_loss_agg += task_query_loss

                # Get gradients of this task's query loss w.r.t. its adapted learner's parameters
                if episodes_seen % 1000 == 0: # Only calculate if we are logging this step
                    # retain_graph=True is needed as task_query_loss is part of meta_loss_agg graph
                    task_grads_raw = torch.autograd.grad(task_query_loss, learner.parameters(), retain_graph=True, allow_unused=True)
                    # Filter out None gradients (e.g., for non-trainable params or parts of model not used by this task_query_loss)
                    task_grad_vec = torch.cat([g.view(-1) for g in task_grads_raw if g is not None])
                    if task_grad_vec.nelement() > 0: # Ensure we have some gradients
                        task_query_grad_vecs_for_batch.append(task_grad_vec)
                    elif verbose:
                        print(f"    [GradAlign] Warning: No gradients found for task {t} query loss w.r.t adapted learner params.")

            meta_loss_agg /= tasks_per_meta_batch
            meta_loss_agg.backward() # This populates .grad for meta.parameters()
            
            # Gradient Alignment Calculation (if it's a logging step)
            current_alignment_for_step = None
            if episodes_seen % 1000 == 0 and task_query_grad_vecs_for_batch:
                if verbose: print(f"    [GradAlign DEBUG] Attempting to calculate alignment. tasks_in_batch: {len(task_query_grad_vecs_for_batch)}")
                
                # Access base model parameters if meta is MetaSGD for consistent grad alignment
                params_for_meta_grad = meta.parameters()
                if isinstance(meta, l2l.algorithms.MetaSGD):
                    # This ensures we get the actual model parameters, not the MetaSGD-specific ones (like per-param LRs)
                    params_for_meta_grad = meta.module.parameters()
                
                meta_grad_list = [p.grad.detach().view(-1) for p in params_for_meta_grad if p.grad is not None]
                
                if not meta_grad_list:
                    if verbose: print("    [GradAlign DEBUG] meta_grad_list is EMPTY. All p.grad might be None for meta params.")
                else:
                    if verbose: print(f"    [GradAlign DEBUG] meta_grad_list has {len(meta_grad_list)} items.")
                    meta_grad_vec = torch.cat(meta_grad_list)
                    
                    if meta_grad_vec.nelement() > 0:
                        if verbose: print(f"    [GradAlign DEBUG] meta_grad_vec has {meta_grad_vec.nelement()} elements.")
                        alignments_for_tasks_in_batch = []
                        for task_idx, task_grad_vec in enumerate(task_query_grad_vecs_for_batch):
                            if verbose: print(f"    [GradAlign DEBUG] Task {task_idx}: task_grad_vec has {task_grad_vec.nelement()} elements.")
                            # Ensure parameter vectors are of the same size for cosine similarity
                            # This might not always be true if learner.parameters() and meta.parameters() differ in structure
                            # or if some meta params didn't get grads. For MetaSGD, they should be congruent.
                            if task_grad_vec.nelement() == meta_grad_vec.nelement(): # This should be true now
                                alignment = F.cosine_similarity(meta_grad_vec, task_grad_vec, dim=0)
                                alignments_for_tasks_in_batch.append(alignment.item())
                                if verbose: print(f"    [GradAlign DEBUG] Task {task_idx}: Computed alignment: {alignment.item()}")
                            # else: # This 'else' for mismatch should not be hit now
                                # if verbose: print(f"    [GradAlign] Warning: Mismatch in grad vec elements for task {task_idx}. Meta: {meta_grad_vec.nelement()}, Task: {task_grad_vec.nelement()}")
                        
                        if alignments_for_tasks_in_batch:
                            current_alignment_for_step = np.mean(alignments_for_tasks_in_batch)
                            if verbose: print(f"    [GradAlign DEBUG] Calculated current_alignment_for_step: {current_alignment_for_step}")
                        else:
                            if verbose: print("    [GradAlign DEBUG] alignments_for_tasks_in_batch is EMPTY.")
                    else:
                        if verbose: print("    [GradAlign DEBUG] meta_grad_vec is EMPTY after torch.cat.")

            clip(meta.parameters(), 1.0) # Clip after .grad is populated and used
            optimizer.step()
            episodes_seen += len(X_s) # Assumes len(X_s) is tasks_per_meta_batch

            if episodes_seen % 1000 == 0:
                # Log gradient alignment
                grad_alignments_log.append(current_alignment_for_step if current_alignment_for_step is not None else np.nan)
                
                meta_val_loss, meta_val_acc = evaluate(
                    meta, val_dataset, criterion, device, [0, adaptation_steps]
                )
                val_losses.append(meta_val_loss)
                val_accuracies.append(meta_val_acc)
                no_improve_epochs += 1

                is_new_best = False
                if meta_val_loss < best_model_info["loss"]:
                    best_model_info["loss"] = meta_val_loss
                    best_model_info["state"] = copy.deepcopy(meta.state_dict()) # Deepcopy here
                    best_model_info["episode"] = episodes_seen
                    no_improve_epochs = 0
                    is_new_best = True
                    # Save checkpoint for new best model (also a deepcopy)
                    periodic_checkpoints.append((episodes_seen, copy.deepcopy(meta.state_dict()))) # Deepcopy here
                    if verbose:
                        print(f"    New best model at episode {episodes_seen}. Val Loss: {meta_val_loss:.4f}. Saved checkpoint.")
                
                if (episodes_seen - last_periodic_checkpoint_episode >= PERIODIC_CHECKPOINT_INTERVAL) and not is_new_best:
                    periodic_checkpoints.append((episodes_seen, copy.deepcopy(meta.state_dict()))) # Deepcopy here
                    last_periodic_checkpoint_episode = episodes_seen
                    if verbose:
                        print(f"    Periodic checkpoint saved at episode {episodes_seen}.")

                if verbose:
                    alignment_str = f"{current_alignment_for_step:.4f}" if current_alignment_for_step is not None else "N/A"
                    print(
                        f"Episodes {episodes_seen}, GradAlign: {alignment_str}, "
                        f"Meta-Train Loss: {meta_loss_agg.item():.4f}, Meta-Val Loss: {meta_val_loss:.4f}; "
                        f"Meta-Val Acc: {meta_val_acc:.4f} ({time.time() - start_time:.2f}s) {'*' if is_new_best else ''}"
                    )
                if no_improve_epochs >= patience:
                    if verbose: print(f"No validation improvement after {patience} checks. Stopping early...")
                    stop_early = True; break
                if np.isnan(meta_val_loss):
                    if verbose: print(f"Meta-training diverged. Stopping early...")
                    stop_early = True; break
                start_time = time.time() # Reset timer for next log interval
        if stop_early:
            break
            
    return val_losses, val_accuracies, grad_alignments_log, best_model_info["state"], periodic_checkpoints


def hyper_search(
    experiment,
    m,
    data_type,
    outer_lr,
    train_loader,
    val_dataset,
    test_dataset,
    device,
    epochs: int = 1,
    tasks_per_meta_batch: int = 4,
    adaptation_steps: int = 1,
    channels: int = 1,
    bits: int = 8,
    n_output: int = 1,
    # Add patience to hyper_search signature if you want to control it there too
    # For now, it will use meta_train's default or whatever meta_train is changed to.
):
    print("--- Hyperparameter Search ---")
    best_index, best_val = 0, np.inf
    search_start_time = time.time() # Timer for the whole search
    for index_loop_val in INDICES: # Renamed index to avoid conflict with module
        model_init_start_time = time.time()
        model = init_model(m, data_type, index_loop_val, channels=channels, bits=bits, n_output=n_output)
        model = model.to(device)
        # Assuming MetaSGD for hyper_search as well, as per main.py logic
        # If --first-order and inner_lr should apply here, this needs more args.
        meta = l2l.algorithms.MetaSGD(model, lr=1e-3, first_order=False).to(device) 
        criterion = nn.MSELoss() if experiment == "mod" else nn.BCEWithLogitsLoss() if experiment == "concept" else nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(meta.parameters(), lr=outer_lr)
        
        # meta_train now returns 5 items
        # For hyper_search, checkpoints from meta_train are not typically kept, only best_val_loss matters.
        # Grad alignments are also not typically analyzed for each hyper_search step.
        val_losses_hyper, _, _, _, _ = meta_train( # Ignore grad_alignments and checkpoints for hyper
            meta,
            train_loader,
            val_dataset,
            criterion,
            optimizer,
            device,
            epochs, # epochs for hyper_search
            tasks_per_meta_batch,
            adaptation_steps,
            patience=5, # Shorter patience for hyper_search is fine
            verbose=False # Usually less verbose for hyper_search
        )
        current_min_loss = min(val_losses_hyper) if val_losses_hyper else float('inf')
        print(
            f"Results: Model={m}, Parameter Index={index_loop_val}, Val Loss={current_min_loss:.3f} ({time.time() - model_init_start_time:.2f}s)"
        )
        if val_losses_hyper and current_min_loss < best_val:
            best_index = index_loop_val
            best_val = current_min_loss
    print(f"Hyperparameter search finished in {time.time() - search_start_time:.2f}s. Best index: {best_index}, Best val loss: {best_val:.3f}")
    return best_index
