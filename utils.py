import os
import random
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
import csv
import pandas as pd
from pathlib import Path


def collate_concept(batch, device="cpu"):
    # Each item in batch, after BaseMetaDataset.__getitem__ for MetaBitConceptsDataset, is:
    # item[0]: X_s_processed_tensor (from original task_data[0])
    # item[1]: y_s_tensor (from original task_data[2])
    # item[2]: X_q_processed_tensor (from original task_data[3])
    # item[3]: y_q_tensor (from original task_data[5])

    X_s_list = [item[0].to(device) for item in batch]
    y_s_list = [item[1].to(device) for item in batch]
    X_q_list = [item[2].to(device) for item in batch]
    y_q_list = [item[3].to(device) for item in batch]

    # Pad sequences: batch_first=True makes the output (batch_size, max_len, num_features)
    # For labels, padding_value=0.0 is used. This might be okay if 0 is a neutral/non-class for BCE.
    # If labels are strictly 0 or 1, padding with 0 means these will be treated as class 0 instances.
    X_s_padded = pad_sequence(X_s_list, batch_first=True, padding_value=0.0)
    y_s_padded = pad_sequence(y_s_list, batch_first=True, padding_value=0.0)
    X_q_padded = pad_sequence(X_q_list, batch_first=True, padding_value=0.0)
    y_q_padded = pad_sequence(y_q_list, batch_first=True, padding_value=0.0)

    return X_s_padded, y_s_padded, X_q_padded, y_q_padded


def collate_default(batch, device="cpu"):
    def move_to_device(data):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, list):
            return [move_to_device(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(move_to_device(item) for item in data)
        elif isinstance(data, dict):
            return {key: move_to_device(value) for key, value in data.items()}
        else:
            return data

    batch = default_collate(batch)
    return move_to_device(batch)


def get_collate(experiment: str, device="cpu"):
    if experiment in ["concept", "mod"]:
        return lambda batch: collate_concept(batch, device=device)
    else:
        return lambda batch: collate_default(batch, device=device)


def set_random_seeds(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 1 gpu


def save_model(meta, save_dir="state_dicts", file_prefix="meta_learning"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(meta.state_dict(), f"{save_dir}/{file_prefix}.pth")
    print(f"Model saved to {save_dir}/{file_prefix}.pth")


def calculate_accuracy(predictions, targets):
    if predictions.shape[1] > 1:
        predictions = predictions.argmax(dim=1)
    else:
        predictions = (predictions > 0.0).float()
    correct = (predictions == targets).sum().item()
    accuracy = correct / targets.numel()
    return accuracy

# ============================================================================
# HESSIAN TOOLS FOR LOSS LANDSCAPE ANALYSIS
# ============================================================================

def top_eigenvalue(loss_fn, params, num_iterations=20, tolerance=1e-8):
    """
    Compute the top eigenvalue of the Hessian using power iteration.
    
    Args:
        loss_fn: Function that computes loss given current parameters
        params: Current model parameters (flattened tensor)
        num_iterations: Number of power iteration steps
        tolerance: Convergence tolerance
        
    Returns:
        Top eigenvalue (lambda_max)
    """
    device = params.device
    
    # Initialize random vector
    v = torch.randn_like(params)
    v = v / torch.norm(v)
    
    lambda_old = 0.0
    
    for i in range(num_iterations):
        # Compute Hessian-vector product using double backward
        grad = torch.autograd.grad(loss_fn(), params, create_graph=True)[0]
        
        # Hessian-vector product
        Hv = torch.autograd.grad(grad, params, grad_outputs=v, retain_graph=True)[0]
        
        # Power iteration update
        v_new = Hv / torch.norm(Hv)
        lambda_new = torch.dot(v, Hv).item()
        
        # Check convergence
        if abs(lambda_new - lambda_old) < tolerance:
            break
            
        v = v_new
        lambda_old = lambda_new
    
    return lambda_new

def hessian_trace_sqr(loss_fn, params, num_samples=50):
    """
    Estimate Tr(H^2) using Hutchinson's estimator.
    
    Args:
        loss_fn: Function that computes loss given current parameters
        params: Current model parameters (flattened tensor)
        num_samples: Number of random vectors for estimation
        
    Returns:
        Estimate of trace of Hessian squared
    """
    device = params.device
    trace_estimate = 0.0
    
    for i in range(num_samples):
        # Random Rademacher vector
        z = torch.randint_like(params, low=0, high=2, dtype=torch.float32)
        z = 2 * z - 1  # Convert {0,1} to {-1,1}
        
        # First order gradient
        grad = torch.autograd.grad(loss_fn(), params, create_graph=True)[0]
        
        # Hessian-vector product Hz
        Hz = torch.autograd.grad(grad, params, grad_outputs=z, retain_graph=True)[0]
        
        # H^2 z = H(Hz)
        H2z = torch.autograd.grad(grad, params, grad_outputs=Hz, retain_graph=True)[0]
        
        # z^T H^2 z
        trace_estimate += torch.dot(z, H2z).item()
    
    return trace_estimate / num_samples

def geodesic_length(theta_start, theta_end):
    """
    Compute geodesic length between two parameter vectors.
    For neural networks, this is approximated as Euclidean distance.
    
    Args:
        theta_start: Starting parameter vector
        theta_end: Ending parameter vector
        
    Returns:
        Geodesic length (Euclidean distance)
    """
    return torch.norm(theta_end - theta_start).item()

def extract_model_parameters(model):
    """
    Extract flattened parameter vector from PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Flattened parameter tensor
    """
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    return torch.cat(params)

def create_loss_closure(model, X_support, y_support, criterion):
    """
    Create a loss closure function for Hessian computation.
    
    Args:
        model: PyTorch model
        X_support: Support set inputs
        y_support: Support set targets
        criterion: Loss criterion
        
    Returns:
        Loss closure function
    """
    def loss_fn():
        model.zero_grad()
        predictions = model(X_support)
        if predictions.dim() > 1 and predictions.size(1) == 1:
            predictions = predictions.squeeze(1)
        loss = criterion(predictions, y_support)
        return loss
    return loss_fn

# ============================================================================
# TRAJECTORY I/O FOR LANDSCAPE LOGGING
# ============================================================================

def initialize_trajectory_log(log_path, method_name):
    """
    Initialize trajectory CSV log file with headers.
    
    Args:
        log_path: Path to CSV file
        method_name: Method name (e.g., 'Meta-SGD', 'SGD')
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    headers = [
        'step', 'theta_norm', 'loss', 'accuracy', 
        'lambda_max', 'hessian_trace_sqr', 'geodesic_length_from_start'
    ]
    
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    
    print(f"Initialized trajectory log: {log_path}")

def log_landscape_checkpoint(log_path, step, model, loss_val, accuracy_val, 
                           X_support, y_support, criterion, theta_start=None):
    """
    Log landscape metrics for current training step.
    
    Args:
        log_path: Path to CSV file
        step: Current training step
        model: Current model
        loss_val: Current loss value
        accuracy_val: Current accuracy value  
        X_support: Support set for Hessian computation
        y_support: Support set targets
        criterion: Loss criterion
        theta_start: Starting parameters for geodesic length
    """
    try:
        # Extract current parameters
        theta_current = extract_model_parameters(model)
        theta_norm = torch.norm(theta_current).item()
        
        # Create loss closure
        loss_fn = create_loss_closure(model, X_support, y_support, criterion)
        
        # Compute Hessian metrics (may be expensive)
        try:
            lambda_max = top_eigenvalue(loss_fn, theta_current, num_iterations=10)
        except:
            lambda_max = float('nan')
            
        try:
            hessian_trace = hessian_trace_sqr(loss_fn, theta_current, num_samples=20)
        except:
            hessian_trace = float('nan')
            
        # Compute geodesic length from start
        if theta_start is not None:
            geo_length = geodesic_length(theta_start, theta_current)
        else:
            geo_length = 0.0
        
        # Write to CSV
        row = [step, theta_norm, loss_val, accuracy_val, lambda_max, hessian_trace, geo_length]
        
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
    except Exception as e:
        print(f"Warning: Failed to log landscape checkpoint at step {step}: {e}")

def save_checkpoint_dict(checkpoint_path, step, model, optimizer, loss_val, 
                        accuracy_val, lambda_max=None, hessian_trace=None, geo_length=None):
    """
    Save comprehensive checkpoint dictionary.
    
    Args:
        checkpoint_path: Path to save checkpoint
        step: Current training step
        model: PyTorch model
        optimizer: Optimizer state
        loss_val: Current loss
        accuracy_val: Current accuracy
        lambda_max: Top Hessian eigenvalue (optional)
        hessian_trace: Hessian trace squared (optional)
        geo_length: Geodesic length from start (optional)
    """
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_val,
        'accuracy': accuracy_val,
        'theta': extract_model_parameters(model).cpu(),
        'lambda_max': lambda_max,
        'hessian_trace_sqr': hessian_trace,
        'geodesic_length': geo_length
    }
    
    torch.save(checkpoint, checkpoint_path)

def load_trajectory_csv(log_path):
    """
    Load trajectory data from CSV file.
    
    Args:
        log_path: Path to CSV file
        
    Returns:
        DataFrame with trajectory data
    """
    if not os.path.exists(log_path):
        return pd.DataFrame()
    
    return pd.read_csv(log_path)
