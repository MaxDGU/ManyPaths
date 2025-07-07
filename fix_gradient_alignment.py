#!/usr/bin/env python3
"""
Quick fix for gradient alignment computation in training.py
This script shows the corrected logic that should be applied
"""

import torch
import torch.nn.functional as F
import numpy as np

def compute_gradient_alignment_fixed(task_query_loss, learner, meta, meta_loss_agg):
    """
    Fixed gradient alignment computation that handles MetaSGD parameter structure correctly
    
    Args:
        task_query_loss: Loss for individual task
        learner: The adapted learner (MetaSGD instance)
        meta: The meta-learner (MetaSGD instance)  
        meta_loss_agg: The aggregated meta loss
        
    Returns:
        float: Gradient alignment score or np.nan if computation fails
    """
    
    try:
        # Step 1: Compute task gradients w.r.t. base model parameters only
        # We want gradients w.r.t. the actual model parameters, not MetaSGD's lr parameters
        if hasattr(learner, 'module'):
            task_params = learner.module.parameters()
        else:
            task_params = learner.parameters()
            
        task_grads_raw = torch.autograd.grad(
            task_query_loss, 
            task_params, 
            retain_graph=True, 
            allow_unused=True
        )
        
        # Flatten task gradients
        task_grad_vec = torch.cat([g.view(-1) for g in task_grads_raw if g is not None])
        
        if task_grad_vec.nelement() == 0:
            return np.nan
            
        # Step 2: Compute meta gradients w.r.t. base model parameters only  
        # Clear any existing gradients
        for param in meta.parameters():
            if param.grad is not None:
                param.grad.zero_()
                
        # Compute meta gradients
        meta_loss_agg.backward(retain_graph=True)
        
        # Extract meta gradients from base model parameters only
        if hasattr(meta, 'module'):
            meta_params = meta.module.parameters()
        else:
            meta_params = meta.parameters()
            
        meta_grad_list = [p.grad.detach().view(-1) for p in meta_params if p.grad is not None]
        
        if not meta_grad_list:
            return np.nan
            
        meta_grad_vec = torch.cat(meta_grad_list)
        
        # Step 3: Check dimension compatibility
        if task_grad_vec.nelement() != meta_grad_vec.nelement():
            print(f"Warning: Gradient dimension mismatch - Task: {task_grad_vec.nelement()}, Meta: {meta_grad_vec.nelement()}")
            return np.nan
            
        # Step 4: Compute cosine similarity
        alignment = F.cosine_similarity(meta_grad_vec, task_grad_vec, dim=0)
        
        return alignment.item()
        
    except Exception as e:
        print(f"Gradient alignment computation failed: {e}")
        return np.nan

def demo_fix():
    """Demonstrate the fix with the same setup as the debug script"""
    import learn2learn as l2l
    from models import MLP
    
    # Create model and meta-learner
    device = torch.device("cpu") 
    model = MLP(n_input=8, n_output=1, n_hidden=32, n_layers=3).to(device)
    meta = l2l.algorithms.MetaSGD(model, lr=0.01, first_order=True).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} base parameters")
    print(f"✅ Meta has {sum(p.numel() for p in meta.parameters())} total parameters")
    print(f"✅ Meta.module has {sum(p.numel() for p in meta.module.parameters())} base parameters")
    
    # Generate synthetic data
    batch_size = 10
    X_s = torch.randn(batch_size, 8)
    y_s = torch.randint(0, 2, (batch_size, 1)).float()
    X_q = torch.randn(batch_size, 8)
    y_q = torch.randint(0, 2, (batch_size, 1)).float()
    
    # Simulate meta-training step
    meta.train()
    
    # Create optimizer and zero gradients
    optimizer = torch.optim.AdamW(meta.parameters(), lr=1e-3)
    optimizer.zero_grad()
    
    # Simulate multiple tasks for meta-batch
    meta_loss_agg = 0.0
    tasks_per_meta_batch = 4
    alignments = []
    
    for t in range(tasks_per_meta_batch):
        # Clone learner for this task
        learner = meta.clone()
        
        # Adaptation step
        support_pred = learner(X_s)
        support_loss = criterion(support_pred, y_s)
        learner.adapt(support_loss)
        
        # Query step
        query_pred = learner(X_q)
        task_query_loss = criterion(query_pred, y_q)
        meta_loss_agg += task_query_loss
        
        # Compute gradient alignment for this task
        alignment = compute_gradient_alignment_fixed(
            task_query_loss, learner, meta, task_query_loss
        )
        
        if not np.isnan(alignment):
            alignments.append(alignment)
            print(f"✅ Task {t+1} gradient alignment: {alignment:.4f}")
        else:
            print(f"❌ Task {t+1} gradient alignment failed")
    
    # Finalize meta update
    meta_loss_agg /= tasks_per_meta_batch
    meta_loss_agg.backward()
    optimizer.step()
    
    if alignments:
        avg_alignment = np.mean(alignments)
        print(f"🎉 Average gradient alignment: {avg_alignment:.4f}")
        return avg_alignment
    else:
        print("❌ No successful gradient alignment computations")
        return None

if __name__ == "__main__":
    print("🔧 Testing fixed gradient alignment computation...")
    result = demo_fix()
    if result is not None:
        print(f"✅ Fixed gradient alignment works: {result:.4f}")
    else:
        print("❌ Still having issues with gradient alignment") 