#!/usr/bin/env python3
"""
Debug gradient alignment computation to see why it's returning N/A
"""

import torch
import torch.nn.functional as F
import numpy as np
import learn2learn as l2l
from models import MLP

def debug_gradient_alignment():
    """Debug gradient alignment computation step by step"""
    
    print("🔍 Debugging gradient alignment computation...")
    
    # Create a simple model and meta-learner
    device = torch.device("cpu")
    model = MLP(n_input=8, n_output=1, n_hidden=32, n_layers=3).to(device)
    meta = l2l.algorithms.MetaSGD(model, lr=0.01, first_order=True).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Generate simple synthetic data
    batch_size = 10
    X_s = torch.randn(batch_size, 8)
    y_s = torch.randint(0, 2, (batch_size, 1)).float()
    X_q = torch.randn(batch_size, 8)
    y_q = torch.randint(0, 2, (batch_size, 1)).float()
    
    print(f"✅ Synthetic data generated: {X_s.shape} support, {X_q.shape} query")
    
    # Simulate meta-training step
    meta.train()
    
    # Step 1: Adaptation
    learner = meta.clone()
    print(f"✅ Learner cloned")
    
    # Forward pass on support
    support_pred = learner(X_s)
    support_loss = criterion(support_pred, y_s)
    print(f"✅ Support loss: {support_loss.item():.4f}")
    
    # Adapt
    learner.adapt(support_loss)
    print(f"✅ Adaptation completed")
    
    # Step 2: Query loss
    query_pred = learner(X_q)
    task_query_loss = criterion(query_pred, y_q)
    print(f"✅ Query loss: {task_query_loss.item():.4f}")
    
    # Step 3: Compute task gradients
    print("🔍 Computing task gradients...")
    try:
        task_grads_raw = torch.autograd.grad(task_query_loss, learner.parameters(), retain_graph=True, allow_unused=True)
        print(f"✅ Task gradients computed: {len(task_grads_raw)} gradients")
        
        # Check which gradients are None
        none_grads = sum(1 for g in task_grads_raw if g is None)
        print(f"✅ Non-None gradients: {len(task_grads_raw) - none_grads}/{len(task_grads_raw)}")
        
        # Flatten task gradients
        task_grad_vec = torch.cat([g.view(-1) for g in task_grads_raw if g is not None])
        print(f"✅ Task gradient vector: {task_grad_vec.shape} elements")
        
    except Exception as e:
        print(f"❌ Task gradient computation failed: {e}")
        return None
    
    # Step 4: Meta gradient computation
    print("🔍 Computing meta gradients...")
    
    # Create meta optimizer and compute meta gradients
    optimizer = torch.optim.AdamW(meta.parameters(), lr=1e-3)
    optimizer.zero_grad()
    
    try:
        task_query_loss.backward()
        print(f"✅ Meta backward completed")
        
        # Extract meta gradients
        meta_grad_list = [p.grad.detach().view(-1) for p in meta.parameters() if p.grad is not None]
        print(f"✅ Meta gradients extracted: {len(meta_grad_list)} gradients")
        
        if meta_grad_list:
            meta_grad_vec = torch.cat(meta_grad_list)
            print(f"✅ Meta gradient vector: {meta_grad_vec.shape} elements")
            
            # Step 5: Check dimension compatibility
            print("🔍 Checking dimension compatibility...")
            print(f"Task grad elements: {task_grad_vec.nelement()}")
            print(f"Meta grad elements: {meta_grad_vec.nelement()}")
            
            if task_grad_vec.nelement() == meta_grad_vec.nelement():
                print("✅ Dimensions match!")
                
                # Step 6: Compute cosine similarity
                print("🔍 Computing cosine similarity...")
                alignment = F.cosine_similarity(meta_grad_vec, task_grad_vec, dim=0)
                print(f"✅ Gradient alignment: {alignment.item():.4f}")
                
                return alignment.item()
            else:
                print("❌ Dimension mismatch!")
                print(f"Task grad shape: {task_grad_vec.shape}")
                print(f"Meta grad shape: {meta_grad_vec.shape}")
                
                # Debug: print parameter names and shapes
                print("🔍 Debugging parameter shapes...")
                print("Task gradient parameters:")
                for i, param in enumerate(learner.parameters()):
                    print(f"  {i}: {param.shape}")
                print("Meta gradient parameters:")
                for i, param in enumerate(meta.parameters()):
                    print(f"  {i}: {param.shape}")
                return None
        else:
            print("❌ No meta gradients found!")
            return None
    
    except Exception as e:
        print(f"❌ Meta gradient computation failed: {e}")
        return None

if __name__ == "__main__":
    result = debug_gradient_alignment()
    if result is not None:
        print(f"🎉 Gradient alignment computation successful: {result:.4f}")
    else:
        print("❌ Gradient alignment computation failed") 