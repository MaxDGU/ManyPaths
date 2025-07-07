#!/usr/bin/env python3
"""
Verbose debugging test for della to identify exactly where the hang occurs
"""

import sys
import os
import time
import subprocess
import signal
import traceback

def print_debug(message, force_flush=True):
    """Print debug message with timestamp and flush immediately"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    if force_flush:
        sys.stdout.flush()

def test_step_by_step():
    """Test each import and initialization step individually"""
    
    print_debug("🚀 Starting verbose step-by-step test...")
    print_debug(f"Python version: {sys.version}")
    print_debug(f"Working directory: {os.getcwd()}")
    print_debug(f"PID: {os.getpid()}")
    
    # Step 1: Basic imports
    print_debug("Step 1: Testing basic imports...")
    try:
        import torch
        print_debug(f"✅ PyTorch imported: {torch.__version__}")
        print_debug(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print_debug(f"✅ CUDA devices: {torch.cuda.device_count()}")
    except Exception as e:
        print_debug(f"❌ PyTorch import failed: {e}")
        return False
    
    # Step 2: Project imports (one by one)
    print_debug("Step 2: Testing project imports one by one...")
    try:
        print_debug("  2a: Importing constants...")
        import constants
        print_debug("  ✅ Constants imported")
        
        print_debug("  2b: Importing utils...")
        from utils import set_random_seeds
        print_debug("  ✅ Utils imported")
        
        print_debug("  2c: Importing models...")
        from models import MLP
        print_debug("  ✅ Models imported")
        
        print_debug("  2d: Importing initialization...")
        from initialization import init_model, init_dataset, init_misc
        print_debug("  ✅ Initialization imported")
        
    except Exception as e:
        print_debug(f"❌ Project imports failed: {e}")
        traceback.print_exc()
        return False
    
    # Step 3: The problematic learn2learn import
    print_debug("Step 3: Testing learn2learn import (this might hang)...")
    import_start_time = time.time()
    try:
        print_debug("  3a: Starting learn2learn import...")
        import learn2learn as l2l
        import_duration = time.time() - import_start_time
        print_debug(f"  ✅ learn2learn imported in {import_duration:.2f}s: {l2l.__version__}")
    except Exception as e:
        print_debug(f"❌ learn2learn import failed after {time.time() - import_start_time:.2f}s: {e}")
        traceback.print_exc()
        return False
    
    # Step 4: Model creation
    print_debug("Step 4: Testing model creation...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_debug(f"  4a: Device selected: {device}")
        
        model = MLP(n_input=8, n_output=1, n_hidden=32, n_layers=3).to(device)
        print_debug(f"  ✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        meta = l2l.algorithms.MetaSGD(model, lr=0.01, first_order=True).to(device)
        print_debug(f"  ✅ MetaSGD created with {sum(p.numel() for p in meta.parameters())} total parameters")
        
    except Exception as e:
        print_debug(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return False
    
    # Step 5: Dataset initialization (this might be slow/hanging)
    print_debug("Step 5: Testing dataset initialization...")
    try:
        print_debug("  5a: Setting random seeds...")
        set_random_seeds(42)
        print_debug("  ✅ Random seeds set")
        
        print_debug("  5b: Initializing misc values...")
        alphabet, bits_for_model, channels, n_output = init_misc('concept', 'asian', num_concept_features_override=8)
        print_debug(f"  ✅ Misc initialized: bits={bits_for_model}, channels={channels}, n_output={n_output}")
        
        print_debug("  5c: Initializing datasets (this might be slow)...")
        dataset_start_time = time.time()
        
        # Use None paths to avoid file I/O issues
        train_dataset, test_dataset, val_dataset = init_dataset(
            'concept', 
            model_arch='mlp',
            data_type='bits', 
            skip_param=1,
            alphabet=alphabet, 
            num_concept_features=8,
            pcfg_max_depth=3,
            save_train_path=None,  # Don't save to avoid I/O issues
            save_val_path=None,
            save_test_path=None
        )
        
        dataset_duration = time.time() - dataset_start_time
        print_debug(f"  ✅ Datasets initialized in {dataset_duration:.2f}s")
        print_debug(f"  ✅ Train dataset size: {len(train_dataset)}")
        
    except Exception as e:
        print_debug(f"❌ Dataset initialization failed after {time.time() - dataset_start_time:.2f}s: {e}")
        traceback.print_exc()
        return False
    
    # Step 6: DataLoader creation
    print_debug("Step 6: Testing DataLoader creation...")
    try:
        from torch.utils.data import DataLoader
        from utils import get_collate
        
        collate_fn = get_collate('concept', device)
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            drop_last=True,
            pin_memory=False,  # Disable pin_memory since collate_fn moves tensors to CUDA
            collate_fn=collate_fn,
        )
        print_debug("  ✅ DataLoader created")
        
        # Test one batch
        print_debug("  6a: Testing one batch iteration...")
        batch_start_time = time.time()
        for i, (X_s, y_s, X_q, y_q) in enumerate(train_loader):
            batch_duration = time.time() - batch_start_time
            print_debug(f"  ✅ First batch loaded in {batch_duration:.2f}s: X_s shape {X_s[0].shape}")
            break
            
    except Exception as e:
        print_debug(f"❌ DataLoader testing failed: {e}")
        traceback.print_exc()
        return False
    
    # Step 7: Training step simulation
    print_debug("Step 7: Testing minimal training step...")
    try:
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(meta.parameters(), lr=1e-3)
        
        print_debug("  7a: Starting adaptation...")
        learner = meta.clone()
        support_pred = learner(X_s[0])
        support_loss = criterion(support_pred, y_s[0])
        learner.adapt(support_loss)
        print_debug("  ✅ Adaptation completed")
        
        print_debug("  7b: Testing query step...")
        query_pred = learner(X_q[0])
        task_query_loss = criterion(query_pred, y_q[0])
        print_debug(f"  ✅ Query loss: {task_query_loss.item():.4f}")
        
        print_debug("  7c: Testing gradient computation...")
        optimizer.zero_grad()
        task_query_loss.backward()
        optimizer.step()
        print_debug("  ✅ Gradient step completed")
        
    except Exception as e:
        print_debug(f"❌ Training step failed: {e}")
        traceback.print_exc()
        return False
    
    print_debug("🎉 All tests passed! No obvious hanging issues detected.")
    return True

def run_timeout_test(timeout_seconds=300):
    """Run the test with a timeout to catch hangs"""
    print_debug(f"Starting timeout test with {timeout_seconds}s limit...")
    
    def timeout_handler(signum, frame):
        print_debug(f"❌ Test timed out after {timeout_seconds}s")
        raise TimeoutError(f"Test timed out after {timeout_seconds}s")
    
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        success = test_step_by_step()
        signal.alarm(0)  # Cancel timeout
        return success
    except TimeoutError as e:
        print_debug(f"❌ Timeout error: {e}")
        return False
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        print_debug(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print_debug("=" * 60)
    print_debug("VERBOSE DEBUGGING TEST FOR DELLA HANGING ISSUE")
    print_debug("=" * 60)
    
    success = run_timeout_test(timeout_seconds=600)  # 10 minute timeout
    
    if success:
        print_debug("✅ All tests completed successfully!")
        print_debug("💡 If this worked but main.py still hangs, the issue is in main.py's argument parsing or other logic")
    else:
        print_debug("❌ Test failed or timed out")
        print_debug("💡 Check the last successful step to identify the hanging point")
    
    print_debug("=" * 60)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 