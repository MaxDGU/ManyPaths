#!/usr/bin/env python3
"""
Test gradient alignment experiments on della with extensive verbose debugging
"""

import sys
import os
import time
import subprocess
import signal

def print_debug(message, force_flush=True):
    """Print debug message with timestamp and flush immediately"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] DEBUG: {message}")
    if force_flush:
        sys.stdout.flush()

def run_verbose_test():
    """Run the main.py test with verbose output and monitoring"""
    
    print_debug("🚀 Starting verbose main.py test...")
    
    # Very minimal test configuration with verbose flags if available
    test_cmd = [
        sys.executable, 'main.py',
        '--experiment', 'concept',
        '--m', 'mlp',
        '--data-type', 'bits',
        '--num-concept-features', '8',
        '--pcfg-max-depth', '3',
        '--adaptation-steps', '1',
        '--first-order',
        '--epochs', '3',  # Very short for testing
        '--tasks_per_meta_batch', '2',  # Smaller batch
        '--outer_lr', '1e-3',
        '--seed', '42',
        '--save',
        '--no_hyper_search',  # Skip hyperparameter search to avoid issues
        '--verbose-debug'  # Enable verbose debugging output
    ]
    
    print_debug(f"Command: {' '.join(test_cmd)}")
    
    # Run with timeout and capture output
    timeout_seconds = 600  # 10 minutes
    
    try:
        print_debug(f"Starting command with {timeout_seconds}s timeout...")
        start_time = time.time()
        
        # Use Popen for real-time output
        process = subprocess.Popen(
            test_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        print_debug(f"Process started with PID: {process.pid}")
        
        # Monitor output in real-time
        stdout_lines = []
        stderr_lines = []
        last_output_time = time.time()
        
        while True:
            # Check if process has finished
            if process.poll() is not None:
                break
                
            # Check for timeout
            current_time = time.time()
            if current_time - start_time > timeout_seconds:
                print_debug(f"❌ Killing process due to {timeout_seconds}s timeout")
                process.kill()
                process.wait()
                return False, "Timeout", stdout_lines, stderr_lines
            
            # Read output with timeout
            try:
                # Check for new stdout
                if process.stdout:
                    import select
                    if select.select([process.stdout], [], [], 1):  # 1 second timeout
                        line = process.stdout.readline()
                        if line:
                            line = line.strip()
                            stdout_lines.append(line)
                            print_debug(f"STDOUT: {line}")
                            last_output_time = current_time
                
                # Check for new stderr
                if process.stderr:
                    if select.select([process.stderr], [], [], 0.1):  # 0.1 second timeout
                        line = process.stderr.readline()
                        if line:
                            line = line.strip()
                            stderr_lines.append(line)
                            print_debug(f"STDERR: {line}")
                            last_output_time = current_time
                            
            except Exception as e:
                print_debug(f"Error reading process output: {e}")
                break
            
            # Check for stalled output (no output for 120 seconds)
            if current_time - last_output_time > 120:
                print_debug(f"❌ No output for 120s, process appears stalled")
                print_debug(f"Last output was at: {time.strftime('%H:%M:%S', time.localtime(last_output_time))}")
                process.kill()
                process.wait()
                return False, "Stalled", stdout_lines, stderr_lines
        
        # Process finished
        return_code = process.returncode
        duration = time.time() - start_time
        
        # Get any remaining output
        remaining_stdout, remaining_stderr = process.communicate()
        if remaining_stdout:
            for line in remaining_stdout.strip().split('\n'):
                if line:
                    stdout_lines.append(line)
                    print_debug(f"STDOUT: {line}")
        if remaining_stderr:
            for line in remaining_stderr.strip().split('\n'):
                if line:
                    stderr_lines.append(line)
                    print_debug(f"STDERR: {line}")
        
        print_debug(f"Process completed in {duration:.2f}s with return code {return_code}")
        
        success = return_code == 0
        reason = "Success" if success else f"Error (code {return_code})"
        
        return success, reason, stdout_lines, stderr_lines
        
    except Exception as e:
        print_debug(f"❌ Exception running test: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Exception: {e}", [], []

def check_gradient_data(results_dir):
    """Check if gradient alignment data was generated"""
    import pandas as pd
    
    if not os.path.exists(results_dir):
        return False, "Results directory not found"
    
    trajectory_files = [f for f in os.listdir(results_dir) if f.endswith('_trajectory.csv')]
    if not trajectory_files:
        return False, "No trajectory files found"
    
    # Look for the most recent trajectory file
    latest_file = max(trajectory_files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
    trajectory_path = os.path.join(results_dir, latest_file)
    
    try:
        df = pd.read_csv(trajectory_path)
        
        if 'grad_alignment' not in df.columns:
            return False, "No gradient alignment column"
        
        grad_data = df['grad_alignment'].dropna()
        if len(grad_data) == 0:
            return False, "Gradient alignment column is empty"
        
        return True, f"Found {len(grad_data)} gradient alignment data points (range: {grad_data.min():.3f} to {grad_data.max():.3f})"
        
    except Exception as e:
        return False, f"Error reading trajectory file: {e}"

def main():
    """Main test function with extensive logging"""
    
    print_debug("=" * 80)
    print_debug("VERBOSE GRADIENT ALIGNMENT TEST FOR DELLA")
    print_debug("=" * 80)
    print_debug(f"Working directory: {os.getcwd()}")
    print_debug(f"Python executable: {sys.executable}")
    print_debug(f"Environment variables:")
    for key in ['CUDA_VISIBLE_DEVICES', 'SLURM_JOB_ID', 'SLURM_NODEID']:
        print_debug(f"  {key}: {os.environ.get(key, 'Not set')}")
    
    # Run the test
    success, reason, stdout_lines, stderr_lines = run_verbose_test()
    
    print_debug("=" * 80)
    print_debug("TEST RESULTS SUMMARY")
    print_debug("=" * 80)
    
    if success:
        print_debug("✅ Test completed successfully!")
        
        # Check gradient alignment data
        success, message = check_gradient_data("results")  # Use default results directory
        
        if success:
            print_debug(f"✅ Gradient alignment data: {message}")
            print_debug("🎉 All tests passed! Gradient alignment is working.")
        else:
            print_debug(f"⚠️  Gradient alignment issue: {message}")
            print_debug("💡 Test ran but gradient alignment data is missing/invalid")
            
    else:
        print_debug(f"❌ Test failed: {reason}")
        
        if stdout_lines:
            print_debug("📝 Last few stdout lines:")
            for line in stdout_lines[-10:]:  # Last 10 lines
                print_debug(f"  STDOUT: {line}")
        
        if stderr_lines:
            print_debug("📝 Last few stderr lines:")
            for line in stderr_lines[-10:]:  # Last 10 lines
                print_debug(f"  STDERR: {line}")
        
        # Provide debugging hints based on where it failed
        if not stdout_lines:
            print_debug("💡 No stdout at all - likely hanging at import or argument parsing")
        elif any("learn2learn" in line for line in stdout_lines):
            print_debug("💡 learn2learn imported - hanging likely in dataset/training logic")
        elif any("Dataset" in line for line in stdout_lines):
            print_debug("💡 Dataset mentioned - hanging likely in training loop")
        else:
            print_debug("💡 Check the last output line to identify hanging point")
    
    print_debug("=" * 80)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 