#!/usr/bin/env python3
"""
Quick test to verify gradient alignment fix works.
"""

import subprocess
import sys
import os

def test_gradient_alignment():
    """Test gradient alignment with a minimal run."""
    print("🧪 TESTING GRADIENT ALIGNMENT FIX")
    print("=" * 50)
    
    # Run a minimal experiment to test gradient alignment
    cmd = [
        sys.executable, 'main.py',
        '--experiment', 'concept',
        '--m', 'mlp',
        '--data-type', 'bits',
        '--num-concept-features', '8',
        '--pcfg-max-depth', '3',
        '--adaptation-steps', '1',
        '--first-order',
        '--epochs', '1',  # Just 1 epoch
        '--seed', '99',
        '--save',
        '--no_hyper_search',
        '--tasks_per_meta_batch', '2'  # Small batch for speed
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✅ Test run completed successfully")
            
            # Check for gradient alignment in output
            stdout = result.stdout
            if 'GradAlign:' in stdout:
                # Find all gradient alignment values
                grad_align_lines = [line for line in stdout.split('\n') if 'GradAlign:' in line]
                
                has_non_na_values = False
                for line in grad_align_lines:
                    if 'GradAlign:' in line:
                        # Extract the value after GradAlign:
                        try:
                            grad_align_part = line.split('GradAlign:')[1].split(',')[0].strip()
                            if grad_align_part != 'N/A':
                                has_non_na_values = True
                                print(f"✅ Found gradient alignment value: {grad_align_part}")
                        except:
                            pass
                
                if has_non_na_values:
                    print("🎉 GRADIENT ALIGNMENT FIX WORKS!")
                    return True
                else:
                    print("❌ All gradient alignment values are still N/A")
                    return False
            else:
                print("❌ No gradient alignment output found")
                return False
        else:
            print(f"❌ Test run failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test run timed out")
        return False
    except Exception as e:
        print(f"❌ Test run failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_gradient_alignment()
    sys.exit(0 if success else 1) 