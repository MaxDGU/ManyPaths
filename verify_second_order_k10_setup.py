#!/usr/bin/env python3
"""
Verification Script for Second-Order Meta-SGD K=10 Setup

Validates the new experimental configuration before running on della.
"""

import os
import sys
from pathlib import Path

def check_slurm_script():
    """Verify the SLURM script exists and has correct configuration."""
    print("🔍 CHECKING SLURM SCRIPT CONFIGURATION")
    print("=" * 50)
    
    slurm_file = "run_concept_10step_second_order.slurm"
    
    if not os.path.exists(slurm_file):
        print(f"❌ SLURM script not found: {slurm_file}")
        return False
    
    print(f"✅ SLURM script found: {slurm_file}")
    
    # Read and verify key configurations
    with open(slurm_file, 'r') as f:
        content = f.read()
    
    # Check key configurations
    checks = [
        ("--array=0-8", "Array job range for 9 experiments"),
        ("ORDERS_LIST=(0)", "Second-order only configuration"),
        ("ADAPTATION_STEPS=10", "K=10 adaptation steps"),
        ("--time=12:00:00", "Sufficient time allocation"),
        ("tensorflow", "Correct conda environment"),
        ("--no_hyper_search", "Consistent with other experiments"),
        ("--hyper-index", "Uses same hyperparameter index")
    ]
    
    for check, description in checks:
        if check in content:
            print(f"✅ {description}: Found '{check}'")
        else:
            print(f"❌ {description}: Missing '{check}'")
            return False
    
    # Verify no --first-order flag (should default to second-order)
    if "--first-order" in content:
        print("❌ Found --first-order flag - should be removed for second-order")
        return False
    else:
        print("✅ No --first-order flag found (correct for second-order)")
    
    return True

def check_integration_compatibility():
    """Check compatibility with existing analysis pipeline."""
    print("\n🔧 CHECKING INTEGRATION COMPATIBILITY")
    print("=" * 50)
    
    # Check if aggregate_seed_results.py exists
    if os.path.exists("aggregate_seed_results.py"):
        print("✅ Found aggregate_seed_results.py for result processing")
    else:
        print("⚠️  aggregate_seed_results.py not found - may need manual integration")
    
    # Check results directory structure
    results_dirs = ["results", "saved_models", "saved_datasets"]
    for dir_name in results_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory exists: {dir_name}")
        else:
            print(f"⚠️  Directory will be created: {dir_name}")
    
    return True

def predict_experiment_coverage():
    """Show what experiments will be run."""
    print("\n📊 EXPERIMENT COVERAGE PREDICTION")
    print("=" * 50)
    
    features = [8, 16, 32]
    depths = [3, 5, 7]
    
    print("Second-Order Meta-SGD K=10 experiments to be run:")
    
    experiment_id = 0
    for feature_idx, features_val in enumerate(features):
        for depth_idx, depth_val in enumerate(depths):
            print(f"  Array[{experiment_id}]: F{features_val}_D{depth_val}_2ndOrd_K10")
            experiment_id += 1
    
    print(f"\nTotal experiments: {len(features) * len(depths)} = 9")
    print("Estimated completion time: 35-45 hours")
    
    # Predict result filenames
    print("\nExpected result files (examples):")
    for features_val in [8, 16, 32]:
        filename = f"concept_mlp_14_bits_feats{features_val}_depth3_adapt10_2ndOrd_seed0_trajectory.csv"
        print(f"  {filename}")

def check_experimental_matrix_completion():
    """Show how this completes the experimental matrix."""
    print("\n🎯 EXPERIMENTAL MATRIX COMPLETION")
    print("=" * 50)
    
    print("Before (incomplete matrix):")
    print("| Algorithm             | K=1 | K=10 |")
    print("|----------------------|-----|------|") 
    print("| SGD Baseline         | ✅   | N/A  |")
    print("| First-Order Meta-SGD | ✅   | ✅    |")
    print("| Second-Order Meta-SGD| ✅   | ❌    |")
    
    print("\nAfter (complete matrix):")
    print("| Algorithm             | K=1 | K=10 |")
    print("|----------------------|-----|------|")
    print("| SGD Baseline         | ✅   | N/A  |") 
    print("| First-Order Meta-SGD | ✅   | ✅    |")
    print("| Second-Order Meta-SGD| ✅   | ✅    |")
    
    print("\n🎉 This completes the 2×2 meta-learning comparison matrix!")

def generate_execution_commands():
    """Generate the exact commands needed for execution."""
    print("\n🚀 EXECUTION COMMANDS")
    print("=" * 50)
    
    print("1. LOCAL TESTING (optional):")
    print("python main.py \\")
    print("    --experiment concept \\")
    print("    --m mlp \\")
    print("    --data-type bits \\")
    print("    --num-concept-features 8 \\")
    print("    --pcfg-max-depth 3 \\")
    print("    --adaptation-steps 10 \\")
    print("    --epochs 100 \\")
    print("    --seed 0 \\")
    print("    --save \\")
    print("    --no_hyper_search \\")
    print("    --hyper-index 14")
    print("# Note: No --first-order flag for second-order")
    
    print("\n2. GIT INTEGRATION:")
    print("git add run_concept_10step_second_order.slurm SECOND_ORDER_K10_PLAN.md verify_second_order_k10_setup.py")
    print("git commit -m 'Add second-order meta-SGD K=10 experiments'")
    print("git push origin master")
    
    print("\n3. DELLA EXECUTION:")
    print("cd /scratch/gpfs/mg7411/ManyPaths")
    print("git pull origin master")
    print("sbatch run_concept_10step_second_order.slurm")
    print("squeue -u mg7411 | grep k10_2nd")

def main():
    """Run all verification checks."""
    print("🧪 SECOND-ORDER META-SGD K=10 SETUP VERIFICATION")
    print("=" * 60)
    
    all_checks_passed = True
    
    # Run verification checks
    if not check_slurm_script():
        all_checks_passed = False
    
    if not check_integration_compatibility():
        all_checks_passed = False
    
    # Show predictions and plans
    predict_experiment_coverage()
    check_experimental_matrix_completion()
    generate_execution_commands()
    
    # Final status
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED - Ready for execution!")
        print("🎯 This will complete your experimental matrix for the ICML paper.")
    else:
        print("❌ Some checks failed - please review and fix issues before proceeding.")
    
    return all_checks_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 