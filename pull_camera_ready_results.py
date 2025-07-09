#!/usr/bin/env python3
"""
Pull Camera-Ready Results from Della and Run Analysis

This script pulls the latest camera_ready_array results from Della and runs
the clean analysis pipeline to generate publication-ready figures.
"""

import subprocess
import os
import sys
import shutil
from pathlib import Path
import argparse
import time

class DellaResultsPuller:
    """Pulls and processes camera-ready results from Della"""
    
    def __init__(self, della_user: str = "mg7411", local_results_dir: str = "camera_ready_fresh"):
        self.della_user = della_user
        self.local_results_dir = Path(local_results_dir)
        self.della_path = f"/scratch/gpfs/{della_user}/ManyPaths"
        self.local_results_dir.mkdir(exist_ok=True)
        
    def check_della_connection(self):
        """Check if we can connect to Della"""
        try:
            result = subprocess.run(
                ["ssh", f"{self.della_user}@della-gpu.princeton.edu", "echo 'Connected'"], 
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print("✅ Della connection successful")
                return True
            else:
                print(f"❌ Della connection failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("❌ Della connection timeout")
            return False
        except Exception as e:
            print(f"❌ Della connection error: {e}")
            return False
    
    def check_job_status(self):
        """Check status of camera_ready_array jobs"""
        try:
            result = subprocess.run([
                "ssh", f"{self.della_user}@della-gpu.princeton.edu", 
                "cd /scratch/gpfs/mg7411/ManyPaths && squeue -u mg7411 | grep camera_ready"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                print("🔄 Camera-ready jobs still running:")
                print(result.stdout)
                return "running"
            else:
                print("✅ No camera-ready jobs running (likely completed)")
                return "completed"
                
        except Exception as e:
            print(f"⚠️  Could not check job status: {e}")
            return "unknown"
    
    def list_camera_ready_files(self):
        """List camera_ready_array files on Della"""
        try:
            result = subprocess.run([
                "ssh", f"{self.della_user}@della-gpu.princeton.edu", 
                f"cd {self.della_path} && find logs/ -name 'camera_ready_array_*_*.out' | head -20"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                if files and files[0]:
                    print(f"📁 Found {len(files)} camera-ready files on Della:")
                    for file in files[:5]:  # Show first 5
                        print(f"   - {file}")
                    if len(files) > 5:
                        print(f"   ... and {len(files) - 5} more")
                    return files
                else:
                    print("📂 No camera-ready files found on Della")
                    return []
            else:
                print(f"❌ Failed to list files: {result.stderr}")
                return []
                
        except Exception as e:
            print(f"❌ Error listing files: {e}")
            return []
    
    def pull_results(self, compress: bool = True):
        """Pull results from Della"""
        print("📥 Pulling camera-ready results from Della...")
        
        try:
            if compress:
                # Create compressed archive on Della
                print("   Creating compressed archive on Della...")
                compress_cmd = [
                    "ssh", f"{self.della_user}@della-gpu.princeton.edu",
                    f"cd {self.della_path} && tar -czf camera_ready_results.tar.gz "
                    f"logs/camera_ready_array_*_*.out results/ saved_models/ || true"
                ]
                
                result = subprocess.run(compress_cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"⚠️  Warning during compression: {result.stderr}")
                
                # Pull compressed file
                print("   Downloading compressed results...")
                scp_cmd = [
                    "scp", f"{self.della_user}@della-gpu.princeton.edu:{self.della_path}/camera_ready_results.tar.gz",
                    str(self.local_results_dir)
                ]
                
                result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    print("   ✅ Download successful")
                    
                    # Extract
                    print("   Extracting results...")
                    extract_cmd = ["tar", "-xzf", str(self.local_results_dir / "camera_ready_results.tar.gz"),
                                  "-C", str(self.local_results_dir)]
                    
                    result = subprocess.run(extract_cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        print("   ✅ Extraction successful")
                        return True
                    else:
                        print(f"   ❌ Extraction failed: {result.stderr}")
                        return False
                else:
                    print(f"   ❌ Download failed: {result.stderr}")
                    return False
            else:
                # Direct rsync (alternative method)
                print("   Using rsync to pull results...")
                rsync_cmd = [
                    "rsync", "-avz", "--progress",
                    f"{self.della_user}@della-gpu.princeton.edu:{self.della_path}/logs/camera_ready_array_*_*.out",
                    str(self.local_results_dir / "logs/")
                ]
                
                os.makedirs(self.local_results_dir / "logs", exist_ok=True)
                result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    print("   ✅ Rsync successful")
                    return True
                else:
                    print(f"   ❌ Rsync failed: {result.stderr}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error pulling results: {e}")
            return False
    
    def run_analysis(self):
        """Run camera-ready analysis on pulled results"""
        print("🔬 Running camera-ready analysis...")
        
        try:
            # Import and run the analysis
            sys.path.insert(0, str(Path.cwd()))
            from camera_ready_master_analysis import CameraReadyAnalyzer
            
            # Create analyzer pointing to pulled results
            analyzer = CameraReadyAnalyzer(
                results_dir=str(self.local_results_dir / "results"),
                output_dir="camera_ready_final"
            )
            
            # Override logs directory to point to pulled logs
            analyzer.logs_dir = str(self.local_results_dir / "logs")
            
            # Run analysis
            comparison_data = analyzer.run_full_analysis()
            
            print("✅ Analysis complete!")
            return comparison_data
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return None
    
    def cleanup(self):
        """Clean up temporary files"""
        tar_file = self.local_results_dir / "camera_ready_results.tar.gz"
        if tar_file.exists():
            tar_file.unlink()
            print("🧹 Cleaned up temporary tar file")

def main():
    parser = argparse.ArgumentParser(description='Pull and analyze camera-ready results from Della')
    parser.add_argument('--user', default='mg7411', help='Della username')
    parser.add_argument('--local-dir', default='camera_ready_fresh', help='Local directory for results')
    parser.add_argument('--check-only', action='store_true', help='Only check job status, don\'t pull')
    parser.add_argument('--pull-only', action='store_true', help='Only pull results, don\'t analyze')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze existing results')
    parser.add_argument('--no-compress', action='store_true', help='Use rsync instead of tar compression')
    
    args = parser.parse_args()
    
    puller = DellaResultsPuller(args.user, args.local_dir)
    
    print("🎯 Camera-Ready Results Puller")
    print("=" * 40)
    
    # Check connection
    if not puller.check_della_connection():
        print("❌ Cannot connect to Della. Please check your SSH setup.")
        sys.exit(1)
    
    # Check job status
    job_status = puller.check_job_status()
    
    if args.check_only:
        print(f"📊 Job status: {job_status}")
        sys.exit(0)
    
    # List available files
    files = puller.list_camera_ready_files()
    
    if not files and not args.analyze_only:
        print("❌ No camera-ready files found. Jobs may not be complete yet.")
        sys.exit(1)
    
    # Pull results
    if not args.analyze_only:
        print("\n" + "="*40)
        success = puller.pull_results(compress=not args.no_compress)
        
        if not success:
            print("❌ Failed to pull results from Della")
            sys.exit(1)
        
        if args.pull_only:
            print("✅ Results pulled successfully")
            sys.exit(0)
    
    # Run analysis
    if not args.pull_only:
        print("\n" + "="*40)
        comparison_data = puller.run_analysis()
        
        if comparison_data is None:
            print("❌ Analysis failed")
            sys.exit(1)
    
    # Cleanup
    puller.cleanup()
    
    print("\n" + "="*40)
    print("🎉 Camera-ready pipeline complete!")
    print(f"📁 Results saved to: camera_ready_final/")
    print("📊 Check the following files:")
    print("   - camera_ready_final/clean_trajectories.pdf")
    print("   - camera_ready_final/k_comparison.pdf")
    print("   - camera_ready_final/statistical_summary.csv")
    print("   - camera_ready_final/camera_ready_report.md")

if __name__ == "__main__":
    main() 