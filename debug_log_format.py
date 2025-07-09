#!/usr/bin/env python3
"""
Debug script to examine camera-ready log file format
"""

import re
from pathlib import Path

def debug_log_format():
    # Find a camera-ready log file
    log_files = list(Path(".").glob("camera_ready_array_*_*.out"))
    
    if not log_files:
        print("No camera-ready log files found")
        return
    
    log_file = log_files[0]
    print(f"🔍 Examining log file: {log_file}")
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    print(f"\n📊 File size: {len(content)} characters")
    print(f"📊 Line count: {len(content.splitlines())}")
    
    # Show first 50 lines
    lines = content.splitlines()
    print(f"\n📝 First 50 lines:")
    for i, line in enumerate(lines[:50]):
        print(f"{i+1:3d}: {line}")
    
    print(f"\n📝 Last 20 lines:")
    for i, line in enumerate(lines[-20:], len(lines)-20):
        print(f"{i+1:3d}: {line}")
    
    # Look for specific patterns
    print(f"\n🔍 Looking for configuration patterns:")
    
    # Configuration patterns
    features_matches = re.findall(r'(num.concept.features?|features?)[:\s=]+(\d+)', content, re.IGNORECASE)
    depth_matches = re.findall(r'(pcfg.max.depth|depth)[:\s=]+(\d+)', content, re.IGNORECASE)
    adapt_matches = re.findall(r'(adaptation.steps?|adapt)[:\s=]+(\d+)', content, re.IGNORECASE)
    seed_matches = re.findall(r'seed[:\s=]+(\d+)', content, re.IGNORECASE)
    
    print(f"Features matches: {features_matches}")
    print(f"Depth matches: {depth_matches}")
    print(f"Adaptation matches: {adapt_matches}")
    print(f"Seed matches: {seed_matches}")
    
    # Trajectory patterns
    print(f"\n🔍 Looking for trajectory patterns:")
    episode_matches = re.findall(r'(Episode|episode)[:\s]+(\d+)', content)
    accuracy_matches = re.findall(r'(accuracy|acc)[:\s]+([\d.]+)', content, re.IGNORECASE)
    validation_matches = re.findall(r'(validation|val)[^:]*accuracy[:\s]+([\d.]+)', content, re.IGNORECASE)
    
    print(f"Episode matches (first 10): {episode_matches[:10]}")
    print(f"Accuracy matches (first 10): {accuracy_matches[:10]}")
    print(f"Validation matches (first 10): {validation_matches[:10]}")
    
    # Final performance patterns
    print(f"\n🔍 Looking for final performance patterns:")
    final_matches = re.findall(r'(final|Final)[^:]*accuracy[:\s]+([\d.]+)', content, re.IGNORECASE)
    test_matches = re.findall(r'(test|Test)[^:]*accuracy[:\s]+([\d.]+)', content, re.IGNORECASE)
    
    print(f"Final accuracy matches: {final_matches}")
    print(f"Test accuracy matches: {test_matches}")
    
    # Look for SLURM array info
    print(f"\n🔍 Looking for SLURM array info:")
    array_matches = re.findall(r'SLURM_ARRAY_TASK_ID[:\s=]+(\d+)', content)
    job_matches = re.findall(r'SLURM_JOB_ID[:\s=]+(\d+)', content)
    
    print(f"Array task ID matches: {array_matches}")
    print(f"Job ID matches: {job_matches}")

if __name__ == "__main__":
    debug_log_format() 