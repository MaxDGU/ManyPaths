#!/usr/bin/env python3
"""
Debug Cache Format
==================

Quick script to inspect the format of cached tasks to understand
the structure and fix loading issues.
"""

import torch
import os

def inspect_cache_file(cache_path):
    """Inspect a cached task file to understand its structure."""
    
    print(f"🔍 Inspecting cache file: {cache_path}")
    print("=" * 60)
    
    if not os.path.exists(cache_path):
        print(f"❌ Cache file not found: {cache_path}")
        return
    
    try:
        # Load cache file
        cached_data = torch.load(cache_path, map_location='cpu', weights_only=False)
        
        print(f"📦 Cache data type: {type(cached_data)}")
        
        if isinstance(cached_data, (tuple, list)) and len(cached_data) == 2:
            cached_tasks, meta_info = cached_data
            print(f"📋 Meta info: {meta_info}")
        else:
            cached_tasks = cached_data
            print("📋 No separate meta info found")
        
        print(f"📊 Number of tasks: {len(cached_tasks)}")
        
        # Inspect first few tasks
        for i in range(min(3, len(cached_tasks))):
            print(f"\n🎯 Task {i}:")
            task = cached_tasks[i]
            
            print(f"   Type: {type(task)}")
            
            if isinstance(task, dict):
                print(f"   Keys: {list(task.keys())}")
                for key, value in task.items():
                    if hasattr(value, 'shape'):
                        print(f"   {key}: {type(value)} shape={value.shape} dtype={value.dtype}")
                    else:
                        print(f"   {key}: {type(value)} = {value}")
                        
            elif isinstance(task, (tuple, list)):
                print(f"   Length: {len(task)}")
                for j, item in enumerate(task):
                    if hasattr(item, 'shape'):
                        print(f"   Item {j}: {type(item)} shape={item.shape} dtype={item.dtype}")
                    else:
                        print(f"   Item {j}: {type(item)} = {str(item)[:100]}")
            else:
                print(f"   Value: {str(task)[:200]}")
    
    except Exception as e:
        print(f"❌ Error loading cache: {e}")

def main():
    """Main inspection function."""
    
    cache_files = [
        "data/concept_cache/pcfg_tasks_f8_d3_s2p3n_q5p5n_t10000.pt",
        "data/concept_cache/pcfg_tasks_f8_d5_s2p3n_q5p5n_t10000.pt", 
        "data/concept_cache/pcfg_tasks_f32_d3_s2p3n_q5p5n_t10000.pt"
    ]
    
    for cache_path in cache_files:
        inspect_cache_file(cache_path)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main() 