import os
import re
import shutil
import argparse
from pathlib import Path

def organize_checkpoints(source_dir, target_base_dir, operation='copy'):
    """
    Organizes checkpoint files from a source directory into a structured target directory.

    Filename pattern expected:
    concept_mlp_<INDEX>_bits_feats<F>_depth<D>_adapt<A>_<ORDER>Ord_seed<S>_epoch_<E>.pt
    
    Target structure:
    <target_base_dir>/feats<F>_depth<D>_adapt<A>_<ORDER>Ord/seed<S>/<original_filename.pt>
    """

    # Regex to capture the necessary parts for folder hierarchy
    pattern = re.compile(
        r"concept_mlp_(?P<index>[^_]+)_bits_"
        r"(?P<feats>feats\d+)_"
        r"(?P<depth>depth\d+)_"
        r"(?P<adapt>adapt\d+)_"
        r"(?P<order>(?:1st|2nd)Ord)_"
        r"(?P<seed>seed\d+)_epoch_\d+\.pt$"
    )

    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    Path(target_base_dir).mkdir(parents=True, exist_ok=True)
    print(f"Scanning source directory: {source_dir}")
    print(f"Organizing into base directory: {target_base_dir}")
    print(f"Operation: {operation}")

    organized_count = 0
    skipped_count = 0
    unmatched_count = 0

    for filename in os.listdir(source_dir):
        match = pattern.match(filename)
        if match:
            parts = match.groupdict()
            
            primary_folder_name = f"{parts['feats']}_{parts['depth']}_{parts['adapt']}_{parts['order']}"
            seed_folder_name = parts['seed']
            
            target_subdir = Path(target_base_dir) / primary_folder_name / seed_folder_name
            target_subdir.mkdir(parents=True, exist_ok=True)
            
            source_filepath = Path(source_dir) / filename
            target_filepath = target_subdir / filename
            
            try:
                if operation == 'copy':
                    shutil.copy2(source_filepath, target_filepath)
                elif operation == 'move':
                    shutil.move(str(source_filepath), str(target_filepath))
                else:
                    print(f"Unknown operation: {operation}. Skipping {filename}")
                    skipped_count +=1
                    continue
                organized_count += 1
                if organized_count % 50 == 0:
                    print(f"  ... processed {organized_count} files ...")

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                skipped_count += 1
        else:
            unmatched_count += 1
            if unmatched_count <= 20: # Print first few non-matching files for diagnostics
                print(f"Skipping non-matching file: {filename}")
            elif unmatched_count == 21:
                print("  ... (suppressing further non-matching file messages) ...")

    print(f"\nOrganization complete.")
    print(f"Successfully {operation}ed {organized_count} files.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} files due to errors.")
    if unmatched_count > 0:
        print(f"{unmatched_count} files did not match the expected pattern.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize concept learning checkpoint files.")
    parser.add_argument("--source_dir", type=str, required=True, 
                        help="Source directory containing the checkpoint files.")
    parser.add_argument("--target_base_dir", type=str, required=True,
                        help="Target base directory where organized subfolders will be created.")
    parser.add_argument("--operation", type=str, choices=['copy', 'move'], default='copy',
                        help="Operation to perform: 'copy' or 'move' files (default: 'copy').")
    
    args = parser.parse_args()
    organize_checkpoints(args.source_dir, args.target_base_dir, args.operation)

# Example usage (run this on a machine with access to the files, e.g., Della login node):
# python organize_checkpoints.py \ 
#   --source_dir "/scratch/gpfs/mg7411/ManyPaths/saved_models/checkpoints" \ 
#   --target_base_dir "/scratch/gpfs/mg7411/ManyPaths/organized_checkpoints" \ 
#   --operation copy 