import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_trajectory_data(results_basedir="results", run_name="run1"):
    """Loads all trajectory CSVs from the specified subdirectory."""
    results_subdir = os.path.join(results_basedir, run_name) # Parameterized
    results_dir = os.path.join(os.getcwd(), results_subdir)
    if not os.path.isdir(results_dir):
        print(f"Error: Directory not found: {results_dir}")
        return {}

    all_files = os.listdir(results_dir)
    csv_files = [f for f in all_files if f.endswith("_trajectory.csv")]
    
    trajectories = {}
    # Regex to parse filename: e.g., concept_mlp_14_bits_feats8_depth5_adapt1_1stOrd_seed0_trajectory.csv
    pattern = re.compile(r"feats(\d+)_depth(\d+)_adapt1_(\w+)Ord_seed0_trajectory.csv")

    for fname in csv_files:
        match = pattern.search(fname)
        if match:
            features = int(match.group(1))
            depth = int(match.group(2))
            order_str = match.group(3) # "1st" or "2nd"
            order = 1 if order_str == "1st" else 2
            
            key = (features, depth, order)
            
            try:
                df = pd.read_csv(os.path.join(results_dir, fname))
                # Add parameters to the dataframe for easier plotting with seaborn
                df['features'] = features
                df['depth'] = depth
                df['order'] = order
                df['order_str'] = f"{order_str} Order" # For plot legends
                # df will now also contain 'grad_alignment' if the new main.py saved it
                trajectories[key] = df
            except Exception as e:
                print(f"Error loading {fname}: {e}")
    
    print(f"Loaded {len(trajectories)} trajectory files from {results_subdir}.")
    return trajectories

def print_summary_stats(trajectories):
    """Prints summary statistics for each loaded trajectory."""
    if not trajectories:
        print("No trajectories loaded to summarize.")
        return

    print("\n--- Summary Statistics ---")
    for key, df in sorted(trajectories.items()): # Sort by key for consistent order
        max_acc = df['val_accuracy'].max()
        final_acc = df['val_accuracy'].iloc[-1]
        max_acc_step = df['log_step'][df['val_accuracy'].idxmax()]
        num_steps = len(df)
        print(f"Params (Feats, Depth, Order): {key} -> Steps: {num_steps}, Max Acc: {max_acc:.4f} at step {max_acc_step}, Final Acc: {final_acc:.4f}")

def plot_learning_curves(trajectories_dict, fixed_features, fixed_depth):
    """Plots 1st vs 2nd order for a given feature and depth."""
    plt.figure(figsize=(10, 6))
    
    key_1st = (fixed_features, fixed_depth, 1)
    key_2nd = (fixed_features, fixed_depth, 2)
    
    found_1st = False
    if key_1st in trajectories_dict:
        df_1st = trajectories_dict[key_1st]
        plt.plot(df_1st['log_step'] * 1000, df_1st['val_accuracy'], label=f'1st Order (F={fixed_features}, D={fixed_depth})', marker='.')
        found_1st = True
        
    found_2nd = False
    if key_2nd in trajectories_dict:
        df_2nd = trajectories_dict[key_2nd]
        plt.plot(df_2nd['log_step'] * 1000, df_2nd['val_accuracy'], label=f'2nd Order (F={fixed_features}, D={fixed_depth})', marker='.')
        found_2nd = True
        
    if not found_1st and not found_2nd:
        print(f"No data found for features={fixed_features}, depth={fixed_depth}")
        plt.close()
        return

    plt.title(f'Validation Accuracy (Features: {fixed_features}, Depth: {fixed_depth})')
    plt.xlabel('Training Episodes (Meta-Batches Seen)')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.4, 1.0) # Adjust if needed
    # Save the plot
    plot_filename = f"results/run1_plots/Acc_F{fixed_features}_D{fixed_depth}_1st_vs_2nd.png"
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename)
    print(f"Saved plot: {plot_filename}")
    plt.show()


def plot_all_curves_faceted(all_dfs_list, run_name="run1"):
    """Plots all accuracy curves faceted by features, depth, and order using seaborn."""
    if not all_dfs_list:
        print("No dataframes to plot for accuracy."); return
    combined_df = pd.concat(all_dfs_list, ignore_index=True)
    combined_df['episodes'] = combined_df['log_step'] * 1000

    g = sns.FacetGrid(combined_df, row='features', col='depth', hue='order_str', margin_titles=True, sharey=True)
    g.map(sns.lineplot, 'episodes', 'val_accuracy', marker=".")
    g.add_legend()
    g.set_axis_labels("Training Episodes", "Validation Accuracy")
    g.fig.suptitle('Validation Accuracy Trajectories', y=1.02)
    
    plot_out_dir = os.path.join("results", f"{run_name}_plots") # Parameterized
    os.makedirs(plot_out_dir, exist_ok=True)
    plot_filename = os.path.join(plot_out_dir, "All_Accuracy_Trajectories_Faceted.png")
    plt.savefig(plot_filename)
    print(f"Saved accuracy faceted plot to: {plot_filename}")
    plt.show()

def plot_all_grad_alignment_faceted(all_dfs_list, run_name="run1"):
    """Plots all gradient alignment curves faceted by features, depth, and order."""
    if not all_dfs_list:
        print("No dataframes to plot for gradient alignment."); return
    
    combined_df = pd.concat(all_dfs_list, ignore_index=True)
    if 'grad_alignment' not in combined_df.columns:
        print("'grad_alignment' column not found in the trajectory data. Skipping alignment plots.")
        return
        
    combined_df['episodes'] = combined_df['log_step'] * 1000
    
    # Handle potential NaNs in grad_alignment if some steps didn't log it
    # For plotting, it's often fine, seaborn might ignore them or you can drop/fill.
    # combined_df.dropna(subset=['grad_alignment'], inplace=True) # Optional: drop rows where alignment is NaN

    g = sns.FacetGrid(combined_df, row='features', col='depth', hue='order_str', margin_titles=True, sharey=False) # sharey=False might be better for alignment
    g.map(sns.lineplot, 'episodes', 'grad_alignment', marker=".")
    g.add_legend()
    g.set_axis_labels("Training Episodes", "Avg. Gradient Alignment (Task Query vs Meta)")
    g.fig.suptitle('Gradient Alignment Trajectories', y=1.02)
    
    plot_out_dir = os.path.join("results", f"{run_name}_plots") # Parameterized
    os.makedirs(plot_out_dir, exist_ok=True)
    plot_filename = os.path.join(plot_out_dir, "All_GradAlignments_Faceted.png")
    plt.savefig(plot_filename)
    print(f"Saved gradient alignment faceted plot to: {plot_filename}")
    plt.show()

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser(description="Analyze trajectory results from meta-learning runs.")
    cli_parser.add_argument("--run-name", type=str, default="run1", help="Name of the run subdirectory (e.g., run1, run2) under results/.")
    cli_parser.add_argument("--results-basedir", type=str, default="results", help="Base directory where run subdirectories are located.")
    args = cli_parser.parse_args()

    trajectories = load_trajectory_data(results_basedir=args.results_basedir, run_name=args.run_name)
    print_summary_stats(trajectories)

    all_dfs = [df for df in trajectories.values()]

    if all_dfs:
        plot_all_curves_faceted(all_dfs, run_name=args.run_name)
        plot_all_grad_alignment_faceted(all_dfs, run_name=args.run_name)
        
        # The individual plot_learning_curves might also be parameterized or a new one for alignment made
        # unique_feature_depth_pairs = sorted(list(set((key[0], key[1]) for key in trajectories.keys())))
        # print("\nGenerating individual comparison plots (1st vs 2nd order for accuracy)...")
        # for feat, dep in unique_feature_depth_pairs:
            # plot_learning_curves(trajectories, fixed_features=feat, fixed_depth=dep) # This would need run_name for output path too
    else:
        print("No trajectory data loaded, skipping plotting.")
