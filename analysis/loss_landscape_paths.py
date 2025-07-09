#!/usr/bin/env python3
"""
Loss Landscape Path Analysis for Meta-SGD vs SGD

This script loads checkpoints from landscape logging experiments, computes PCA
on parameter deltas, samples loss grids, and creates comparative visualizations 
showing optimization paths over real loss landscapes.

Usage:
    python analysis/loss_landscape_paths.py --tier easy
    python analysis/loss_landscape_paths.py --tier medium  
    python analysis/loss_landscape_paths.py --tier complex
    python analysis/loss_landscape_paths.py --all-tiers

Created for camera-ready submission pipeline.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from initialization import init_model, init_misc
from models import *
from constants import DEFAULT_INDEX
from utils import extract_model_parameters, create_loss_closure
from pcfg import sample_concept_from_pcfg, evaluate_pcfg_concept

# Set matplotlib style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class LandscapePathAnalyzer:
    """Analyzes and visualizes optimization paths over loss landscapes."""
    
    def __init__(self, results_dir="results", output_dir="figs"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Complexity tier configurations
        self.tier_configs = {
            'easy': {'features': 8, 'depth': 3},
            'medium': {'features': 16, 'depth': 5}, 
            'complex': {'features': 32, 'depth': 7}
        }
        
    def load_checkpoints(self, tier, seed=1):
        """Load Meta-SGD and SGD checkpoints for a given tier."""
        config = self.tier_configs[tier]
        features, depth = config['features'], config['depth']
        
        # Define checkpoint paths
        meta_sgd_dir = self.results_dir / "checkpoints" / "meta_sgd" / f"concept_mlp_{features}_bits_feats{features}_depth{depth}"
        sgd_dir = self.results_dir / "baseline_sgd" / f"run_baseline_seed{seed}" / "checkpoints" / "sgd" / f"feats{features}_depth{depth}"
        
        meta_sgd_checkpoints = []
        sgd_checkpoints = []
        
        # Load Meta-SGD checkpoints
        if meta_sgd_dir.exists():
            for ckpt_file in sorted(meta_sgd_dir.glob("step*.pt")):
                try:
                    checkpoint = torch.load(ckpt_file, map_location='cpu')
                    meta_sgd_checkpoints.append(checkpoint)
                except Exception as e:
                    print(f"Warning: Failed to load Meta-SGD checkpoint {ckpt_file}: {e}")
        
        # Load SGD checkpoints  
        if sgd_dir.exists():
            for ckpt_file in sorted(sgd_dir.glob("step*.pt")):
                try:
                    checkpoint = torch.load(ckpt_file, map_location='cpu')
                    sgd_checkpoints.append(checkpoint)
                except Exception as e:
                    print(f"Warning: Failed to load SGD checkpoint {ckpt_file}: {e}")
        
        print(f"Loaded {len(meta_sgd_checkpoints)} Meta-SGD and {len(sgd_checkpoints)} SGD checkpoints for {tier}")
        return meta_sgd_checkpoints, sgd_checkpoints
    
    def load_trajectory_csvs(self, tier, seed=1):
        """Load trajectory CSV files for Hessian metrics."""
        config = self.tier_configs[tier]
        features, depth = config['features'], config['depth']
        
        # Define trajectory paths
        meta_sgd_csv = self.results_dir / f"concept_mlp_14_bits_feats{features}_depth{depth}_adapt10_1stOrd_seed{seed}_landscape_trajectory.csv"
        sgd_csv = self.results_dir / "baseline_sgd" / f"run_baseline_seed{seed}" / f"concept_mlp_bits_feats{features}_depth{depth}_sgd_baseline_seed{seed}_landscape_trajectory.csv"
        
        meta_sgd_df = pd.DataFrame()
        sgd_df = pd.DataFrame()
        
        if meta_sgd_csv.exists():
            meta_sgd_df = pd.read_csv(meta_sgd_csv)
            
        if sgd_csv.exists():
            sgd_df = pd.read_csv(sgd_csv)
            
        print(f"Loaded trajectory CSVs: Meta-SGD ({len(meta_sgd_df)} steps), SGD ({len(sgd_df)} steps)")
        return meta_sgd_df, sgd_df
    
    def extract_parameter_trajectories(self, checkpoints):
        """Extract parameter vectors from checkpoint sequence."""
        trajectories = []
        for checkpoint in checkpoints:
            if 'theta' in checkpoint:
                trajectories.append(checkpoint['theta'].numpy())
            elif 'model_state_dict' in checkpoint:
                # Reconstruct theta from state dict
                state_dict = checkpoint['model_state_dict']
                params = []
                for key in sorted(state_dict.keys()):
                    params.append(state_dict[key].view(-1).numpy())
                trajectories.append(np.concatenate(params))
        
        return np.array(trajectories) if trajectories else np.array([])
    
    def compute_pca_basis(self, meta_sgd_traj, sgd_traj):
        """Compute PCA basis from combined parameter trajectories."""
        if len(meta_sgd_traj) == 0 or len(sgd_traj) == 0:
            print("Warning: Empty trajectories, cannot compute PCA")
            return None, None, None
            
        # Combine trajectories for PCA
        all_trajectories = np.vstack([meta_sgd_traj, sgd_traj])
        
        # Compute parameter deltas from starting point
        start_point = all_trajectories[0]
        deltas = all_trajectories - start_point
        
        # Remove zero deltas
        non_zero_mask = np.any(deltas != 0, axis=1)
        if not np.any(non_zero_mask):
            print("Warning: No parameter changes detected")
            return None, None, None
            
        deltas = deltas[non_zero_mask]
        
        # Standardize and apply PCA
        scaler = StandardScaler()
        deltas_scaled = scaler.fit_transform(deltas)
        
        pca = PCA(n_components=2)
        pca.fit(deltas_scaled)
        
        # PCA basis vectors in original parameter space
        basis_alpha = scaler.inverse_transform(pca.components_[0].reshape(1, -1))[0]
        basis_beta = scaler.inverse_transform(pca.components_[1].reshape(1, -1))[0]
        
        explained_variance = pca.explained_variance_ratio_
        print(f"PCA explained variance: PC1={explained_variance[0]:.3f}, PC2={explained_variance[1]:.3f}")
        
        return basis_alpha, basis_beta, start_point
    
    def sample_loss_grid(self, tier, basis_alpha, basis_beta, start_point, grid_size=50, grid_range=2.0):
        """Sample loss values on a 2D grid around parameter space."""
        config = self.tier_configs[tier]
        features, depth = config['features'], config['depth']
        
        # Initialize model and sample concept
        device = torch.device('cpu')
        alphabet, bits_for_model, channels, n_output = init_misc("concept", None, features)
        model = init_model("mlp", "bits", index=DEFAULT_INDEX, verbose=False, 
                          channels=channels, bits=bits_for_model, n_output=n_output).to(device)
        
        # Sample a concept for loss evaluation
        concept = sample_concept_from_pcfg(depth)
        
        # Generate a small support set
        X_support = torch.randn(32, features)  # 32 examples
        y_support = torch.tensor([evaluate_pcfg_concept(concept, x.numpy()) for x in X_support], dtype=torch.float32)
        
        criterion = nn.BCEWithLogitsLoss()
        
        # Create grid
        alpha_vals = np.linspace(-grid_range, grid_range, grid_size)
        beta_vals = np.linspace(-grid_range, grid_range, grid_size)
        loss_grid = np.zeros((grid_size, grid_size))
        
        print(f"Sampling {grid_size}x{grid_size} loss grid for {tier} tier...")
        
        for i, alpha in enumerate(alpha_vals):
            for j, beta in enumerate(beta_vals):
                # Compute parameter point
                param_point = start_point + alpha * basis_alpha + beta * basis_beta
                param_tensor = torch.tensor(param_point, dtype=torch.float32)
                
                # Set model parameters
                param_idx = 0
                with torch.no_grad():
                    for param in model.parameters():
                        param_size = param.numel()
                        param.data = param_tensor[param_idx:param_idx + param_size].view(param.shape)
                        param_idx += param_size
                
                # Evaluate loss
                try:
                    model.eval()
                    with torch.no_grad():
                        pred = model(X_support)
                        loss = criterion(pred, y_support)
                        loss_grid[i, j] = loss.item()
                except:
                    loss_grid[i, j] = np.nan
        
        return alpha_vals, beta_vals, loss_grid
    
    def project_trajectory_to_pca(self, trajectory, basis_alpha, basis_beta, start_point):
        """Project parameter trajectory onto PCA coordinates."""
        if len(trajectory) == 0:
            return np.array([]), np.array([])
            
        deltas = trajectory - start_point
        
        # Project onto PCA basis
        alpha_coords = np.dot(deltas, basis_alpha) / np.dot(basis_alpha, basis_alpha)
        beta_coords = np.dot(deltas, basis_beta) / np.dot(basis_beta, basis_beta)
        
        return alpha_coords, beta_coords
    
    def create_landscape_visualization(self, tier, seed=1):
        """Create comprehensive landscape visualization for a tier."""
        print(f"\n🎨 Creating landscape visualization for {tier} tier (seed {seed})")
        
        # Load data
        meta_sgd_checkpoints, sgd_checkpoints = self.load_checkpoints(tier, seed)
        meta_sgd_df, sgd_df = self.load_trajectory_csvs(tier, seed)
        
        if len(meta_sgd_checkpoints) == 0 or len(sgd_checkpoints) == 0:
            print(f"Insufficient checkpoint data for {tier} tier")
            return
        
        # Extract trajectories
        meta_sgd_traj = self.extract_parameter_trajectories(meta_sgd_checkpoints)
        sgd_traj = self.extract_parameter_trajectories(sgd_checkpoints)
        
        # Compute PCA basis
        basis_alpha, basis_beta, start_point = self.compute_pca_basis(meta_sgd_traj, sgd_traj)
        if basis_alpha is None:
            print(f"Cannot compute PCA for {tier} tier")
            return
        
        # Sample loss grid
        alpha_vals, beta_vals, loss_grid = self.sample_loss_grid(tier, basis_alpha, basis_beta, start_point)
        
        # Project trajectories to PCA space
        meta_alpha, meta_beta = self.project_trajectory_to_pca(meta_sgd_traj, basis_alpha, basis_beta, start_point)
        sgd_alpha, sgd_beta = self.project_trajectory_to_pca(sgd_traj, basis_alpha, basis_beta, start_point)
        
        # Create visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Main loss landscape plot
        ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
        
        # Contour plot
        contour = ax_main.contour(alpha_vals, beta_vals, loss_grid.T, levels=20, colors='gray', alpha=0.4)
        ax_main.contourf(alpha_vals, beta_vals, loss_grid.T, levels=20, cmap='viridis', alpha=0.6)
        
        # Plot optimization paths
        if len(meta_alpha) > 0:
            ax_main.plot(meta_alpha, meta_beta, 'o-', color='teal', linewidth=3, 
                        markersize=8, label='Meta-SGD', alpha=0.8)
        if len(sgd_alpha) > 0:
            ax_main.plot(sgd_alpha, sgd_beta, 'o-', color='red', linewidth=3,
                        markersize=8, label='SGD', alpha=0.8)
        
        # Mark start and end points
        ax_main.plot(0, 0, 'ko', markersize=12, label='Start')
        if len(meta_alpha) > 0:
            ax_main.plot(meta_alpha[-1], meta_beta[-1], 's', color='teal', markersize=12, label='Meta-SGD End')
        if len(sgd_alpha) > 0:
            ax_main.plot(sgd_alpha[-1], sgd_beta[-1], 's', color='red', markersize=12, label='SGD End')
        
        ax_main.set_xlabel('PC1 (α)', fontsize=14)
        ax_main.set_ylabel('PC2 (β)', fontsize=14)
        ax_main.set_title(f'Loss Landscape - {tier.title()} Tier\n'
                         f'F{self.tier_configs[tier]["features"]}D{self.tier_configs[tier]["depth"]} (Seed {seed})', 
                         fontsize=16)
        ax_main.legend(fontsize=12)
        ax_main.grid(True, alpha=0.3)
        
        # 3D surface plot
        ax_3d = plt.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=2, projection='3d')
        Alpha, Beta = np.meshgrid(alpha_vals, beta_vals)
        surface = ax_3d.plot_surface(Alpha, Beta, loss_grid.T, cmap='viridis', alpha=0.7, edgecolor='none')
        
        # Plot 3D paths
        if len(meta_alpha) > 0:
            meta_losses = [loss_grid[np.argmin(np.abs(alpha_vals - a)), np.argmin(np.abs(beta_vals - b))] 
                          for a, b in zip(meta_alpha, meta_beta)]
            ax_3d.plot(meta_alpha, meta_beta, meta_losses, 'o-', color='teal', linewidth=4, markersize=6)
        
        if len(sgd_alpha) > 0:
            sgd_losses = [loss_grid[np.argmin(np.abs(alpha_vals - a)), np.argmin(np.abs(beta_vals - b))] 
                         for a, b in zip(sgd_alpha, sgd_beta)]
            ax_3d.plot(sgd_alpha, sgd_beta, sgd_losses, 'o-', color='red', linewidth=4, markersize=6)
        
        ax_3d.set_xlabel('PC1 (α)')
        ax_3d.set_ylabel('PC2 (β)')
        ax_3d.set_zlabel('Loss')
        ax_3d.set_title('3D Loss Surface')
        
        # Hessian sharpness plot
        ax_sharp = plt.subplot2grid((3, 4), (2, 0))
        if not meta_sgd_df.empty and 'lambda_max' in meta_sgd_df.columns:
            ax_sharp.plot(meta_sgd_df['step'], meta_sgd_df['lambda_max'], 'o-', color='teal', label='Meta-SGD')
        if not sgd_df.empty and 'lambda_max' in sgd_df.columns:
            ax_sharp.plot(sgd_df['step'], sgd_df['lambda_max'], 'o-', color='red', label='SGD')
        ax_sharp.set_xlabel('Training Step')
        ax_sharp.set_ylabel('λ_max')
        ax_sharp.set_title('Hessian Sharpness')
        ax_sharp.legend()
        ax_sharp.grid(True, alpha=0.3)
        
        # Geodesic length plot
        ax_geo = plt.subplot2grid((3, 4), (2, 1))
        if not meta_sgd_df.empty and 'geodesic_length_from_start' in meta_sgd_df.columns:
            ax_geo.plot(meta_sgd_df['step'], meta_sgd_df['geodesic_length_from_start'], 'o-', color='teal', label='Meta-SGD')
        if not sgd_df.empty and 'geodesic_length_from_start' in sgd_df.columns:
            ax_geo.plot(sgd_df['step'], sgd_df['geodesic_length_from_start'], 'o-', color='red', label='SGD')
        ax_geo.set_xlabel('Training Step')
        ax_geo.set_ylabel('Path Length')
        ax_geo.set_title('Geodesic Length')
        ax_geo.legend()
        ax_geo.grid(True, alpha=0.3)
        
        # Final accuracy comparison
        ax_acc = plt.subplot2grid((3, 4), (2, 2))
        methods = ['Meta-SGD', 'SGD']
        final_accs = []
        
        if not meta_sgd_df.empty and 'accuracy' in meta_sgd_df.columns:
            final_accs.append(meta_sgd_df['accuracy'].iloc[-1])
        else:
            final_accs.append(0)
            
        if not sgd_df.empty and 'accuracy' in sgd_df.columns:
            final_accs.append(sgd_df['accuracy'].iloc[-1])
        else:
            final_accs.append(0)
        
        bars = ax_acc.bar(methods, final_accs, color=['teal', 'red'], alpha=0.7)
        ax_acc.set_ylabel('Final Accuracy')
        ax_acc.set_title('Final Performance')
        ax_acc.set_ylim(0, 1)
        for bar, acc in zip(bars, final_accs):
            ax_acc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{acc:.3f}', ha='center', va='bottom')
        
        # Path length comparison
        ax_path = plt.subplot2grid((3, 4), (2, 3))
        path_lengths = []
        if len(meta_alpha) > 0:
            meta_path_length = np.sum(np.sqrt(np.diff(meta_alpha)**2 + np.diff(meta_beta)**2))
            path_lengths.append(meta_path_length)
        else:
            path_lengths.append(0)
            
        if len(sgd_alpha) > 0:
            sgd_path_length = np.sum(np.sqrt(np.diff(sgd_alpha)**2 + np.diff(sgd_beta)**2))
            path_lengths.append(sgd_path_length)
        else:
            path_lengths.append(0)
        
        bars = ax_path.bar(methods, path_lengths, color=['teal', 'red'], alpha=0.7)
        ax_path.set_ylabel('Path Length')
        ax_path.set_title('Optimization Path Length')
        for bar, length in zip(bars, path_lengths):
            ax_path.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{length:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / f"landscape_{tier}.svg"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved landscape visualization: {output_path}")
        
        plt.show()
        return fig
    
    def create_combined_visualization(self):
        """Create combined visualization for all tiers."""
        print("\n🎨 Creating combined landscape visualization")
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        for idx, tier in enumerate(['easy', 'medium', 'complex']):
            ax = axes[idx]
            
            # Load and process data for this tier
            meta_sgd_checkpoints, sgd_checkpoints = self.load_checkpoints(tier)
            
            if len(meta_sgd_checkpoints) == 0 or len(sgd_checkpoints) == 0:
                ax.text(0.5, 0.5, f'No data for {tier} tier', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            meta_sgd_traj = self.extract_parameter_trajectories(meta_sgd_checkpoints)
            sgd_traj = self.extract_parameter_trajectories(sgd_checkpoints)
            
            basis_alpha, basis_beta, start_point = self.compute_pca_basis(meta_sgd_traj, sgd_traj)
            if basis_alpha is None:
                continue
                
            alpha_vals, beta_vals, loss_grid = self.sample_loss_grid(tier, basis_alpha, basis_beta, start_point, grid_size=30)
            
            meta_alpha, meta_beta = self.project_trajectory_to_pca(meta_sgd_traj, basis_alpha, basis_beta, start_point)
            sgd_alpha, sgd_beta = self.project_trajectory_to_pca(sgd_traj, basis_alpha, basis_beta, start_point)
            
            # Plot landscape and paths
            contour = ax.contourf(alpha_vals, beta_vals, loss_grid.T, levels=15, cmap='viridis', alpha=0.6)
            
            if len(meta_alpha) > 0:
                ax.plot(meta_alpha, meta_beta, 'o-', color='teal', linewidth=3, 
                       markersize=6, label='Meta-SGD', alpha=0.8)
            if len(sgd_alpha) > 0:
                ax.plot(sgd_alpha, sgd_beta, 'o-', color='red', linewidth=3,
                       markersize=6, label='SGD', alpha=0.8)
            
            ax.plot(0, 0, 'ko', markersize=10, label='Start')
            
            config = self.tier_configs[tier]
            ax.set_title(f'{tier.title()} (F{config["features"]}D{config["depth"]})', fontsize=16)
            ax.set_xlabel('PC1 (α)', fontsize=14)
            if idx == 0:
                ax.set_ylabel('PC2 (β)', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Loss Landscape Navigation: Meta-SGD vs SGD Across Complexity Tiers', fontsize=20)
        plt.tight_layout()
        
        # Save combined figure
        output_path = self.output_dir / "fig_loss_landscapes_all.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined visualization: {output_path}")
        
        plt.show()
        return fig

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Loss Landscape Path Analysis')
    parser.add_argument('--tier', choices=['easy', 'medium', 'complex'], 
                       help='Analyze specific complexity tier')
    parser.add_argument('--all-tiers', action='store_true',
                       help='Create combined analysis for all tiers')
    parser.add_argument('--seed', type=int, default=1,
                       help='Seed for analysis')
    parser.add_argument('--results-dir', default='results',
                       help='Results directory')
    parser.add_argument('--output-dir', default='figs',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    analyzer = LandscapePathAnalyzer(args.results_dir, args.output_dir)
    
    if args.all_tiers:
        analyzer.create_combined_visualization()
    elif args.tier:
        analyzer.create_landscape_visualization(args.tier, args.seed)
    else:
        # Default: analyze all tiers individually
        for tier in ['easy', 'medium', 'complex']:
            try:
                analyzer.create_landscape_visualization(tier, args.seed)
            except Exception as e:
                print(f"Failed to analyze {tier} tier: {e}")
                continue
        
        # Also create combined visualization
        analyzer.create_combined_visualization()
    
    print("\n✅ Loss landscape path analysis complete!")

if __name__ == "__main__":
    main() 