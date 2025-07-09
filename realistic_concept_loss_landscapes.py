#!/usr/bin/env python3
"""
Realistic Concept Loss Landscapes

Generate realistic loss landscapes using actual PCFG-generated Boolean concepts.
This creates loss surface visualizations that reflect the true topology of 
concept learning in your domain, rather than artificial landscapes.

Features:
- Uses your actual PCFG concept generation pipeline
- Generates complete datasets (all 2^N possible inputs)
- Trains MLPs to convergence on specific concepts
- Creates 1D, 2D, and 3D loss landscape visualizations
- Analyzes how concept complexity affects landscape structure
- Compares landscapes across different concept types
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
import random
import copy
from typing import List, Tuple, Dict, Any
import json
from datetime import datetime

# Import your concept generation system
from pcfg import sample_concept_from_pcfg, evaluate_pcfg_concept
from concept_visualization_for_paper import expression_to_string

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ConceptMLP(nn.Module):
    """MLP architecture consistent with your main training pipeline."""
    
    def __init__(self, n_input=8, n_output=1, n_hidden=32, n_layers=3):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(n_input, n_hidden))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(n_hidden, n_output))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class RealisticConceptLandscapeAnalyzer:
    """
    Generate realistic loss landscapes from actual concept learning tasks.
    """
    
    def __init__(self, output_dir="figures/concept_landscapes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.concepts_analyzed = []
        self.landscape_data = {}
        
    def generate_concept_data(self, num_features=8, max_depth=3, seed=42):
        """Generate a complete dataset for a PCFG concept."""
        
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
        print(f"🎯 Generating concept (F{num_features}D{max_depth}, seed={seed})...")
        
        # Try to get a concept with reasonable complexity for visualization
        best_concept = None
        for attempt in range(100):
            expr, literals, depth = sample_concept_from_pcfg(num_features, max_depth=max_depth)
            if 2 <= literals <= min(8, num_features):  # Reasonable complexity
                best_concept = (expr, literals, depth)
                break
        
        if best_concept is None:
            expr, literals, depth = sample_concept_from_pcfg(num_features, max_depth=max_depth)
            best_concept = (expr, literals, depth)
        
        expr, literals, depth = best_concept
        expr_str = expression_to_string(expr)
        
        print(f"  📝 Concept: {expr_str}")
        print(f"  📊 Complexity: {literals} literals, depth {depth}")
        
        # Generate complete dataset (all 2^num_features possible inputs)
        all_inputs = []
        all_labels = []
        
        for i in range(2**num_features):
            input_vec = np.array([int(x) for x in f"{i:0{num_features}b}"])
            label = evaluate_pcfg_concept(expr, input_vec)
            all_inputs.append(input_vec)
            all_labels.append(float(label))
        
        # Convert to tensors and scale to [-1, 1] as in your main training
        X = torch.tensor(np.array(all_inputs), dtype=torch.float32) * 2.0 - 1.0
        y = torch.tensor(np.array(all_labels), dtype=torch.float32).unsqueeze(1)
        
        positive_count = sum(all_labels)
        total_count = len(all_labels)
        
        print(f"  📈 Dataset: {total_count} samples")
        print(f"  ✅ Positive: {positive_count} ({100*positive_count/total_count:.1f}%)")
        print(f"  ❌ Negative: {total_count-positive_count} ({100*(total_count-positive_count)/total_count:.1f}%)")
        
        concept_info = {
            'expr': expr,
            'expr_str': expr_str,
            'literals': literals,
            'depth': depth,
            'num_features': num_features,
            'max_depth': max_depth,
            'seed': seed,
            'positive_ratio': positive_count / total_count
        }
        
        return X, y, concept_info
    
    def train_concept_model(self, X, y, concept_info, n_hidden=32, n_layers=3, 
                           epochs=2000, lr=0.05, verbose=True):
        """Train MLP to convergence on the concept."""
        
        if verbose:
            print(f"🚂 Training MLP on concept (hidden={n_hidden}, layers={n_layers})...")
        
        model = ConceptMLP(
            n_input=concept_info['num_features'], 
            n_output=1, 
            n_hidden=n_hidden, 
            n_layers=n_layers
        )
        criterion = nn.BCEWithLogitsLoss()
        
        losses = []
        accuracies = []
        
        # Manual training loop to avoid SymPy dependency issues
        for epoch in range(epochs):
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            
            # Manual parameter update
            with torch.no_grad():
                for param in model.parameters():
                    param -= lr * param.grad
                    param.grad.zero_()
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = torch.sigmoid(outputs) > 0.5
                accuracy = (predictions == (y > 0.5)).float().mean()
            
            losses.append(loss.item())
            accuracies.append(accuracy.item())
            
            if verbose and epoch % 400 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {accuracy.item():.4f}")
        
        final_loss = losses[-1]
        final_acc = accuracies[-1]
        
        if verbose:
            print(f"  🎯 Final: Loss = {final_loss:.4f}, Accuracy = {final_acc:.4f}")
        
        training_info = {
            'final_loss': final_loss,
            'final_accuracy': final_acc,
            'n_hidden': n_hidden,
            'n_layers': n_layers,
            'epochs': epochs,
            'lr': lr
        }
        
        return model, losses, accuracies, training_info
    
    def generate_random_direction(self, model):
        """Generate a random direction in parameter space."""
        direction = []
        for param in model.parameters():
            direction.append(torch.randn_like(param))
        return direction
    
    def normalize_direction(self, direction):
        """Normalize direction to unit length."""
        total_norm = 0
        for d in direction:
            total_norm += (d ** 2).sum()
        total_norm = torch.sqrt(total_norm)
        
        normalized = []
        for d in direction:
            normalized.append(d / total_norm)
        return normalized
    
    def make_orthogonal(self, dir1, dir2):
        """Make dir2 orthogonal to dir1 using Gram-Schmidt."""
        dot_product = sum((d1 * d2).sum() for d1, d2 in zip(dir1, dir2))
        orthogonal = []
        for i in range(len(dir2)):
            orthogonal.append(dir2[i] - dot_product * dir1[i])
        return self.normalize_direction(orthogonal)
    
    def compute_loss_at_point(self, model, X, y, directions, coefficients):
        """Compute loss at a point: theta_0 + sum(coeff_i * dir_i)."""
        
        # Save original parameters
        original_params = []
        for param in model.parameters():
            original_params.append(param.data.clone())
        
        # Move to new point
        for param_idx, param in enumerate(model.parameters()):
            displacement = torch.zeros_like(param)
            for dir_idx, coeff in enumerate(coefficients):
                displacement += coeff * directions[dir_idx][param_idx]
            param.data = param.data + displacement
        
        # Compute loss
        with torch.no_grad():
            outputs = model(X)
            loss = nn.BCEWithLogitsLoss()(outputs, y)
        
        # Restore original parameters
        for param, orig in zip(model.parameters(), original_params):
            param.data = orig
        
        return loss.item()
    
    def create_1d_landscape(self, model, X, y, steps=100, distance=2.0, direction=None):
        """Create 1D loss landscape along a direction."""
        
        if direction is None:
            direction = self.normalize_direction(self.generate_random_direction(model))
        
        alphas = np.linspace(-distance, distance, steps)
        losses = []
        
        for alpha in alphas:
            loss = self.compute_loss_at_point(model, X, y, [direction], [alpha])
            losses.append(loss)
        
        return alphas, losses
    
    def create_2d_landscape(self, model, X, y, steps=50, distance=1.5):
        """Create 2D loss landscape using two orthogonal random directions."""
        
        # Generate two orthogonal directions
        dir1 = self.normalize_direction(self.generate_random_direction(model))
        dir2 = self.normalize_direction(self.generate_random_direction(model))
        dir2 = self.make_orthogonal(dir1, dir2)
        
        # Create grid
        alphas = np.linspace(-distance, distance, steps)
        betas = np.linspace(-distance, distance, steps)
        
        losses = np.zeros((len(betas), len(alphas)))
        
        print(f"  Computing {steps}x{steps} loss surface...")
        for i, beta in enumerate(betas):
            for j, alpha in enumerate(alphas):
                loss = self.compute_loss_at_point(model, X, y, [dir1, dir2], [alpha, beta])
                losses[i, j] = loss
            if (i + 1) % 10 == 0:
                print(f"    Progress: {i+1}/{len(betas)} rows completed")
        
        return alphas, betas, losses
    
    def analyze_landscape_properties(self, alphas, losses):
        """Analyze mathematical properties of 1D landscape."""
        
        # Find local minima
        local_minima = []
        for i in range(1, len(losses) - 1):
            if losses[i] < losses[i-1] and losses[i] < losses[i+1]:
                local_minima.append((alphas[i], losses[i]))
        
        # Estimate curvature (second derivative)
        curvatures = []
        for i in range(1, len(losses) - 1):
            curvature = losses[i+1] - 2*losses[i] + losses[i-1]
            curvatures.append(curvature)
        
        # Global minimum
        global_min_idx = np.argmin(losses)
        global_minimum = (alphas[global_min_idx], losses[global_min_idx])
        
        properties = {
            'local_minima_count': len(local_minima),
            'local_minima': local_minima,
            'global_minimum': global_minimum,
            'mean_curvature': np.mean(curvatures),
            'max_curvature': np.max(curvatures),
            'loss_range': np.max(losses) - np.min(losses),
            'loss_variance': np.var(losses)
        }
        
        return properties
    
    def visualize_concept_landscape(self, model, X, y, concept_info, training_info):
        """Create comprehensive landscape visualization for a concept."""
        
        print(f"🗺️  Creating landscape visualizations...")
        
        # Create output directory for this concept
        concept_dir = self.output_dir / f"concept_F{concept_info['num_features']}D{concept_info['max_depth']}_L{concept_info['literals']}_seed{concept_info['seed']}"
        concept_dir.mkdir(parents=True, exist_ok=True)
        
        # 1D landscapes (multiple random directions)
        print("  📈 Creating 1D landscapes...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"1D Loss Landscapes\nConcept: {concept_info['expr_str'][:50]}...\n"
                    f"Complexity: {concept_info['literals']} literals, depth {concept_info['depth']}", 
                    fontsize=12)
        
        landscape_properties = []
        for i in range(4):
            ax = axes[i//2, i%2]
            alphas, losses = self.create_1d_landscape(model, X, y, steps=200, distance=3.0)
            properties = self.analyze_landscape_properties(alphas, losses)
            landscape_properties.append(properties)
            
            ax.plot(alphas, losses, 'b-', linewidth=2, alpha=0.8)
            ax.scatter([0], [properties['global_minimum'][1]], color='red', s=100, zorder=5, label='Trained model')
            
            # Mark local minima
            if properties['local_minima']:
                min_alphas, min_losses = zip(*properties['local_minima'])
                ax.scatter(min_alphas, min_losses, color='orange', s=60, zorder=4, label='Local minima')
            
            ax.set_xlabel('Parameter displacement (α)')
            ax.set_ylabel('Loss')
            ax.set_title(f"Direction {i+1} ({properties['local_minima_count']} local minima)")
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(concept_dir / "1d_landscapes.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2D landscape
        print("  🗺️  Creating 2D landscape...")
        alphas, betas, losses_2d = self.create_2d_landscape(model, X, y, steps=40, distance=2.0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Contour plot
        contour = ax1.contour(alphas, betas, losses_2d, levels=20, alpha=0.6)
        contourf = ax1.contourf(alphas, betas, losses_2d, levels=50, cmap='viridis', alpha=0.8)
        ax1.scatter([0], [0], color='red', s=200, marker='*', zorder=5, label='Trained model')
        ax1.set_xlabel('Direction 1 (α)')
        ax1.set_ylabel('Direction 2 (β)')
        ax1.set_title('2D Loss Landscape (Contour)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(contourf, ax=ax1, label='Loss')
        
        # 3D surface
        Alpha, Beta = np.meshgrid(alphas, betas)
        ax2 = fig.add_subplot(122, projection='3d')
        surface = ax2.plot_surface(Alpha, Beta, losses_2d, cmap='viridis', alpha=0.8)
        ax2.scatter([0], [0], [losses_2d[len(betas)//2, len(alphas)//2]], 
                   color='red', s=200, zorder=5)
        ax2.set_xlabel('Direction 1 (α)')
        ax2.set_ylabel('Direction 2 (β)')
        ax2.set_zlabel('Loss')
        ax2.set_title('2D Loss Landscape (3D Surface)')
        
        plt.tight_layout()
        plt.savefig(concept_dir / "2d_landscape.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Summary statistics
        avg_properties = {
            'avg_local_minima': np.mean([p['local_minima_count'] for p in landscape_properties]),
            'avg_curvature': np.mean([p['mean_curvature'] for p in landscape_properties]),
            'avg_loss_range': np.mean([p['loss_range'] for p in landscape_properties]),
            'min_loss_2d': np.min(losses_2d),
            'max_loss_2d': np.max(losses_2d),
            'loss_variance_2d': np.var(losses_2d)
        }
        
        # Save analysis data
        analysis_data = {
            'concept_info': concept_info,
            'training_info': training_info,
            'landscape_properties_1d': landscape_properties,
            'avg_properties': avg_properties,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(concept_dir / "analysis_data.json", 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        print(f"  💾 Saved to: {concept_dir}")
        
        return analysis_data
    
    def compare_concept_complexities(self, feature_depth_configs=None):
        """Compare landscapes across different concept complexities."""
        
        if feature_depth_configs is None:
            feature_depth_configs = [
                (8, 3),   # Simple
                (8, 5),   # Medium  
                (16, 3),  # More features
                (16, 5),  # Complex
            ]
        
        print("🔬 Comparing landscapes across concept complexities...")
        
        all_results = []
        
        for num_features, max_depth in feature_depth_configs:
            print(f"\n📊 Analyzing F{num_features}D{max_depth}...")
            
            # Generate concept and data
            X, y, concept_info = self.generate_concept_data(
                num_features=num_features, 
                max_depth=max_depth, 
                seed=42
            )
            
            # Train model
            model, losses, accuracies, training_info = self.train_concept_model(
                X, y, concept_info, verbose=False
            )
            
            # Analyze landscape
            analysis_data = self.visualize_concept_landscape(
                model, X, y, concept_info, training_info
            )
            
            all_results.append(analysis_data)
        
        # Create comparison plot
        self._create_complexity_comparison_plot(all_results)
        
        return all_results
    
    def _create_complexity_comparison_plot(self, all_results):
        """Create comparison plots across different complexities."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Landscape Properties vs Concept Complexity", fontsize=14)
        
        # Extract data for plotting
        complexities = []
        avg_minima = []
        avg_curvatures = []
        loss_ranges = []
        final_accuracies = []
        
        for result in all_results:
            concept = result['concept_info']
            complexity_score = concept['literals'] * concept['depth'] / concept['num_features']
            complexities.append(complexity_score)
            
            avg_minima.append(result['avg_properties']['avg_local_minima'])
            avg_curvatures.append(result['avg_properties']['avg_curvature'])
            loss_ranges.append(result['avg_properties']['avg_loss_range'])
            final_accuracies.append(result['training_info']['final_accuracy'])
        
        # Plot 1: Local minima vs complexity
        axes[0,0].scatter(complexities, avg_minima, s=100, alpha=0.7)
        axes[0,0].set_xlabel('Complexity Score (literals × depth / features)')
        axes[0,0].set_ylabel('Average Local Minima Count')
        axes[0,0].set_title('Landscape Complexity vs Concept Complexity')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Curvature vs complexity
        axes[0,1].scatter(complexities, avg_curvatures, s=100, alpha=0.7, color='orange')
        axes[0,1].set_xlabel('Complexity Score')
        axes[0,1].set_ylabel('Average Curvature')
        axes[0,1].set_title('Landscape Curvature vs Concept Complexity')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Loss range vs complexity
        axes[1,0].scatter(complexities, loss_ranges, s=100, alpha=0.7, color='green')
        axes[1,0].set_xlabel('Complexity Score')
        axes[1,0].set_ylabel('Loss Range')
        axes[1,0].set_title('Loss Landscape Range vs Concept Complexity')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Final accuracy vs complexity
        axes[1,1].scatter(complexities, final_accuracies, s=100, alpha=0.7, color='red')
        axes[1,1].set_xlabel('Complexity Score')
        axes[1,1].set_ylabel('Final Training Accuracy')
        axes[1,1].set_title('Learning Success vs Concept Complexity')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "complexity_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Comparison saved to: {self.output_dir}/complexity_comparison.png")

def main():
    """Main analysis pipeline."""
    
    print("🚀 Realistic Concept Loss Landscape Analysis")
    print("=" * 50)
    
    analyzer = RealisticConceptLandscapeAnalyzer()
    
    # Option 1: Analyze a single concept in detail
    print("\n🎯 Single Concept Analysis")
    X, y, concept_info = analyzer.generate_concept_data(num_features=8, max_depth=3, seed=42)
    model, losses, accuracies, training_info = analyzer.train_concept_model(X, y, concept_info)
    analysis_data = analyzer.visualize_concept_landscape(model, X, y, concept_info, training_info)
    
    # Option 2: Compare across complexities
    print("\n🔬 Multi-Complexity Comparison")
    comparison_results = analyzer.compare_concept_complexities()
    
    print(f"\n💾 All results saved to: {analyzer.output_dir}")
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main() 