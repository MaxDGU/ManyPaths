#!/usr/bin/env python3
"""
Visualize Loss Landscape for Simple Boolean Concepts

This script creates loss landscape visualizations for MLPs learning the simplest 
boolean concepts to understand the topological structure of concept space.

Focus: F8D3 (8 features, depth 3) - the lowest complexity configuration
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pcfg import sample_concept_from_pcfg, evaluate_pcfg_concept
from concept_visualization_for_paper import expression_to_string
import os
import random
from pathlib import Path
import copy

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SimpleMLP(nn.Module):
    """Simple MLP for boolean concept learning."""
    
    def __init__(self, n_input=8, n_hidden=32, n_layers=3):
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
        layers.append(nn.Linear(n_hidden, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def generate_simple_concept_data(seed=42):
    """Generate data for the simplest concept (F8D3)."""
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate a simple concept with F8D3
    print("🎯 Generating simple concept (F8D3)...")
    
    # Try to get a concept with 2-4 literals for good visualization
    best_concept = None
    for attempt in range(100):
        expr, literals, depth = sample_concept_from_pcfg(8, max_depth=3)
        if 2 <= literals <= 4:
            best_concept = (expr, literals, depth)
            break
    
    if best_concept is None:
        expr, literals, depth = sample_concept_from_pcfg(8, max_depth=3)
        best_concept = (expr, literals, depth)
    
    expr, literals, depth = best_concept
    expr_str = expression_to_string(expr)
    
    print(f"  📝 Concept: {expr_str}")
    print(f"  📊 Complexity: {literals} literals, depth {depth}")
    
    # Generate complete dataset (all 2^8 = 256 possible inputs)
    all_inputs = []
    all_labels = []
    
    for i in range(2**8):
        input_vec = np.array([int(x) for x in f"{i:08b}"])
        label = evaluate_pcfg_concept(expr, input_vec)
        all_inputs.append(input_vec)
        all_labels.append(float(label))
    
    # Convert to tensors
    X = torch.tensor(np.array(all_inputs), dtype=torch.float32)
    y = torch.tensor(np.array(all_labels), dtype=torch.float32).unsqueeze(1)
    
    # Scale inputs to [-1, 1] as in main training
    X = X * 2.0 - 1.0
    
    positive_count = sum(all_labels)
    total_count = len(all_labels)
    
    print(f"  📈 Dataset: {total_count} total samples")
    print(f"  ✅ Positive: {positive_count} ({100*positive_count/total_count:.1f}%)")
    print(f"  ❌ Negative: {total_count-positive_count} ({100*(total_count-positive_count)/total_count:.1f}%)")
    
    return X, y, expr_str, literals, depth

def train_simple_mlp(X, y, n_hidden=32, n_layers=3, epochs=1000, lr=0.01):
    """Train MLP to convergence on the concept."""
    
    print(f"🚂 Training MLP (hidden={n_hidden}, layers={n_layers})...")
    
    model = SimpleMLP(n_input=8, n_hidden=n_hidden, n_layers=n_layers)
    criterion = nn.BCEWithLogitsLoss()
    
    # Use SGD instead of Adam to avoid SymPy dependency issues
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = torch.sigmoid(outputs) > 0.5
            accuracy = (predictions == (y > 0.5)).float().mean()
        
        losses.append(loss.item())
        accuracies.append(accuracy.item())
        
        if epoch % 200 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {accuracy.item():.4f}")
    
    print(f"  🎯 Final: Loss = {losses[-1]:.4f}, Accuracy = {accuracies[-1]:.4f}")
    
    return model, losses, accuracies

def compute_loss_at_point(model, X, y, direction1, direction2, alpha, beta):
    """Compute loss at a point in the parameter space."""
    
    # Save original parameters
    original_params = []
    for param in model.parameters():
        original_params.append(param.data.clone())
    
    # Move to new point: theta = theta_0 + alpha * d1 + beta * d2
    for param, d1, d2 in zip(model.parameters(), direction1, direction2):
        param.data = param.data + alpha * d1 + beta * d2
    
    # Compute loss
    with torch.no_grad():
        outputs = model(X)
        loss = nn.BCEWithLogitsLoss()(outputs, y)
    
    # Restore original parameters
    for param, orig in zip(model.parameters(), original_params):
        param.data = orig
    
    return loss.item()

def generate_random_direction(model):
    """Generate a random direction in parameter space."""
    direction = []
    for param in model.parameters():
        direction.append(torch.randn_like(param))
    return direction

def normalize_direction(direction):
    """Normalize direction to unit length."""
    total_norm = 0
    for d in direction:
        total_norm += (d ** 2).sum()
    total_norm = torch.sqrt(total_norm)
    
    normalized = []
    for d in direction:
        normalized.append(d / total_norm)
    return normalized

def create_loss_landscape_2d(model, X, y, steps=50, distance=1.0):
    """Create 2D loss landscape around the trained model."""
    
    print("🗺️  Creating 2D loss landscape...")
    
    # Generate two random orthogonal directions
    dir1 = normalize_direction(generate_random_direction(model))
    dir2 = normalize_direction(generate_random_direction(model))
    
    # Make directions orthogonal using Gram-Schmidt
    dot_product = sum((d1 * d2).sum() for d1, d2 in zip(dir1, dir2))
    for i in range(len(dir2)):
        dir2[i] = dir2[i] - dot_product * dir1[i]
    dir2 = normalize_direction(dir2)
    
    # Create grid
    alphas = np.linspace(-distance, distance, steps)
    betas = np.linspace(-distance, distance, steps)
    
    losses = np.zeros((steps, steps))
    
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            losses[i, j] = compute_loss_at_point(model, X, y, dir1, dir2, alpha, beta)
            
        if i % 10 == 0:
            print(f"  Progress: {i+1}/{steps} rows completed")
    
    return alphas, betas, losses

def create_loss_landscape_1d(model, X, y, steps=100, distance=2.0):
    """Create 1D loss landscape along a random direction."""
    
    print("📏 Creating 1D loss landscape...")
    
    # Generate random direction
    direction = normalize_direction(generate_random_direction(model))
    
    # Create points along the direction
    alphas = np.linspace(-distance, distance, steps)
    losses = []
    
    for alpha in alphas:
        loss = compute_loss_at_point(model, X, y, direction, 
                                   [torch.zeros_like(d) for d in direction], alpha, 0)
        losses.append(loss)
    
    return alphas, losses

def visualize_concept_landscape(X, y, expr_str, literals, depth):
    """Create comprehensive visualization of the concept landscape."""
    
    print("🎨 Creating comprehensive landscape visualization...")
    
    # Train multiple MLPs with different architectures
    architectures = [
        {"n_hidden": 16, "n_layers": 2, "name": "Small (16h, 2l)"},
        {"n_hidden": 32, "n_layers": 3, "name": "Medium (32h, 3l)"},
        {"n_hidden": 64, "n_layers": 4, "name": "Large (64h, 4l)"}
    ]
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # Main title
    fig.suptitle(f'Loss Landscape Analysis: Boolean Concept Learning\n' + 
                f'Concept: {expr_str} ({literals} literals, depth {depth})', 
                fontsize=16, fontweight='bold')
    
    for i, arch in enumerate(architectures):
        print(f"\n🏗️  Training {arch['name']} architecture...")
        
        # Train model
        model, losses, accuracies = train_simple_mlp(
            X, y, 
            n_hidden=arch['n_hidden'], 
            n_layers=arch['n_layers'],
            epochs=800
        )
        
        # Row 1: Training curves
        ax1 = plt.subplot(3, 3, i + 1)
        epochs = range(len(losses))
        ax1.plot(epochs, losses, 'b-', label='Loss', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_title(f'{arch["name"]}\nTraining Dynamics')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(epochs, accuracies, 'r-', label='Accuracy', alpha=0.7)
        ax1_twin.set_ylabel('Accuracy', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        ax1_twin.set_ylim([0, 1])
        
        # Row 2: 1D Loss landscape
        ax2 = plt.subplot(3, 3, i + 4)
        alphas, losses_1d = create_loss_landscape_1d(model, X, y, steps=50, distance=1.5)
        ax2.plot(alphas, losses_1d, 'g-', linewidth=2)
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Trained Model')
        ax2.set_xlabel('Distance from Trained Model')
        ax2.set_ylabel('Loss')
        ax2.set_title('1D Loss Landscape\n(Random Direction)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Row 3: 2D Loss landscape
        ax3 = plt.subplot(3, 3, i + 7)
        alphas_2d, betas_2d, losses_2d = create_loss_landscape_2d(model, X, y, steps=30, distance=1.0)
        
        # Create contour plot
        X_grid, Y_grid = np.meshgrid(alphas_2d, betas_2d)
        contour = ax3.contourf(X_grid, Y_grid, losses_2d.T, levels=20, cmap='viridis', alpha=0.8)
        ax3.contour(X_grid, Y_grid, losses_2d.T, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        
        # Mark the trained model location
        ax3.plot(0, 0, 'r*', markersize=15, label='Trained Model')
        
        ax3.set_xlabel('Direction 1')
        ax3.set_ylabel('Direction 2')
        ax3.set_title('2D Loss Landscape\n(Around Trained Model)')
        ax3.legend()
        
        # Add colorbar
        plt.colorbar(contour, ax=ax3, shrink=0.8, label='Loss')
    
    plt.tight_layout()
    return fig

def analyze_concept_complexity_effect():
    """Analyze how concept complexity affects loss landscape topology."""
    
    print("🔬 Analyzing concept complexity effects on loss landscape...")
    
    # Generate concepts of different complexities
    concepts = []
    
    # Simple concept (target 2-3 literals)
    for attempt in range(50):
        expr, literals, depth = sample_concept_from_pcfg(8, max_depth=3)
        if 2 <= literals <= 3:
            concepts.append((expr, literals, depth, "Simple"))
            break
    
    # Medium concept (target 4-5 literals)
    for attempt in range(50):
        expr, literals, depth = sample_concept_from_pcfg(8, max_depth=5)
        if 4 <= literals <= 5:
            concepts.append((expr, literals, depth, "Medium"))
            break
    
    # Complex concept (target 6+ literals)
    for attempt in range(50):
        expr, literals, depth = sample_concept_from_pcfg(8, max_depth=7)
        if literals >= 6:
            concepts.append((expr, literals, depth, "Complex"))
            break
    
    print(f"  📊 Generated {len(concepts)} concepts for comparison")
    
    fig, axes = plt.subplots(2, len(concepts), figsize=(6*len(concepts), 10))
    if len(concepts) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (expr, literals, depth, complexity) in enumerate(concepts):
        expr_str = expression_to_string(expr)
        print(f"\n  🎯 Analyzing {complexity} concept: {expr_str}")
        
        # Generate data
        X, y, _, _, _ = generate_simple_concept_data(seed=42+i)
        
        # Use same concept but regenerate labels
        all_inputs = []
        all_labels = []
        
        for j in range(2**8):
            input_vec = np.array([int(x) for x in f"{j:08b}"])
            label = evaluate_pcfg_concept(expr, input_vec)
            all_inputs.append(input_vec)
            all_labels.append(float(label))
        
        X = torch.tensor(np.array(all_inputs), dtype=torch.float32) * 2.0 - 1.0
        y = torch.tensor(np.array(all_labels), dtype=torch.float32).unsqueeze(1)
        
        # Train model
        model, losses, accuracies = train_simple_mlp(X, y, epochs=600)
        
        # 1D landscape
        alphas, losses_1d = create_loss_landscape_1d(model, X, y, steps=50)
        axes[0, i].plot(alphas, losses_1d, linewidth=2)
        axes[0, i].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[0, i].set_title(f'{complexity} Concept\n{expr_str[:50]}...\n({literals} literals)')
        axes[0, i].set_xlabel('Distance from Solution')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].grid(True, alpha=0.3)
        
        # 2D landscape
        alphas_2d, betas_2d, losses_2d = create_loss_landscape_2d(model, X, y, steps=25)
        X_grid, Y_grid = np.meshgrid(alphas_2d, betas_2d)
        contour = axes[1, i].contourf(X_grid, Y_grid, losses_2d.T, levels=15, cmap='viridis')
        axes[1, i].plot(0, 0, 'r*', markersize=12)
        axes[1, i].set_title(f'2D Landscape\n{complexity}')
        axes[1, i].set_xlabel('Direction 1')
        axes[1, i].set_ylabel('Direction 2')
        
        plt.colorbar(contour, ax=axes[1, i], shrink=0.8)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to run complete loss landscape analysis."""
    
    print("🌄 Boolean Concept Loss Landscape Analysis")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("figures/loss_landscapes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate simple concept data
    X, y, expr_str, literals, depth = generate_simple_concept_data(seed=42)
    
    # Create comprehensive landscape visualization
    fig1 = visualize_concept_landscape(X, y, expr_str, literals, depth)
    fig1.savefig(output_dir / "simple_concept_landscape_comprehensive.png", 
                dpi=300, bbox_inches='tight')
    fig1.savefig(output_dir / "simple_concept_landscape_comprehensive.pdf", 
                bbox_inches='tight')
    print(f"\n💾 Saved comprehensive landscape to {output_dir}")
    
    # Analyze complexity effects
    fig2 = analyze_concept_complexity_effect()
    fig2.savefig(output_dir / "concept_complexity_landscape_comparison.png", 
                dpi=300, bbox_inches='tight')
    fig2.savefig(output_dir / "concept_complexity_landscape_comparison.pdf", 
                bbox_inches='tight')
    print(f"💾 Saved complexity comparison to {output_dir}")
    
    print("\n🎉 Loss landscape analysis complete!")
    print(f"📁 Results saved in: {output_dir}")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main() 