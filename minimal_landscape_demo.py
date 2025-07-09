#!/usr/bin/env python3
"""
Minimal Loss Landscape Demo for Boolean Concepts

Manual implementation to avoid PyTorch optimizer dependency issues.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pcfg import sample_concept_from_pcfg, evaluate_pcfg_concept
from concept_visualization_for_paper import expression_to_string
import random
from pathlib import Path

class MinimalMLP(nn.Module):
    """Minimal MLP with manual training."""
    
    def __init__(self, n_input=8, n_hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_simple_concept():
    """Generate a simple concept."""
    
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    print("🎯 Generating simple concept...")
    
    # Get a simple concept
    expr, literals, depth = sample_concept_from_pcfg(8, max_depth=3)
    expr_str = expression_to_string(expr)
    
    print(f"  📝 Concept: {expr_str}")
    print(f"  📊 Complexity: {literals} literals, depth {depth}")
    
    # Generate complete dataset
    all_inputs = []
    all_labels = []
    
    for i in range(2**8):
        input_vec = np.array([int(x) for x in f"{i:08b}"])
        label = evaluate_pcfg_concept(expr, input_vec)
        all_inputs.append(input_vec)
        all_labels.append(float(label))
    
    X = torch.tensor(np.array(all_inputs), dtype=torch.float32) * 2.0 - 1.0
    y = torch.tensor(np.array(all_labels), dtype=torch.float32).unsqueeze(1)
    
    positive_count = sum(all_labels)
    print(f"  📈 Dataset: 256 samples, {positive_count} positive ({100*positive_count/256:.1f}%)")
    
    return X, y, expr_str, literals, depth

def manual_train(model, X, y, lr=0.01, epochs=500):
    """Manual training to avoid optimizer issues."""
    
    print("🚂 Training MLP manually...")
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Manual parameter update
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad
                param.grad.zero_()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            with torch.no_grad():
                predictions = torch.sigmoid(outputs) > 0.5
                accuracy = (predictions == (y > 0.5)).float().mean()
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {accuracy.item():.4f}")
    
    print(f"  🎯 Final Loss: {losses[-1]:.4f}")
    return model, losses

def compute_loss_along_line(model, X, y, direction, steps=50, distance=1.0):
    """Compute loss along a line in parameter space."""
    
    # Save original parameters
    original_params = {}
    for name, param in model.named_parameters():
        original_params[name] = param.data.clone()
    
    # Normalize direction
    total_norm = 0
    for d in direction.values():
        total_norm += (d ** 2).sum()
    total_norm = torch.sqrt(total_norm)
    
    normalized_dir = {}
    for name, d in direction.items():
        normalized_dir[name] = d / total_norm
    
    # Compute losses
    alphas = np.linspace(-distance, distance, steps)
    losses = []
    
    criterion = nn.BCEWithLogitsLoss()
    
    for alpha in alphas:
        # Move parameters
        for name, param in model.named_parameters():
            param.data = original_params[name] + alpha * normalized_dir[name]
        
        # Compute loss
        with torch.no_grad():
            outputs = model(X)
            loss = criterion(outputs, y)
            losses.append(loss.item())
    
    # Restore parameters
    for name, param in model.named_parameters():
        param.data = original_params[name]
    
    return alphas, losses

def generate_random_direction(model):
    """Generate random direction."""
    direction = {}
    for name, param in model.named_parameters():
        direction[name] = torch.randn_like(param)
    return direction

def visualize_simple_landscape(model, X, y, expr_str, literals):
    """Create simple landscape visualization."""
    
    print("🗺️  Creating loss landscape...")
    
    # Generate random directions
    directions = [generate_random_direction(model) for _ in range(3)]
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot landscapes in different directions
    for i, direction in enumerate(directions):
        if i < 3:
            row = i // 2
            col = i % 2
            
            alphas, losses = compute_loss_along_line(model, X, y, direction, steps=30, distance=0.5)
            
            axes[row, col].plot(alphas, losses, 'b-', linewidth=2, marker='o', markersize=3)
            axes[row, col].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Trained Model')
            axes[row, col].set_xlabel('Distance from Solution')
            axes[row, col].set_ylabel('Loss')
            axes[row, col].set_title(f'1D Loss Landscape\nDirection {i+1}')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend()
    
    # Info panel
    info_text = f"""Concept Analysis

{expr_str}

Complexity:
• {literals} literals
• 8 boolean features
• 256 total samples

The loss landscape shows the 
topological structure around 
the trained solution in 
different random directions."""
    
    axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes, 
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    axes[1, 1].set_title('Concept Information')
    axes[1, 1].axis('off')
    
    plt.suptitle('Boolean Concept Loss Landscape Topology\n' + 
                f'Simple Concept: {literals} Literals', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def analyze_landscape_topology(alphas_list, losses_list, expr_str):
    """Analyze the topological properties of the loss landscape."""
    
    print("🔬 Analyzing landscape topology...")
    
    topology_insights = []
    
    for i, (alphas, losses) in enumerate(zip(alphas_list, losses_list)):
        # Find local minima
        losses_arr = np.array(losses)
        local_minima = []
        
        for j in range(1, len(losses_arr) - 1):
            if losses_arr[j] < losses_arr[j-1] and losses_arr[j] < losses_arr[j+1]:
                local_minima.append((alphas[j], losses_arr[j]))
        
        # Analyze shape
        center_idx = len(losses) // 2
        center_loss = losses[center_idx]
        
        # Check if it's convex around the solution
        left_slope = (losses[center_idx] - losses[center_idx - 5]) / (alphas[center_idx] - alphas[center_idx - 5])
        right_slope = (losses[center_idx + 5] - losses[center_idx]) / (alphas[center_idx + 5] - alphas[center_idx])
        
        convex_like = left_slope < 0 and right_slope > 0
        
        topology_insights.append({
            'direction': i + 1,
            'local_minima_count': len(local_minima),
            'convex_like': convex_like,
            'center_loss': center_loss,
            'loss_range': max(losses) - min(losses)
        })
    
    # Print insights
    print("  📊 Topology Analysis:")
    for insight in topology_insights:
        print(f"    Direction {insight['direction']}: {insight['local_minima_count']} local minima, " + 
              f"{'convex-like' if insight['convex_like'] else 'non-convex'}, " + 
              f"loss range: {insight['loss_range']:.3f}")
    
    return topology_insights

def main():
    """Main demo."""
    
    print("🌄 Minimal Boolean Concept Loss Landscape Demo")
    print("=" * 50)
    
    # Generate data
    X, y, expr_str, literals, depth = generate_simple_concept()
    
    # Create and train model
    model = MinimalMLP(n_input=8, n_hidden=16)
    model, losses = manual_train(model, X, y, lr=0.05, epochs=300)
    
    # Analyze landscape in multiple directions
    print("\n🗺️  Analyzing landscape in multiple directions...")
    
    directions = [generate_random_direction(model) for _ in range(3)]
    alphas_list = []
    losses_list = []
    
    for i, direction in enumerate(directions):
        alphas, direction_losses = compute_loss_along_line(model, X, y, direction, steps=30, distance=0.5)
        alphas_list.append(alphas)
        losses_list.append(direction_losses)
        print(f"  ✅ Analyzed direction {i+1}")
    
    # Topology analysis
    topology_insights = analyze_landscape_topology(alphas_list, losses_list, expr_str)
    
    # Visualize
    fig = visualize_simple_landscape(model, X, y, expr_str, literals)
    
    # Save results
    output_dir = Path("figures/loss_landscapes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_dir / "minimal_boolean_concept_landscape.png", 
                dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / "minimal_boolean_concept_landscape.pdf", 
                bbox_inches='tight')
    
    print(f"\n💾 Saved to {output_dir}")
    
    # Summary
    print("\n🎯 Summary:")
    print(f"   • Concept: {expr_str}")
    print(f"   • Complexity: {literals} literals, depth {depth}")
    print(f"   • Landscape shows {'mostly convex' if sum(t['convex_like'] for t in topology_insights) >= 2 else 'complex'} topology")
    print("   • Boolean concept space has discrete structure affecting loss geometry")
    
    print("\n🎉 Landscape analysis complete!")
    plt.show()

if __name__ == "__main__":
    main() 