#!/usr/bin/env python3
"""
Boolean Concept Manifold Visualization

Generate an SVG figure showing a 3D cube representing the boolean concept space
with points colored by concept evaluation (teal = True, gray = False).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import torch
from pcfg import sample_concept_from_pcfg, evaluate_pcfg_concept
from concept_visualization_for_paper import expression_to_string

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

def generate_concept_data():
    """Generate a sample Boolean concept and evaluate it on all 8-bit inputs."""
    
    print("🎯 Generating Boolean concept for visualization...")
    
    # Generate a concept with 8 features, moderate complexity
    num_features = 8
    max_depth = 5
    
    # Try to get a concept with 3-6 literals for good visualization
    best_concept = None
    for attempt in range(100):
        expr, literals, depth = sample_concept_from_pcfg(num_features, max_depth=max_depth)
        if 3 <= literals <= 6:
            best_concept = (expr, literals, depth)
            break
    
    if best_concept is None:
        expr, literals, depth = sample_concept_from_pcfg(num_features, max_depth=max_depth)
        best_concept = (expr, literals, depth)
    
    expr, literals, depth = best_concept
    expr_str = expression_to_string(expr)
    
    print(f"  📝 Concept: {expr_str}")
    print(f"  📊 Complexity: {literals} literals, depth {depth}")
    
    # Generate all 2^8 = 256 possible 8-bit inputs
    all_inputs = []
    all_labels = []
    
    for i in range(2**8):
        input_vec = np.array([int(x) for x in f"{i:08b}"])
        label = evaluate_pcfg_concept(expr, input_vec)
        all_inputs.append(input_vec)
        all_labels.append(label)
    
    # Convert to numpy arrays
    inputs = np.array(all_inputs)
    labels = np.array(all_labels)
    
    positive_count = np.sum(labels)
    total_count = len(labels)
    
    print(f"  📈 Dataset: {total_count} total samples")
    print(f"  ✅ Positive: {positive_count} ({100*positive_count/total_count:.1f}%)")
    print(f"  ❌ Negative: {total_count-positive_count} ({100*(total_count-positive_count)/total_count:.1f}%)")
    
    return inputs, labels, expr_str, literals, depth

def create_manifold_visualization():
    """Create the Boolean Concept Manifold visualization."""
    
    # Generate concept data
    inputs, labels, expr_str, literals, depth = generate_concept_data()
    
    # Extract first 3 features for 3D visualization
    f1 = inputs[:, 0]  # First feature
    f2 = inputs[:, 1]  # Second feature  
    f3 = inputs[:, 2]  # Third feature
    
    # Add slight jitter to avoid overlap
    jitter_amount = 0.03
    f1_jitter = f1 + np.random.normal(0, jitter_amount, size=len(f1))
    f2_jitter = f2 + np.random.normal(0, jitter_amount, size=len(f2))
    f3_jitter = f3 + np.random.normal(0, jitter_amount, size=len(f3))
    
    # Create figure with specific style
    plt.style.use('default')  # Clean, minimal style
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set font to Helvetica (or closest available)
    plt.rcParams['font.family'] = ['Helvetica', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['font.size'] = 8
    
    # Create cube wireframe (faint gray gridlines)
    cube_edges = [
        [[0, 1], [0, 0], [0, 0]], [[0, 0], [0, 1], [0, 0]], [[0, 0], [0, 0], [0, 1]],  # Bottom edges
        [[1, 1], [0, 1], [0, 0]], [[1, 1], [0, 0], [0, 1]], [[0, 1], [1, 1], [0, 0]],  # Top edges
        [[0, 1], [1, 1], [1, 1]], [[1, 1], [0, 1], [1, 1]], [[1, 1], [1, 1], [0, 1]],  # Additional edges
        [[0, 0], [0, 0], [0, 1]], [[0, 0], [1, 1], [0, 0]], [[1, 1], [0, 0], [0, 0]],  # Vertical edges
    ]
    
    # Draw cube edges
    for edge in cube_edges:
        ax.plot(edge[0], edge[1], edge[2], 'gray', alpha=0.2, linewidth=0.5)
    
    # Plot points colored by concept evaluation
    positive_mask = labels == True
    negative_mask = labels == False
    
    # Plot positive points (teal)
    ax.scatter(f1_jitter[positive_mask], f2_jitter[positive_mask], f3_jitter[positive_mask], 
               c='#008B8B', s=20, alpha=0.8, label='Positive')
    
    # Plot negative points (light gray)
    ax.scatter(f1_jitter[negative_mask], f2_jitter[negative_mask], f3_jitter[negative_mask], 
               c='#D3D3D3', s=20, alpha=0.6, label='Negative')
    
    # Set labels and title
    ax.set_xlabel('f1', fontsize=8)
    ax.set_ylabel('f2', fontsize=8)
    ax.set_zlabel('f3', fontsize=8)
    ax.set_title('Boolean Concept Manifold', fontsize=12, fontweight='bold', pad=20)
    
    # Set axis limits to show cube clearly
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_zlim(-0.1, 1.1)
    
    # Set ticks to show binary values
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_zticks([0, 1])
    
    # Minimize perspective distortion
    ax.view_init(elev=20, azim=45)
    ax.dist = 10
    
    # Add legend (small, positioned nicely)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#008B8B', 
                   markersize=6, label='Positive'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#D3D3D3', 
                   markersize=6, label='Negative')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=True, 
              fancybox=True, shadow=True)
    
    # Remove pane backgrounds for cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Make pane edges more subtle
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Add concept information as text
    concept_info = f"Concept: {expr_str}\nFeatures: 8, Literals: {literals}, Depth: {depth}"
    fig.text(0.02, 0.98, concept_info, fontsize=8, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    return fig, expr_str

def main():
    """Generate the Boolean Concept Manifold SVG figure."""
    
    print("🎨 Creating Boolean Concept Manifold Visualization...")
    
    # Create the visualization
    fig, expr_str = create_manifold_visualization()
    
    # Save as SVG
    output_path = "boolean_concept_manifold.svg"
    fig.savefig(output_path, format='svg', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Saved Boolean Concept Manifold to {output_path}")
    print(f"   Concept: {expr_str}")
    
    # Also save as PNG for backup
    fig.savefig("boolean_concept_manifold.png", format='png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("🎉 Visualization complete!")
    
    # Display the figure
    plt.show()

if __name__ == "__main__":
    main() 