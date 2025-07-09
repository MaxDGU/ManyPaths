#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
import matplotlib.patches as mpatches

def create_gradient_learning_diagram():
    """Create a clear diagram showing Meta-SGD's gradient learning vs SGD's fixed approach"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create a 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, height_ratios=[1, 1.2])
    
    # Top left: SGD fixed gradient approach
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create a simple loss landscape
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (X**2 + Y**2) + 0.3 * np.sin(3*X) * np.cos(3*Y)
    
    # Plot contours
    contours = ax1.contour(X, Y, Z, levels=15, colors='gray', alpha=0.5)
    
    # SGD path - fixed step sizes and standard gradient directions
    sgd_x = np.array([-1.5, -1.2, -0.9, -0.7, -0.5, -0.3, -0.1, 0.0])
    sgd_y = np.array([-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, -0.1, 0.0])
    
    # Plot SGD path
    ax1.plot(sgd_x, sgd_y, 'ro-', linewidth=3, markersize=8, label='SGD Path')
    
    # Add fixed step size arrows
    for i in range(len(sgd_x)-1):
        dx = sgd_x[i+1] - sgd_x[i]
        dy = sgd_y[i+1] - sgd_y[i]
        ax1.arrow(sgd_x[i], sgd_y[i], dx*0.8, dy*0.8, 
                 head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
    
    ax1.set_title('SGD: Fixed Gradient Strategy\n• Fixed step sizes\n• Standard gradient directions', 
                  fontweight='bold', color='red')
    ax1.set_xlabel('Parameter 1')
    ax1.set_ylabel('Parameter 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top right: Meta-SGD adaptive gradient approach
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Same landscape
    contours = ax2.contour(X, Y, Z, levels=15, colors='gray', alpha=0.5)
    
    # Meta-SGD path - adaptive step sizes and learned gradient directions
    meta_x = np.array([-1.3, -0.8, -0.4, -0.1, 0.0])
    meta_y = np.array([-1.1, -0.6, -0.2, -0.05, 0.0])
    
    # Plot Meta-SGD path
    ax2.plot(meta_x, meta_y, 'go-', linewidth=3, markersize=8, label='Meta-SGD Path')
    
    # Add adaptive step size arrows (varying sizes)
    step_sizes = [1.0, 0.7, 1.2, 0.5]  # Varying step sizes
    for i, step_size in enumerate(step_sizes):
        dx = (meta_x[i+1] - meta_x[i]) * step_size
        dy = (meta_y[i+1] - meta_y[i]) * step_size
        arrow_width = 0.08 + 0.04 * step_size  # Varying arrow thickness
        ax2.arrow(meta_x[i], meta_y[i], dx*0.8, dy*0.8, 
                 head_width=arrow_width, head_length=arrow_width, 
                 fc='green', ec='green', alpha=0.7)
    
    ax2.set_title('Meta-SGD: Learned Gradient Strategy\n• Adaptive step sizes\n• Learned gradient directions', 
                  fontweight='bold', color='green')
    ax2.set_xlabel('Parameter 1')
    ax2.set_ylabel('Parameter 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bottom: Detailed mechanism comparison
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Create mechanism diagram
    mechanism_text = """
🔄 GRADIENT LEARNING MECHANISMS: Meta-SGD vs SGD

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          OPTIMIZATION UPDATE EQUATION                                          │
│                                                                                                                │
│  SGD:        θ_{t+1} = θ_t - α ∇L(θ_t)                    [Fixed learning rate α, standard gradient]         │
│                                                                                                                │
│  Meta-SGD:   θ_{t+1} = θ_t - α_t(θ_t) • ∇_t(θ_t)         [Learned α_t AND learned ∇_t]                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

🎯 WHAT META-SGD LEARNS:

1️⃣ ADAPTIVE STEP SIZES (α_t):
   • Small steps in flat regions (avoid overshooting)
   • Large steps in steep regions (faster convergence)
   • Concept-specific step size policies
   
2️⃣ GRADIENT DIRECTIONS (∇_t):
   • Beyond standard loss gradients
   • Learned optimal search directions
   • Concept-aware directional policies

3️⃣ DYNAMIC COORDINATION:
   • Step size + direction work together
   • Adaptive to local landscape topology
   • Context-dependent optimization strategy

🧠 CONCEPT LEARNING IMPLICATIONS:

Simple Concepts (F8D3):
• SGD: Fixed small steps, standard gradients → slow but steady
• Meta-SGD: Learned larger steps when safe → faster convergence

Medium Concepts (F8D5):  
• SGD: Fixed strategy struggles with complexity → gets stuck
• Meta-SGD: Adaptive strategy navigates complexity → finds better solutions

Complex Concepts (F32D3):
• SGD: Fixed approach overwhelmed by ruggedness → poor performance  
• Meta-SGD: Learned navigation strategies → handles complexity better

🎉 KEY INSIGHT: Meta-SGD doesn't just optimize parameters—it learns HOW TO OPTIMIZE!
    """
    
    ax3.text(0.02, 0.98, mechanism_text, transform=ax3.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Meta-SGD Gradient Learning Mechanisms: Beyond Standard Optimization', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig('meta_sgd_gradient_learning_diagram.svg', dpi=300, bbox_inches='tight')
    plt.savefig('meta_sgd_gradient_learning_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Meta-SGD gradient learning diagram created!")
    print("Key insight: Meta-SGD learns both WHEN to take big steps and WHERE to step!")

def create_step_size_adaptation_comparison():
    """Create a focused comparison of step size adaptation"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Simulate optimization trajectories
    steps = np.arange(100)
    
    # SGD: fixed step sizes
    sgd_steps = np.ones(100) * 0.01  # Fixed learning rate
    sgd_loss = 1.0 * np.exp(-0.02 * steps) + 0.1 * np.sin(0.5 * steps) * np.exp(-0.01 * steps)
    
    # Meta-SGD: adaptive step sizes
    meta_steps = 0.01 * (1 + 0.5 * np.sin(0.3 * steps) + 0.3 * np.exp(-0.05 * steps))
    meta_loss = 1.0 * np.exp(-0.04 * steps) + 0.05 * np.sin(0.3 * steps) * np.exp(-0.02 * steps)
    
    # Plot step sizes
    ax1.plot(steps, sgd_steps, 'r-', linewidth=3, label='SGD: Fixed Step Size', alpha=0.8)
    ax1.plot(steps, meta_steps, 'g-', linewidth=3, label='Meta-SGD: Adaptive Step Size', alpha=0.8)
    
    ax1.set_title('Step Size Evolution', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Optimization Step')
    ax1.set_ylabel('Step Size (Learning Rate)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add annotations
    ax1.annotate('Fixed throughout\noptimization', xy=(50, 0.01), xytext=(30, 0.016),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, color='red', fontweight='bold')
    
    ax1.annotate('Adapts based on\nlocal landscape', xy=(70, 0.013), xytext=(75, 0.018),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                fontsize=10, color='green', fontweight='bold')
    
    # Plot resulting loss curves
    ax2.plot(steps, sgd_loss, 'r-', linewidth=3, label='SGD Loss', alpha=0.8)
    ax2.plot(steps, meta_loss, 'g-', linewidth=3, label='Meta-SGD Loss', alpha=0.8)
    
    ax2.set_title('Resulting Loss Convergence', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Optimization Step')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Add annotations
    ax2.annotate(f'Final: {sgd_loss[-1]:.3f}', xy=(95, sgd_loss[-1]), xytext=(70, sgd_loss[-1]*2),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, color='red', fontweight='bold')
    
    ax2.annotate(f'Final: {meta_loss[-1]:.3f}', xy=(95, meta_loss[-1]), xytext=(70, meta_loss[-1]*0.5),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                fontsize=10, color='green', fontweight='bold')
    
    plt.suptitle('Step Size Adaptation Impact on Concept Learning', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('step_size_adaptation_comparison.svg', dpi=300, bbox_inches='tight')
    plt.savefig('step_size_adaptation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Step size adaptation comparison created!")

if __name__ == "__main__":
    create_gradient_learning_diagram()
    create_step_size_adaptation_comparison() 