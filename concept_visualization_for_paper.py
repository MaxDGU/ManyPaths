#!/usr/bin/env python3
"""
Concept Visualization for Paper

Creates publication-quality visualizations of:
1. PCFG concept examples across complexity levels
2. Feature×Depth complexity grid  
3. Concept distribution statistics
4. Example concept evaluations

For camera-ready submission figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pcfg import sample_concept_from_pcfg, evaluate_pcfg_concept, DEFAULT_MAX_DEPTH
import random
from collections import defaultdict
import pandas as pd

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def expression_to_string(expr):
    """Convert PCFG expression to readable string."""
    if isinstance(expr, str):
        # Literal: F0_T -> "x₀", F0_F -> "¬x₀"
        parts = expr.split('_')
        feature_idx = int(parts[0][1:])
        if parts[1] == 'T':
            return f"x_{feature_idx}"
        else:
            return f"¬x_{feature_idx}"
    
    op = expr[0]
    if op == 'NOT':
        child_str = expression_to_string(expr[1])
        return f"¬({child_str})"
    elif op == 'AND':
        left_str = expression_to_string(expr[1])
        right_str = expression_to_string(expr[2])
        return f"({left_str} ∧ {right_str})"
    elif op == 'OR':
        left_str = expression_to_string(expr[1])
        right_str = expression_to_string(expr[2])
        return f"({left_str} ∨ {right_str})"
    else:
        return str(expr)

def generate_concept_examples():
    """Generate representative concept examples across complexity levels."""
    
    examples = []
    
    # Simple concepts (depth 1-2, few literals)
    np.random.seed(42)
    random.seed(42)
    
    # Generate examples for different feature counts and depths
    configs = [
        (8, 3, "F8_D3"),
        (16, 3, "F16_D3"), 
        (32, 3, "F32_D3"),
        (8, 5, "F8_D5"),
        (16, 5, "F16_D5"),
        (32, 5, "F32_D5"),
        (8, 7, "F8_D7"),
        (16, 7, "F16_D7"),
        (32, 7, "F32_D7")
    ]
    
    for num_features, max_depth, config_name in configs:
        # Generate 5 examples per configuration
        config_examples = []
        for _ in range(5):
            expr, literals, depth = sample_concept_from_pcfg(num_features, max_depth=max_depth)
            config_examples.append({
                'config': config_name,
                'num_features': num_features,
                'max_depth': max_depth,
                'expression': expr,
                'literals': literals,
                'actual_depth': depth,
                'expression_str': expression_to_string(expr)
            })
        examples.extend(config_examples)
    
    return examples

def create_complexity_grid_visualization():
    """Create the features×depth complexity grid visualization."""
    
    # Generate complexity statistics
    features_list = [8, 16, 32]
    depths_list = [3, 5, 7]
    
    # Collect complexity statistics
    complexity_data = []
    
    np.random.seed(42)
    random.seed(42)
    
    for num_features in features_list:
        for max_depth in depths_list:
            # Generate many samples to get statistics
            literals_counts = []
            actual_depths = []
            
            for _ in range(1000):  # Large sample for good statistics
                expr, literals, depth = sample_concept_from_pcfg(num_features, max_depth=max_depth)
                literals_counts.append(literals)
                actual_depths.append(depth)
            
            complexity_data.append({
                'num_features': num_features,
                'max_depth': max_depth,
                'mean_literals': np.mean(literals_counts),
                'std_literals': np.std(literals_counts),
                'mean_depth': np.mean(actual_depths),
                'std_depth': np.std(actual_depths),
                'min_literals': np.min(literals_counts),
                'max_literals': np.max(literals_counts),
                'min_depth': np.min(actual_depths),
                'max_depth_actual': np.max(actual_depths)
            })
    
    df = pd.DataFrame(complexity_data)
    
    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PCFG Concept Complexity Grid', fontsize=16, fontweight='bold')
    
    # 1. Mean number of literals heatmap
    pivot_literals = df.pivot(index='max_depth', columns='num_features', values='mean_literals')
    sns.heatmap(pivot_literals, annot=True, fmt='.1f', cmap='viridis', 
                ax=axes[0,0], cbar_kws={'label': 'Mean Literals'})
    axes[0,0].set_title('(a) Mean Number of Literals')
    axes[0,0].set_xlabel('Number of Features')
    axes[0,0].set_ylabel('Max PCFG Depth')
    
    # 2. Mean actual depth heatmap  
    pivot_depth = df.pivot(index='max_depth', columns='num_features', values='mean_depth')
    sns.heatmap(pivot_depth, annot=True, fmt='.1f', cmap='plasma',
                ax=axes[0,1], cbar_kws={'label': 'Mean Depth'})
    axes[0,1].set_title('(b) Mean Actual Depth')
    axes[0,1].set_xlabel('Number of Features')
    axes[0,1].set_ylabel('Max PCFG Depth')
    
    # 3. Complexity distribution by configuration
    df['config'] = df['num_features'].astype(str) + 'F_D' + df['max_depth'].astype(str)
    
    # Scatter plot: literals vs depth
    colors = sns.color_palette("husl", len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        axes[1,0].scatter(row['mean_literals'], row['mean_depth'], 
                         s=100, color=colors[i], label=f"F{row['num_features']}_D{row['max_depth']}")
        # Add error bars
        axes[1,0].errorbar(row['mean_literals'], row['mean_depth'],
                          xerr=row['std_literals'], yerr=row['std_depth'],
                          color=colors[i], alpha=0.6)
    
    axes[1,0].set_xlabel('Mean Number of Literals')
    axes[1,0].set_ylabel('Mean Actual Depth')
    axes[1,0].set_title('(c) Complexity Trade-offs')
    axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 4. Complexity ranges
    x_pos = np.arange(len(df))
    axes[1,1].bar(x_pos - 0.2, df['mean_literals'], 0.4, 
                  yerr=df['std_literals'], label='Literals', alpha=0.7)
    axes[1,1].bar(x_pos + 0.2, df['mean_depth'], 0.4,
                  yerr=df['std_depth'], label='Depth', alpha=0.7) 
    
    axes[1,1].set_xlabel('Configuration')
    axes[1,1].set_ylabel('Mean Value')
    axes[1,1].set_title('(d) Complexity Statistics')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels([f"F{row['num_features']}_D{row['max_depth']}" 
                              for _, row in df.iterrows()], rotation=45)
    axes[1,1].legend()
    
    plt.tight_layout()
    return fig, df

def create_concept_examples_figure():
    """Create figure showing example concepts across complexity levels."""
    
    examples = generate_concept_examples()
    df = pd.DataFrame(examples)
    
    # Select representative examples
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Representative PCFG Concept Examples', fontsize=16, fontweight='bold')
    
    configs = ["F8_D3", "F16_D3", "F32_D3",
               "F8_D5", "F16_D5", "F32_D5", 
               "F8_D7", "F16_D7", "F32_D7"]
    
    for i, config in enumerate(configs):
        row = i // 3
        col = i % 3
        
        config_examples = df[df['config'] == config]
        
        # Select examples with different complexities
        sorted_examples = config_examples.sort_values(['literals', 'actual_depth'])
        selected_indices = [0, len(sorted_examples)//2, len(sorted_examples)-1]
        
        texts = []
        for j, idx in enumerate(selected_indices):
            if idx < len(sorted_examples):
                example = sorted_examples.iloc[idx]
                complexity_label = f"L={example['literals']}, D={example['actual_depth']}"
                texts.append(f"{complexity_label}:\n{example['expression_str']}")
        
        # Display as text in subplot
        axes[row, col].text(0.05, 0.95, '\n\n'.join(texts), 
                           transform=axes[row, col].transAxes,
                           fontsize=10, verticalalignment='top',
                           fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        axes[row, col].set_title(f'{config} (Features={config_examples.iloc[0]["num_features"]}, MaxDepth={config_examples.iloc[0]["max_depth"]})')
        axes[row, col].set_xlim(0, 1)
        axes[row, col].set_ylim(0, 1)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig

def create_concept_evaluation_demo():
    """Create demonstration of concept evaluation on example inputs."""
    
    # Create a specific example
    np.random.seed(42)
    random.seed(42)
    
    # Generate a moderately complex concept
    expr, literals, depth = sample_concept_from_pcfg(8, max_depth=4)
    expr_str = expression_to_string(expr)
    
    # Generate test inputs and evaluate
    test_inputs = []
    evaluations = []
    
    # Generate some specific interesting inputs
    for i in range(16):
        # Generate diverse test cases
        if i < 8:
            # First 8: systematic patterns
            input_vec = np.array([int(x) for x in f"{i:08b}"])
        else:
            # Random inputs
            input_vec = np.random.randint(0, 2, 8)
        
        evaluation = evaluate_pcfg_concept(expr, input_vec)
        
        test_inputs.append(input_vec.copy())
        evaluations.append(evaluation)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Concept visualization
    ax1.text(0.05, 0.95, f"Example Concept:\n{expr_str}\n\nLiterals: {literals}\nDepth: {depth}",
             transform=ax1.transAxes, fontsize=14, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    ax1.set_title('PCFG Concept Definition', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Right: Evaluation matrix
    eval_matrix = np.array([[int(bit) for bit in input_vec] + [int(eval)] 
                           for input_vec, eval in zip(test_inputs, evaluations)])
    
    im = ax2.imshow(eval_matrix, cmap='RdYlBu', aspect='auto')
    
    # Labels
    feature_labels = [f'x_{i}' for i in range(8)] + ['Output']
    ax2.set_xticks(range(9))
    ax2.set_xticklabels(feature_labels)
    ax2.set_ylabel('Test Input #')
    ax2.set_title('Concept Evaluation Examples', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(test_inputs)):
        for j in range(9):
            text = ax2.text(j, i, f'{eval_matrix[i, j]}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Value')
    
    plt.tight_layout()
    return fig

def create_experiment_overview():
    """Create overview of experimental configuration grid."""
    
    # Current experimental status (based on della analysis)
    experiments = [
        {"config": "F8_D3", "features": 8, "depth": 3, "k1_seeds": 6, "k10_seeds": 4, "status": "Complete"},
        {"config": "F8_D5", "features": 8, "depth": 5, "k1_seeds": 1, "k10_seeds": 1, "status": "Partial"}, 
        {"config": "F8_D7", "features": 8, "depth": 7, "k1_seeds": 1, "k10_seeds": 1, "status": "Partial"},
        {"config": "F16_D3", "features": 16, "depth": 3, "k1_seeds": 1, "k10_seeds": 1, "status": "Partial"},
        {"config": "F16_D5", "features": 16, "depth": 5, "k1_seeds": 1, "k10_seeds": 1, "status": "Partial"},
        {"config": "F16_D7", "features": 16, "depth": 7, "k1_seeds": 1, "k10_seeds": 1, "status": "Partial"},
        {"config": "F32_D3", "features": 32, "depth": 3, "k1_seeds": 1, "k10_seeds": 1, "status": "Partial"},
        {"config": "F32_D5", "features": 32, "depth": 5, "k1_seeds": 1, "k10_seeds": 1, "status": "Partial"},
        {"config": "F32_D7", "features": 32, "depth": 7, "k1_seeds": 1, "k10_seeds": 1, "status": "Partial"}
    ]
    
    df = pd.DataFrame(experiments)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Experimental Configuration Overview', fontsize=16, fontweight='bold')
    
    # 1. Experiment grid status
    status_pivot = df.pivot(index='depth', columns='features', values='status')
    status_map = {'Complete': 2, 'Partial': 1, 'Missing': 0}
    status_numeric = status_pivot.replace(status_map)
    
    sns.heatmap(status_numeric, annot=status_pivot.values, fmt='', 
                cmap=['red', 'yellow', 'green'], 
                ax=axes[0], cbar_kws={'label': 'Status'})
    axes[0].set_title('(a) Experiment Status Grid')
    axes[0].set_xlabel('Number of Features')
    axes[0].set_ylabel('PCFG Max Depth')
    
    # 2. K=1 seeds available
    k1_pivot = df.pivot(index='depth', columns='features', values='k1_seeds')
    sns.heatmap(k1_pivot, annot=True, fmt='d', cmap='Blues',
                ax=axes[1], cbar_kws={'label': 'K=1 Seeds'})
    axes[1].set_title('(b) K=1 Seeds Available')
    axes[1].set_xlabel('Number of Features')
    axes[1].set_ylabel('PCFG Max Depth')
    
    # 3. K=10 seeds available  
    k10_pivot = df.pivot(index='depth', columns='features', values='k10_seeds')
    sns.heatmap(k10_pivot, annot=True, fmt='d', cmap='Reds',
                ax=axes[2], cbar_kws={'label': 'K=10 Seeds'})
    axes[2].set_title('(c) K=10 Seeds Available')
    axes[2].set_xlabel('Number of Features')
    axes[2].set_ylabel('PCFG Max Depth')
    
    plt.tight_layout()
    return fig, df

def main():
    """Generate all concept visualizations for the paper."""
    
    print("🎨 Generating Concept Visualizations for Paper...")
    
    # Create output directory
    import os
    os.makedirs('figures/concept_visualizations', exist_ok=True)
    
    # 1. Complexity grid analysis
    print("📊 Creating complexity grid visualization...")
    fig1, complexity_df = create_complexity_grid_visualization()
    fig1.savefig('figures/concept_visualizations/complexity_grid.png', dpi=300, bbox_inches='tight')
    fig1.savefig('figures/concept_visualizations/complexity_grid.pdf', bbox_inches='tight')
    
    # 2. Concept examples
    print("📝 Creating concept examples figure...")
    fig2 = create_concept_examples_figure()
    fig2.savefig('figures/concept_visualizations/concept_examples.png', dpi=300, bbox_inches='tight')
    fig2.savefig('figures/concept_visualizations/concept_examples.pdf', bbox_inches='tight')
    
    # 3. Evaluation demonstration
    print("🔍 Creating concept evaluation demo...")
    fig3 = create_concept_evaluation_demo()
    fig3.savefig('figures/concept_visualizations/evaluation_demo.png', dpi=300, bbox_inches='tight')
    fig3.savefig('figures/concept_visualizations/evaluation_demo.pdf', bbox_inches='tight')
    
    # 4. Experiment overview
    print("🧪 Creating experiment overview...")
    fig4, exp_df = create_experiment_overview()
    fig4.savefig('figures/concept_visualizations/experiment_overview.png', dpi=300, bbox_inches='tight')
    fig4.savefig('figures/concept_visualizations/experiment_overview.pdf', bbox_inches='tight')
    
    # Save complexity statistics
    complexity_df.to_csv('figures/concept_visualizations/complexity_statistics.csv', index=False)
    exp_df.to_csv('figures/concept_visualizations/experiment_status.csv', index=False)
    
    print("✅ All visualizations saved to figures/concept_visualizations/")
    print("\nGenerated files:")
    print("  - complexity_grid.png/pdf: PCFG complexity analysis")
    print("  - concept_examples.png/pdf: Representative concept examples") 
    print("  - evaluation_demo.png/pdf: Concept evaluation demonstration")
    print("  - experiment_overview.png/pdf: Current experimental status")
    print("  - complexity_statistics.csv: Numerical complexity data")
    print("  - experiment_status.csv: Experiment status data")
    
    plt.show()

if __name__ == "__main__":
    main() 