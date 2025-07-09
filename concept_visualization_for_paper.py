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
from matplotlib.gridspec import GridSpec

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
    """Create demonstration of concept evaluation on example inputs with consistent visualization."""
    
    # Create a specific, controlled example
    np.random.seed(42)
    random.seed(42)
    
    # Generate a concept with exactly 5 features for clear demonstration
    num_features = 5
    max_depth = 3
    
    # Keep generating until we get a concept that uses a reasonable number of literals
    attempts = 0
    while attempts < 20:
        expr, literals, depth = sample_concept_from_pcfg(num_features, max_depth=max_depth)
        expr_str = expression_to_string(expr)
        
        # Accept concepts with 3-5 literals for good demonstration
        if 3 <= literals <= 5:
            break
        attempts += 1
    
    # Generate systematic test inputs (all possible combinations for 5 features = 32 combinations)
    test_inputs = []
    evaluations = []
    
    # Generate all 2^5 = 32 possible input combinations for complete demonstration
    for i in range(2**num_features):
        input_vec = np.array([int(x) for x in f"{i:0{num_features}b}"])
        evaluation = evaluate_pcfg_concept(expr, input_vec)
        test_inputs.append(input_vec.copy())
        evaluations.append(evaluation)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Left: Concept definition with clear complexity explanation
    concept_text = f"""Example Concept:
{expr_str}

Complexity Measures:
• Literals Used: {literals} (unique variables)
• Parse Tree Depth: {depth}
• Input Dimensionality: {num_features} features

The concept uses {literals} out of {num_features} 
available boolean features (x₀, x₁, ..., x₄).
Each input is a {num_features}-bit vector."""
    
    ax1.text(0.05, 0.95, concept_text,
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    ax1.set_title('PCFG Concept Definition', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Right: Discrete boolean visualization (not continuous heatmap)
    # Create a grid showing inputs and outputs with proper discrete visualization
    
    # Select a representative subset for clarity (first 16 combinations)
    display_count = min(16, len(test_inputs))
    
    # Create a table-style visualization
    ax2.set_xlim(-0.5, num_features + 0.5)
    ax2.set_ylim(-0.5, display_count - 0.5)
    
    # Draw grid and fill cells
    for i in range(display_count):
        for j in range(num_features):
            # Input features
            color = 'lightgreen' if test_inputs[i][j] == 1 else 'lightcoral'
            rect = plt.Rectangle((j-0.4, i-0.4), 0.8, 0.8, 
                               facecolor=color, edgecolor='black', linewidth=1)
            ax2.add_patch(rect)
            
            # Add text
            text_color = 'black'
            symbol = '1' if test_inputs[i][j] == 1 else '0'
            ax2.text(j, i, symbol, ha='center', va='center', 
                    fontsize=12, fontweight='bold', color=text_color)
        
        # Output column
        output_color = 'gold' if evaluations[i] else 'lightgray'
        rect = plt.Rectangle((num_features-0.4, i-0.4), 0.8, 0.8,
                           facecolor=output_color, edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        
        # Output symbol
        output_symbol = '✓' if evaluations[i] else '✗'
        ax2.text(num_features, i, output_symbol, ha='center', va='center',
                fontsize=14, fontweight='bold', color='black')
    
    # Labels and formatting
    feature_labels = [f'x₍{i}₎' for i in range(num_features)] + ['Output']
    ax2.set_xticks(range(num_features + 1))
    ax2.set_xticklabels(feature_labels, fontsize=12)
    ax2.set_yticks(range(display_count))
    ax2.set_yticklabels([f'#{i}' for i in range(display_count)])
    ax2.set_ylabel('Test Cases', fontsize=12)
    ax2.set_title(f'Concept Evaluation Examples (showing {display_count}/{len(test_inputs)} cases)', 
                  fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='black', label='Input = 1'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', edgecolor='black', label='Input = 0'),
        plt.Rectangle((0, 0), 1, 1, facecolor='gold', edgecolor='black', label='Output = True'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='black', label='Output = False')
    ]
    ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Add summary statistics
    true_count = sum(evaluations)
    false_count = len(evaluations) - true_count
    summary_text = f"""Evaluation Summary:
• Total inputs: {len(evaluations)}
• True outputs: {true_count} ({100*true_count/len(evaluations):.1f}%)
• False outputs: {false_count} ({100*false_count/len(evaluations):.1f}%)"""
    
    ax2.text(1.05, 0.3, summary_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    # Remove spines and ticks for cleaner look
    ax2.set_xticks(range(num_features + 1))
    ax2.set_yticks(range(display_count))
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
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

def create_final_concept_complexity_figure():
    """Create the final, optimized concept complexity figure for ICML paper.
    
    This figure is specifically designed to help ICML readers quickly understand:
    1. What makes a concept complex (concrete examples)
    2. How our experimental parameters relate to actual complexity
    3. The key insight: depth drives complexity more than feature count
    """
    
    # Set controlled seed for reproducible examples
    np.random.seed(42)
    random.seed(42)
    
    # Create figure with clean, academic layout
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.2, 0.8, 0.8], hspace=0.35, wspace=0.25)
    
    # ========================================
    # TOP ROW: COMPLEXITY PROGRESSION EXAMPLES
    # ========================================
    
    complexity_examples = [
        {"name": "Simple", "config": "F8D3", "features": 8, "depth": 3, "target_literals": 3, "color": "lightblue"},
        {"name": "Medium", "config": "F16D5", "features": 16, "depth": 5, "target_literals": 5, "color": "lightgreen"}, 
        {"name": "Complex", "config": "F32D7", "features": 32, "depth": 7, "target_literals": 7, "color": "lightcoral"}
    ]
    
    generated_examples = []
    
    for i, example in enumerate(complexity_examples):
        ax = fig.add_subplot(gs[0, i])
        
        # Generate concept with target complexity
        best_concept = None
        best_diff = float('inf')
        
        for attempt in range(100):
            expr, literals, depth = sample_concept_from_pcfg(example['features'], max_depth=example['depth'])
            diff = abs(literals - example['target_literals'])
            
            if diff < best_diff:
                best_diff = diff
                best_concept = (expr, literals, depth)
                
            if diff <= 1:  # Accept close matches
                break
        
        if best_concept:
            expr, literals, depth = best_concept
            expr_str = expression_to_string(expr)
            generated_examples.append((expr, literals, depth, expr_str))
            
            # Create concept info box
            concept_info = f"""{example['name']} Concept ({example['config']})

{expr_str}

Complexity Measures:
• Literals: {literals}
• Parse Depth: {depth}
• Feature Space: {example['features']}

This concept uses {literals} of {example['features']} boolean variables.
The logical structure has depth {depth}."""
            
            ax.text(0.05, 0.95, concept_info, transform=ax.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=example['color'], alpha=0.8))
            
            ax.set_title(f'{example["name"]} Concept\n({example["config"]}: {example["features"]} features, max depth {example["depth"]})', 
                        fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f"Failed to generate\n{example['name']} concept", 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(f'{example["name"]} Concept', fontsize=14, fontweight='bold')
        
        ax.axis('off')
    
    # ========================================
    # MIDDLE ROW: EXPERIMENTAL COMPLEXITY MAP
    # ========================================
    
    ax_complexity = fig.add_subplot(gs[1, :])
    
    # Complexity data from our CSV
    complexity_data = [
        {"config": "F8D3", "features": 8, "depth": 3, "mean_literals": 2.791, "mean_depth": 2.802},
        {"config": "F16D3", "features": 16, "depth": 3, "mean_literals": 2.711, "mean_depth": 2.752},
        {"config": "F32D3", "features": 32, "depth": 3, "mean_literals": 2.761, "mean_depth": 2.797},
        {"config": "F8D5", "features": 8, "depth": 5, "mean_literals": 4.764, "mean_depth": 3.833},
        {"config": "F16D5", "features": 16, "depth": 5, "mean_literals": 4.712, "mean_depth": 3.874},
        {"config": "F32D5", "features": 32, "depth": 5, "mean_literals": 4.751, "mean_depth": 3.797},
        {"config": "F8D7", "features": 8, "depth": 7, "mean_literals": 7.424, "mean_depth": 4.803},
        {"config": "F16D7", "features": 16, "depth": 7, "mean_literals": 7.802, "mean_depth": 4.776},
        {"config": "F32D7", "features": 32, "depth": 7, "mean_literals": 7.341, "mean_depth": 4.746}
    ]
    
    # Create color and marker maps
    depth_colors = {3: '#2E86AB', 5: '#A23B72', 7: '#F18F01'}  # Professional colors
    feature_markers = {8: 'o', 16: 's', 32: '^'}
    feature_sizes = {8: 120, 16: 150, 32: 180}
    
    # Plot complexity relationships
    for data in complexity_data:
        depth = data['depth']
        features = data['features']
        
        scatter = ax_complexity.scatter(
            data['mean_literals'], data['mean_depth'], 
            c=depth_colors[depth], marker=feature_markers[features], 
            s=feature_sizes[features], alpha=0.8, edgecolors='white', linewidth=2
        )
        
        # Add configuration labels
        ax_complexity.annotate(
            data['config'], 
            (data['mean_literals'], data['mean_depth']),
            xytext=(8, 8), textcoords='offset points',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
        )
    
    # Draw trend lines to show depth effect
    for depth in [3, 5, 7]:
        depth_data = [d for d in complexity_data if d['depth'] == depth]
        if len(depth_data) >= 2:
            x_vals = [d['mean_literals'] for d in depth_data]
            y_vals = [d['mean_depth'] for d in depth_data]
            ax_complexity.plot(x_vals, y_vals, '--', color=depth_colors[depth], alpha=0.5, linewidth=2)
    
    ax_complexity.set_xlabel('Mean Literals Used', fontsize=14, fontweight='bold')
    ax_complexity.set_ylabel('Mean Parse Tree Depth', fontsize=14, fontweight='bold')
    ax_complexity.set_title('Experimental Configuration Complexity Map', fontsize=16, fontweight='bold')
    ax_complexity.grid(True, alpha=0.3, linestyle='--')
    
    # Enhanced legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12, label='8 Features'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=12, label='16 Features'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=12, label='32 Features'),
        Line2D([0], [0], marker='o', color=depth_colors[3], markersize=12, label='Max Depth 3'),
        Line2D([0], [0], marker='o', color=depth_colors[5], markersize=12, label='Max Depth 5'),
        Line2D([0], [0], marker='o', color=depth_colors[7], markersize=12, label='Max Depth 7')
    ]
    ax_complexity.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.9)
    
    # ========================================
    # BOTTOM ROW: KEY INSIGHTS
    # ========================================
    
    # Left: Complexity scaling insight
    ax_insight1 = fig.add_subplot(gs[2, 0])
    
    insight1_text = """🔑 Key Insight #1: Depth Drives Complexity

• Max Depth 3 → ~3 literals avg
• Max Depth 5 → ~5 literals avg  
• Max Depth 7 → ~7 literals avg

Feature count (8, 16, 32) has minimal 
impact on concept structure complexity.

This allows us to study complexity 
scaling independently of input 
dimensionality."""
    
    ax_insight1.text(0.05, 0.95, insight1_text, transform=ax_insight1.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F4FD", alpha=0.9))
    ax_insight1.set_title('Complexity Scaling', fontsize=14, fontweight='bold')
    ax_insight1.axis('off')
    
    # Middle: Experimental design insight
    ax_insight2 = fig.add_subplot(gs[2, 1])
    
    insight2_text = """🎯 Experimental Design

Our 9 configurations span:
• 3× complexity levels (D3, D5, D7)
• 3× feature dimensions (8, 16, 32)
• 36 total experiments planned

This design isolates:
- Concept complexity effects
- Input dimensionality effects
- Adaptation strategy effects (K=1 vs K=10)"""
    
    ax_insight2.text(0.05, 0.95, insight2_text, transform=ax_insight2.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F8E8", alpha=0.9))
    ax_insight2.set_title('Experimental Strategy', fontsize=14, fontweight='bold')
    ax_insight2.axis('off')
    
    # Right: Paper contribution insight
    ax_insight3 = fig.add_subplot(gs[2, 2])
    
    insight3_text = """📊 Paper Contributions

1. More gradient steps → better 
   generalization across complexity

2. Robust data efficiency: meta-SGD 
   outperforms SGD especially at 
   higher complexity levels

3. Mechanistic insights: gradient 
   alignment explains when and why 
   meta-learning succeeds

Clear complexity scaling enables 
precise theoretical analysis."""
    
    ax_insight3.text(0.05, 0.95, insight3_text, transform=ax_insight3.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF2E8", alpha=0.9))
    ax_insight3.set_title('Paper Contributions', fontsize=14, fontweight='bold')
    ax_insight3.axis('off')
    
    # ========================================
    # OVERALL TITLE
    # ========================================
    
    fig.suptitle('PCFG Concept Complexity Scale: From Simple to Complex Logical Structures\n' + 
                'Understanding the Relationship Between Grammar Depth and Concept Complexity', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig

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
    
    # 5. Final concept complexity figure
    print("🎨 Creating final concept complexity figure...")
    fig5 = create_final_concept_complexity_figure()
    fig5.savefig('figures/concept_visualizations/final_concept_complexity_figure.png', dpi=300, bbox_inches='tight')
    fig5.savefig('figures/concept_visualizations/final_concept_complexity_figure.pdf', bbox_inches='tight')
    
    # Save complexity statistics
    complexity_df.to_csv('figures/concept_visualizations/complexity_statistics.csv', index=False)
    exp_df.to_csv('figures/concept_visualizations/experiment_status.csv', index=False)
    
    print("✅ All visualizations saved to figures/concept_visualizations/")
    print("\nGenerated files:")
    print("  - complexity_grid.png/pdf: PCFG complexity analysis")
    print("  - concept_examples.png/pdf: Representative concept examples") 
    print("  - evaluation_demo.png/pdf: Concept evaluation demonstration")
    print("  - experiment_overview.png/pdf: Current experimental status")
    print("  - final_concept_complexity_figure.png/pdf: Final concept complexity figure")
    print("  - complexity_statistics.csv: Numerical complexity data")
    print("  - experiment_status.csv: Experiment status data")
    
    plt.show()

if __name__ == "__main__":
    main() 