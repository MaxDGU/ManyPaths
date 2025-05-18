import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects
from sklearn.decomposition import PCA
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define model colors for consistent visualization
MODEL_COLORS = {
    "mlp": "#4C72B0",   # Blue
    "cnn": "#C44E52",   # Red
    "lstm": "#55A868",  # Green
    "transformer": "#8172B3"  # Purple
}

# Define network size markers
SIZE_MARKERS = {
    "small": "o",      # Circle for small networks
    "medium": "s",     # Square for medium networks
    "large": "^",      # Triangle for large networks
    "xlarge": "D"      # Diamond for extra large networks
}

def get_model_configs():
    """Define model configurations of increasing complexity."""
    configs = {
        "mlp": [
            {"name": "small", "n_hidden": 32, "n_layers": 2, "color": MODEL_COLORS["mlp"]},
            {"name": "medium", "n_hidden": 64, "n_layers": 4, "color": MODEL_COLORS["mlp"]},
            {"name": "large", "n_hidden": 128, "n_layers": 8, "color": MODEL_COLORS["mlp"]},
            {"name": "xlarge", "n_hidden": 256, "n_layers": 12, "color": MODEL_COLORS["mlp"]},
        ],
        "cnn": [
            {"name": "small", "n_hiddens": [16, 16], "n_layers": 3, "color": MODEL_COLORS["cnn"]},
            {"name": "medium", "n_hiddens": [32, 32, 16], "n_layers": 4, "color": MODEL_COLORS["cnn"]},
            {"name": "large", "n_hiddens": [64, 64, 32, 16], "n_layers": 5, "color": MODEL_COLORS["cnn"]},
            {"name": "xlarge", "n_hiddens": [128, 64, 64, 32, 16], "n_layers": 6, "color": MODEL_COLORS["cnn"]},
        ],
        "lstm": [
            {"name": "small", "n_hidden": 32, "n_layers": 1, "color": MODEL_COLORS["lstm"]},
            {"name": "medium", "n_hidden": 64, "n_layers": 2, "color": MODEL_COLORS["lstm"]},
            {"name": "large", "n_hidden": 128, "n_layers": 3, "color": MODEL_COLORS["lstm"]},
            {"name": "xlarge", "n_hidden": 256, "n_layers": 4, "color": MODEL_COLORS["lstm"]},
        ],
        "transformer": [
            {"name": "small", "d_model": 32, "nhead": 2, "num_layers": 1, "color": MODEL_COLORS["transformer"]},
            {"name": "medium", "d_model": 64, "nhead": 4, "num_layers": 2, "color": MODEL_COLORS["transformer"]},
            {"name": "large", "d_model": 128, "nhead": 8, "num_layers": 3, "color": MODEL_COLORS["transformer"]},
            {"name": "xlarge", "d_model": 256, "nhead": 8, "num_layers": 4, "color": MODEL_COLORS["transformer"]},
        ]
    }
    return configs

def generate_concept(bits, scale=1.0):
    """Generate a simple concept image based on 4 bits."""
    if not (len(bits) == 4):
        raise ValueError("Bits must be length 4.")

    # Initialize a blank grid
    grid_image = np.ones((32, 32, 3), dtype=np.float32) * 255

    # Extract bits
    color = (1, 2) if bits[0] == 1 else (0, 1)
    shape = bits[1] == 1
    size = 4 if bits[2] == 1 else 10
    style = bits[3] == 1

    if shape:
        grid_image[size : 32 - size, size : 32 - size, color] = 0
        if style == 1:
            grid_image[size : 32 - size, size : 32 - size : 2, color] = 200
    else:
        for i in range(32 - 2 * size):
            grid_image[
                32 - (size + i + 1), i // 2 + size : 32 - i // 2 - size, color
            ] = 0
        if style == 1:
            for i in range(0, 32, 1):
                for j in range(0, 32, 2):
                    if grid_image[i, j, color].any() == 0:
                        grid_image[i, j, color] = 200
    
    grid_image = grid_image / scale
    return grid_image

def generate_synthetic_data():
    """Generate synthetic data to simulate concept learning experiment results."""
    model_configs = get_model_configs()
    
    # Simulate concept learning data
    all_results = {}
    
    # For each model architecture
    for model_type, configs in model_configs.items():
        results = {}
        
        # For each model size configuration
        for config in configs:
            size_name = config["name"]
            
            # Generate 16 concept examples (all possible 4-bit combinations)
            concept_activations = {}
            concept_labels = []
            
            # Synthetic last layer activations
            # We'll simulate more separation for larger models
            separation_factor = {
                "small": 0.5,
                "medium": 1.0,
                "large": 2.0,
                "xlarge": 3.0
            }[size_name]
            
            # Set the separation factor based on model type as well
            architecture_factor = {
                "mlp": 1.0,
                "cnn": 1.5,
                "lstm": 0.8,
                "transformer": 1.6
            }[model_type]
            
            # Combined factor
            factor = separation_factor * architecture_factor
            
            # Create synthetic latent representations that reflect concept learning quality
            # We'll represent 16 concepts (4 bits, 2^4 = 16)
            simulated_activations = []
            for i in range(16):
                # Get the concept bits
                bits = [int(x) for x in f"{i:04b}"]
                
                # Create a concept label (binary classification)
                # Simple rule: concept is "1" if first two bits are the same
                concept_label = 1.0 if bits[0] == bits[1] else 0.0
                concept_labels.append(concept_label)
                
                # Create a base vector that has some pattern related to the concept
                base_vector = np.array([
                    bits[0] * 2 - 1,  # Convert 0/1 to -1/1
                    bits[1] * 2 - 1,
                    bits[2] * 2 - 1,
                    bits[3] * 2 - 1
                ])
                
                # Expand to higher dimensions
                latent_dim = {
                    "small": 16,
                    "medium": 32,
                    "large": 64,
                    "xlarge": 128
                }[size_name]
                
                # Create a random projection matrix
                if i == 0:
                    # Initialize on first concept
                    projection = np.random.randn(4, latent_dim)
                
                # Project the base vector to higher dimensions
                high_dim = base_vector @ projection
                
                # Add noise, with less noise for better models
                noise = np.random.randn(latent_dim) / factor
                
                # Final latent representation
                latent_repr = high_dim + noise
                
                simulated_activations.append(latent_repr)
            
            # Convert to numpy array
            last_layer_activations = np.array(simulated_activations)
            
            # Store in format similar to what we'd get from a real model
            concept_activations['last_layer'] = last_layer_activations
            
            # Calculate parameter counts based on model architecture and size
            if model_type == "mlp":
                n_params = 1000 * config["n_hidden"] * config["n_layers"]
            elif model_type == "cnn":
                n_params = 1000 * sum(config["n_hiddens"]) * config["n_layers"]
            elif model_type == "lstm":
                n_params = 4000 * config["n_hidden"] * config["n_layers"]
            else:  # transformer
                n_params = 5000 * config["d_model"] * config["num_layers"]
            
            # Store results
            results[size_name] = {
                'activations': concept_activations,
                'labels': np.array(concept_labels),
                'n_params': n_params,
                'config': config
            }
        
        all_results[model_type] = results
    
    return all_results

def calculate_concept_separation(activations, labels):
    """Calculate a metric for how well concepts are separated in the latent space."""
    # Get the last layer activations
    if not activations:
        return 0.0
        
    last_layer_key = sorted(activations.keys())[-1]
    last_layer_activations = activations[last_layer_key]
    
    # Flatten activations if needed
    if len(last_layer_activations.shape) > 2:
        last_layer_activations = last_layer_activations.reshape(last_layer_activations.shape[0], -1)
    
    # Calculate average distance between points with different labels
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return 0.0
    
    # Calculate cluster separation using between-class vs within-class distance
    between_class_distance = 0
    within_class_distance = 0
    n_between_pairs = 0
    n_within_pairs = 0
    
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            dist = np.linalg.norm(last_layer_activations[i] - last_layer_activations[j])
            
            if labels[i] == labels[j]:
                within_class_distance += dist
                n_within_pairs += 1
            else:
                between_class_distance += dist
                n_between_pairs += 1
    
    if n_within_pairs == 0 or n_between_pairs == 0:
        return 0.0
    
    avg_within_class = within_class_distance / n_within_pairs
    avg_between_class = between_class_distance / n_between_pairs
    
    if avg_within_class == 0:
        return 0.0
    
    separation_score = avg_between_class / avg_within_class
    return separation_score

def calculate_silhouette_score(activations, labels):
    """Calculate silhouette score for clusters in the latent space."""
    if not activations:
        return 0.0
        
    last_layer_key = sorted(activations.keys())[-1]
    last_layer_activations = activations[last_layer_key]
    
    # Flatten activations if needed
    if len(last_layer_activations.shape) > 2:
        last_layer_activations = last_layer_activations.reshape(last_layer_activations.shape[0], -1)
    
    # Reduce dimensions to 2D for visualization using PCA
    try:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(last_layer_activations)
        
        # Calculate a simplified silhouette-like score
        unique_labels = np.unique(labels)
        
        # If only one label, silhouette is undefined
        if len(unique_labels) <= 1:
            return 0.0
        
        # Calculate centroids for each class
        centroids = {}
        for label in unique_labels:
            mask = labels == label
            centroids[label] = X_pca[mask].mean(axis=0)
        
        # Calculate silhouette-like score for each point
        silhouette_values = []
        for i, (point, label) in enumerate(zip(X_pca, labels)):
            # Calculate average distance to points in same class (a)
            same_class_points = X_pca[labels == label]
            if len(same_class_points) <= 1:
                continue  # Skip if this is the only point in its class
                
            a = np.mean([np.linalg.norm(point - other) for other in same_class_points if not np.array_equal(point, other)])
            
            # Calculate minimum average distance to points in different classes (b)
            b_values = []
            for other_label in unique_labels:
                if other_label == label:
                    continue
                other_class_points = X_pca[labels == other_label]
                if len(other_class_points) == 0:
                    continue
                b_values.append(np.mean([np.linalg.norm(point - other) for other in other_class_points]))
            
            if not b_values:
                continue  # Skip if no other classes
            
            b = min(b_values)
            
            # Calculate silhouette for this point
            s = (b - a) / max(a, b)
            silhouette_values.append(s)
        
        # Return average silhouette score
        return np.mean(silhouette_values) if silhouette_values else 0.0
    
    except Exception as e:
        print(f"Error calculating silhouette score: {e}")
        return 0.0

def visualize_concepts():
    """Create a grid visualization of all 16 possible concepts."""
    # Create a figure to hold the grid
    grid_size = (4, 4)
    cell_size = 32
    grid_image = np.ones((grid_size[0] * cell_size, grid_size[1] * cell_size, 3), dtype=np.float32) * 255
    
    # Generate all 16 concepts
    for i in range(16):
        bits = [int(x) for x in f"{i:04b}"]
        concept_image = generate_concept(bits, scale=255.0)
        
        # Calculate position in grid
        row = i // 4
        col = i % 4
        
        # Place the concept image in the grid
        grid_image[row*cell_size:(row+1)*cell_size, col*cell_size:(col+1)*cell_size] = concept_image
    
    return grid_image

def create_latent_space_visualization(all_results):
    """Create a comprehensive latent space visualization."""
    # Create visualization directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Use a dark background for the visualization
    plt.style.use('dark_background')
    background_color = '#0E1117'
    fig = plt.figure(figsize=(22, 18), facecolor=background_color)
    
    # Create a grid layout for the plots
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.2])
    
    # 1. 3D scatter plot of model complexity vs concept separation
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Gather data for the 3D plot
    x, y, z = [], [], []  # Size, Depth, Concept Separation
    c, s = [], []         # Colors, Sizes
    markers, labels = [], []
    
    for model_type, results in all_results.items():
        for size_name, data in results.items():
            # Extract key information
            config = data['config']
            activations = data['activations']
            true_labels = data['labels']
            n_params = data['n_params']
            
            # Calculate concept separation in latent space
            concept_separation = calculate_concept_separation(activations, true_labels)
            
            # Add data point for 3D scatter
            if model_type == "mlp":
                x.append(config['n_hidden'])
                y.append(config['n_layers'])
            elif model_type == "cnn":
                x.append(max(config['n_hiddens']))
                y.append(config['n_layers'])
            elif model_type == "lstm":
                x.append(config['n_hidden'])
                y.append(config['n_layers'])
            else:  # transformer
                x.append(config['d_model'])
                y.append(config['num_layers'])
                
            z.append(concept_separation)
            c.append(MODEL_COLORS[model_type])
            s.append(np.log2(n_params) * 10)  # Size based on parameter count
            markers.append(SIZE_MARKERS[size_name])
            labels.append(f"{model_type}-{size_name}")
    
    # Create scatter plot with appropriate markers
    for i in range(len(x)):
        ax1.scatter(
            x[i], y[i], z[i], 
            s=s[i], c=c[i], 
            marker=markers[i], 
            label=labels[i],
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5
        )
    
    # Add parameter count annotations
    for i in range(len(x)):
        txt = ax1.text(
            x[i], y[i], z[i], 
            f"{all_results[labels[i].split('-')[0]][labels[i].split('-')[1]]['n_params']}", 
            color='white', 
            fontsize=8
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
    
    # Set labels and title
    ax1.set_xlabel('Hidden Size / Feature Maps', labelpad=10, color='white')
    ax1.set_ylabel('Layers', labelpad=10, color='white')
    ax1.set_zlabel('Concept Separation', labelpad=10, color='white')
    title = ax1.set_title('Network Architecture vs Concept Separation', 
                         pad=20, color='cyan', fontsize=14, fontweight='bold')
    title.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])
    
    # Customize 3D plot appearance
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.tick_params(colors='white')
    
    # 2. Concept formation visualization
    ax2 = fig.add_subplot(gs[0, 1:])
    
    # For each model type, show the biggest model's concept representation
    concept_images = []
    concept_labels = []
    
    # Extract PCA embeddings for the largest model of each type
    embeddings_by_model = {}
    for model_type in all_results.keys():
        xlarge_data = all_results[model_type]['xlarge']
        activations = xlarge_data['activations']
        labels = xlarge_data['labels']
        
        # Get the last layer activations
        last_layer_key = sorted(activations.keys())[-1]
        last_layer_activations = activations[last_layer_key]
        
        # Flatten activations if needed
        if len(last_layer_activations.shape) > 2:
            last_layer_activations = last_layer_activations.reshape(last_layer_activations.shape[0], -1)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2, random_state=42)
        embeddings = pca.fit_transform(last_layer_activations)
        
        embeddings_by_model[model_type] = {
            'embeddings': embeddings,
            'labels': labels
        }
    
    # Create a 2x2 grid within ax2 to show PCA embeddings
    gs_inner = GridSpec(2, 2, figure=fig, 
                       left=ax2.get_position().x0, 
                       right=ax2.get_position().x1,
                       bottom=ax2.get_position().y0, 
                       top=ax2.get_position().y1)
    
    axes = []
    for i, (model_type, position) in enumerate([
        ('mlp', (0, 0)), 
        ('cnn', (0, 1)), 
        ('lstm', (1, 0)), 
        ('transformer', (1, 1))
    ]):
        ax = fig.add_subplot(gs_inner[position])
        axes.append(ax)
        
        if model_type in embeddings_by_model:
            embeddings = embeddings_by_model[model_type]['embeddings']
            labels = embeddings_by_model[model_type]['labels']
            
            # Create scatter plot colored by concept (label)
            scatter = ax.scatter(
                embeddings[:, 0], 
                embeddings[:, 1], 
                c=labels, 
                cmap='viridis',
                alpha=0.8,
                s=100,
                edgecolors='white',
                linewidth=0.5
            )
            
            # Add method label
            ax.set_title(f"{model_type.upper()}", color='white')
            ax.set_xlabel("PC 1", color='white')
            ax.set_ylabel("PC 2", color='white')
            
            # Add annotations for each concept point
            for j, (x, y, label) in enumerate(zip(embeddings[:, 0], embeddings[:, 1], labels)):
                if j % 3 == 0:  # Annotate every 3rd point to avoid clutter
                    bits = [int(b) for b in f"{j:04b}"][:4]
                    ax.text(x, y, f"{bits}", fontsize=6, color='white',
                           path_effects=[path_effects.withStroke(linewidth=1, foreground='black')])
            
            ax.tick_params(colors='white')
            ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add overall title for the PCA section
    fig.text(0.6, 0.95, "Concept Representation in Latent Space", 
             color='cyan', fontsize=14, fontweight='bold',
             path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
    
    # Add color bar
    cbar_ax = fig.add_axes([0.92, 0.66, 0.02, 0.2])
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Concept Class', rotation=270, labelpad=20, color='white')
    
    # 3. Concept learning curves
    ax3 = fig.add_subplot(gs[1, :2])
    
    # Prepare data for learning curves
    model_sizes = ['small', 'medium', 'large', 'xlarge']
    x_values = np.arange(len(model_sizes))
    width = 0.2
    
    for i, model_type in enumerate(all_results.keys()):
        separations = []
        for size in model_sizes:
            if size in all_results[model_type]:
                data = all_results[model_type][size]
                activations = data['activations']
                labels = data['labels']
                separation = calculate_concept_separation(activations, labels)
                separations.append(separation)
            else:
                separations.append(0)
        
        # Plot bars with offset
        ax3.bar(
            x_values + i * width - 0.3, 
            separations, 
            width=width, 
            color=MODEL_COLORS[model_type], 
            label=model_type.upper(),
            alpha=0.8,
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add text annotations
        for j, v in enumerate(separations):
            ax3.text(
                x_values[j] + i * width - 0.3, 
                v + 0.05, 
                f"{v:.2f}", 
                ha='center', 
                color='white', 
                fontsize=8,
                path_effects=[path_effects.withStroke(linewidth=1, foreground='black')]
            )
    
    # Customize learning curve plot
    ax3.set_xticks(x_values)
    ax3.set_xticklabels(model_sizes)
    ax3.set_xlabel("Model Size", color='white')
    ax3.set_ylabel("Concept Separation Score", color='white')
    ax3.set_title("Concept Separation by Architecture and Size", 
                 color='cyan', fontsize=14, fontweight='bold',
                 path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
    ax3.tick_params(colors='white')
    ax3.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax3.legend(loc='upper left', frameon=True, facecolor='black', edgecolor='cyan')
    
    # 4. Parameter count comparison with silhouette scores
    ax4 = fig.add_subplot(gs[1, 2])
    
    # Prepare data
    param_counts = []
    silhouette_scores = []
    model_colors = []
    model_markers = []
    model_labels = []
    
    for model_type in all_results.keys():
        for size, data in all_results[model_type].items():
            param_counts.append(data['n_params'])
            activations = data['activations']
            labels = data['labels']
            
            # Calculate silhouette score for concept clustering
            silhouette = calculate_silhouette_score(activations, labels)
            silhouette_scores.append(silhouette)
            
            model_colors.append(MODEL_COLORS[model_type])
            model_markers.append(SIZE_MARKERS[size])
            model_labels.append(f"{model_type}-{size}")
    
    # Create scatter plot
    for i in range(len(param_counts)):
        ax4.scatter(
            np.log10(param_counts[i]), 
            silhouette_scores[i], 
            color=model_colors[i], 
            marker=model_markers[i], 
            s=80, 
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5,
            label=model_labels[i]
        )
    
    # Add text annotations
    for i, (x, y, label) in enumerate(zip(np.log10(param_counts), silhouette_scores, model_labels)):
        if i % 4 == 0:  # Annotate every 4th point to prevent clutter
            ax4.text(
                x, y, label, 
                fontsize=6, 
                color='white',
                path_effects=[path_effects.withStroke(linewidth=1, foreground='black')]
            )
    
    # Add trend line
    z = np.polyfit(np.log10(param_counts), silhouette_scores, 1)
    p = np.poly1d(z)
    ax4.plot(
        np.log10(param_counts), 
        p(np.log10(param_counts)), 
        "r--", 
        alpha=0.5,
        linewidth=1,
        color='white'
    )
    
    # Customize parameter plot
    ax4.set_xlabel("Log10(Parameters)", color='white')
    ax4.set_ylabel("Silhouette Score", color='white')
    ax4.set_title("Model Parameters vs. Concept Clustering Quality", 
                 color='cyan', fontsize=14, fontweight='bold',
                 path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
    ax4.tick_params(colors='white')
    ax4.grid(True, linestyle='--', alpha=0.3)
    
    # 5. Create a grid of concept visualizations
    ax5 = fig.add_subplot(gs[2, :])
    
    # Generate 16 concept images
    concept_grid = visualize_concepts()
    ax5.imshow(concept_grid)
    ax5.axis('off')
    ax5.set_title("Concept Space: Visual Patterns for Binary Features", 
                 color='cyan', fontsize=14, fontweight='bold',
                 pad=20,
                 path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
    
    # Add explanatory text
    fig.text(0.5, 0.01, 
            "Concepts are defined by 4 binary features: color, shape, size, and style", 
            ha='center', color='white', fontsize=12)
    
    # Add a watermark
    fig.text(0.5, 0.005, 
            "Latent Space Evolution in Concept Learning", 
            ha='center', color='gray', alpha=0.5, fontsize=10)
    
    # Add a unified legend for model types and sizes
    handles = []
    labels = []
    
    # Model type legend
    for model_type, color in MODEL_COLORS.items():
        handle = plt.Line2D([0], [0], color=color, lw=4, label=model_type.upper())
        handles.append(handle)
        labels.append(model_type.upper())
    
    # Size marker legend
    for size, marker in SIZE_MARKERS.items():
        handle = plt.Line2D([0], [0], marker=marker, color='white', 
                           markersize=8, linestyle='None', label=size)
        handles.append(handle)
        labels.append(size)
    
    # Add the unified legend
    leg = fig.legend(
        handles, labels, 
        loc='lower center', 
        ncol=len(handles), 
        frameon=True, 
        facecolor='black', 
        edgecolor='cyan',
        fontsize=10,
        bbox_to_anchor=(0.5, 0.02)
    )
    
    # Final layout adjustments
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig('visualizations/concept_latent_space_evolution.png', 
               dpi=300, 
               facecolor=background_color, 
               bbox_inches='tight')
    plt.close()
    
    print("Visualization saved to visualizations/concept_latent_space_evolution.png")

def main():
    """Main function to run the visualization pipeline."""
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Generate synthetic data
    all_results = generate_synthetic_data()
    
    # Create the comprehensive visualization
    create_latent_space_visualization(all_results)

if __name__ == "__main__":
    main() 