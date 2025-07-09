#!/usr/bin/env python3
"""
Concept-Complexity Spectrum Figure

Generate a 16×9 cm figure showing the distribution of 10,000 Boolean concepts
across the complexity spectrum with example binary trees.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import random
from collections import defaultdict
from pcfg import sample_concept_from_pcfg, evaluate_pcfg_concept
from concept_visualization_for_paper import expression_to_string

def generate_concept_complexity_data(n_concepts=10000):
    """Generate complexity data for n_concepts Boolean concepts."""
    
    print(f"🎯 Generating {n_concepts} Boolean concepts for complexity analysis...")
    
    literals_list = []
    depths_list = []
    concepts = []
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    for i in range(n_concepts):
        if i % 1000 == 0:
            print(f"  Generated {i}/{n_concepts} concepts...")
        
        # Vary parameters to get good distribution
        num_features = random.choice([8, 16, 32])
        max_depth = random.choice([3, 5, 7, 9])
        
        expr, literals, depth = sample_concept_from_pcfg(num_features, max_depth=max_depth)
        
        literals_list.append(literals)
        depths_list.append(depth)
        concepts.append((expr, literals, depth))
    
    print(f"✅ Generated {n_concepts} concepts")
    print(f"   Literals range: {min(literals_list)} - {max(literals_list)}")
    print(f"   Depth range: {min(depths_list)} - {max(depths_list)}")
    
    return literals_list, depths_list, concepts

def create_density_heatmap(literals_list, depths_list):
    """Create density heatmap data from literals and depths."""
    
    # Create 2D histogram
    max_literals = max(literals_list)
    max_depth = max(depths_list)
    
    # Create density grid
    density_grid = defaultdict(int)
    for lit, dep in zip(literals_list, depths_list):
        density_grid[(lit, dep)] += 1
    
    return density_grid, max_literals, max_depth

def find_example_concepts(concepts, target_points):
    """Find example concepts close to target (literals, depth) points."""
    
    examples = {}
    
    for target_lit, target_dep in target_points:
        best_concept = None
        best_distance = float('inf')
        
        for expr, literals, depth in concepts:
            distance = abs(literals - target_lit) + abs(depth - target_dep)
            if distance < best_distance:
                best_distance = distance
                best_concept = (expr, literals, depth)
        
        examples[(target_lit, target_dep)] = best_concept
        print(f"  Example for ({target_lit}, {target_dep}): found ({best_concept[1]}, {best_concept[2]})")
    
    return examples

def draw_binary_tree(svg, expr, x, y, width, height, title):
    """Draw a binary tree representation of a concept expression."""
    
    # Create group for this tree
    tree_group = ET.SubElement(svg, "g")
    
    # Add title
    title_elem = ET.SubElement(tree_group, "text")
    title_elem.set("x", str(x + width/2))
    title_elem.set("y", str(y - 5))
    title_elem.set("class", "tree-title")
    title_elem.text = title
    
    # Add border
    border = ET.SubElement(tree_group, "rect")
    border.set("x", str(x))
    border.set("y", str(y))
    border.set("width", str(width))
    border.set("height", str(height))
    border.set("fill", "none")
    border.set("stroke", "#E0E0E0")
    border.set("stroke-width", "0.5")
    
    # Simplified tree drawing - just show the structure
    node_radius = 8
    level_height = height / 4
    
    def draw_node(expr, node_x, node_y, level=0, width_available=width-20):
        """Recursively draw tree nodes."""
        
        if isinstance(expr, str):
            # Leaf node (literal)
            circle = ET.SubElement(tree_group, "circle")
            circle.set("cx", str(node_x))
            circle.set("cy", str(node_y))
            circle.set("r", str(node_radius))
            circle.set("fill", "#B3E5E5")
            circle.set("stroke", "#0F9D9D")
            circle.set("stroke-width", "1")
            
            # Add literal text (simplified)
            text = ET.SubElement(tree_group, "text")
            text.set("x", str(node_x))
            text.set("y", str(node_y + 3))
            text.set("class", "node-text")
            if expr.startswith('F'):
                parts = expr.split('_')
                text.text = f"x{parts[0][1:]}"
            else:
                text.text = "x"
            
            return node_x, node_y
        
        else:
            # Internal node (operator)
            op = expr[0]
            
            # Draw operator node
            circle = ET.SubElement(tree_group, "circle")
            circle.set("cx", str(node_x))
            circle.set("cy", str(node_y))
            circle.set("r", str(node_radius))
            circle.set("fill", "#0F9D9D")
            circle.set("stroke", "#0F9D9D")
            circle.set("stroke-width", "1")
            
            # Add operator text
            text = ET.SubElement(tree_group, "text")
            text.set("x", str(node_x))
            text.set("y", str(node_y + 3))
            text.set("class", "node-text")
            text.set("fill", "white")
            if op == 'AND':
                text.text = "∧"
            elif op == 'OR':
                text.text = "∨"
            elif op == 'NOT':
                text.text = "¬"
            else:
                text.text = op
            
            # Draw children
            if op == 'NOT':
                # Unary operator
                child_x = node_x
                child_y = node_y + level_height
                
                # Draw connection line
                line = ET.SubElement(tree_group, "line")
                line.set("x1", str(node_x))
                line.set("y1", str(node_y + node_radius))
                line.set("x2", str(child_x))
                line.set("y2", str(child_y - node_radius))
                line.set("stroke", "#666666")
                line.set("stroke-width", "1")
                
                draw_node(expr[1], child_x, child_y, level+1, width_available)
                
            else:
                # Binary operator
                left_x = node_x - width_available/4
                right_x = node_x + width_available/4
                child_y = node_y + level_height
                
                # Draw connection lines
                line1 = ET.SubElement(tree_group, "line")
                line1.set("x1", str(node_x))
                line1.set("y1", str(node_y + node_radius))
                line1.set("x2", str(left_x))
                line1.set("y2", str(child_y - node_radius))
                line1.set("stroke", "#666666")
                line1.set("stroke-width", "1")
                
                line2 = ET.SubElement(tree_group, "line")
                line2.set("x1", str(node_x))
                line2.set("y1", str(node_y + node_radius))
                line2.set("x2", str(right_x))
                line2.set("y2", str(child_y - node_radius))
                line2.set("stroke", "#666666")
                line2.set("stroke-width", "1")
                
                draw_node(expr[1], left_x, child_y, level+1, width_available/2)
                draw_node(expr[2], right_x, child_y, level+1, width_available/2)
    
    # Start drawing from root
    root_x = x + width/2
    root_y = y + 30
    draw_node(expr, root_x, root_y)

def create_concept_complexity_spectrum_svg():
    """Create the concept complexity spectrum figure."""
    
    # Canvas dimensions: 16 cm × 9 cm
    width_cm = 16
    height_cm = 9
    width_px = width_cm * 96 / 2.54  # ~605px
    height_px = height_cm * 96 / 2.54  # ~340px
    
    # Colors (ICML palette)
    deep_teal = "#0F9D9D"
    light_teal = "#B3E5E5"
    axis_color = "#666666"
    
    # Margins
    margin = 40
    plot_width = width_px * 0.6  # 60% for main plot
    inset_width = 75  # 2 cm in pixels
    inset_height = 60
    
    # Generate data
    literals_list, depths_list, concepts = generate_concept_complexity_data(10000)
    density_grid, max_literals, max_depth = create_density_heatmap(literals_list, depths_list)
    
    # Find example concepts
    target_points = [(2, 1), (5, 3), (9, 5)]
    examples = find_example_concepts(concepts, target_points)
    
    # Create SVG
    svg = ET.Element("svg")
    svg.set("xmlns", "http://www.w3.org/2000/svg")
    svg.set("width", f"{width_px:.1f}")
    svg.set("height", f"{height_px:.1f}")
    svg.set("viewBox", f"0 0 {width_px:.1f} {height_px:.1f}")
    
    # Add styles
    style = ET.SubElement(svg, "style")
    style.text = """
    .title { font-family: Helvetica, Arial, sans-serif; font-size: 12pt; font-weight: bold; text-anchor: middle; }
    .axis-label { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; text-anchor: middle; fill: #333; }
    .tick-label { font-family: Helvetica, Arial, sans-serif; font-size: 8pt; text-anchor: middle; fill: #666; }
    .tree-title { font-family: Helvetica, Arial, sans-serif; font-size: 8pt; text-anchor: middle; fill: #333; font-weight: bold; }
    .node-text { font-family: Helvetica, Arial, sans-serif; font-size: 6pt; text-anchor: middle; fill: #333; }
    """
    
    # Title
    title = ET.SubElement(svg, "text")
    title.set("x", str(width_px / 2))
    title.set("y", "25")
    title.set("class", "title")
    title.text = "Concept-Complexity Spectrum"
    
    # Main plot area
    plot_x = margin
    plot_y = 50
    plot_height = height_px - plot_y - margin
    
    # Create scales
    x_scale = plot_width / max_literals
    y_scale = plot_height / max_depth
    
    # Draw axes
    # X-axis
    x_axis = ET.SubElement(svg, "line")
    x_axis.set("x1", str(plot_x))
    x_axis.set("y1", str(plot_y + plot_height))
    x_axis.set("x2", str(plot_x + plot_width))
    x_axis.set("y2", str(plot_y + plot_height))
    x_axis.set("stroke", axis_color)
    x_axis.set("stroke-width", "1")
    
    # Y-axis
    y_axis = ET.SubElement(svg, "line")
    y_axis.set("x1", str(plot_x))
    y_axis.set("y1", str(plot_y))
    y_axis.set("x2", str(plot_x))
    y_axis.set("y2", str(plot_y + plot_height))
    y_axis.set("stroke", axis_color)
    y_axis.set("stroke-width", "1")
    
    # X-axis ticks and labels
    for i in range(0, max_literals + 1, 2):
        tick_x = plot_x + i * x_scale
        
        # Tick mark
        tick = ET.SubElement(svg, "line")
        tick.set("x1", str(tick_x))
        tick.set("y1", str(plot_y + plot_height))
        tick.set("x2", str(tick_x))
        tick.set("y2", str(plot_y + plot_height + 3))
        tick.set("stroke", axis_color)
        tick.set("stroke-width", "1")
        
        # Label
        label = ET.SubElement(svg, "text")
        label.set("x", str(tick_x))
        label.set("y", str(plot_y + plot_height + 15))
        label.set("class", "tick-label")
        label.text = str(i)
    
    # Y-axis ticks and labels
    for i in range(0, max_depth + 1, 1):
        tick_y = plot_y + plot_height - i * y_scale
        
        # Tick mark
        tick = ET.SubElement(svg, "line")
        tick.set("x1", str(plot_x - 3))
        tick.set("y1", str(tick_y))
        tick.set("x2", str(plot_x))
        tick.set("y2", str(tick_y))
        tick.set("stroke", axis_color)
        tick.set("stroke-width", "1")
        
        # Label
        label = ET.SubElement(svg, "text")
        label.set("x", str(plot_x - 10))
        label.set("y", str(tick_y + 3))
        label.set("class", "tick-label")
        label.text = str(i)
    
    # Axis labels
    x_label = ET.SubElement(svg, "text")
    x_label.set("x", str(plot_x + plot_width / 2))
    x_label.set("y", str(height_px - 10))
    x_label.set("class", "axis-label")
    x_label.text = "Number of Literals"
    
    y_label = ET.SubElement(svg, "text")
    y_label.set("x", str(15))
    y_label.set("y", str(plot_y + plot_height / 2))
    y_label.set("class", "axis-label")
    y_label.set("transform", f"rotate(-90, 15, {plot_y + plot_height / 2})")
    y_label.text = "Parse Tree Depth"
    
    # Draw density points
    max_density = max(density_grid.values())
    
    for (literals, depth), count in density_grid.items():
        if literals <= max_literals and depth <= max_depth:
            x = plot_x + literals * x_scale
            y = plot_y + plot_height - depth * y_scale
            
            # Calculate color intensity based on density
            intensity = count / max_density
            
            # Interpolate between light_teal and deep_teal
            if intensity < 0.1:
                color = "#E5F5F5"
                radius = 1
            elif intensity < 0.3:
                color = "#CCF0F0"
                radius = 1.5
            elif intensity < 0.6:
                color = "#99E0E0"
                radius = 2
            else:
                color = deep_teal
                radius = 2.5
            
            circle = ET.SubElement(svg, "circle")
            circle.set("cx", str(x))
            circle.set("cy", str(y))
            circle.set("r", str(radius))
            circle.set("fill", color)
            circle.set("stroke", "none")
            circle.set("opacity", str(min(1.0, 0.3 + 0.7 * intensity)))
    
    # Draw insets on the right
    inset_x = plot_x + plot_width + 20
    inset_spacing = (height_px - 80) / 3
    
    for i, (target_lit, target_dep) in enumerate(target_points):
        inset_y = 60 + i * inset_spacing
        
        if (target_lit, target_dep) in examples:
            expr, actual_lit, actual_dep = examples[(target_lit, target_dep)]
            title = f"({actual_lit}, {actual_dep})"
            draw_binary_tree(svg, expr, inset_x, inset_y, inset_width, inset_height, title)
            
            # Draw connection line to main plot
            main_x = plot_x + actual_lit * x_scale
            main_y = plot_y + plot_height - actual_dep * y_scale
            
            connection = ET.SubElement(svg, "line")
            connection.set("x1", str(main_x))
            connection.set("y1", str(main_y))
            connection.set("x2", str(inset_x))
            connection.set("y2", str(inset_y + inset_height/2))
            connection.set("stroke", "#CCCCCC")
            connection.set("stroke-width", "0.5")
            connection.set("stroke-dasharray", "2,2")
    
    return svg

def save_svg(svg_element, filename):
    """Save SVG element to file with proper formatting."""
    rough_string = ET.tostring(svg_element, 'unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Remove empty lines and fix formatting
    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    formatted_xml = '\n'.join(lines)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(formatted_xml)

def main():
    """Generate the concept complexity spectrum figure."""
    
    print("🎨 Creating Concept-Complexity Spectrum Figure...")
    
    # Create SVG
    svg = create_concept_complexity_spectrum_svg()
    
    # Save as SVG
    output_file = "concept_complexity_spectrum.svg"
    save_svg(svg, output_file)
    
    print(f"✅ Saved concept complexity spectrum as {output_file}")
    print("🎉 Figure complete!")

if __name__ == "__main__":
    main() 