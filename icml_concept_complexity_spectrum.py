#!/usr/bin/env python3
"""
ICML Concept Complexity Spectrum Figure

Generate a refined 16×10.5 cm figure showing the distribution of Boolean concepts
across the complexity spectrum with improved styling and tree insets.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import random
from collections import defaultdict
from pcfg import sample_concept_from_pcfg, evaluate_pcfg_concept
from concept_visualization_for_paper import expression_to_string

def generate_concept_complexity_data(n_concepts=10000, max_literals=50, max_depth=10):
    """Generate complexity data for n_concepts Boolean concepts with specified limits."""
    
    print(f"🎯 Generating {n_concepts} Boolean concepts for ICML figure...")
    
    literals_list = []
    depths_list = []
    concepts = []
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    for i in range(n_concepts):
        if i % 1000 == 0:
            print(f"  Generated {i}/{n_concepts} concepts...")
        
        # Vary parameters to get good distribution within limits
        num_features = random.choice([8, 16, 32])
        concept_max_depth = random.choice([3, 5, 7, 9, 10])
        
        expr, literals, depth = sample_concept_from_pcfg(num_features, max_depth=concept_max_depth)
        
        # Filter to keep within desired ranges
        if literals <= max_literals and depth <= max_depth:
            literals_list.append(literals)
            depths_list.append(depth)
            concepts.append((expr, literals, depth))
    
    print(f"✅ Generated {len(concepts)} valid concepts")
    print(f"   Literals range: {min(literals_list)} - {max(literals_list)}")
    print(f"   Depth range: {min(depths_list)} - {max(depths_list)}")
    
    return literals_list, depths_list, concepts

def find_target_concepts(concepts, targets):
    """Find concepts closest to specified (literals, depth) targets."""
    
    examples = {}
    
    for target_lit, target_dep in targets:
        best_concept = None
        best_distance = float('inf')
        
        for expr, literals, depth in concepts:
            distance = abs(literals - target_lit) + abs(depth - target_dep)
            if distance < best_distance:
                best_distance = distance
                best_concept = (expr, literals, depth)
        
        examples[(target_lit, target_dep)] = best_concept
        print(f"  Target ({target_lit}, {target_dep}): found ({best_concept[1]}, {best_concept[2]})")
    
    return examples

def draw_concept_tree(svg, expr, x, y, width, height, title):
    """Draw a clean binary tree representation of a concept expression."""
    
    # Create group for this tree
    tree_group = ET.SubElement(svg, "g")
    
    # Add title
    title_elem = ET.SubElement(tree_group, "text")
    title_elem.set("x", str(x + width/2))
    title_elem.set("y", str(y - 8))
    title_elem.set("class", "inset-title")
    title_elem.text = title
    
    # Add border
    border = ET.SubElement(tree_group, "rect")
    border.set("x", str(x))
    border.set("y", str(y))
    border.set("width", str(width))
    border.set("height", str(height))
    border.set("fill", "none")
    border.set("stroke", "#CCCCCC")
    border.set("stroke-width", "0.6")
    
    # Tree drawing parameters
    node_radius = 6
    level_height = height / 5
    
    def draw_node(expr, node_x, node_y, level=0, width_available=width-16):
        """Recursively draw tree nodes."""
        
        if isinstance(expr, str):
            # Leaf node (literal)
            circle = ET.SubElement(tree_group, "circle")
            circle.set("cx", str(node_x))
            circle.set("cy", str(node_y))
            circle.set("r", str(node_radius))
            circle.set("fill", "#E6F7F7")
            circle.set("stroke", "#0F9D9D")
            circle.set("stroke-width", "0.8")
            
            # Add literal text
            text = ET.SubElement(tree_group, "text")
            text.set("x", str(node_x))
            text.set("y", str(node_y + 2))
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
            circle.set("stroke-width", "0.8")
            
            # Add operator text
            text = ET.SubElement(tree_group, "text")
            text.set("x", str(node_x))
            text.set("y", str(node_y + 2))
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
                line.set("stroke-width", "0.6")
                
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
                line1.set("stroke-width", "0.6")
                
                line2 = ET.SubElement(tree_group, "line")
                line2.set("x1", str(node_x))
                line2.set("y1", str(node_y + node_radius))
                line2.set("x2", str(right_x))
                line2.set("y2", str(child_y - node_radius))
                line2.set("stroke", "#666666")
                line2.set("stroke-width", "0.6")
                
                if len(expr) > 1:
                    draw_node(expr[1], left_x, child_y, level+1, width_available/2)
                if len(expr) > 2:
                    draw_node(expr[2], right_x, child_y, level+1, width_available/2)
    
    # Start drawing from root
    root_x = x + width/2
    root_y = y + 20
    draw_node(expr, root_x, root_y)

def create_icml_concept_complexity_spectrum():
    """Create the ICML-quality concept complexity spectrum figure."""
    
    # Canvas dimensions: 16 cm × 10.5 cm
    width_cm = 16
    height_cm = 10.5
    width_px = width_cm * 96 / 2.54  # ~605px
    height_px = height_cm * 96 / 2.54  # ~396px
    
    # Colors
    teal = "#0F9D9D"
    light_gray = "#CCCCCC"
    
    # Layout parameters
    margin_left = 50
    margin_right = 120  # Space for insets
    margin_top = 40
    margin_bottom = 50
    
    plot_width = width_px - margin_left - margin_right
    plot_height = height_px - margin_top - margin_bottom
    
    # Axis ranges
    max_literals = 50
    max_depth = 10
    
    # Generate data
    literals_list, depths_list, concepts = generate_concept_complexity_data(
        n_concepts=10000, max_literals=max_literals, max_depth=max_depth
    )
    
    # Find target concepts
    targets = [(1, 1), (5, 4), (9, 7)]
    examples = find_target_concepts(concepts, targets)
    
    # Create SVG
    svg = ET.Element("svg")
    svg.set("xmlns", "http://www.w3.org/2000/svg")
    svg.set("width", f"{width_px:.1f}")
    svg.set("height", f"{height_px:.1f}")
    svg.set("viewBox", f"0 0 {width_px:.1f} {height_px:.1f}")
    
    # Add styles
    style = ET.SubElement(svg, "style")
    style.text = """
    .title { font-family: Helvetica, Arial, sans-serif; font-size: 11pt; font-weight: bold; text-anchor: middle; }
    .axis-label { font-family: Helvetica, Arial, sans-serif; font-size: 9pt; font-weight: bold; text-anchor: middle; }
    .tick-label { font-family: Helvetica, Arial, sans-serif; font-size: 8pt; text-anchor: middle; fill: #666; }
    .inset-title { font-family: Helvetica, Arial, sans-serif; font-size: 11pt; font-weight: bold; text-anchor: middle; }
    .node-text { font-family: Helvetica, Arial, sans-serif; font-size: 6pt; text-anchor: middle; }
    """
    
    # Title
    title = ET.SubElement(svg, "text")
    title.set("x", str(width_px / 2))
    title.set("y", "25")
    title.set("class", "title")
    title.text = "Concept Complexity Spectrum"
    
    # Main plot area
    plot_x = margin_left
    plot_y = margin_top
    
    # Create scales
    x_scale = plot_width / max_literals
    y_scale = plot_height / max_depth
    
    # Draw grid lines (light gray)
    for i in range(0, max_literals + 1, 10):
        grid_x = plot_x + i * x_scale
        grid_line = ET.SubElement(svg, "line")
        grid_line.set("x1", str(grid_x))
        grid_line.set("y1", str(plot_y))
        grid_line.set("x2", str(grid_x))
        grid_line.set("y2", str(plot_y + plot_height))
        grid_line.set("stroke", light_gray)
        grid_line.set("stroke-width", "0.3")
    
    for i in range(0, max_depth + 1, 2):
        grid_y = plot_y + plot_height - i * y_scale
        grid_line = ET.SubElement(svg, "line")
        grid_line.set("x1", str(plot_x))
        grid_line.set("y1", str(grid_y))
        grid_line.set("x2", str(plot_x + plot_width))
        grid_line.set("y2", str(grid_y))
        grid_line.set("stroke", light_gray)
        grid_line.set("stroke-width", "0.3")
    
    # Draw axes
    # X-axis
    x_axis = ET.SubElement(svg, "line")
    x_axis.set("x1", str(plot_x))
    x_axis.set("y1", str(plot_y + plot_height))
    x_axis.set("x2", str(plot_x + plot_width))
    x_axis.set("y2", str(plot_y + plot_height))
    x_axis.set("stroke", "black")
    x_axis.set("stroke-width", "0.8")
    
    # Y-axis
    y_axis = ET.SubElement(svg, "line")
    y_axis.set("x1", str(plot_x))
    y_axis.set("y1", str(plot_y))
    y_axis.set("x2", str(plot_x))
    y_axis.set("y2", str(plot_y + plot_height))
    y_axis.set("stroke", "black")
    y_axis.set("stroke-width", "0.8")
    
    # X-axis ticks and labels
    for i in range(0, max_literals + 1, 10):
        tick_x = plot_x + i * x_scale
        
        # Tick mark
        tick = ET.SubElement(svg, "line")
        tick.set("x1", str(tick_x))
        tick.set("y1", str(plot_y + plot_height))
        tick.set("x2", str(tick_x))
        tick.set("y2", str(plot_y + plot_height + 4))
        tick.set("stroke", "black")
        tick.set("stroke-width", "0.6")
        
        # Label
        label = ET.SubElement(svg, "text")
        label.set("x", str(tick_x))
        label.set("y", str(plot_y + plot_height + 16))
        label.set("class", "tick-label")
        label.text = str(i)
    
    # Y-axis ticks and labels
    for i in range(0, max_depth + 1, 2):
        tick_y = plot_y + plot_height - i * y_scale
        
        # Tick mark
        tick = ET.SubElement(svg, "line")
        tick.set("x1", str(plot_x - 4))
        tick.set("y1", str(tick_y))
        tick.set("x2", str(plot_x))
        tick.set("y2", str(tick_y))
        tick.set("stroke", "black")
        tick.set("stroke-width", "0.6")
        
        # Label
        label = ET.SubElement(svg, "text")
        label.set("x", str(plot_x - 12))
        label.set("y", str(tick_y + 3))
        label.set("class", "tick-label")
        label.text = str(i)
    
    # Axis labels
    x_label = ET.SubElement(svg, "text")
    x_label.set("x", str(plot_x + plot_width / 2))
    x_label.set("y", str(height_px - 15))
    x_label.set("class", "axis-label")
    x_label.text = "Number of Literals"
    
    y_label = ET.SubElement(svg, "text")
    y_label.set("x", str(20))
    y_label.set("y", str(plot_y + plot_height / 2))
    y_label.set("class", "axis-label")
    y_label.set("transform", f"rotate(-90, 20, {plot_y + plot_height / 2})")
    y_label.text = "Parse Tree Depth"
    
    # Draw scatter points with transparency
    for literals, depth in zip(literals_list, depths_list):
        if literals <= max_literals and depth <= max_depth:
            x = plot_x + literals * x_scale
            y = plot_y + plot_height - depth * y_scale
            
            circle = ET.SubElement(svg, "circle")
            circle.set("cx", str(x))
            circle.set("cy", str(y))
            circle.set("r", "1.5")
            circle.set("fill", teal)
            circle.set("stroke", "none")
            circle.set("opacity", "0.2")
    
    # Draw insets on the right
    inset_width = 80
    inset_height = 70
    inset_x = plot_x + plot_width + 25
    inset_spacing = (plot_height - 40) / 3
    
    for i, (target_lit, target_dep) in enumerate(targets):
        inset_y = plot_y + 20 + i * inset_spacing
        
        if (target_lit, target_dep) in examples:
            expr, actual_lit, actual_dep = examples[(target_lit, target_dep)]
            title = f"({actual_lit}, {actual_dep})"
            draw_concept_tree(svg, expr, inset_x, inset_y, inset_width, inset_height, title)
            
            # Draw dotted connection line to main plot
            main_x = plot_x + actual_lit * x_scale
            main_y = plot_y + plot_height - actual_dep * y_scale
            
            # Highlight the connected point
            highlight = ET.SubElement(svg, "circle")
            highlight.set("cx", str(main_x))
            highlight.set("cy", str(main_y))
            highlight.set("r", "3")
            highlight.set("fill", "none")
            highlight.set("stroke", teal)
            highlight.set("stroke-width", "1.5")
            
            # Connection line
            connection = ET.SubElement(svg, "line")
            connection.set("x1", str(main_x))
            connection.set("y1", str(main_y))
            connection.set("x2", str(inset_x))
            connection.set("y2", str(inset_y + inset_height/2))
            connection.set("stroke", "#999999")
            connection.set("stroke-width", "0.6")
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
    """Generate the ICML concept complexity spectrum figure."""
    
    print("🎨 Creating ICML Concept-Complexity Spectrum Figure...")
    
    # Create SVG
    svg = create_icml_concept_complexity_spectrum()
    
    # Save as SVG
    output_file = "icml_concept_complexity_spectrum.svg"
    save_svg(svg, output_file)
    
    print(f"✅ Saved ICML concept complexity spectrum as {output_file}")
    print("🎉 Publication-quality figure complete!")

if __name__ == "__main__":
    main() 