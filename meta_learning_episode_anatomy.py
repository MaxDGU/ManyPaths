#!/usr/bin/env python3
"""
Anatomy of a Meta-Learning Episode Figure

Generate a clean ICML-style figure showing the structure of a single task
in the ManyPaths meta-learning dataset.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import random
from pcfg import sample_concept_from_pcfg, evaluate_pcfg_concept
from concept_visualization_for_paper import expression_to_string

def generate_episode_data():
    """Generate a sample meta-learning episode with Boolean concept and examples."""
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate a Boolean concept
    num_features = 8
    max_depth = 4
    expr, literals, depth = sample_concept_from_pcfg(num_features, max_depth=max_depth)
    
    # Generate all possible 8-bit inputs
    all_inputs = []
    all_labels = []
    
    for i in range(2**num_features):
        binary_input = [(i >> j) & 1 for j in range(num_features)]
        label = evaluate_pcfg_concept(expr, binary_input)
        all_inputs.append(binary_input)
        all_labels.append(label)
    
    # Sample support and query examples
    positive_indices = [i for i, label in enumerate(all_labels) if label]
    negative_indices = [i for i, label in enumerate(all_labels) if not label]
    
    # Sample support set (2 positive, 1 negative)
    support_pos_idx = random.sample(positive_indices, min(2, len(positive_indices)))
    support_neg_idx = random.sample(negative_indices, min(1, len(negative_indices)))
    
    # Sample query set (1 positive, 1 negative) 
    remaining_pos = [i for i in positive_indices if i not in support_pos_idx]
    remaining_neg = [i for i in negative_indices if i not in support_neg_idx]
    
    query_pos_idx = random.sample(remaining_pos, min(1, len(remaining_pos)))
    query_neg_idx = random.sample(remaining_neg, min(1, len(remaining_neg)))
    
    # Collect examples
    support_examples = []
    for idx in support_pos_idx:
        support_examples.append((all_inputs[idx], True, 'S'))
    for idx in support_neg_idx:
        support_examples.append((all_inputs[idx], False, 'S'))
        
    query_examples = []
    for idx in query_pos_idx:
        query_examples.append((all_inputs[idx], True, 'Q'))
    for idx in query_neg_idx:
        query_examples.append((all_inputs[idx], False, 'Q'))
    
    # Create expression string
    expr_string = expression_to_string(expr)
    
    return expr, expr_string, support_examples, query_examples

def draw_binary_tree(svg, expr, x, y, width, height):
    """Draw a clean binary tree representation."""
    
    tree_group = ET.SubElement(svg, "g")
    
    node_radius = 12
    level_height = height / 4
    
    def draw_node(expr, node_x, node_y, level=0, width_available=width-20):
        """Recursively draw tree nodes."""
        
        if isinstance(expr, str):
            # Leaf node (variable)
            circle = ET.SubElement(tree_group, "circle")
            circle.set("cx", str(node_x))
            circle.set("cy", str(node_y))
            circle.set("r", str(node_radius))
            circle.set("fill", "white")
            circle.set("stroke", "#333333")
            circle.set("stroke-width", "0.5")
            
            text = ET.SubElement(tree_group, "text")
            text.set("x", str(node_x))
            text.set("y", str(node_y + 3))
            text.set("class", "node-text")
            if expr.startswith('F'):
                parts = expr.split('_')
                text.text = f"x{parts[0][1:]}"
            else:
                text.text = "x₀"
            
        else:
            # Internal node (operator)
            op = expr[0]
            
            # Draw operator node
            rect = ET.SubElement(tree_group, "rect")
            rect.set("x", str(node_x - node_radius))
            rect.set("y", str(node_y - node_radius/2))
            rect.set("width", str(node_radius * 2))
            rect.set("height", str(node_radius))
            rect.set("fill", "#0F9D9D")
            rect.set("stroke", "#333333")
            rect.set("stroke-width", "0.5")
            rect.set("rx", "3")
            
            # Add operator text
            text = ET.SubElement(tree_group, "text")
            text.set("x", str(node_x))
            text.set("y", str(node_y + 3))
            text.set("class", "node-text")
            text.set("fill", "white")
            if op == 'AND':
                text.text = "AND"
            elif op == 'OR':
                text.text = "OR"
            elif op == 'NOT':
                text.text = "NOT"
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
                line.set("y1", str(node_y + node_radius/2))
                line.set("x2", str(child_x))
                line.set("y2", str(child_y - node_radius))
                line.set("stroke", "#333333")
                line.set("stroke-width", "0.5")
                
                draw_node(expr[1], child_x, child_y, level+1, width_available)
                
            else:
                # Binary operator
                left_x = node_x - width_available/4
                right_x = node_x + width_available/4
                child_y = node_y + level_height
                
                # Draw connection lines
                line1 = ET.SubElement(tree_group, "line")
                line1.set("x1", str(node_x))
                line1.set("y1", str(node_y + node_radius/2))
                line1.set("x2", str(left_x))
                line1.set("y2", str(child_y - node_radius))
                line1.set("stroke", "#333333")
                line1.set("stroke-width", "0.5")
                
                line2 = ET.SubElement(tree_group, "line")
                line2.set("x1", str(node_x))
                line2.set("y1", str(node_y + node_radius/2))
                line2.set("x2", str(right_x))
                line2.set("y2", str(child_y - node_radius))
                line2.set("stroke", "#333333")
                line2.set("stroke-width", "0.5")
                
                if len(expr) > 1:
                    draw_node(expr[1], left_x, child_y, level+1, width_available/2)
                if len(expr) > 2:
                    draw_node(expr[2], right_x, child_y, level+1, width_available/2)
    
    # Start drawing from root
    root_x = x + width/2
    root_y = y + 40
    draw_node(expr, root_x, root_y)

def create_meta_learning_episode_figure():
    """Create the meta-learning episode anatomy figure."""
    
    # Canvas dimensions: 15 cm × 9 cm (increased height for better spacing)
    width_cm = 15
    height_cm = 9
    width_px = width_cm * 96 / 2.54  # ~567px
    height_px = height_cm * 96 / 2.54  # ~340px
    
    # Colors
    teal = "#0F9D9D"
    light_gray = "#DDDDDD"
    text_gray = "#333333"
    
    # Layout
    padding = 19  # 0.5cm
    panel_width = (width_px - 3*padding) / 2
    panel_height = height_px - 3*padding
    
    # Generate episode data
    expr, expr_string, support_examples, query_examples = generate_episode_data()
    
    # Create SVG
    svg = ET.Element("svg")
    svg.set("xmlns", "http://www.w3.org/2000/svg")
    svg.set("width", f"{width_px:.1f}")
    svg.set("height", f"{height_px:.1f}")
    svg.set("viewBox", f"0 0 {width_px:.1f} {height_px:.1f}")
    
    # Add styles
    style = ET.SubElement(svg, "style")
    style.text = """
    .main-title { font-family: Helvetica, Arial, sans-serif; font-size: 11pt; font-weight: bold; text-anchor: middle; }
    .panel-title { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; font-weight: bold; text-anchor: middle; }
    .label-text { font-family: Helvetica, Arial, sans-serif; font-size: 8pt; text-anchor: middle; fill: #333; }
    .node-text { font-family: Helvetica, Arial, sans-serif; font-size: 8pt; text-anchor: middle; fill: #333; }
    .example-text { font-family: Helvetica, Arial, sans-serif; font-size: 8pt; text-anchor: middle; }
    .caption-text { font-family: Helvetica, Arial, sans-serif; font-size: 8pt; text-anchor: middle; fill: #666; }
    """
    
    # Main title
    main_title = ET.SubElement(svg, "text")
    main_title.set("x", str(width_px / 2))
    main_title.set("y", str(padding - 5))
    main_title.set("class", "main-title")
    main_title.text = "Anatomy of a Meta-Learning Episode"
    
    # Left panel: Boolean Concept
    left_panel_x = padding
    left_panel_y = padding + 15
    
    # Left panel title
    left_title = ET.SubElement(svg, "text")
    left_title.set("x", str(left_panel_x + panel_width/2))
    left_title.set("y", str(left_panel_y))
    left_title.set("class", "panel-title")
    left_title.text = "Boolean Concept"
    
    # Draw binary tree - reduced height to avoid overlap with caption
    tree_y = left_panel_y + 20
    tree_height = panel_height - 110  # Increased margin for caption
    draw_binary_tree(svg, expr, left_panel_x, tree_y, panel_width, tree_height)
    
    # Caption under tree - moved further down to avoid overlap
    caption_y = left_panel_y + panel_height - 30
    caption1 = ET.SubElement(svg, "text")
    caption1.set("x", str(left_panel_x + panel_width/2))
    caption1.set("y", str(caption_y))
    caption1.set("class", "caption-text")
    caption1.text = "Sampled from PCFG Grammar"
    
    # Split long expression into multiple lines if needed
    caption2 = ET.SubElement(svg, "text")
    caption2.set("x", str(left_panel_x + panel_width/2))
    caption2.set("y", str(caption_y + 15))
    caption2.set("class", "caption-text")
    # Show full expression without truncation
    if len(expr_string) > 35:
        # Split into two lines for very long expressions
        mid_point = len(expr_string) // 2
        # Find a good break point near the middle
        break_point = mid_point
        for i in range(mid_point - 5, mid_point + 5):
            if i < len(expr_string) and expr_string[i] in ['∧', '∨', ')', '(']:
                break_point = i + 1
                break
        
        line1 = expr_string[:break_point].strip()
        line2 = expr_string[break_point:].strip()
        
        caption2.text = f"({line1}"
        
        caption3 = ET.SubElement(svg, "text")
        caption3.set("x", str(left_panel_x + panel_width/2))
        caption3.set("y", str(caption_y + 30))
        caption3.set("class", "caption-text")
        caption3.text = f"{line2})"
    else:
        caption2.text = f"({expr_string})"
    
    # Vertical divider
    divider_x = left_panel_x + panel_width + padding/2
    divider = ET.SubElement(svg, "line")
    divider.set("x1", str(divider_x))
    divider.set("y1", str(left_panel_y))
    divider.set("x2", str(divider_x))
    divider.set("y2", str(left_panel_y + panel_height))
    divider.set("stroke", "#CCCCCC")
    divider.set("stroke-width", "0.3")
    
    # Right panel: Support & Query Examples
    right_panel_x = divider_x + padding/2
    right_panel_y = left_panel_y
    
    # Right panel title
    right_title = ET.SubElement(svg, "text")
    right_title.set("x", str(right_panel_x + panel_width/2))
    right_title.set("y", str(right_panel_y))
    right_title.set("class", "panel-title")
    right_title.text = "Support & Query Examples"
    
    # Examples grid
    all_examples = support_examples + query_examples
    example_width = 120
    example_height = 25
    grid_start_y = right_panel_y + 30
    
    for i, (input_vec, is_positive, example_type) in enumerate(all_examples):
        row = i // 2
        col = i % 2
        
        ex_x = right_panel_x + col * (example_width + 15)
        ex_y = grid_start_y + row * (example_height + 15)
        
        # Example rectangle
        color = teal if is_positive else light_gray
        text_color = "white" if is_positive else "#333333"
        
        rect = ET.SubElement(svg, "rect")
        rect.set("x", str(ex_x))
        rect.set("y", str(ex_y))
        rect.set("width", str(example_width))
        rect.set("height", str(example_height))
        rect.set("fill", color)
        rect.set("stroke", "#333333")
        rect.set("stroke-width", "0.5")
        rect.set("rx", "3")
        
        # Binary string
        binary_str = " ".join(map(str, input_vec))
        text = ET.SubElement(svg, "text")
        text.set("x", str(ex_x + example_width/2))
        text.set("y", str(ex_y + example_height/2 + 3))
        text.set("class", "example-text")
        text.set("fill", text_color)
        text.text = binary_str
        
        # S/Q label above
        label = ET.SubElement(svg, "text")
        label.set("x", str(ex_x + example_width/2))
        label.set("y", str(ex_y - 5))
        label.set("class", "label-text")
        label.text = example_type
    
    # Legend - positioned properly to avoid overlap
    legend_x = right_panel_x + 10
    legend_y = grid_start_y + (len(all_examples)//2 + 1) * (example_height + 15) + 20
    
    # Positive legend
    pos_rect = ET.SubElement(svg, "rect")
    pos_rect.set("x", str(legend_x))
    pos_rect.set("y", str(legend_y))
    pos_rect.set("width", "15")
    pos_rect.set("height", "12")
    pos_rect.set("fill", teal)
    pos_rect.set("stroke", "#333333")
    pos_rect.set("stroke-width", "0.5")
    
    pos_label = ET.SubElement(svg, "text")
    pos_label.set("x", str(legend_x + 20))
    pos_label.set("y", str(legend_y + 9))
    pos_label.set("class", "label-text")
    pos_label.set("text-anchor", "start")
    pos_label.text = "Positive"
    
    # Negative legend
    neg_rect = ET.SubElement(svg, "rect")
    neg_rect.set("x", str(legend_x))
    neg_rect.set("y", str(legend_y + 18))
    neg_rect.set("width", "15")
    neg_rect.set("height", "12")
    neg_rect.set("fill", light_gray)
    neg_rect.set("stroke", "#333333")
    neg_rect.set("stroke-width", "0.5")
    
    neg_label = ET.SubElement(svg, "text")
    neg_label.set("x", str(legend_x + 20))
    neg_label.set("y", str(legend_y + 27))
    neg_label.set("class", "label-text")
    neg_label.set("text-anchor", "start")
    neg_label.text = "Negative"
    
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
    """Generate the meta-learning episode anatomy figure."""
    
    print("🎨 Creating Meta-Learning Episode Anatomy Figure...")
    
    # Create SVG
    svg = create_meta_learning_episode_figure()
    
    # Save as SVG
    output_file = "meta_learning_episode_anatomy.svg"
    save_svg(svg, output_file)
    
    print(f"✅ Saved anatomy figure as {output_file}")
    print("🎉 Figure complete!")

if __name__ == "__main__":
    main() 