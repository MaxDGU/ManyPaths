#!/usr/bin/env python3
"""
ICML Publication-Quality Nested Optimization Diagram

Generate the "Nested Optimisation in ManyPaths" diagram as SVG
according to precise ICML formatting specifications.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_icml_nested_optimization_svg():
    """Create the ICML-quality nested optimization diagram as SVG."""
    
    # Canvas dimensions: 16 cm × 11 cm (convert to pixels at 96 DPI)
    width_cm = 16
    height_cm = 11
    width_px = width_cm * 96 / 2.54  # ~605px
    height_px = height_cm * 96 / 2.54  # ~417px
    
    # Colors
    accent_1 = "#0F9D9D"  # Teal
    accent_2 = "#BFBFBF"  # Mid-grey
    black = "#000000"
    grey_90 = "#1A1A1A"  # 90% grey for labels
    
    # Margins
    margin = 0.7 * 96 / 2.54  # 0.7 cm in pixels
    
    # Create SVG root
    svg = ET.Element("svg")
    svg.set("xmlns", "http://www.w3.org/2000/svg")
    svg.set("width", f"{width_px:.1f}")
    svg.set("height", f"{height_px:.1f}")
    svg.set("viewBox", f"0 0 {width_px:.1f} {height_px:.1f}")
    
    # Add styles
    style = ET.SubElement(svg, "style")
    style.text = """
    .title { font-family: Arial, Helvetica, sans-serif; font-size: 11pt; font-weight: bold; text-anchor: middle; }
    .subtitle { font-family: Arial, Helvetica, sans-serif; font-size: 9pt; text-anchor: middle; fill: #1A1A1A; }
    .label-8pt { font-family: Arial, Helvetica, sans-serif; font-size: 8pt; text-anchor: middle; fill: #1A1A1A; }
    .label-9pt { font-family: Arial, Helvetica, sans-serif; font-size: 9pt; text-anchor: middle; }
    .annotation { font-family: Arial, Helvetica, sans-serif; font-size: 8pt; fill: #1A1A1A; }
    .italic { font-style: italic; }
    .bold { font-weight: bold; }
    """
    
    # Complexity data
    complexity_data = [
        {"config": "F8D3", "literals": 2.8, "depth": 2.8, "label": "Simple"},
        {"config": "F16D3", "literals": 2.7, "depth": 2.8, "label": "Simple+"},
        {"config": "F8D5", "literals": 4.8, "depth": 3.8, "label": "Medium"},
        {"config": "F16D5", "literals": 4.7, "depth": 3.9, "label": "Medium+"},
        {"config": "F8D7", "literals": 7.4, "depth": 4.8, "label": "Complex"}
    ]
    
    # 1. Title & Subtitle
    title_y = margin + 20
    title = ET.SubElement(svg, "text")
    title.set("x", str(width_px / 2))
    title.set("y", str(title_y))
    title.set("class", "title")
    title.text = "Nested Optimisation in ManyPaths"
    
    subtitle_y = title_y + 20
    subtitle = ET.SubElement(svg, "text")
    subtitle.set("x", str(width_px / 2))
    subtitle.set("y", str(subtitle_y))
    subtitle.set("class", "subtitle")
    subtitle.text = "PCFG-Generated Boolean Concepts → Meta-Learning Pipeline"
    
    # 2. Curriculum Bars (Top Row)
    bars_start_y = subtitle_y + 40
    bars_base_y = bars_start_y + 80
    bar_width = 40
    bar_spacing = (width_px - 2 * margin - 5 * bar_width) / 4
    max_bar_height = 60
    
    # Heights proportional to complexity (F8D7 is tallest)
    heights = [30, 35, 45, 50, 60]  # Incremental heights
    
    for i, (data, height) in enumerate(zip(complexity_data, heights)):
        x = margin + i * (bar_width + bar_spacing)
        y = bars_base_y - height
        
        # Main bar (filled with Accent-2)
        bar = ET.SubElement(svg, "rect")
        bar.set("x", str(x))
        bar.set("y", str(y))
        bar.set("width", str(bar_width))
        bar.set("height", str(height))
        bar.set("fill", accent_2)
        bar.set("stroke", "none")
        bar.set("rx", "2")
        
        # Accent stripe (left edge, 8% of bar width)
        stripe_width = bar_width * 0.08
        stripe = ET.SubElement(svg, "rect")
        stripe.set("x", str(x))
        stripe.set("y", str(y))
        stripe.set("width", str(stripe_width))
        stripe.set("height", str(height))
        stripe.set("fill", accent_1)
        stripe.set("stroke", "none")
        stripe.set("rx", "2")
        
        # Configuration label above bar
        config_text = ET.SubElement(svg, "text")
        config_text.set("x", str(x + bar_width / 2))
        config_text.set("y", str(y - 25))
        config_text.set("class", "label-8pt")
        config_text.text = data["config"]
        
        # Literals/depth in parentheses
        complexity_text = ET.SubElement(svg, "text")
        complexity_text.set("x", str(x + bar_width / 2))
        complexity_text.set("y", str(y - 10))
        complexity_text.set("class", "label-8pt")
        complexity_text.text = f"({data['literals']:.1f}, {data['depth']:.1f})"
        
        # Complexity label below bar
        label_text = ET.SubElement(svg, "text")
        label_text.set("x", str(x + bar_width / 2))
        label_text.set("y", str(bars_base_y + 15))
        label_text.set("class", "label-8pt")
        label_text.text = data["label"]
    
    # 3. Flow Arrows & Inner Loop Box
    inner_loop_y = bars_base_y + 50
    inner_loop_width = 200
    inner_loop_height = 80
    inner_loop_x = (width_px - inner_loop_width) / 2
    
    # Draw curved arrows from bars to inner loop
    for i in range(5):
        bar_x = margin + i * (bar_width + bar_spacing) + bar_width / 2
        bar_top_y = bars_base_y - heights[i]
        
        # Curved path to inner loop center
        path = ET.SubElement(svg, "path")
        control_y = (bar_top_y + inner_loop_y) / 2
        path_d = f"M {bar_x} {bar_top_y} Q {bar_x} {control_y} {inner_loop_x + inner_loop_width/2} {inner_loop_y}"
        path.set("d", path_d)
        path.set("fill", "none")
        path.set("stroke", grey_90)
        path.set("stroke-width", "0.4")
        path.set("marker-end", "url(#arrowhead)")
    
    # Arrow marker definition
    defs = ET.SubElement(svg, "defs")
    marker = ET.SubElement(defs, "marker")
    marker.set("id", "arrowhead")
    marker.set("markerWidth", "10")
    marker.set("markerHeight", "7")
    marker.set("refX", "9")
    marker.set("refY", "3.5")
    marker.set("orient", "auto")
    
    arrow_path = ET.SubElement(marker, "path")
    arrow_path.set("d", "M 0 0 L 10 3.5 L 0 7 z")
    arrow_path.set("fill", grey_90)
    
    # Task Distribution label
    task_label = ET.SubElement(svg, "text")
    task_label.set("x", str(width_px / 2 - 50))
    task_label.set("y", str((bars_base_y + inner_loop_y) / 2 + 10))
    task_label.set("class", "annotation italic")
    task_label.text = "Task Distribution"
    
    # 4. Inner Loop Box
    inner_box = ET.SubElement(svg, "rect")
    inner_box.set("x", str(inner_loop_x))
    inner_box.set("y", str(inner_loop_y))
    inner_box.set("width", str(inner_loop_width))
    inner_box.set("height", str(inner_loop_height))
    inner_box.set("fill", "none")
    inner_box.set("stroke", accent_1)
    inner_box.set("stroke-width", "0.6")
    inner_box.set("rx", "8")
    
    # MLP Icon (3 layers: 3→3→1 circles)
    mlp_center_x = inner_loop_x + inner_loop_width / 2
    mlp_center_y = inner_loop_y + inner_loop_height / 2
    
    # Layer positions
    layer_1_x = mlp_center_x - 40
    layer_2_x = mlp_center_x
    layer_3_x = mlp_center_x + 40
    
    # Layer 1 (3 circles)
    for i in range(3):
        circle = ET.SubElement(svg, "circle")
        circle.set("cx", str(layer_1_x))
        circle.set("cy", str(mlp_center_y - 15 + i * 15))
        circle.set("r", "4")
        circle.set("fill", "none")
        circle.set("stroke", accent_1)
        circle.set("stroke-width", "0.6")
    
    # Layer 2 (3 circles)
    for i in range(3):
        circle = ET.SubElement(svg, "circle")
        circle.set("cx", str(layer_2_x))
        circle.set("cy", str(mlp_center_y - 15 + i * 15))
        circle.set("r", "4")
        circle.set("fill", "none")
        circle.set("stroke", accent_1)
        circle.set("stroke-width", "0.6")
    
    # Layer 3 (1 circle)
    circle = ET.SubElement(svg, "circle")
    circle.set("cx", str(layer_3_x))
    circle.set("cy", str(mlp_center_y))
    circle.set("r", "4")
    circle.set("fill", "none")
    circle.set("stroke", accent_1)
    circle.set("stroke-width", "0.6")
    
    # Connection lines between layers
    for i in range(3):
        for j in range(3):
            line = ET.SubElement(svg, "line")
            line.set("x1", str(layer_1_x + 4))
            line.set("y1", str(mlp_center_y - 15 + i * 15))
            line.set("x2", str(layer_2_x - 4))
            line.set("y2", str(mlp_center_y - 15 + j * 15))
            line.set("stroke", grey_90)
            line.set("stroke-width", "0.2")
            line.set("opacity", "0.3")
    
    for i in range(3):
        line = ET.SubElement(svg, "line")
        line.set("x1", str(layer_2_x + 4))
        line.set("y1", str(mlp_center_y - 15 + i * 15))
        line.set("x2", str(layer_3_x - 4))
        line.set("y2", str(mlp_center_y))
        line.set("stroke", grey_90)
        line.set("stroke-width", "0.2")
        line.set("opacity", "0.3")
    
    # MLP caption
    mlp_caption = ET.SubElement(svg, "text")
    mlp_caption.set("x", str(mlp_center_x))
    mlp_caption.set("y", str(inner_loop_y + inner_loop_height + 15))
    mlp_caption.set("class", "label-8pt")
    mlp_caption.text = "MLP"
    
    # Annotation left of inner loop
    adapt_annotation = ET.SubElement(svg, "text")
    adapt_annotation.set("x", str(inner_loop_x - 10))
    adapt_annotation.set("y", str(mlp_center_y))
    adapt_annotation.set("class", "annotation")
    adapt_annotation.set("text-anchor", "end")
    adapt_annotation.text = "Adapt MLP parameters for each concept"
    
    # 5. Downstream Arrow
    meta_opt_y = inner_loop_y + inner_loop_height + 40
    
    arrow_line = ET.SubElement(svg, "line")
    arrow_line.set("x1", str(mlp_center_x))
    arrow_line.set("y1", str(inner_loop_y + inner_loop_height))
    arrow_line.set("x2", str(mlp_center_x))
    arrow_line.set("y2", str(meta_opt_y))
    arrow_line.set("stroke", grey_90)
    arrow_line.set("stroke-width", "0.4")
    arrow_line.set("marker-end", "url(#arrowhead)")
    
    # Gradients label
    gradients_label = ET.SubElement(svg, "text")
    gradients_label.set("x", str(mlp_center_x + 20))
    gradients_label.set("y", str((inner_loop_y + inner_loop_height + meta_opt_y) / 2))
    gradients_label.set("class", "annotation italic")
    gradients_label.text = "Gradients"
    
    # 6. Meta-Optimizer Block
    meta_width = width_px * 0.8
    meta_height = 40
    meta_x = (width_px - meta_width) / 2
    
    meta_box = ET.SubElement(svg, "rect")
    meta_box.set("x", str(meta_x))
    meta_box.set("y", str(meta_opt_y))
    meta_box.set("width", str(meta_width))
    meta_box.set("height", str(meta_height))
    meta_box.set("fill", accent_1)
    meta_box.set("fill-opacity", "0.08")
    meta_box.set("stroke", accent_1)
    meta_box.set("stroke-width", "0.6")
    meta_box.set("rx", "8")
    
    # Meta-optimizer label
    meta_label = ET.SubElement(svg, "text")
    meta_label.set("x", str(meta_x + meta_width / 2))
    meta_label.set("y", str(meta_opt_y + meta_height / 2 + 3))
    meta_label.set("class", "label-9pt bold")
    meta_label.text = "Meta-Optimizer (MetaSGD)"
    
    # Update annotation under meta-optimizer
    update_annotation = ET.SubElement(svg, "text")
    update_annotation.set("x", str(meta_x))
    update_annotation.set("y", str(meta_opt_y + meta_height + 15))
    update_annotation.set("class", "annotation")
    update_annotation.text = "Update meta-parameters using gradients"
    
    # 7. Step Labels (Unicode circled numbers)
    step_x = margin - 20
    
    # Step 1
    step1 = ET.SubElement(svg, "text")
    step1.set("x", str(step_x))
    step1.set("y", str(bars_start_y + 30))
    step1.set("class", "label-9pt")
    step1.set("fill", accent_1)
    step1.text = "①"
    
    # Step 2
    step2 = ET.SubElement(svg, "text")
    step2.set("x", str(step_x))
    step2.set("y", str(inner_loop_y + inner_loop_height / 2))
    step2.set("class", "label-9pt")
    step2.set("fill", accent_1)
    step2.text = "②"
    
    # Step 3
    step3 = ET.SubElement(svg, "text")
    step3.set("x", str(step_x))
    step3.set("y", str(meta_opt_y + meta_height / 2))
    step3.set("class", "label-9pt")
    step3.set("fill", accent_1)
    step3.text = "③"
    
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
    """Generate the ICML nested optimization diagram."""
    
    print("🎨 Creating ICML Publication-Quality Nested Optimization Diagram...")
    
    # Create SVG
    svg = create_icml_nested_optimization_svg()
    
    # Save as SVG
    output_file = "icml_nested_optimization.svg"
    save_svg(svg, output_file)
    
    print(f"✅ Saved ICML diagram as {output_file}")
    print("🎉 Publication-quality diagram complete!")

if __name__ == "__main__":
    main() 