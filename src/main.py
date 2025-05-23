import os
import sys
from datetime import datetime
import shutil
import plotly.graph_objects as go
import numpy as np
import colorsys
# Change absolute imports to relative imports
from seating_manager import SeatingManager
from genetic_algorithm import GeneticAlgorithm
from visualizer import Visualizer
# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.weights as weights

def setup_output_directory():
    """Create output directory with timestamp if it doesn't exist."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(weights.OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def create_3d_seating_visualization(seating_manager, solution_mapping, output_dir):
    """
    Create an interactive 3D network visualization of the seating arrangement.
    
    Args:
        seating_manager: SeatingManager instance with guest and table information
        solution_mapping: Dictionary mapping guest IDs to table IDs
        output_dir: Directory to save the visualization
    """
    # Create reverse mapping (table -> list of guests)
    table_guests = {}
    for guest_id, table_id in solution_mapping.items():
        if table_id not in table_guests:
            table_guests[table_id] = []
        table_guests[table_id].append(guest_id)
    
    # Initialize lists for node positions and edge data
    nodes_x, nodes_y, nodes_z = [], [], []
    node_labels, node_colors, node_sizes = [], [], []
    
    # Separate lists for different types of edges
    table_connections = {'x': [], 'y': [], 'z': []}
    positive_relationships = {'x': [], 'y': [], 'z': [], 'width': []}
    negative_relationships = {'x': [], 'y': [], 'z': [], 'width': []}
    
    # Calculate table positions in a circle
    num_tables = len(seating_manager.tables)
    table_radius = 5
    table_angles = np.linspace(0, 2*np.pi, num_tables, endpoint=False)
    table_positions = {
        i: (table_radius * np.cos(angle), table_radius * np.sin(angle), 0)
        for i, angle in enumerate(table_angles)
    }
    
    # Generate distinct colors for each table using HSV color space
    table_colors = {}
    for i in range(num_tables):
        hue = i / num_tables
        # Convert HSV to RGB (using full saturation and value)
        rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(hue, 0.8, 0.8))
        table_colors[i] = f'rgb{rgb}'
    
    # Add table nodes
    for table_id, pos in table_positions.items():
        nodes_x.append(pos[0])
        nodes_y.append(pos[1])
        nodes_z.append(pos[2])
        node_labels.append(f"Table {table_id}")
        node_colors.append(table_colors[table_id])
        node_sizes.append(30)  # Larger size for tables
    
    # Calculate different heights for each table's guests
    base_height = 2
    height_increment = 1.5
    
    # Add guest nodes and relationship edges
    for table_id, guests in table_guests.items():
        table_pos = table_positions[table_id]
        num_guests = len(guests)
        
        # Assign specific height for this table's guests
        guest_height = base_height + (table_id * height_increment)
        guest_radius = 2
        guest_angles = np.linspace(0, 2*np.pi, num_guests, endpoint=False)
        
        # Add guest nodes
        for i, guest_id in enumerate(guests):
            angle = guest_angles[i]
            guest_x = table_pos[0] + guest_radius * np.cos(angle)
            guest_y = table_pos[1] + guest_radius * np.sin(angle)
            guest_z = guest_height
            
            # Add guest node with table's color
            nodes_x.append(guest_x)
            nodes_y.append(guest_y)
            nodes_z.append(guest_z)
            node_labels.append(f"Guest {guest_id}")
            node_colors.append(table_colors[table_id])
            node_sizes.append(15)  # Smaller size for guests
            
            # Connect guest to table
            table_connections['x'].extend([guest_x, table_pos[0], None])
            table_connections['y'].extend([guest_y, table_pos[1], None])
            table_connections['z'].extend([guest_z, table_pos[2], None])
            
            # Add relationships between guests at the same table
            guest_obj = seating_manager.guests[guest_id]
            for j, other_guest_id in enumerate(guests[i+1:], i+1):
                relationship = guest_obj.get_relationship(other_guest_id)
                if relationship != 0:  # Only show non-zero relationships
                    other_angle = guest_angles[j]
                    other_x = table_pos[0] + guest_radius * np.cos(other_angle)
                    other_y = table_pos[1] + guest_radius * np.sin(other_angle)
                    other_z = guest_height
                    
                    # Add relationship edge to appropriate list
                    if relationship > 0:
                        positive_relationships['x'].extend([guest_x, other_x, None])
                        positive_relationships['y'].extend([guest_y, other_y, None])
                        positive_relationships['z'].extend([guest_z, other_z, None])
                        # Scale width directly with relationship strength, but with smaller factor
                        line_width = abs(relationship) / 20  # Reduced scaling factor
                        positive_relationships['width'].append(line_width)
                    else:
                        negative_relationships['x'].extend([guest_x, other_x, None])
                        negative_relationships['y'].extend([guest_y, other_y, None])
                        negative_relationships['z'].extend([guest_z, other_z, None])
                        # Scale width directly with relationship strength, but with smaller factor
                        line_width = abs(relationship) / 20  # Reduced scaling factor
                        negative_relationships['width'].append(line_width)
    
    # Create the 3D network visualization
    fig = go.Figure()
    
    # Add table connection edges (using table colors)
    for table_id in range(num_tables):
        table_pos = table_positions[table_id]
        guests = table_guests.get(table_id, [])
        if guests:
            guest_coords = []
            for guest_id in guests:
                idx = node_labels.index(f"Guest {guest_id}")
                guest_coords.append((nodes_x[idx], nodes_y[idx], nodes_z[idx]))
            
            for guest_pos in guest_coords:
                fig.add_trace(go.Scatter3d(
                    x=[table_pos[0], guest_pos[0]],
                    y=[table_pos[1], guest_pos[1]],
                    z=[table_pos[2], guest_pos[2]],
                    mode='lines',
                    line=dict(color=table_colors[table_id], width=1),
                    hoverinfo='none',
                    showlegend=False
                ))
    
    # Add positive relationship edges
    if positive_relationships['x']:
        # Create individual traces for each relationship line
        for i in range(0, len(positive_relationships['x']), 3):  # Step by 3 because each line segment has 3 points (start, end, None)
            fig.add_trace(go.Scatter3d(
                x=positive_relationships['x'][i:i+3],
                y=positive_relationships['y'][i:i+3],
                z=positive_relationships['z'][i:i+3],
                mode='lines',
                line=dict(
                    color='rgba(0, 255, 0, 0.5)',
                    width=positive_relationships['width'][i//3]  # Use the width for this specific line
                ),
                hoverinfo='none',
                showlegend=True if i == 0 else False,  # Only show in legend once
                name='Positive Relationships'
            ))
    
    # Add negative relationship edges
    if negative_relationships['x']:
        # Create individual traces for each relationship line
        for i in range(0, len(negative_relationships['x']), 3):  # Step by 3 because each line segment has 3 points (start, end, None)
            fig.add_trace(go.Scatter3d(
                x=negative_relationships['x'][i:i+3],
                y=negative_relationships['y'][i:i+3],
                z=negative_relationships['z'][i:i+3],
                mode='lines',
                line=dict(
                    color='rgba(255, 0, 0, 0.5)',
                    width=negative_relationships['width'][i//3]  # Use the width for this specific line
                ),
                hoverinfo='none',
                showlegend=True if i == 0 else False,  # Only show in legend once
                name='Negative Relationships'
            ))
    
    # Add nodes (tables and guests)
    fig.add_trace(go.Scatter3d(
        x=nodes_x,
        y=nodes_y,
        z=nodes_z,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            symbol='circle'
        ),
        text=node_labels,
        hoverinfo='text',
        name='Nodes'
    ))
    
    # Update layout
    fig.update_layout(
        title='3D Seating Arrangement Visualization',
        showlegend=True,
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)  # Adjust camera angle for better view
            )
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # Save the visualization
    fig.write_html(os.path.join(output_dir, '3d_visualization.html'))

def main():
    """
    Run the wedding seating arrangement algorithm with configured parameters.
    Results and visualizations will be saved to the outputs directory.
    """
    # Setup output directory
    output_dir = setup_output_directory()
    print(f"Output will be saved to: {output_dir}")
    
    # Create seating manager and load data
    seating_manager = SeatingManager()
    relationship_matrix = seating_manager.load_guests_from_csv(weights.INPUT_FILE)
    
    print(f"Loaded {seating_manager.guest_count} guests from {weights.INPUT_FILE}")
    
    # Create tables
    seating_manager.create_tables(weights.NUM_TABLES, weights.TABLE_CAPACITY)
    print(f"Created {weights.NUM_TABLES} tables with capacity {weights.TABLE_CAPACITY}")
    
    # Visualize relationship matrix if requested
    if weights.GENERATE_VISUALIZATIONS:
        Visualizer.plot_relationship_matrix(relationship_matrix)
        # Move the generated file to output directory
        shutil.move('relationship_matrix.png', 
                   os.path.join(output_dir, f'relationship_matrix.{weights.VISUALIZATION_FORMAT}'))
    
    # Run genetic algorithm
    print("\nRunning genetic algorithm...")
    print(f"Population size: {weights.POPULATION_SIZE}")
    print(f"Max generations: {weights.MAX_GENERATIONS}")
    print(f"Mutation rate: {weights.MUTATION_RATE}")
    print(f"Elite size: {weights.ELITE_SIZE}")
    
    ga = GeneticAlgorithm(
        seating_manager=seating_manager,
        population_size=weights.POPULATION_SIZE,
        mutation_rate=weights.MUTATION_RATE,
        elite_size=weights.ELITE_SIZE,
        max_generations=weights.MAX_GENERATIONS
    )
    
    best_solution, best_fitness = ga.run()
    print(f"\nBest solution found with fitness: {best_fitness}")
    
    # Apply the solution and get the mapping
    solution_mapping = ga.apply_solution(best_solution)
    
    # Validate the solution meets all constraints
    is_valid, message = seating_manager.validate_arrangement()
    if is_valid:
        print("Solution is valid: All guests are assigned to exactly one table.")
    else:
        print(f"WARNING - Invalid solution: {message}")
    
    # Save table assignments to file
    solution_path = os.path.join(output_dir, "solution.csv")
    save_solution(solution_mapping, solution_path)
    
    # Print and save table assignments
    assignments_path = os.path.join(output_dir, "assignments.txt")
    with open(assignments_path, 'w') as f:
        # Redirect stdout to file
        original_stdout = sys.stdout
        sys.stdout = f
        Visualizer.print_table_assignments(seating_manager, solution_mapping)
        sys.stdout = original_stdout
    
    # Visualize the solution if requested
    if weights.GENERATE_VISUALIZATIONS:
        Visualizer.plot_seating_arrangement(seating_manager, solution_mapping)
        # Move the generated file to output directory
        shutil.move('seating_arrangement.png', 
                   os.path.join(output_dir, f'seating_arrangement.{weights.VISUALIZATION_FORMAT}'))
        
        # Create 3D network visualization
        create_3d_seating_visualization(seating_manager, solution_mapping, output_dir)
    
    print(f"\nAll outputs have been saved to: {output_dir}")

def save_solution(solution_mapping, filename):
    """
    Save the solution to a CSV file.
    
    Args:
        solution_mapping (dict): Mapping of guest IDs to table IDs.
        filename (str): Output file name.
    """
    with open(filename, 'w') as f:
        f.write("Guest ID,Table ID\n")
        for guest_id, table_id in sorted(solution_mapping.items()):
            f.write(f"{guest_id},{table_id}\n")
    print(f"Solution saved to {filename}")

if __name__ == "__main__":
    main() 