import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

class Visualizer:
    """
    Class to visualize seating arrangements and relationship scores.
    """
    
    @staticmethod
    def plot_seating_arrangement(seating_manager, solution_mapping):
        """
        Plot the seating arrangement.
        
        Args:
            seating_manager: The SeatingManager instance.
            solution_mapping: A dictionary mapping guest IDs to table IDs.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a colormap for tables
        num_tables = len(seating_manager.tables)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_tables))
        
        # Group guests by table
        tables_guests = {}
        for guest_id, table_id in solution_mapping.items():
            if table_id not in tables_guests:
                tables_guests[table_id] = []
            tables_guests[table_id].append(guest_id)
        
        # Create a graph
        G = nx.Graph()
        
        # Add nodes (guests)
        for guest_id, guest in seating_manager.guests.items():
            G.add_node(guest_id, label=f"Guest {guest_id}")
        
        # Add edges (relationships)
        for guest_id, guest in seating_manager.guests.items():
            for other_id, relationship in guest.relationships.items():
                if relationship != 0 and other_id in seating_manager.guests:
                    # Only add positive relationships for clarity
                    if relationship > 0:
                        G.add_edge(guest_id, other_id, weight=relationship)
        
        # Compute layout
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # Draw nodes
        for table_id, guest_ids in tables_guests.items():
            nx.draw_networkx_nodes(G, pos, nodelist=guest_ids, 
                                  node_color=[colors[table_id]]*len(guest_ids),
                                  node_size=300, alpha=0.8)
        
        # Draw edges with varying thickness based on relationship strength
        edge_weights = [G[u][v]['weight']/1000 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        # Add legend for tables
        for table_id in range(num_tables):
            ax.plot([], [], 'o', color=colors[table_id], label=f'Table {table_id}')
        ax.legend(loc='upper right')
        
        # Set title and show
        plt.title('Wedding Seating Arrangement')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('seating_arrangement.png')
        plt.show()
    
    @staticmethod
    def plot_relationship_matrix(relationship_matrix):
        """
        Plot the relationship matrix as a heatmap.
        
        Args:
            relationship_matrix: The matrix of relationship values.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create a custom colormap (red for negative, white for neutral, green for positive)
        colors = [(0.8, 0, 0), (1, 1, 1), (0, 0.8, 0)]
        cmap = LinearSegmentedColormap.from_list('relationship_cmap', colors, N=256)
        
        # Determine max absolute value for color scaling
        max_val = max(np.abs(np.min(relationship_matrix)), np.abs(np.max(relationship_matrix)))
        
        # Plot heatmap
        im = ax.imshow(relationship_matrix, cmap=cmap, vmin=-max_val, vmax=max_val)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Relationship Value')
        
        # Add labels
        plt.title('Guest Relationship Matrix')
        plt.xlabel('Guest ID')
        plt.ylabel('Guest ID')
        
        # Show ticks for every 5 guests
        ax.set_xticks(np.arange(0, len(relationship_matrix), 5))
        ax.set_yticks(np.arange(0, len(relationship_matrix), 5))
        ax.set_xticklabels(np.arange(1, len(relationship_matrix) + 1, 5))
        ax.set_yticklabels(np.arange(1, len(relationship_matrix) + 1, 5))
        
        plt.savefig('relationship_matrix.png')
        plt.show()
    
    @staticmethod
    def print_table_assignments(seating_manager, solution_mapping):
        """
        Print the table assignments in a readable format.
        
        Args:
            seating_manager: The SeatingManager instance.
            solution_mapping: A dictionary mapping guest IDs to table IDs.
        """
        # Group guests by table
        tables_guests = {}
        for guest_id, table_id in solution_mapping.items():
            if table_id not in tables_guests:
                tables_guests[table_id] = []
            tables_guests[table_id].append(guest_id)
        
        # Print results
        print("\n===== SEATING ARRANGEMENT =====")
        
        # Validate each guest is assigned to exactly one table
        all_seated_guests = set()
        for guests in tables_guests.values():
            all_seated_guests.update(guests)
        
        if len(all_seated_guests) != len(seating_manager.guests):
            print("WARNING: Not all guests are assigned to a table!")
        
        for table_id in sorted(tables_guests.keys()):
            print(f"\nTable {table_id} ({len(tables_guests[table_id])} guests):")
            
            # Sort guests by ID for readability
            guests = sorted(tables_guests[table_id])
            for guest_id in guests:
                print(f"  Guest {guest_id}")
            
            # Calculate table happiness
            table_happiness = 0
            for i, guest1_id in enumerate(guests):
                guest1 = seating_manager.guests[guest1_id]
                for guest2_id in guests[i+1:]:
                    table_happiness += guest1.get_relationship(guest2_id)
                    table_happiness += seating_manager.guests[guest2_id].get_relationship(guest1_id)
            
            print(f"  Table Happiness: {table_happiness}")
        
        print(f"\nTotal Happiness: {seating_manager.calculate_total_happiness()}")
        print(f"Total Guests: {len(seating_manager.guests)}")
        print(f"Assigned Guests: {len(all_seated_guests)}")
        
        # Check for any guests assigned to multiple tables
        guest_table_count = {}
        for guest_id in seating_manager.guests:
            guest_table_count[guest_id] = 0
            
        for table_guests in tables_guests.values():
            for guest_id in table_guests:
                guest_table_count[guest_id] += 1
                
        multiple_assignments = [g for g, count in guest_table_count.items() if count > 1]
        unassigned = [g for g, count in guest_table_count.items() if count == 0]
        
        if multiple_assignments:
            print(f"\nWARNING: {len(multiple_assignments)} guests assigned to multiple tables!")
            for guest_id in multiple_assignments:
                tables = [t for t, guests in tables_guests.items() if guest_id in guests]
                print(f"  Guest {guest_id} assigned to tables: {tables}")
                
        if unassigned:
            print(f"\nWARNING: {len(unassigned)} guests not assigned to any table!")
            for guest_id in unassigned:
                print(f"  Guest {guest_id} not assigned") 