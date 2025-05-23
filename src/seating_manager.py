import csv
import numpy as np
from guest import Guest
from table import Table

class SeatingManager:
    """
    Manages the seating arrangement for a wedding, including loading data
    and calculating the overall happiness score.
    
    Attributes:
        guests (dict): Dictionary mapping guest IDs to Guest objects.
        tables (list): List of Table objects.
        guest_count (int): Total number of guests.
        table_count (int): Total number of tables.
    """
    
    def __init__(self):
        """Initialize a new SeatingManager."""
        self.guests = {}
        self.tables = []
        self.guest_count = 0
        self.table_count = 0
        
    def load_guests_from_csv(self, csv_file):
        """
        Load guest relationships from a CSV file.
        
        Args:
            csv_file (str): Path to the CSV file containing relationship data.
        """
        relationship_matrix = []
        
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            # Skip header row
            next(csv_reader)
            
            for row in csv_reader:
                guest_id = int(row[0])
                # Create guest if it doesn't exist
                if guest_id not in self.guests:
                    self.guests[guest_id] = Guest(guest_id)
                    
                # Parse relationship values and add them to the guest
                for other_id, value in enumerate(row[1:], start=1):
                    if other_id != guest_id:  # Skip self-relationships
                        # Add all relationships, converting empty values to 0
                        self.guests[guest_id].add_relationship(other_id, int(value) if value else 0)
                
                relationship_matrix.append([int(val) if val else 0 for val in row[1:]])
        
        self.guest_count = len(self.guests)
        return np.array(relationship_matrix)
    
    def create_tables(self, num_tables, capacity=8):
        """
        Create a specified number of tables.
        
        Args:
            num_tables (int): Number of tables to create.
            capacity (int, optional): Capacity of each table. Defaults to 8.
        """
        self.tables = [Table(i, capacity) for i in range(num_tables)]
        self.table_count = num_tables
    
    def calculate_total_happiness(self):
        """
        Calculate the total happiness of the current seating arrangement.
        
        Returns:
            int: Total happiness score across all tables.
        """
        return sum(table.calculate_table_happiness() for table in self.tables)
    
    def get_guest_table_mapping(self):
        """
        Get a mapping of guest IDs to their table IDs.
        
        Returns:
            dict: Dictionary with guest IDs as keys and table IDs as values.
        """
        mapping = {}
        for table in self.tables:
            for guest in table.guests:
                mapping[guest.id] = table.id
        return mapping
    
    def clear_tables(self):
        """Remove all guests from all tables."""
        for table in self.tables:
            table.guests = []
            
    def validate_arrangement(self):
        """
        Validate that the current seating arrangement is valid:
        1. Each guest is assigned to exactly one table
        2. No guest is left unassigned
        
        Returns:
            tuple: (is_valid, error_message)
        """
        # Get current arrangement
        mapping = self.get_guest_table_mapping()
        
        # Check if all guests are assigned
        if len(mapping) != self.guest_count:
            missing = set(self.guests.keys()) - set(mapping.keys())
            return False, f"Not all guests are assigned. Missing: {missing}"
            
        # Check if any guest is assigned to multiple tables
        guest_count = {}
        for table in self.tables:
            for guest in table.guests:
                if guest.id not in guest_count:
                    guest_count[guest.id] = 0
                guest_count[guest.id] += 1
                
        multiple = [g for g, count in guest_count.items() if count > 1]
        if multiple:
            return False, f"Some guests are assigned to multiple tables: {multiple}"
            
        return True, "Valid arrangement" 