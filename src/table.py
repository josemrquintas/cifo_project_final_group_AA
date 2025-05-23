class Table:
    """
    Represents a table at the wedding with a set of guests.
    
    Attributes:
        id (int): Unique identifier for the table.
        capacity (int): Maximum number of guests the table can seat.
        guests (list): List of Guest objects seated at this table.
    """
    
    def __init__(self, id, capacity=8):
        """
        Initialize a new Table.
        
        Args:
            id (int): Unique identifier for the table.
            capacity (int, optional): Maximum number of guests. Defaults to 8.
        """
        self.id = id
        self.capacity = capacity
        self.guests = []
        
    def add_guest(self, guest):
        """
        Add a guest to the table if there is space available.
        
        Args:
            guest (Guest): The guest to be seated at this table.
            
        Returns:
            bool: True if guest was added, False if table is full.
        """
        if len(self.guests) < self.capacity:
            self.guests.append(guest)
            return True
        return False
        
    def remove_guest(self, guest_id):
        """
        Remove a guest from the table.
        
        Args:
            guest_id (int): ID of the guest to remove.
            
        Returns:
            Guest: The removed guest, or None if not found.
        """
        for i, guest in enumerate(self.guests):
            if guest.id == guest_id:
                return self.guests.pop(i)
        return None
    
    def is_full(self):
        """
        Check if the table is full.
        
        Returns:
            bool: True if the table is at capacity, False otherwise.
        """
        return len(self.guests) >= self.capacity
    
    def calculate_table_happiness(self):
        """
        Calculate the total happiness score for this table based on guest relationships.
        
        Returns:
            int: Total happiness score for the table.
        """
        happiness = 0
        for i, guest1 in enumerate(self.guests):
            for guest2 in self.guests[i+1:]:  # Avoid counting relationships twice
                happiness += guest1.get_relationship(guest2.id)
                happiness += guest2.get_relationship(guest1.id)
        return happiness
    
    def __str__(self):
        return f"Table {self.id}: {len(self.guests)}/{self.capacity} guests"
    
    def __repr__(self):
        return f"Table({self.id}, capacity={self.capacity})" 