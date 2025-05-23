class Guest:
    """
    Represents a wedding guest with their relationships to other guests.
    
    Attributes:
        id (int): Unique identifier for the guest.
        name (str): Name or role of the guest.
        relationships (dict): Dictionary mapping guest IDs to relationship values.
    """
    
    def __init__(self, id, name="Guest"):
        """
        Initialize a new Guest.
        
        Args:
            id (int): Unique identifier for the guest.
            name (str, optional): Name or role of the guest. Defaults to "Guest".
        """
        self.id = id
        self.name = name
        self.relationships = {}
        
    def add_relationship(self, other_id, value):
        """
        Add or update a relationship with another guest.
        
        Args:
            other_id (int): ID of the other guest.
            value (int): Relationship value (positive for positive relationships,
                         negative for negative relationships).
        """
        self.relationships[other_id] = value
        
    def get_relationship(self, other_id):
        """
        Get the relationship value with another guest.
        
        Args:
            other_id (int): ID of the other guest.
            
        Returns:
            int: Relationship value, 0 if no relationship exists.
        """
        return self.relationships.get(other_id, 0)
    
    def __str__(self):
        return f"Guest {self.id}: {self.name}"
    
    def __repr__(self):
        return f"Guest({self.id}, {self.name})" 