import random
import numpy as np
import copy
import sys
import os

# Add the project root to the path so we can import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import weights

class GeneticAlgorithm:
    """
    Implements a genetic algorithm to solve the wedding seating problem.
    
    Attributes:
        seating_manager: The SeatingManager instance to work with.
        population_size: Size of the population.
        mutation_rate: Probability of mutation for each gene.
        elite_size: Number of best individuals to keep in each generation.
        max_generations: Maximum number of generations to run.
    """
    
    def __init__(self, seating_manager, population_size=weights.POPULATION_SIZE, 
                 mutation_rate=weights.MUTATION_RATE, elite_size=weights.ELITE_SIZE, 
                 max_generations=weights.MAX_GENERATIONS):
        """
        Initialize the GeneticAlgorithm.
        
        Args:
            seating_manager: SeatingManager instance.
            population_size (int): Size of the population. Default is from weights.
            mutation_rate (float): Mutation probability (0-1). Default is from weights.
            elite_size (int): Number of elite individuals. Default is from weights.
            max_generations (int): Maximum generations to run. Default is from weights.
        """
        self.seating_manager = seating_manager
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.max_generations = max_generations
        self.num_tables = len(self.seating_manager.tables)
        self.guest_ids = list(self.seating_manager.guests.keys())
        self.last_generation = 0  # Track the last generation number
        
    def create_individual(self):
        """
        Create a random individual (seating arrangement) that ensures:
        1. Each guest is assigned to exactly one table
        2. No guest is left unassigned
        3. No table exceeds its capacity
        
        Returns:
            dict: A dictionary mapping guest IDs to table IDs
        """
        # Create a dictionary mapping guest IDs to table assignments
        individual = {}
        
        # Keep track of table capacities
        table_counts = {i: 0 for i in range(self.num_tables)}
        
        # Create list of guests to assign randomly
        guests_to_assign = self.guest_ids.copy()
        random.shuffle(guests_to_assign)
        
        # First pass: assign guests to tables with available capacity
        for guest_id in guests_to_assign:
            # Find tables that aren't full
            available_tables = [
                table_id for table_id, count in table_counts.items()
                if count < self.seating_manager.tables[table_id].capacity
            ]
            
            if available_tables:
                # Randomly choose from available tables
                table_id = random.choice(available_tables)
                individual[guest_id] = table_id
                table_counts[table_id] += 1
            else:
                # No tables with capacity - need to redistribute
                # Find table with minimum overflow
                min_overflow_table = min(range(self.num_tables), 
                                      key=lambda t: table_counts[t] - self.seating_manager.tables[t].capacity)
                individual[guest_id] = min_overflow_table
                table_counts[min_overflow_table] += 1
        
        # Second pass: fix any capacity violations
        return self.repair_table_capacity(individual)
    
    def initialize_population(self):
        """
        Initialize a population of random valid individuals.
        
        Returns:
            list: A list of individuals (dictionaries).
        """
        return [self.create_individual() for _ in range(self.population_size)]
    
    def calculate_fitness(self, individual):
        """
        Calculate the fitness (happiness) of an individual.
        
        Args:
            individual: A seating arrangement (dict mapping guest_id to table_id).
            
        Returns:
            int: The fitness value (happiness score).
        """
        # Clear all tables
        self.seating_manager.clear_tables()
        
        # Validate that all guests are assigned to exactly one table
        if len(individual) != len(self.guest_ids):
            # Invalid solution - return extremely low fitness
            return float('-inf')
        
        # Assign guests to tables according to the individual
        for guest_id, table_id in individual.items():
            if guest_id in self.seating_manager.guests and table_id < self.num_tables:
                table = self.seating_manager.tables[table_id]
                table.add_guest(self.seating_manager.guests[guest_id])
        
        # Calculate total happiness
        return self.seating_manager.calculate_total_happiness()
    
    def select_parents(self, population, fitness_scores):
        """Select parents for the next generation using tournament selection."""
        # Sort population by fitness
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
        
        # Select elites
        elites = sorted_population[:self.elite_size]
        
        # Create a list to store selected parents
        selected_parents = elites.copy()
        
        # Use tournament selection with configured tournament size
        tournament_size = weights.TOURNAMENT_SIZE
        
        # Fill the rest of the population with parents selected using weighted selection methods
        while len(selected_parents) < self.population_size:
            # Choose selection method based on weights
            selection_method = random.choices(
                ['tournament', 'roulette'],
                weights=[weights.TOURNAMENT_SELECTION_WEIGHT, weights.ROULETTE_WHEEL_SELECTION_WEIGHT],
                k=1
            )[0]
            
            if selection_method == 'tournament':
                parent = self._tournament_selection(sorted_population, tournament_size)
            else:  # roulette wheel selection
                parent = self._roulette_wheel_selection(sorted_population, fitness_scores)
                
            selected_parents.append(parent)
            
        return selected_parents
    
    def _tournament_selection(self, population, tournament_size=3):
        """Select parents using tournament selection."""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: self.calculate_fitness(x))
    
    def _roulette_wheel_selection(self, population, fitness_scores):
        """Select parents using roulette wheel selection with fitness scaling."""
        # Handle case where all fitness scores are the same
        if len(set(fitness_scores)) == 1:
            return random.choice(population)
            
        # Apply fitness scaling to prevent premature convergence
        min_fitness = min(fitness_scores)
        scaled_fitness = [f - min_fitness + 1 for f in fitness_scores]  # Add 1 to avoid zero probabilities
        total_scaled_fitness = sum(scaled_fitness)
            
        # Calculate selection probabilities using scaled fitness
        selection_probs = [f/total_scaled_fitness for f in scaled_fitness]
        
        # Select parent based on probabilities
        selected = random.choices(population, weights=selection_probs, k=1)[0]
        
        # Always repair the selected individual to ensure it's valid
        repaired = self.repair_individual(selected)
        repaired = self.repair_table_capacity(repaired)
        
        return repaired
    
    # REPAIR METHODS
    
    def repair_individual(self, individual):
        """
        Repair an individual to ensure it meets all constraints:
        1. Each guest is assigned to exactly one table
        2. No guest is left unassigned
        3. All table assignments are valid
        
        Args:
            individual: The individual to repair (dict).
            
        Returns:
            dict: A repaired individual.
        """
        repaired = individual.copy()
        
        # Ensure all guests are assigned
        for guest_id in self.guest_ids:
            if guest_id not in repaired:
                # Assign to a random table if not assigned
                repaired[guest_id] = random.randint(0, self.num_tables - 1)
        
        # Ensure all table assignments are valid
        for guest_id in repaired:
            if repaired[guest_id] >= self.num_tables:
                # Assign to a valid table if current assignment is invalid
                repaired[guest_id] = random.randint(0, self.num_tables - 1)
        
        return repaired
    
    def repair_table_capacity(self, individual):
        """
        Repair an individual to ensure no table exceeds its capacity.
        
        Args:
            individual: The individual to repair (dict).
            
        Returns:
            dict: A repaired individual.
        """
        repaired = individual.copy()
        
        # Count guests at each table
        table_counts = {}
        for table_id in range(self.num_tables):
            table_counts[table_id] = 0
        
        for guest_id, table_id in repaired.items():
            table_counts[table_id] += 1
        
        # Check if any table exceeds capacity
        for table_id, count in table_counts.items():
            if count > self.seating_manager.tables[table_id].capacity:
                # Find guests at this table
                guests_at_table = [g for g, t in repaired.items() if t == table_id]
                
                # Randomly select guests to move to other tables
                excess = count - self.seating_manager.tables[table_id].capacity
                guests_to_move = random.sample(guests_at_table, excess)
                
                # Move selected guests to random tables
                for guest_id in guests_to_move:
                    # Find a table with available capacity
                    available_tables = [t for t in range(self.num_tables) 
                                      if table_counts[t] < self.seating_manager.tables[t].capacity]
                    
                    if available_tables:
                        new_table = random.choice(available_tables)
                        repaired[guest_id] = new_table
                        table_counts[new_table] += 1
                        table_counts[table_id] -= 1
        
        return repaired
    
    # CROSSOVER OPERATORS
    
    def uniform_crossover(self, parent1, parent2):
        """
        Uniform crossover: For each guest, randomly choose which parent to inherit from.
        
        Args:
            parent1: First parent (dict).
            parent2: Second parent (dict).
            
        Returns:
            dict: A new individual (child).
        """
        child = {}
        
        # Randomly decide for each guest whether to inherit from parent1 or parent2
        for guest_id in self.guest_ids:
            if random.random() < 0.5:
                child[guest_id] = parent1[guest_id]
            else:
                child[guest_id] = parent2[guest_id]
        
        # Repair the child to ensure it's valid
        child = self.repair_individual(child)
        child = self.repair_table_capacity(child)
        
        return child
    
    def single_point_crossover(self, parent1, parent2):
        """
        Single-point crossover: Choose a random point and swap all assignments after that point.
        
        Args:
            parent1: First parent (dict).
            parent2: Second parent (dict).
            
        Returns:
            dict: A new individual (child).
        """
        child = {}
        
        # Choose a random crossover point
        crossover_point = random.randint(0, len(self.guest_ids) - 1)
        
        # Take assignments from parent1 before the crossover point
        for i, guest_id in enumerate(self.guest_ids):
            if i < crossover_point:
                child[guest_id] = parent1[guest_id]
            else:
                child[guest_id] = parent2[guest_id]
        
        # Repair the child to ensure it's valid
        child = self.repair_individual(child)
        child = self.repair_table_capacity(child)
        
        return child
    
    def table_based_crossover(self, parent1, parent2):
        """
        Table-based crossover: Inherit entire tables from either parent.
        
        Args:
            parent1: First parent (dict).
            parent2: Second parent (dict).
            
        Returns:
            dict: A new individual (child).
        """
        child = {}
        
        # Group guests by table for each parent
        parent1_tables = {}
        parent2_tables = {}
        
        for guest_id in self.guest_ids:
            table1 = parent1[guest_id]
            table2 = parent2[guest_id]
            
            if table1 not in parent1_tables:
                parent1_tables[table1] = []
            if table2 not in parent2_tables:
                parent2_tables[table2] = []
                
            parent1_tables[table1].append(guest_id)
            parent2_tables[table2].append(guest_id)
        
        # For each table in parent1, decide whether to keep it or replace with parent2's table
        for table_id in range(self.num_tables):
            if random.random() < 0.5 and table_id in parent1_tables:
                # Keep parent1's table
                for guest_id in parent1_tables[table_id]:
                    child[guest_id] = table_id
            elif table_id in parent2_tables:
                # Use parent2's table
                for guest_id in parent2_tables[table_id]:
                    child[guest_id] = table_id
        
        # Ensure all guests are assigned
        for guest_id in self.guest_ids:
            if guest_id not in child:
                # Assign to a random table if not already assigned
                child[guest_id] = random.randint(0, self.num_tables - 1)
        
        # Repair the child to ensure it's valid
        child = self.repair_individual(child)
        child = self.repair_table_capacity(child)
        
        return child
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents using a randomly selected operator.
        
        Args:
            parent1: First parent (dict).
            parent2: Second parent (dict).
            
        Returns:
            dict: A new individual (child).
        """
        # Randomly select a crossover operator based on configured probabilities
        crossover_type = random.choices(
            ['uniform', 'single_point', 'table_based'],
            weights=[weights.UNIFORM_CROSSOVER_PROB, 
                    weights.SINGLE_POINT_CROSSOVER_PROB, 
                    weights.TABLE_BASED_CROSSOVER_PROB],
            k=1
        )[0]
        
        if crossover_type == 'uniform':
            return self.uniform_crossover(parent1, parent2)
        elif crossover_type == 'single_point':
            return self.single_point_crossover(parent1, parent2)
        else:
            return self.table_based_crossover(parent1, parent2)
    
    # MUTATION OPERATORS
    
    def random_mutation(self, individual):
        """
        Random mutation: For each guest, with probability mutation_rate,
        assign to a random table.
        
        Args:
            individual: The individual to mutate (dict).
            
        Returns:
            dict: Mutated individual.
        """
        mutated = individual.copy()
        
        for guest_id in self.guest_ids:
            if random.random() < self.mutation_rate:
                # Assign to a random table
                mutated[guest_id] = random.randint(0, self.num_tables - 1)
        
        # Repair the mutated individual to ensure it's valid
        mutated = self.repair_individual(mutated)
        mutated = self.repair_table_capacity(mutated)
        
        return mutated
    
    def swap_mutation(self, individual):
        """
        Swap mutation: With probability mutation_rate, swap two guests' table assignments.
        
        Args:
            individual: The individual to mutate (dict).
            
        Returns:
            dict: Mutated individual.
        """
        mutated = individual.copy()
        
        if random.random() < self.mutation_rate:
            # Select two random guests
            guest1, guest2 = random.sample(self.guest_ids, 2)
            
            # Swap their table assignments
            table1 = mutated[guest1]
            table2 = mutated[guest2]
            mutated[guest1] = table2
            mutated[guest2] = table1
        
        # Repair the mutated individual to ensure it's valid
        mutated = self.repair_individual(mutated)
        mutated = self.repair_table_capacity(mutated)
        
        return mutated
    
    def table_shift_mutation(self, individual):
        """
        Table shift mutation: With probability mutation_rate, shift all guests
        at a table to a different table.
        
        Args:
            individual: The individual to mutate (dict).
            
        Returns:
            dict: Mutated individual.
        """
        mutated = individual.copy()
        
        if random.random() < self.mutation_rate:
            # Select a random table
            table_id = random.randint(0, self.num_tables - 1)
            
            # Find all guests at this table
            guests_at_table = [g for g in self.guest_ids if mutated[g] == table_id]
            
            if guests_at_table:
                # Select a different table
                new_table = random.randint(0, self.num_tables - 1)
                while new_table == table_id:
                    new_table = random.randint(0, self.num_tables - 1)
                
                # Move all guests to the new table
                for guest_id in guests_at_table:
                    mutated[guest_id] = new_table
        
        # Repair the mutated individual to ensure it's valid
        mutated = self.repair_individual(mutated)
        mutated = self.repair_table_capacity(mutated)
        
        return mutated
    
    def relationship_based_mutation(self, individual):
        """
        Relationship-based mutation: With probability mutation_rate,
        move a guest to a table with guests they have positive relationships with.
        
        Args:
            individual: The individual to mutate (dict).
            
        Returns:
            dict: Mutated individual.
        """
        mutated = individual.copy()
        
        if random.random() < self.mutation_rate:
            # Select a random guest
            guest_id = random.choice(self.guest_ids)
            guest = self.seating_manager.guests[guest_id]
            
            # Find tables with guests this guest has positive relationships with
            table_scores = {}
            for table_id in range(self.num_tables):
                table_scores[table_id] = 0
                
                # Check relationships with guests at this table
                for other_id in self.guest_ids:
                    if mutated[other_id] == table_id and other_id != guest_id:
                        relationship = guest.get_relationship(other_id)
                        if relationship > 0:
                            table_scores[table_id] += relationship
            
            # Move to the table with the highest positive relationship score
            if table_scores:
                best_table = max(table_scores.items(), key=lambda x: x[1])[0]
                mutated[guest_id] = best_table
        
        # Repair the mutated individual to ensure it's valid
        mutated = self.repair_individual(mutated)
        mutated = self.repair_table_capacity(mutated)
        
        return mutated
    
    def mutate(self, individual):
        """
        Mutate an individual by randomly changing some table assignments.
        Uses multiple mutation operators with different probabilities.
        
        Args:
            individual: The individual to mutate.
            
        Returns:
            dict: The mutated individual.
        """
        # Create a copy of the individual
        mutated = individual.copy()
        
        # Choose a mutation operator based on configured probabilities
        mutation_type = random.choices(
            ['random', 'swap', 'table_shift', 'relationship'],
            weights=[weights.RANDOM_MUTATION_WEIGHT, 
                    weights.SWAP_MUTATION_WEIGHT, 
                    weights.TABLE_SHIFT_MUTATION_WEIGHT, 
                    weights.RELATIONSHIP_MUTATION_WEIGHT],
            k=1
        )[0]
        
        if mutation_type == 'random':
            # Random mutation: assign each guest to a random table with probability mutation_rate
            for guest_id in mutated:
                if random.random() < self.mutation_rate:
                    mutated[guest_id] = random.randint(0, self.num_tables - 1)
        elif mutation_type == 'swap':
            # Swap mutation: swap the table assignments of two guests with probability mutation_rate
            if random.random() < self.mutation_rate and len(mutated) >= 2:
                guest1, guest2 = random.sample(list(mutated.keys()), 2)
                mutated[guest1], mutated[guest2] = mutated[guest2], mutated[guest1]
        elif mutation_type == 'table_shift':
            # Table shift mutation: shift all guests at a table to a different table with probability mutation_rate
            if random.random() < self.mutation_rate and self.num_tables > 1:
                # Find a table with guests
                tables_with_guests = set(mutated.values())
                if tables_with_guests:
                    source_table = random.choice(list(tables_with_guests))
                    target_table = random.choice([t for t in range(self.num_tables) if t != source_table])
                    
                    # Move all guests from source table to target table
                    for guest_id, table_id in mutated.items():
                        if table_id == source_table:
                            mutated[guest_id] = target_table
        else:  # relationship-based mutation
            # Relationship-based mutation: move a guest to a table with guests they have positive relationships with
            if random.random() < self.mutation_rate:
                # Select a random guest
                guest_id = random.choice(list(mutated.keys()))
                
                # Find tables with guests that have positive relationships with the selected guest
                positive_relationship_tables = set()
                for other_guest_id, other_table_id in mutated.items():
                    if other_guest_id != guest_id:
                        # Fix: Use the correct method to get relationship value
                        relationship_value = self.seating_manager.guests[guest_id].get_relationship(other_guest_id)
                        if relationship_value > 0:
                            positive_relationship_tables.add(other_table_id)
                
                # If there are tables with positive relationships, move the guest to one of them
                if positive_relationship_tables:
                    mutated[guest_id] = random.choice(list(positive_relationship_tables))
        
        # Repair the individual to ensure it's valid
        mutated = self.repair_individual(mutated)
        
        return mutated
    
    def create_next_generation(self, population, fitness_scores):
        """
        Create the next generation through selection, crossover, and mutation,
        ensuring all solutions are valid.
        
        Args:
            population: Current population.
            fitness_scores: Fitness scores for the current population.
            
        Returns:
            list: The next generation population.
        """
        # Select parents
        parents = self.select_parents(population, fitness_scores)
        
        # Create next generation starting with elites
        population_fitness = list(zip(population, fitness_scores))
        population_fitness.sort(key=lambda x: x[1], reverse=True)
        next_generation = [individual for individual, _ in population_fitness[:self.elite_size]]
        
        # Fill the rest of the population with children from crossover and mutation
        while len(next_generation) < self.population_size:
            # Select two random parents from the selected parents
            parent1, parent2 = random.sample(parents, 2)
            
            # Apply crossover with configured probability
            if random.random() < weights.CROSSOVER_RATE:
                child = self.crossover(parent1, parent2)
            else:
                # If no crossover, clone the better parent
                child = parent1.copy()
            
            # Apply mutation with adaptive rate
            child = self.mutate(child)
            
            next_generation.append(child)
        
        return next_generation
    
    def run(self):
        """
        Run the genetic algorithm to find the best seating arrangement.
        
        Returns:
            tuple: The best individual and its fitness score.
        """
        # Initialize population
        population = self.initialize_population()
        
        best_individual = None
        best_fitness = float('-inf')
        generations_without_improvement = 0
        max_generations_without_improvement = weights.MAX_GENERATIONS_WITHOUT_IMPROVEMENT
        base_mutation_rate = self.mutation_rate  # Store the initial mutation rate
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.last_generation = generation  # Update last generation number
            
            # Calculate fitness for each individual
            fitness_scores = [self.calculate_fitness(individual) for individual in population]
            
            # Update best individual if better
            current_best_idx = fitness_scores.index(max(fitness_scores))
            current_best_individual = population[current_best_idx]
            current_best_fitness = fitness_scores[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_individual = current_best_individual
                best_fitness = current_best_fitness
                generations_without_improvement = 0
                # Reset mutation rate back to baseline when we find an improvement
                if self.mutation_rate != base_mutation_rate:
                    self.mutation_rate = base_mutation_rate
                    print(f"Found improvement - resetting mutation rate to {self.mutation_rate}")
                print(f"Generation {generation}: New best fitness = {best_fitness}")
            else:
                generations_without_improvement += 1
                
            # Early stopping if no improvement for a while
            if generations_without_improvement >= max_generations_without_improvement:
                print(f"Stopping early at generation {generation} due to no improvement for {max_generations_without_improvement} generations")
                break
                
            # Create next generation
            population = self.create_next_generation(population, fitness_scores)
            
            # Periodically increase mutation rate if stuck and adaptive mutation is enabled
            if (weights.ADAPTIVE_MUTATION and 
                generation % weights.ADAPTIVE_MUTATION_CHECK_INTERVAL == 0 and 
                generations_without_improvement > weights.ADAPTIVE_MUTATION_THRESHOLD):
                old_rate = self.mutation_rate
                self.mutation_rate = min(weights.MAX_MUTATION_RATE, 
                                       self.mutation_rate * weights.ADAPTIVE_MUTATION_INCREASE_FACTOR)
                if old_rate != self.mutation_rate:
                    print(f"Increasing mutation rate to {self.mutation_rate}")
        
        # Reset mutation rate to baseline at the end
        self.mutation_rate = base_mutation_rate
        return best_individual, best_fitness
    
    def apply_solution(self, solution):
        """
        Apply the solution to the seating manager.
        
        Args:
            solution: The best individual found (dict mapping guest_id to table_id).
            
        Returns:
            dict: A mapping of guest IDs to table IDs.
        """
        # Clear all tables
        self.seating_manager.clear_tables()
        
        # Assign guests to tables according to the solution
        for guest_id, table_id in solution.items():
            if guest_id in self.seating_manager.guests and table_id < self.num_tables:
                table = self.seating_manager.tables[table_id]
                table.add_guest(self.seating_manager.guests[guest_id])
        
        return solution 