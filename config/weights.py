"""
Configuration file for genetic algorithm parameters and weights.
Adjust these values to tune the algorithm's behavior.

Current settings were set based on gridsearch results.
"""

# Genetic Algorithm Parameters
POPULATION_SIZE = 500
MAX_GENERATIONS = 1000
MUTATION_RATE = 0.3
ELITE_SIZE = 20

# Selection Parameters
TOURNAMENT_SIZE = 5
TOURNAMENT_SELECTION_WEIGHT = 0.9  # 90% chance of using tournament selection
ROULETTE_WHEEL_SELECTION_WEIGHT = 0.1  # 10% chance of using roulette wheel selection

# Crossover Parameters
CROSSOVER_RATE = 0.8
UNIFORM_CROSSOVER_PROB = 0.3
SINGLE_POINT_CROSSOVER_PROB = 0.3
TABLE_BASED_CROSSOVER_PROB = 0.4

# Mutation Parameters
RANDOM_MUTATION_WEIGHT = 0.2
SWAP_MUTATION_WEIGHT = 0.3
TABLE_SHIFT_MUTATION_WEIGHT = 0.2
RELATIONSHIP_MUTATION_WEIGHT = 0.3

# Early Stopping Parameters
MAX_GENERATIONS_WITHOUT_IMPROVEMENT = 200
ADAPTIVE_MUTATION = True
ADAPTIVE_MUTATION_INCREASE_FACTOR = 1.5
MAX_MUTATION_RATE = 0.8
ADAPTIVE_MUTATION_CHECK_INTERVAL = 20
ADAPTIVE_MUTATION_THRESHOLD = 10

# Table Configuration
NUM_TABLES = 8
TABLE_CAPACITY = 8

# Input/Output Configuration
INPUT_FILE = "inputs/seating_data.csv"
OUTPUT_DIR = "outputs"

# Visualization Settings
GENERATE_VISUALIZATIONS = True
VISUALIZATION_FORMAT = "png"  # Options: "png", "jpg", "pdf" 