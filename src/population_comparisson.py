import time
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import sys
import os

# Add the project root to the path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import weights
from seating_manager import SeatingManager
from genetic_algorithm import GeneticAlgorithm


# Population sizes to compare
POP_SIZES = [500, 1000, 5000, 10000]
MAX_GENERATIONS = weights.MAX_GENERATIONS

fitness_histories = {}
running_times = {}

for pop_size in POP_SIZES:
    print(f"Running GA with population size {pop_size}...")
    # Set up seating manager and GA
    seating_manager = SeatingManager()
    seating_manager.load_guests_from_csv(weights.INPUT_FILE)
    seating_manager.create_tables(weights.NUM_TABLES, weights.TABLE_CAPACITY)

    ga = GeneticAlgorithm(
        seating_manager=deepcopy(seating_manager),
        population_size=pop_size,
        mutation_rate=weights.MUTATION_RATE,
        elite_size=weights.ELITE_SIZE,
        max_generations=MAX_GENERATIONS
    )

    # Track fitness evolution
    fitness_over_time = []
    population = ga.initialize_population()
    start_time = time.time()
    for generation in range(MAX_GENERATIONS):
        fitness_scores = [ga.calculate_fitness(ind) for ind in population]
        best_fitness = max(fitness_scores)
        fitness_over_time.append(best_fitness)
        # Early stopping if no improvement for a while
        if generation > weights.MAX_GENERATIONS_WITHOUT_IMPROVEMENT:
            if all(f == fitness_over_time[-weights.MAX_GENERATIONS_WITHOUT_IMPROVEMENT] for f in fitness_over_time[-weights.MAX_GENERATIONS_WITHOUT_IMPROVEMENT:]):
                break
        population = ga.create_next_generation(population, fitness_scores)
    end_time = time.time()
    fitness_histories[pop_size] = fitness_over_time
    running_times[pop_size] = end_time - start_time
    print(f"Population {pop_size}: Best fitness {fitness_over_time[-1]}, Time {running_times[pop_size]:.2f}s, Generations {len(fitness_over_time)}")

# Plot fitness evolution
plt.figure(figsize=(10, 6))
for pop_size, history in fitness_histories.items():
    plt.plot(history, label=f"Population {pop_size}")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Fitness Evolution for Different Population Sizes")
plt.legend()
plt.tight_layout()
plt.savefig("fitness_evolution_by_population.png")

# Plot running time
plt.figure(figsize=(8, 5))
pop_sizes = list(running_times.keys())
times = [running_times[size] for size in pop_sizes]
plt.bar([str(size) for size in pop_sizes], times, color='skyblue')
plt.xlabel("Population Size")
plt.ylabel("Running Time (seconds)")
plt.title("Running Time for Different Population Sizes")
plt.tight_layout()
plt.savefig("running_time_by_population.png")

print("Plots saved: fitness_evolution_by_population.png, running_time_by_population.png") 