import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from itertools import product

# Add the project root to the path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import weights
from seating_manager import SeatingManager
from genetic_algorithm import GeneticAlgorithm

def run_experiment(params, seating_manager, num_runs=30):
    """
    Run multiple experiments with given GA parameters and return average results.
    
    Args:
        params (dict): Dictionary containing GA parameters
        seating_manager (SeatingManager): The seating manager instance
        num_runs (int): Number of runs to average over
        
    Returns:
        dict: Results including average fitness, best fitness, and convergence stats
    """
    results = []
    best_fitnesses = []
    generations_to_converge = []
    
    for run in range(num_runs):
        # Create GA instance with current parameters
        ga = GeneticAlgorithm(
            seating_manager=deepcopy(seating_manager),
            population_size=params['population_size'],
            mutation_rate=params['mutation_rate'],
            elite_size=params['elite_size'],
            max_generations=weights.MAX_GENERATIONS
        )
        
        # Override crossover rate
        weights.CROSSOVER_RATE = params['crossover_rate']
        
        # Run GA and collect results
        best_solution, best_fitness = ga.run()
        best_fitnesses.append(best_fitness)
        
        # Store results
        results.append({
            **params,  # Include all parameters
            'run': run,
            'best_fitness': best_fitness,
            'generations': ga.last_generation
        })
        generations_to_converge.append(ga.last_generation)
    
    # Calculate aggregate statistics
    avg_result = {
        **params,  # Include all parameters
        'avg_fitness': np.mean(best_fitnesses),
        'std_fitness': np.std(best_fitnesses),
        'max_fitness': max(best_fitnesses),
        'min_fitness': min(best_fitnesses),
        'avg_generations': np.mean(generations_to_converge),
        'std_generations': np.std(generations_to_converge)
    }
    
    return avg_result, results

def grid_search_ga_params():
    """
    Perform grid search over GA parameters and analyze results.
    """
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(weights.OUTPUT_DIR, f"ga_gridsearch_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize seating manager
    seating_manager = SeatingManager()
    relationship_matrix = seating_manager.load_guests_from_csv(weights.INPUT_FILE)
    seating_manager.create_tables(weights.NUM_TABLES, weights.TABLE_CAPACITY)
    
    # Define parameter ranges for grid search
    param_grid = {
        'population_size': [500],
        'mutation_rate': [0.2, 0.3],
        'elite_size': [10, 20],
        'crossover_rate': [0.7, 0.8]
    }
    
    # Generate all combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    
    # Store results
    aggregate_results = []
    detailed_results = []
    
    # Run grid search
    total_combinations = len(param_combinations)
    for i, params in enumerate(param_combinations, 1):
        print(f"\nTesting combination {i}/{total_combinations}")
        print("Parameters:", params)
        
        avg_result, run_results = run_experiment(params, seating_manager)
        aggregate_results.append(avg_result)
        detailed_results.extend(run_results)
        
        print(f"Average fitness: {avg_result['avg_fitness']:.2f} ± {avg_result['std_fitness']:.2f}")
        print(f"Average generations: {avg_result['avg_generations']:.1f} ± {avg_result['std_generations']:.1f}")
    
    # Convert results to DataFrames
    agg_df = pd.DataFrame(aggregate_results)
    detailed_df = pd.DataFrame(detailed_results)
    
    # Save results
    agg_df.to_csv(os.path.join(output_dir, 'aggregate_results.csv'), index=False)
    detailed_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    
    # Create visualizations for each parameter's effect on fitness
    plt.figure(figsize=(15, 10))
    
    # Population size vs Fitness
    plt.subplot(2, 2, 1)
    sns.boxplot(data=detailed_df, x='population_size', y='best_fitness')
    plt.xlabel('Population Size')
    plt.ylabel('Fitness Score')
    plt.title('Population Size vs Fitness')
    
    # Mutation rate vs Fitness
    plt.subplot(2, 2, 2)
    sns.boxplot(data=detailed_df, x='mutation_rate', y='best_fitness')
    plt.xlabel('Mutation Rate')
    plt.ylabel('Fitness Score')
    plt.title('Mutation Rate vs Fitness')
    
    # Elite size vs Fitness
    plt.subplot(2, 2, 3)
    sns.boxplot(data=detailed_df, x='elite_size', y='best_fitness')
    plt.xlabel('Elite Size')
    plt.ylabel('Fitness Score')
    plt.title('Elite Size vs Fitness')
    
    # Crossover rate vs Fitness
    plt.subplot(2, 2, 4)
    sns.boxplot(data=detailed_df, x='crossover_rate', y='best_fitness')
    plt.xlabel('Crossover Rate')
    plt.ylabel('Fitness Score')
    plt.title('Crossover Rate vs Fitness')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_analysis.png'))
    
    # Create interaction plot between mutation rate and population size
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=agg_df, x='mutation_rate', y='avg_fitness', 
                hue='population_size', marker='o')
    plt.title('Interaction between Mutation Rate and Population Size')
    plt.savefig(os.path.join(output_dir, 'mutation_population_interaction.png'))
    
    # Find best configuration
    best_idx = agg_df['avg_fitness'].idxmax()
    best_config = agg_df.iloc[best_idx]
    
    print("\nBest Configuration Found:")
    print(f"Population Size: {best_config['population_size']}")
    print(f"Mutation Rate: {best_config['mutation_rate']:.3f}")
    print(f"Elite Size: {best_config['elite_size']}")
    print(f"Crossover Rate: {best_config['crossover_rate']:.3f}")
    print(f"Average Fitness: {best_config['avg_fitness']:.2f} ± {best_config['std_fitness']:.2f}")
    print(f"Average Generations: {best_config['avg_generations']:.1f} ± {best_config['std_generations']:.1f}")
    
    return best_config

if __name__ == "__main__":
    grid_search_ga_params() 