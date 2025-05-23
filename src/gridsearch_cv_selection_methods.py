import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

# Add the project root to the path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import weights
from seating_manager import SeatingManager
from genetic_algorithm import GeneticAlgorithm

def run_experiment(tournament_weight, seating_manager, num_runs=5):
    """
    Run multiple experiments with given selection weights and return average results.
    
    Args:
        tournament_weight (float): Weight for tournament selection (roulette = 1 - tournament)
        seating_manager (SeatingManager): The seating manager instance
        num_runs (int): Number of runs to average over
        
    Returns:
        dict: Results including average fitness, best fitness, and convergence stats
    """
    roulette_weight = 1.0 - tournament_weight
    results = []
    best_fitnesses = []
    generations_to_converge = []
    
    for run in range(num_runs):
        # Create GA instance with current weights
        ga = GeneticAlgorithm(
            seating_manager=deepcopy(seating_manager),
            population_size=weights.POPULATION_SIZE,
            mutation_rate=weights.MUTATION_RATE,
            elite_size=weights.ELITE_SIZE,
            max_generations=weights.MAX_GENERATIONS
        )
        
        # Override selection weights
        weights.TOURNAMENT_SELECTION_WEIGHT = tournament_weight
        weights.ROULETTE_WHEEL_SELECTION_WEIGHT = roulette_weight
        
        # Run GA and collect results
        best_solution, best_fitness = ga.run()
        best_fitnesses.append(best_fitness)
        
        # Store results
        results.append({
            'tournament_weight': tournament_weight,
            'roulette_weight': roulette_weight,
            'run': run,
            'best_fitness': best_fitness,
            'generations': ga.last_generation
        })
        generations_to_converge.append(ga.last_generation)
    
    # Calculate aggregate statistics
    avg_result = {
        'tournament_weight': tournament_weight,
        'roulette_weight': roulette_weight,
        'avg_fitness': np.mean(best_fitnesses),
        'std_fitness': np.std(best_fitnesses),
        'max_fitness': max(best_fitnesses),
        'min_fitness': min(best_fitnesses),
        'avg_generations': np.mean(generations_to_converge),
        'std_generations': np.std(generations_to_converge)
    }
    
    return avg_result, results

def grid_search_selection_methods():
    """
    Perform grid search over selection method weights and analyze results.
    """
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(weights.OUTPUT_DIR, f"gridsearch_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize seating manager
    seating_manager = SeatingManager()
    relationship_matrix = seating_manager.load_guests_from_csv(weights.INPUT_FILE)
    seating_manager.create_tables(weights.NUM_TABLES, weights.TABLE_CAPACITY)
    
    # Define grid search parameters
    tournament_weights = np.linspace(0, 1, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
    
    # Store results
    aggregate_results = []
    detailed_results = []
    
    # Run grid search
    total_combinations = len(tournament_weights)
    for i, tournament_weight in enumerate(tournament_weights, 1):
        print(f"\nTesting combination {i}/{total_combinations}")
        print(f"Tournament weight: {tournament_weight:.2f}, Roulette weight: {1-tournament_weight:.2f}")
        
        avg_result, run_results = run_experiment(tournament_weight, seating_manager)
        aggregate_results.append(avg_result)
        detailed_results.extend(run_results)
        
        print(f"Average fitness: {avg_result['avg_fitness']:.2f} ± {avg_result['std_fitness']:.2f}")
        print(f"Average generations to converge: {avg_result['avg_generations']:.1f} ± {avg_result['std_generations']:.1f}")
    
    # Convert results to DataFrames
    agg_df = pd.DataFrame(aggregate_results)
    detailed_df = pd.DataFrame(detailed_results)
    
    # Save results
    agg_df.to_csv(os.path.join(output_dir, 'aggregate_results.csv'), index=False)
    detailed_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    
    # Create visualizations
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=agg_df, x='tournament_weight', y='avg_fitness', 
                marker='o', label='Average Fitness')
    plt.fill_between(agg_df['tournament_weight'], 
                    agg_df['avg_fitness'] - agg_df['std_fitness'],
                    agg_df['avg_fitness'] + agg_df['std_fitness'],
                    alpha=0.2)
    plt.xlabel('Tournament Selection Weight')
    plt.ylabel('Fitness Score')
    plt.title('Average Fitness vs Selection Weights')
    
    plt.subplot(1, 2, 2)
    sns.lineplot(data=agg_df, x='tournament_weight', y='avg_generations', 
                marker='o', label='Average Generations')
    plt.fill_between(agg_df['tournament_weight'], 
                    agg_df['avg_generations'] - agg_df['std_generations'],
                    agg_df['avg_generations'] + agg_df['std_generations'],
                    alpha=0.2)
    plt.xlabel('Tournament Selection Weight')
    plt.ylabel('Generations to Converge')
    plt.title('Convergence Speed vs Selection Weights')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'selection_weights_analysis.png'))
    
    # Print best configuration
    best_idx = agg_df['avg_fitness'].idxmax()
    best_config = agg_df.iloc[best_idx]
    print("\nBest Configuration Found:")
    print(f"Tournament Selection Weight: {best_config['tournament_weight']:.2f}")
    print(f"Roulette Wheel Selection Weight: {best_config['roulette_weight']:.2f}")
    print(f"Average Fitness: {best_config['avg_fitness']:.2f} ± {best_config['std_fitness']:.2f}")
    print(f"Average Generations: {best_config['avg_generations']:.1f} ± {best_config['std_generations']:.1f}")
    
    return best_config

if __name__ == "__main__":
    grid_search_selection_methods() 