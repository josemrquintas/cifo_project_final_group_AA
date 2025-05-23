import os
import sys
from datetime import datetime
import shutil
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seating_manager import SeatingManager
from genetic_algorithm import GeneticAlgorithm
from visualizer import Visualizer
import config.weights as weights

def setup_output_directory():
    """Create output directory with timestamp if it doesn't exist."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(weights.OUTPUT_DIR, f"multiple_runs_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def run_single_ga(seating_manager, run_number):
    """
    Run a single instance of the genetic algorithm.
    
    Args:
        seating_manager: SeatingManager instance
        run_number: Number of the current run
    
    Returns:
        tuple: (best_solution, best_fitness, solution_mapping)
    """
    print(f"\nRunning genetic algorithm (Run {run_number})...")
    
    ga = GeneticAlgorithm(
        seating_manager=deepcopy(seating_manager),
        population_size=weights.POPULATION_SIZE,
        mutation_rate=weights.MUTATION_RATE,
        elite_size=weights.ELITE_SIZE,
        max_generations=weights.MAX_GENERATIONS
    )
    
    best_solution, best_fitness = ga.run()
    solution_mapping = ga.apply_solution(best_solution)
    
    return best_solution, best_fitness, solution_mapping

def save_run_results(run_number, best_fitness, solution_mapping, output_dir):
    """Save results for a single run."""
    run_dir = os.path.join(output_dir, f"run_{run_number}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save fitness score
    with open(os.path.join(run_dir, "fitness.txt"), 'w') as f:
        f.write(f"Best Fitness: {best_fitness}\n")
    
    # Save solution mapping
    solution_path = os.path.join(run_dir, "solution.csv")
    with open(solution_path, 'w') as f:
        f.write("Guest ID,Table ID\n")
        for guest_id, table_id in sorted(solution_mapping.items()):
            f.write(f"{guest_id},{table_id}\n")

def analyze_results(fitness_scores, output_dir):
    """
    Create visualizations to analyze the results of multiple runs.
    
    Args:
        fitness_scores: List of fitness scores from all runs
        output_dir: Directory to save the visualizations
    """
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'Run': range(1, len(fitness_scores) + 1),
        'Fitness': fitness_scores
    })
    
    # Save raw data
    df.to_csv(os.path.join(output_dir, "all_fitness_scores.csv"), index=False)
    
    # Create boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, y='Fitness')
    plt.title('Distribution of Fitness Scores Across Runs')
    plt.ylabel('Fitness Score')
    plt.savefig(os.path.join(output_dir, "fitness_boxplot.png"))
    plt.close()
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Fitness', bins=20)
    plt.title('Histogram of Fitness Scores')
    plt.xlabel('Fitness Score')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, "fitness_histogram.png"))
    plt.close()
    
    # Create bar chart of fitness scores
    plt.figure(figsize=(12, 6))
    plt.bar(df['Run'], df['Fitness'], color='skyblue')
    plt.title('Fitness Scores Across Runs')
    plt.xlabel('Run Number')
    plt.ylabel('Fitness Score')
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_dir, "fitness_line_plot.png"))
    plt.close()
    
    # Calculate and save statistics
    stats = {
        'Mean': df['Fitness'].mean(),
        'Median': df['Fitness'].median(),
        'Std Dev': df['Fitness'].std(),
        'Min': df['Fitness'].min(),
        'Max': df['Fitness'].max(),
        'Best Run': df['Run'][df['Fitness'].idxmax()]
    }
    
    with open(os.path.join(output_dir, "statistics.txt"), 'w') as f:
        f.write("Statistical Analysis of Fitness Scores\n")
        f.write("===================================\n\n")
        for stat, value in stats.items():
            f.write(f"{stat}: {value:.2f}\n")

def main():
    """
    Run the genetic algorithm multiple times and analyze the results.
    """
    NUM_RUNS = 30
    
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
    
    # Store results from all runs
    all_fitness_scores = []
    best_overall_fitness = float('-inf')
    best_overall_solution = None
    best_overall_mapping = None
    best_run_number = None
    
    # Run the genetic algorithm multiple times
    for run in range(1, NUM_RUNS + 1):
        best_solution, best_fitness, solution_mapping = run_single_ga(seating_manager, run)
        all_fitness_scores.append(best_fitness)
        
        # Save results for this run
        save_run_results(run, best_fitness, solution_mapping, output_dir)
        
        # Update best overall solution if needed
        if best_fitness > best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall_solution = best_solution
            best_overall_mapping = solution_mapping
            best_run_number = run
    
    # Analyze results
    analyze_results(all_fitness_scores, output_dir)
    
    # Save best overall solution
    best_solution_dir = os.path.join(output_dir, "best_solution")
    os.makedirs(best_solution_dir, exist_ok=True)
    
    # Save best solution mapping
    with open(os.path.join(best_solution_dir, "solution.csv"), 'w') as f:
        f.write("Guest ID,Table ID\n")
        for guest_id, table_id in sorted(best_overall_mapping.items()):
            f.write(f"{guest_id},{table_id}\n")
    
    # Save best solution details
    with open(os.path.join(best_solution_dir, "details.txt"), 'w') as f:
        f.write(f"Best Solution Details\n")
        f.write("===================\n\n")
        f.write(f"Run Number: {best_run_number}\n")
        f.write(f"Fitness Score: {best_overall_fitness}\n")
    
    # Create visualizations for best solution if requested
    if weights.GENERATE_VISUALIZATIONS:
        # Create a copy of seating manager for visualization
        viz_seating_manager = deepcopy(seating_manager)
        viz_seating_manager.apply_solution(best_overall_mapping)
        
        # Generate and save visualizations
        Visualizer.plot_seating_arrangement(viz_seating_manager, best_overall_mapping)
        shutil.move('seating_arrangement.png', 
                   os.path.join(best_solution_dir, f'seating_arrangement.{weights.VISUALIZATION_FORMAT}'))
        
        # Create 3D visualization
        create_3d_seating_visualization(viz_seating_manager, best_overall_mapping, best_solution_dir)
    
    print(f"\nAll outputs have been saved to: {output_dir}")
    print(f"Best solution found in run {best_run_number} with fitness: {best_overall_fitness}")

if __name__ == "__main__":
    main() 