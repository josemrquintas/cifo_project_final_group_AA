# Wedding Seating Arrangement Optimization using Genetic Algorithms

## Project Overview

This repository implements a genetic algorithm to solve the wedding seating arrangement problem. The goal is to assign guests to tables in a way that maximizes overall happiness, based on interpersonal relationships, while respecting table capacity and other constraints. The project includes a flexible configuration system, multiple genetic operators, and comprehensive analysis tools for parameter tuning and performance evaluation.

## How It Works

The genetic algorithm represents each seating arrangement as a mapping of guests to tables. It evolves a population of such arrangements over multiple generations, using selection, crossover, mutation, and repair operators to explore the solution space. The fitness function evaluates each arrangement by summing the relationship scores between guests at the same table. The algorithm supports elitism, adaptive mutation, and hybrid selection strategies to improve convergence and solution quality.

## Repository Structure and Python File Functionality

- **src/genetic_algorithm.py**: Core implementation of the genetic algorithm, including population initialization, selection, crossover, mutation, repair operators, and the main evolutionary loop.
- **src/seating_manager.py**: Manages guest and table data, handles guest assignments, and calculates total happiness for a given arrangement.
- **src/table.py**: Defines the Table class, including capacity management and happiness calculation for guests at a table.
- **src/guest.py**: Defines the Guest class, handling individual guest properties and relationships.
- **src/visualizer.py**: Generates visualizations for the seating arrangement and relationship matrix.
- **src/main.py**: Main entry point for running the genetic algorithm with default settings.
- **src/main_multiple_runs.py**: Executes multiple runs of the algorithm for statistical analysis.
- **src/population_comparisson.py**: Compares the effect of different population sizes on fitness evolution and running time, generating plots for analysis.
- **src/gridsearch_cv_ga_params.py**: Performs grid search over genetic algorithm parameters (population size, mutation rate, elite size, crossover rate) and visualizes their impact on performance.
- **src/gridsearch_cv_ga_params_v2.py**: Enhanced version of parameter grid search with additional metrics and visualizations.
- **src/gridsearch_cv_ga_population_test.py**: Specialized grid search focusing on population size effects.
- **src/gridsearch_cv_selection_methods.py**: Analyzes the effect of different selection method weights (tournament vs. roulette wheel) on fitness and convergence speed.
- **config/weights.py**: Configuration file for all algorithm parameters, operator weights, and experiment settings.
- **inputs/seating_data.csv**: Input file containing guest and relationship data (not included in this repo by default).
- **outputs/**: Directory for saving experiment results, plots, and generated reports.
- **reports/report.md**: Main project report, including problem definition, algorithm details, analysis, and results.

## Setup and Usage

1. **Install dependencies** (if any are required, e.g., numpy, matplotlib, seaborn, pandas):
   ```sh
   pip install -r requirements.txt
   ```
   (Create a requirements.txt if needed.)

2. **Prepare input data**: Place your guest and relationship data in `inputs/seating_data.csv`.

3. **Run the genetic algorithm**:
   ```sh
   python src/main.py
   ```
   (Or use the analysis scripts for parameter tuning and comparison.)

4. **Check Relationship Matrix and Close it**: If GENERATE_VISUALIZATIONS = True, Relationship matrix will show up, close it so algorithm runs

5. **Analyze results**: Generated plots and reports will be saved in the `outputs/` directory.

## Main Features
- Flexible configuration for all algorithm parameters
- Multiple crossover and mutation operators
- Hybrid selection (tournament and roulette wheel)
- Integrated repair mechanisms for solution validity
- Parameter grid search and analysis tools
- Visualization of fitness evolution and parameter effects

## Generated Graphs and Visualizations

This project includes several scripts that generate visualizations to help analyze the performance and behavior of the genetic algorithm under different configurations:

- **Fitness Evolution for Different Population Sizes** (`fitness_evolution_by_population.png`):  
  Shows how the best fitness score evolves over generations for various population sizes (e.g., 500, 1000, 5000, 10,000). This plot helps illustrate the impact of population size on convergence speed and solution quality.

- **Running Time for Different Population Sizes** (`running_time_by_population.png`):  
  Compares the total computation time required for each population size, providing insight into the trade-off between computational cost and solution quality.

- **Parameter Analysis** (`parameter_analysis.png`):  
  Boxplots showing the effect of different genetic algorithm parameters (population size, mutation rate, elite size, crossover rate) on the best fitness scores achieved. Useful for identifying optimal parameter ranges.

- **Mutation Rate and Population Size Interaction** (`mutation_population_interaction.png`):  
  Line plot illustrating how the average fitness score changes with mutation rate for different population sizes, highlighting the interaction between these two parameters.

- **Selection Weights Analysis** (`selection_weights_analysis.png`):  
  Line plots showing how the average fitness and convergence speed vary with different weights assigned to tournament and roulette wheel selection methods.

- **3D Parameter Space Visualization** (`3d_parameter_space.png`):  
  Interactive 3D visualization showing the relationship between population size, mutation rate, and fitness scores. This helps identify optimal parameter combinations in the three-dimensional parameter space.

- **Relationship Matrix** (`relationship_matrix.png`):  
  Heatmap visualization of guest relationships, showing the strength of connections between all pairs of guests.

- **Seating Arrangement Visualization** (`seating_arrangement.png`):  
  Visual representation of the final seating arrangement, showing table assignments and guest relationships.

- **Convergence Analysis** (`convergence_analysis.png`):  
  Plots showing the convergence behavior of the algorithm across multiple runs, including average and best fitness trends.

- **Parameter Sensitivity Analysis** (`parameter_sensitivity.png`):  
  Detailed analysis of how different parameter combinations affect algorithm performance and solution quality.

All generated plots are saved in the `outputs/` directory and are referenced in the main report for detailed analysis.

