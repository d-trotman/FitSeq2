import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr

"""
Comparison script for analyzing the outputs of discrete dilution and chemostat models
of barcoded S. cerevisiae competition dynamics.

This script assumes you have already run both simulations and have CSV files:
- yeast_strain_simulation.csv (discrete model)
- yeast_chemostat_simulation.csv (chemostat model)
"""

# Chemostat parameters (needed for generation calculation)
chemostat_dilution_rate = 0.12  # Dilution rate per hour
generation_time = np.log(2) / chemostat_dilution_rate  # Hours per generation
chemostat_time = 50  # Total simulation time in hours

# Load data from CSV files
print("Loading simulation results...")
discrete_df = pd.read_csv('yeast_strain_simulation.csv')
chemostat_df = pd.read_csv('yeast_chemostat_simulation.csv')

# Extract fitness values (should be the same for both models)
fitness_values = discrete_df['initial_fitness'].values
num_strains = len(fitness_values)
num_timepoints = 10

print(f"Loaded data for {num_strains} strains across {num_timepoints} timepoints.")
print(f"Generation time in chemostat: {generation_time:.2f} hours")

def compare_models(discrete_df, chemostat_df, fitness_values):
    """Compare strain frequencies between discrete and chemostat models."""
    
    num_strains = len(fitness_values)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Extract timepoints
    discrete_timepoints = np.arange(num_timepoints)
    chemostat_timepoints = np.arange(num_timepoints)
    
    # Sort strains by fitness for coloring
    fitness_order = np.argsort(fitness_values)
    
    # Plot top and bottom 5 strains by fitness for each model
    for model_idx, (df, ax, title) in enumerate([
        (discrete_df, ax1, "Discrete Dilution Model"), 
        (chemostat_df, ax2, "Chemostat Model")
    ]):
        
        # Extract frequency data for all strains
        frequency_data = np.zeros((num_strains, num_timepoints))
        for t in range(num_timepoints):
            col_name = f'frequency_t{t}'
            frequency_data[:, t] = df[col_name].values
        
        # Plot top 5 and bottom 5 strains by fitness
        for i in list(fitness_order[-5:]) + list(fitness_order[:5]):
            # Normalize color by fitness
            color_val = (fitness_values[i] - min(fitness_values)) / (max(fitness_values) - min(fitness_values))
            color = plt.cm.viridis(color_val)
            
            # Plot with different line styles for each model
            timepoints = discrete_timepoints if model_idx == 0 else chemostat_timepoints
            ax.plot(timepoints, frequency_data[i], '-', color=color, 
                    alpha=0.7, linewidth=1.5,
                    label=f'Strain {i} (r={fitness_values[i]:.2f})' if model_idx == 0 else None)
        
        # Set y-axis to log scale
        ax.set_yscale('log')
        ax.set_ylabel('Strain Frequency (log scale)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    # Plot the differences between models
    ax3.set_title("Difference in Strain Frequencies (Chemostat - Discrete)")
    
    # Extract the same top and bottom strains for differences
    for i in list(fitness_order[-5:]) + list(fitness_order[:5]):
        # Get frequency data for this strain from both models
        discrete_freq = np.array([discrete_df[f'frequency_t{t}'].iloc[i] for t in range(num_timepoints)])
        chemostat_freq = np.array([chemostat_df[f'frequency_t{t}'].iloc[i] for t in range(num_timepoints)])
        
        # Calculate log-ratio difference to compare relative changes
        # Add a small value to avoid log(0)
        epsilon = 1e-10
        log_ratio = np.log10((chemostat_freq + epsilon) / (discrete_freq + epsilon))
        
        # Normalize color by fitness
        color_val = (fitness_values[i] - min(fitness_values)) / (max(fitness_values) - min(fitness_values))
        color = plt.cm.viridis(color_val)
        
        # Plot the difference
        ax3.plot(discrete_timepoints, log_ratio, '-', color=color, 
                alpha=0.7, linewidth=1.5,
                label=f'Strain {i} (r={fitness_values[i]:.2f})')
    
    ax3.axhline(y=0, color='red', linestyle=':', linewidth=1)
    ax3.set_ylabel('Log10 Frequency Ratio\n(Chemostat/Discrete)')
    ax3.set_xlabel('Time Point')
    
    # Add legend for top and bottom strains
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1), ncol=1)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.close()
    
    print("Comparison plot generated and saved as 'model_comparison.png'")

def analyze_model_differences(discrete_df, chemostat_df, fitness_values):
    """Calculate statistical differences between models."""
    
    # Compare average fitness over time
    discrete_avg_fitness = [discrete_df[f'avg_fitness_t{t}'].iloc[0] for t in range(num_timepoints)]
    chemostat_avg_fitness = [chemostat_df[f'avg_fitness_t{t}'].iloc[0] for t in range(num_timepoints)]
    
    # Calculate correlation between strain frequencies at each timepoint
    correlations = []
    rank_correlations = []
    
    for t in range(num_timepoints):
        discrete_freq = discrete_df[f'frequency_t{t}']
        chemostat_freq = chemostat_df[f'frequency_t{t}']
        
        # Pearson correlation (linear)
        corr = np.corrcoef(discrete_freq, chemostat_freq)[0, 1]
        correlations.append(corr)
        
        # Spearman rank correlation (rank order)
        rank_corr = spearmanr(discrete_freq, chemostat_freq)[0]
        rank_correlations.append(rank_corr)
    
    # Calculate fitness effects
    discrete_final = discrete_df['frequency_t9']
    chemostat_final = chemostat_df['frequency_t9']
    
    # Linear regression of final frequency vs fitness for each model
    discrete_slope, _, discrete_r, _, _ = linregress(fitness_values, np.log10(discrete_final + 1e-10))
    chemostat_slope, _, chemostat_r, _, _ = linregress(fitness_values, np.log10(chemostat_final + 1e-10))
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot average fitness over time
    ax1.plot(range(num_timepoints), discrete_avg_fitness, 'o-', label='Discrete Model')
    ax1.plot(range(num_timepoints), chemostat_avg_fitness, 's-', label='Chemostat Model')
    ax1.set_xlabel('Time Point')
    ax1.set_ylabel('Average Population Fitness')
    ax1.set_title('Average Fitness Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot correlations over time
    ax2.plot(range(num_timepoints), correlations, 'o-', label='Pearson Correlation')
    ax2.plot(range(num_timepoints), rank_correlations, 's-', label='Spearman Rank Correlation')
    ax2.set_xlabel('Time Point')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_title('Model Correlation Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_statistics.png', dpi=300)
    plt.close()
    
    print("Statistical analysis plot saved as 'model_statistics.png'")
    print(f"Fitness effect (slope) in discrete model: {discrete_slope:.3f} (R² = {discrete_r**2:.3f})")
    print(f"Fitness effect (slope) in chemostat model: {chemostat_slope:.3f} (R² = {chemostat_r**2:.3f})")

def compare_by_generations(discrete_df, chemostat_df, fitness_values, chemostat_generation_time, chemostat_hours):
    """Compare models at equivalent generation counts."""
    
    # Calculate generations at each discrete timepoint (5 gens between each point)
    discrete_generations = np.arange(0, 10*5, 5)
    
    # Calculate generations at each chemostat timepoint
    chemostat_timepoints = np.linspace(0, chemostat_hours, 10)
    chemostat_generations = chemostat_timepoints / chemostat_generation_time
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot average fitness vs generations for both models
    discrete_avg_fitness = [discrete_df[f'avg_fitness_t{t}'].iloc[0] for t in range(10)]
    chemostat_avg_fitness = [chemostat_df[f'avg_fitness_t{t}'].iloc[0] for t in range(10)]
    
    ax.plot(discrete_generations, discrete_avg_fitness, 'o-', label='Discrete Model')
    ax.plot(chemostat_generations, chemostat_avg_fitness, 's-', label='Chemostat Model')
    
    ax.set_xlabel('Generations')
    ax.set_ylabel('Average Population Fitness')
    ax.set_title('Average Fitness vs Generations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fitness_by_generations.png', dpi=300)
    plt.close()
    
    print("Generation-normalized comparison saved as 'fitness_by_generations.png'")

def plot_strain_fitness_correlations(discrete_df, chemostat_df, fitness_values):
    """Plot the correlation between strain fitness and final frequency."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Get final frequencies
    discrete_final = discrete_df['frequency_t9']
    chemostat_final = chemostat_df['frequency_t9']
    
    # Plot discrete model
    ax1.scatter(fitness_values, np.log10(discrete_final + 1e-10), alpha=0.7)
    ax1.set_xlabel('Strain Fitness')
    ax1.set_ylabel('Log10 Final Frequency')
    ax1.set_title('Discrete Model: Fitness vs Final Frequency')
    
    # Add regression line
    discrete_slope, discrete_intercept, _, _, _ = linregress(fitness_values, np.log10(discrete_final + 1e-10))
    x_range = np.linspace(min(fitness_values), max(fitness_values), 100)
    ax1.plot(x_range, discrete_slope * x_range + discrete_intercept, 'r-', 
             label=f'Slope = {discrete_slope:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot chemostat model
    ax2.scatter(fitness_values, np.log10(chemostat_final + 1e-10), alpha=0.7)
    ax2.set_xlabel('Strain Fitness')
    ax2.set_ylabel('Log10 Final Frequency')
    ax2.set_title('Chemostat Model: Fitness vs Final Frequency')
    
    # Add regression line
    chemostat_slope, chemostat_intercept, _, _, _ = linregress(fitness_values, np.log10(chemostat_final + 1e-10))
    x_range = np.linspace(min(fitness_values), max(fitness_values), 100)
    ax2.plot(x_range, chemostat_slope * x_range + chemostat_intercept, 'r-', 
             label=f'Slope = {chemostat_slope:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fitness_correlation.png', dpi=300)
    plt.close()
    
    print("Fitness correlation plot saved as 'fitness_correlation.png'")

def extinction_analysis(discrete_df, chemostat_df, population_size):
    """Analyze when strains go extinct in each model."""
    
    # Calculate extinction threshold
    extinction_threshold = 1.0 / population_size
    
    # Count strains below threshold at each timepoint
    discrete_extinct = []
    chemostat_extinct = []
    
    for t in range(num_timepoints):
        discrete_count = sum(discrete_df[f'frequency_t{t}'] < extinction_threshold)
        chemostat_count = sum(chemostat_df[f'frequency_t{t}'] < extinction_threshold)
        
        discrete_extinct.append(discrete_count)
        chemostat_extinct.append(chemostat_count)
    
    # Plot extinction curves
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(range(num_timepoints), discrete_extinct, 'o-', label='Discrete Model')
    ax.plot(range(num_timepoints), chemostat_extinct, 's-', label='Chemostat Model')
    
    ax.set_xlabel('Time Point')
    ax.set_ylabel('Number of Extinct Strains')
    ax.set_title('Strain Extinction Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('extinction_analysis.png', dpi=300)
    plt.close()
    
    print("Extinction analysis plot saved as 'extinction_analysis.png'")

# Run comparison analyses
print("\nComparing models...")
compare_models(discrete_df, chemostat_df, fitness_values)
analyze_model_differences(discrete_df, chemostat_df, fitness_values)
compare_by_generations(discrete_df, chemostat_df, fitness_values, generation_time, chemostat_time)
plot_strain_fitness_correlations(discrete_df, chemostat_df, fitness_values)
extinction_analysis(discrete_df, chemostat_df, 3e7)  # Using the population size of 3*10^7

print("\nAnalysis complete! All visualizations generated.")

