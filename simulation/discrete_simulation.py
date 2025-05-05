import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

"""
Simulation of barcoded S. cerevisiae strains growing in liquid culture
with fitness-based competition dynamics and visualization.
"""

# Set random seed for reproducibility
np.random.seed(42)  # This ensures the same "random" numbers are generated each time the code runs

# Parameters
num_strains = 4000  # Number of different strains
fitness_std_dev = 0.1  # Standard deviation of fitness distribution
bottleneck_size = 3e7  # 3 * 10^7 cells
carrying_capacity = bottleneck_size
generations_between_bottlenecks = 5
dilution_factor = np.log(2) / np.log(18)
num_timepoints = 10

# Step 1: Generate fitness values from Gaussian distribution
# Mean fitness is set to 1.0
fitness_values = np.random.normal(1.0, fitness_std_dev, num_strains)

# Initial equal frequencies for all strains
initial_frequencies = np.ones(num_strains) / num_strains
initial_population = initial_frequencies * bottleneck_size

def calculate_mean_fitness(population, fitness):
    """Calculate average population fitness weighted by population size."""
    return np.sum(fitness * population) / np.sum(population)

def growth_model(t, N, r, K):
    """
    Differential equation model for competitive growth.
    
    Implements: dNi/dt = Ni(ri - ((sum from j=1 to s of rjNj)/K)
    
    Args:
        t: Time (required by solver but not used in equation)
        N: Array of population sizes for each strain
        r: Array of fitness values for each strain
        K: Carrying capacity
    """
    weighted_sum = np.sum(r * N) / K
    dNdt = N * (r - weighted_sum)
    return dNdt

def simulate_growth(initial_population, fitness_values, num_timepoints, carrying_capacity):
    """
    Simulate growth of multiple strains over time with bottlenecks.
    """
    populations = np.zeros((num_timepoints, len(initial_population)))
    populations[0] = initial_population
    
    current_population = initial_population.copy()
    
    for t in range(1, num_timepoints):
        # Simulate growth for specified number of generations
        time_span = [0, generations_between_bottlenecks]
        
        # Solve the differential equation
        solution = solve_ivp(
            fun=lambda t, N: growth_model(t, N, fitness_values, carrying_capacity),
            t_span=time_span,
            y0=current_population,
            method='RK45'
        )
        
        # Get the population after growth
        grown_population = solution.y[:, -1]
        
        # Apply bottleneck (dilution)
        total_population = np.sum(grown_population)
        frequencies = grown_population / total_population
        current_population = frequencies * bottleneck_size * dilution_factor
        
        # Store the population
        populations[t] = current_population
    
    return populations

def plot_strain_frequencies(df, num_strains, num_timepoints, fitness_values):
    """
    Plot the frequency of each strain over time on a log scale.
    
    Args:
        df: DataFrame containing the strain data
        num_strains: Number of strains
        num_timepoints: Number of time points
        fitness_values: Array of fitness values for each strain
    """
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Create time points array
    time_points = list(range(num_timepoints))
    
    # Extract all frequency data at once for efficiency
    frequency_data = np.zeros((num_strains, num_timepoints))
    for t in range(num_timepoints):
        col_name = f'frequency_t{t}'
        frequency_data[:, t] = df[col_name].values
    
    # Sort strains by fitness for coloring
    fitness_order = np.argsort(fitness_values)
    
    # Plot each strain's frequency over time
    for i, strain_idx in enumerate(fitness_order):
        # Normalize color by fitness rank (0 to 1)
        color_val = i / (num_strains - 1)
        color = plt.cm.viridis(color_val)
        
        # Adjust line width slightly based on fitness (thicker for higher fitness)
        linewidth = 0.8 + 1.2 * color_val
        
        # Plot frequency data on ax1
        ax1.plot(time_points, frequency_data[strain_idx], '-', color=color, 
                alpha=0.7, linewidth=linewidth,
                label=f'Strain {strain_idx} (r={fitness_values[strain_idx]:.2f})' 
                      if i >= num_strains - 5 else None)  # Only show top 5 fittest strains in legend
    
    # Set y-axis to log scale
    ax1.set_yscale('log')
    ax1.set_xlabel('Time Point')
    ax1.set_ylabel('Strain Frequency (log scale)')
    
    # Add average fitness line on secondary y-axis
    ax2 = ax1.twinx()
    avg_fitness = [df[f'avg_fitness_t{t}'].iloc[0] for t in range(num_timepoints)]
    ax2.plot(time_points, avg_fitness, '--', color='black', linewidth=2, label='Avg. Fitness')
    ax2.set_ylabel('Average Population Fitness')
    
    # Calculate extinction threshold (frequency equivalent to 1 cell)
    extinction_threshold = 1.0 / bottleneck_size
    
    # Add horizontal line at extinction threshold
    ax1.axhline(y=extinction_threshold, color='red', linestyle=':', linewidth=2)
    
    # Add text label for extinction line - now on the LEFT side
    ax1.text(0, extinction_threshold*1.5, 'Extinction Threshold (0 reads)', 
             color='red', fontsize=10, ha='left', va='bottom')
    
    # Title
    plt.title('Simulated Frequency of S. cerevisiae Strains Over Time')
    
    # Legend for the top 5 fittest strains and average fitness
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Add a colorbar to show fitness scale
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(min(fitness_values), max(fitness_values)))
    sm.set_array([])
    # Explicitly pass ax1 as the axis to associate with
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Strain Fitness')
    
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    
    # Save the figure
    plt.savefig('discrete_frequencies_plot.png', dpi=300)
    plt.close()
    
    print("Plot generated and saved as 'strain_frequencies_plot.png'")

# Step 2: Simulate growth
print("Simulating growth across 10 time points...")
population_data = simulate_growth(initial_population, fitness_values, num_timepoints, carrying_capacity)

# Step 3: Generate CSV with required data
print("Preparing data for CSV output...")
data = []

for strain_idx in range(num_strains):
    strain_data = {
        'strain_id': f'strain_{strain_idx}',
        'initial_fitness': fitness_values[strain_idx]
    }
    
    # Add read data and frequency for each time point
    for t in range(num_timepoints):
        strain_data[f'reads_t{t}'] = int(population_data[t, strain_idx])
        strain_data[f'frequency_t{t}'] = population_data[t, strain_idx] / np.sum(population_data[t])
    
    data.append(strain_data)

# Create DataFrame
df = pd.DataFrame(data)

# Add average population fitness at each time point
for t in range(num_timepoints):
    avg_fitness = calculate_mean_fitness(population_data[t], fitness_values)
    df[f'avg_fitness_t{t}'] = avg_fitness

# Save to CSV
output_file = 'yeast_discrete_simulation.csv'
df.to_csv(output_file, index=False)

print(f"Simulation complete! Data saved to {output_file}")
print(f"{num_strains} strains simulated across {num_timepoints} time points.")

# Generate visualization
print("Generating visualization...")
plot_strain_frequencies(df, num_strains, num_timepoints, fitness_values)