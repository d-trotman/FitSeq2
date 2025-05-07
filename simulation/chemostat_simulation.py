import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

"""
Simulation of barcoded S. cerevisiae strains growing in a chemostat
with fitness-based competition dynamics and continuous dilution.
"""

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_strains = 4000  # Number of different strains
fitness_std_dev = 0.1  # Standard deviation of fitness distribution
chemostat_size = 3e7  # Static population size (3 * 10^7 cells)
dilution_rate = 0.12  # Dilution rate per hour
num_timepoints = 10
total_time = 50 # Total simulation time in hours
time_points = np.linspace(0, total_time, num_timepoints)  # Sample at these time points

# Calculate generation time for this dilution rate (for reference)
generation_time = np.log(2) / dilution_rate  # Hours per generation
gens_between_samples = (time_points[1] - time_points[0]) / generation_time

print(f"Dilution rate: {dilution_rate:.4f} per hour")
print(f"Generation time: {generation_time:.2f} hours")
print(f"Hours between samples: {time_points[1] - time_points[0]:.2f}")
print(f"Generations between samples: {gens_between_samples:.2f}")
print(f"Total simulation time: {total_time:.2f} hours")
print(f"Total generations in simulation: {total_time/generation_time:.2f}")

# Step 1: Generate fitness values from Gaussian distribution
# Mean fitness is set to 1.0
fitness_values = np.random.normal(1.0, fitness_std_dev, num_strains)

# Initial equal frequencies for all strains
initial_frequencies = np.ones(num_strains) / num_strains
initial_population = initial_frequencies * chemostat_size

def calculate_mean_fitness(population, fitness):
    """Calculate average population fitness weighted by population size."""
    return np.sum(fitness * population) / np.sum(population)

def chemostat_growth_model(t, N, r, D):
    """
    Differential equation model for competitive growth in a chemostat.
    
    dNi/dt = Ni * (ri - r*) - D * Ni
    
    where:
    - Ni is the population of strain i
    - ri is the fitness of strain i
    - r* is the average population fitness
    - D is the dilution rate
    """
    # Calculate mean fitness
    total_pop = np.sum(N)
    mean_fitness = np.sum(r * N) / total_pop

    
    # Growth rate for each strain
    dNdt = N * (r - mean_fitness) - D * N
    
    return dNdt

def simulate_chemostat(initial_population, fitness_values, dilution_rate, time_points):
    """
    Simulate growth of multiple strains in a chemostat over time.
    
    Args:
        initial_population: Initial population sizes for each strain
        fitness_values: Fitness values for each strain
        dilution_rate: Dilution rate of the chemostat
        time_points: Time points at which to sample the population
    """
    # Solve the differential equation over the entire time span
    solution = solve_ivp(
        fun=lambda t, N: chemostat_growth_model(t, N, fitness_values, dilution_rate),
        t_span=[0, time_points[-1]],
        y0=initial_population,
        method='RK45',
        t_eval=time_points
    )
    
    # Extract population sizes at each time point
    populations = solution.y.T  # Transpose to get [timepoints, strains]
    
    return populations

def plot_strain_frequencies(df, num_strains, time_points, fitness_values, generation_time):
    """
    Plot the frequency of each strain over time on a log scale.
    """
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Extract all frequency data at once for efficiency
    frequency_data = np.zeros((num_strains, len(time_points)))
    for t in range(len(time_points)):
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
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Strain Frequency (log scale)')
    
    # Add average fitness line on secondary y-axis
    ax2 = ax1.twinx()
    avg_fitness = [df[f'avg_fitness_t{t}'].iloc[0] for t in range(len(time_points))]
    ax2.plot(time_points, avg_fitness, '--', color='black', linewidth=2, label='Avg. Fitness')
    ax2.set_ylabel('Average Population Fitness')
    
    # Calculate extinction threshold (frequency equivalent to 1 cell)
    extinction_threshold = 1.0 / chemostat_size
    
    # Add horizontal line at extinction threshold
    ax1.axhline(y=extinction_threshold, color='red', linestyle=':', linewidth=2)
    
    # Add text label for extinction line
    ax1.text(0, extinction_threshold*1.5, 'Extinction Threshold (0 reads)', 
             color='red', fontsize=10, ha='left', va='bottom')
    
    # Add generation markers on top x-axis
    ax3 = ax1.twiny()
    generation_numbers = time_points / generation_time
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticks(time_points)
    ax3.set_xticklabels([f"{g:.1f}" for g in generation_numbers])
    ax3.set_xlabel("Generations")
    
    # Title
    plt.title('Frequency of S. cerevisiae Strains in Chemostat Over Time')
    
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
    plt.savefig('yeast_chemostat_simulation.png', dpi=300)
    plt.close()
    
    print("Plot generated and saved as 'yeast_chemostat_simulation.png'")

# Simulate growth in chemostat
print("Simulating chemostat growth...")
population_data = simulate_chemostat(initial_population, fitness_values, dilution_rate, time_points)

# Generate CSV with required data
print("Preparing data for CSV output...")
data = []

for strain_idx in range(num_strains):
    strain_data = {
        'strain_id': f'strain_{strain_idx}',
        'initial_fitness': fitness_values[strain_idx]
    }
    
    # Add read data and frequency for each time point
    for t in range(len(time_points)):
        strain_data[f'reads_t{t}'] = int(population_data[t, strain_idx])
        strain_data[f'frequency_t{t}'] = population_data[t, strain_idx] / np.sum(population_data[t])
    
    data.append(strain_data)

# Create DataFrame
df = pd.DataFrame(data)

# Add average population fitness at each time point
for t in range(len(time_points)):
    avg_fitness = calculate_mean_fitness(population_data[t], fitness_values)
    df[f'avg_fitness_t{t}'] = avg_fitness

# Save to CSV
output_file = 'yeast_chemostat_simulation.csv'
df.to_csv(output_file, index=False)

print(f"Simulation complete! Data saved to {output_file}")
print(f"{num_strains} strains simulated across {len(time_points)} time points.")

# Generate visualization
print("Generating visualization...")
plot_strain_frequencies(df, num_strains, time_points, fitness_values, generation_time)