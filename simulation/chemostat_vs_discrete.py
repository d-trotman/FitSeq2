import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the data
chemostat_data = pd.read_csv('yeast_chemostat_simulation.csv')
discrete_data = pd.read_csv('yeast_discrete_simulation.csv')

# Ensure both datasets have the same strains and are in the same order
merged_data = pd.merge(chemostat_data, discrete_data, on='strain_id', suffixes=('_chemo', '_discrete'))

# Create a directory for saving plots
output_dir = 'frequency_plots'
os.makedirs(output_dir, exist_ok=True)

# Generate scatter plots for each time point
time_points = range(10)  # t0 through t9

# Small value to add to zeros to avoid log(0)
epsilon = 1e-15

for t in time_points:
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Column names for the current time point
    freq_col_chemo = f'frequency_t{t}_chemo'
    freq_col_discrete = f'frequency_t{t}_discrete'
    
    # Add epsilon to zero values to handle log scale
    x_values = np.maximum(merged_data[freq_col_chemo], epsilon)
    y_values = np.maximum(merged_data[freq_col_discrete], epsilon)
    
    # Create scatter plot
    plt.scatter(x_values, y_values, alpha=0.6)
    
    # Set log scale for both axes
    plt.xscale('log')
    plt.yscale('log')
    
    # Add diagonal line (y=x) to help visualize differences
    min_val = min(x_values.min(), y_values.min())
    max_val = max(x_values.max(), y_values.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    # Add titles and labels
    plt.title(f'Strain Frequencies at Time Point {t} (Log Scale)', fontsize=14)
    plt.xlabel('Frequency in Chemostat Environment (Log Scale)', fontsize=12)
    plt.ylabel('Frequency in Discrete Environment (Log Scale)', fontsize=12)
    
    # Add grid with log-scale compatible styling
    plt.grid(True, alpha=0.3, which="both")
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{output_dir}/log_frequency_comparison_t{t}.png', dpi=300)
    
    # Close the figure to free memory
    plt.close()

print(f"All log-scale plots have been saved to the '{output_dir}' directory.")

# Create a combined figure showing all time points in a grid
plt.figure(figsize=(20, 16))

for t in time_points:
    plt.subplot(3, 4, t+1)
    
    freq_col_chemo = f'frequency_t{t}_chemo'
    freq_col_discrete = f'frequency_t{t}_discrete'
    
    # Add epsilon to zero values to handle log scale
    x_values = np.maximum(merged_data[freq_col_chemo], epsilon)
    y_values = np.maximum(merged_data[freq_col_discrete], epsilon)
    
    plt.scatter(x_values, y_values, alpha=0.5, s=10)
    
    # Set log scales
    plt.xscale('log')
    plt.yscale('log')
    
    # Add diagonal line
    min_val = min(x_values.min(), y_values.min())
    max_val = max(x_values.max(), y_values.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    plt.title(f'Time Point {t}')
    
    if t % 3 == 0:  # Add y-label only for leftmost plots
        plt.ylabel('Discrete (Log)')
    
    if t >= 6:  # Add x-label only for bottom plots
        plt.xlabel('Chemostat (Log)')
    
    plt.grid(True, alpha=0.3, which="both")

plt.suptitle('Log-scale Comparison of Strain Frequencies Across All Time Points', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust to make room for the suptitle
plt.savefig(f'{output_dir}/all_timepoints_log_comparison.png', dpi=300)
plt.close()

print("Combined log-scale plot of all time points has also been saved.")

