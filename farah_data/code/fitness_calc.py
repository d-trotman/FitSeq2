#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# Set plot style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

def load_csv_data(filename):
    """Load CSV data and filter out rows with all empty cells."""
    df = pd.read_csv(filename)
    # Keep rows that have at least one non-NaN value in fitness columns
    fitness_cols = [col for col in df.columns if col != 'mutant_id']
    mask = df[fitness_cols].notna().any(axis=1)
    return df[mask]

def filter_extreme_values(df, min_val=-1, max_val=1):
    """Filter out rows with fitness values outside the specified range."""
    df_filtered = df.copy()
    
    # Get all columns except mutant_id
    fitness_cols = [col for col in df.columns if col != 'mutant_id']
    
    # For each fitness column, replace values outside range with NaN
    for col in fitness_cols:
        df_filtered.loc[df_filtered[col] > max_val, col] = np.nan
        df_filtered.loc[df_filtered[col] < min_val, col] = np.nan
    
    # Keep only rows that still have at least one valid value
    mask = df_filtered[fitness_cols].notna().any(axis=1)
    return df_filtered[mask]

def calculate_average_fitness(df):
    """Calculate average fitness for each mutant, ignoring the last timepoint."""
    # Get all columns except mutant_id
    fitness_cols = [col for col in df.columns if col != 'mutant_id']
    # Remove the last timepoint
    fitness_cols = fitness_cols[:-1]
    
    # Calculate mean fitness for each mutant
    result_df = df.copy()
    result_df['avg_fitness'] = result_df[fitness_cols].mean(axis=1, skipna=True)
    return result_df[['mutant_id', 'avg_fitness']]

def calculate_component_fitness(switch_df):
    """Calculate component fitness from Switch data for Clim and Nlim cycles, then take absolute values."""
    # Define which timepoints correspond to which environment
    # Corrected: Removed t7-t8 from clim_cycles to avoid overlap
    clim_cycles = ['t0-t1', 't2-t3', 't4-t5', 't6-t7', 't8-t9']
    nlim_cycles = ['t1-t2', 't3-t4', 't5-t6', 't7-t8']
    
    # Filter columns that exist in the dataframe
    existing_clim_cycles = [col for col in clim_cycles if col in switch_df.columns]
    existing_nlim_cycles = [col for col in nlim_cycles if col in switch_df.columns]
    
    # Calculate mean component fitness and then take absolute values
    result_df = switch_df[['mutant_id']].copy()
    result_df['clim_component'] = switch_df[existing_clim_cycles].mean(axis=1, skipna=True).abs()
    result_df['nlim_component'] = switch_df[existing_nlim_cycles].mean(axis=1, skipna=True).abs()
    
    return result_df

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Analyze fitness data across different environments.')
    parser.add_argument('--clim', type=str, 
                       default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/fit_chk_calc/Clim_rep1_individual_mutant_fitness.csv',
                       help='Path to the Carbon-limited fitness data CSV file')
    parser.add_argument('--nlim', type=str, 
                       default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/fit_chk_calc/Nlim_rep1_individual_mutant_fitness.csv',
                       help='Path to the Nitrogen-limited fitness data CSV file')
    parser.add_argument('--switch', type=str, 
                       default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/fit_chk_calc/Switch_rep1_individual_mutant_fitness.csv',
                       help='Path to the Switch environment fitness data CSV file')
    parser.add_argument('--output', type=str, 
                       default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/fit_chk_calc/plots/fitness_comparison.png',
                       help='Path to save the output plot')
    parser.add_argument('--min_fitness', type=float, default=-1,
                       help='Minimum fitness value to include in analysis')
    parser.add_argument('--max_fitness', type=float, default=1,
                       help='Maximum fitness value to include in analysis')
    
    args = parser.parse_args()
    
    print("Loading data...")
    # Load the CSV files and filter out invalid rows
    clim_df = load_csv_data(args.clim)
    nlim_df = load_csv_data(args.nlim)
    switch_df = load_csv_data(args.switch)
    
    print(f"Data loaded. Initial row counts: Clim={len(clim_df)}, Nlim={len(nlim_df)}, Switch={len(switch_df)}")
    
    # Filter extreme values
    print(f"Filtering out fitness values outside range [{args.min_fitness}, {args.max_fitness}]...")
    clim_df = filter_extreme_values(clim_df, args.min_fitness, args.max_fitness)
    nlim_df = filter_extreme_values(nlim_df, args.min_fitness, args.max_fitness)
    switch_df = filter_extreme_values(switch_df, args.min_fitness, args.max_fitness)
    
    print(f"After filtering, row counts: Clim={len(clim_df)}, Nlim={len(nlim_df)}, Switch={len(switch_df)}")
    
    # Calculate average fitness for each mutant in Clim and Nlim
    print("Calculating average fitness...")
    clim_avg = calculate_average_fitness(clim_df)
    nlim_avg = calculate_average_fitness(nlim_df)
    
    # Calculate |Clim fit - Nlim fit|/2 for each mutant
    print("Calculating fitness differences...")
    fitness_diff = pd.merge(clim_avg, nlim_avg, on='mutant_id', suffixes=('_clim', '_nlim'))
    fitness_diff['fitness_diff'] = abs(fitness_diff['avg_fitness_clim'] - fitness_diff['avg_fitness_nlim']) / 2
    
    # Calculate component fitness from Switch data
    print("Calculating component fitness in fluctuating environment...")
    component_fitness = calculate_component_fitness(switch_df)
    
    # Merge datasets for plotting
    plot_data = pd.merge(fitness_diff, component_fitness, on='mutant_id')
    print(f"Final dataset contains {len(plot_data)} mutants.")
    
    # Save the processed data
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    data_output = os.path.join(output_dir, 'processed_fitness_data.csv')
    plot_data.to_csv(data_output, index=False)
    print(f"Processed data saved to {data_output}")
    
    # Print some statistics
    print("\nStatistics:")
    print(f"Fitness Difference (|Clim - Nlim|/2): min={plot_data['fitness_diff'].min():.4f}, "
          f"max={plot_data['fitness_diff'].max():.4f}, mean={plot_data['fitness_diff'].mean():.4f}")
    print(f"Clim Component in Switch: min={plot_data['clim_component'].min():.4f}, "
          f"max={plot_data['clim_component'].max():.4f}, mean={plot_data['clim_component'].mean():.4f}")
    print(f"Nlim Component in Switch: min={plot_data['nlim_component'].min():.4f}, "
          f"max={plot_data['nlim_component'].max():.4f}, mean={plot_data['nlim_component'].mean():.4f}")
    
    # Create scatter plot
    print("Creating scatter plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the data
    ax.scatter(plot_data['fitness_diff'], plot_data['clim_component'], 
               alpha=0.6, label='Carbon-limited Component', color='blue')
    ax.scatter(plot_data['fitness_diff'], plot_data['nlim_component'], 
               alpha=0.6, label='Nitrogen-limited Component', color='green', marker='^')
    
    # Add labels and title
    ax.set_xlabel('|Clim fit - Nlim fit|/2', fontsize=14)
    ax.set_ylabel('Component Fitness in Switch Environment', fontsize=14)
    ax.set_title('Absolute Fitness in Fluctuating Environment vs. Fitness Difference in Static Environments', fontsize=16)
    
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend
    ax.legend(fontsize=12)
    
    # Add reference lines
    ax.axhline(y=0, color='grey', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='grey', linestyle='-', alpha=0.3)
    
    # Set axis limits
    ax.set_xlim(0, plot_data['fitness_diff'].max() * 1.05)
    # Set y-axis limits to show only filtered data
    y_min = min(plot_data['clim_component'].min(), plot_data['nlim_component'].min())
    y_max = max(plot_data['clim_component'].max(), plot_data['nlim_component'].max())
    padding = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # Add linear regression lines
    x = plot_data['fitness_diff']
    y_clim = plot_data['clim_component']
    y_nlim = plot_data['nlim_component']
    
    # Clim regression line
    m_clim, b_clim = np.polyfit(x, y_clim, 1)
    ax.plot(x, m_clim*x + b_clim, color='blue', linestyle='--', alpha=0.5)
    
    # Nlim regression line
    m_nlim, b_nlim = np.polyfit(x, y_nlim, 1)
    ax.plot(x, m_nlim*x + b_nlim, color='green', linestyle='--', alpha=0.5)
    
    # Add regression equations to the plot
    ax.annotate(f'y = {m_clim:.4f}x + {b_clim:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                color='blue', fontsize=10)
    ax.annotate(f'y = {m_nlim:.4f}x + {b_nlim:.4f}', 
                xy=(0.05, 0.90), xycoords='axes fraction',
                color='green', fontsize=10)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Plot saved as '{args.output}'")
    
    # Calculate correlation
    clim_corr = np.corrcoef(plot_data['fitness_diff'], plot_data['clim_component'])[0, 1]
    nlim_corr = np.corrcoef(plot_data['fitness_diff'], plot_data['nlim_component'])[0, 1]
    print(f"\nCorrelation between fitness difference and Clim component: {clim_corr:.4f}")
    print(f"Correlation between fitness difference and Nlim component: {nlim_corr:.4f}")
    
    plt.show()

if __name__ == "__main__":
    main()
