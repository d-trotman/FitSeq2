#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import warnings
from scipy import stats

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
    """Calculate component fitness from Switch data for Clim and Nlim cycles."""
    # Define which timepoints correspond to which environment
    clim_cycles = ['t0-t1', 't2-t3', 't4-t5', 't6-t7', 't8-t9']
    nlim_cycles = ['t1-t2', 't3-t4', 't5-t6', 't7-t8']
    
    # Filter columns that exist in the dataframe
    existing_clim_cycles = [col for col in clim_cycles if col in switch_df.columns]
    existing_nlim_cycles = [col for col in nlim_cycles if col in switch_df.columns]
    
    # Calculate mean component fitness
    result_df = switch_df[['mutant_id']].copy()
    result_df['clim_component'] = switch_df[existing_clim_cycles].mean(axis=1, skipna=True)
    result_df['nlim_component'] = switch_df[existing_nlim_cycles].mean(axis=1, skipna=True)
    
    return result_df

def safe_correlation(x, y):
    """Safely calculate correlation with proper error handling."""
    # Drop any NaN or infinite values
    df = pd.DataFrame({'x': x, 'y': y})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(df) < 2:
        return None
    
    try:
        # Use scipy's pearsonr which is more robust
        corr, p_val = stats.pearsonr(df['x'], df['y'])
        return corr, p_val
    except:
        # Fall back to pandas correlation
        try:
            corr = df['x'].corr(df['y'])
            return corr, None
        except:
            return None, None

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Compare fitness in fluctuating vs. static environments.')
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
                       default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/fit_chk_calc/plots/fitness_environment_differences.png',
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
    
    # Calculate average fitness for each mutant in static environments
    print("Calculating average fitness in static environments...")
    clim_avg = calculate_average_fitness(clim_df)
    nlim_avg = calculate_average_fitness(nlim_df)
    
    # Calculate component fitness from Switch data
    print("Calculating component fitness in fluctuating environment...")
    component_fitness = calculate_component_fitness(switch_df)
    
    # Merge datasets for analysis
    print("Calculating fitness differences between environments...")
    # Merge static Clim data with Switch component data
    clim_merged = pd.merge(clim_avg, component_fitness, on='mutant_id')
    # Rename columns for clarity
    clim_merged = clim_merged.rename(columns={'avg_fitness': 'clim_static'})
    
    # Merge with static Nlim data
    all_merged = pd.merge(clim_merged, nlim_avg, on='mutant_id')
    all_merged = all_merged.rename(columns={'avg_fitness': 'nlim_static'})
    
    # Calculate differences between fluctuating and static environments
    all_merged['clim_diff'] = all_merged['clim_component'] - all_merged['clim_static']
    all_merged['nlim_diff'] = all_merged['nlim_component'] - all_merged['nlim_static']
    
    # Remove any rows with NaN values
    plot_data = all_merged.dropna(subset=['clim_diff', 'nlim_diff'])
    
    print(f"Final dataset contains {len(plot_data)} mutants.")
    
    # Save the processed data
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    data_output = os.path.join(output_dir, 'processed_fitness_environment_differences.csv')
    plot_data.to_csv(data_output, index=False)
    print(f"Processed data saved to {data_output}")
    
    # Print some statistics
    print("\nStatistics:")
    print(f"Clim difference (fluctuating - static): min={plot_data['clim_diff'].min():.4f}, "
          f"max={plot_data['clim_diff'].max():.4f}, mean={plot_data['clim_diff'].mean():.4f}")
    print(f"Nlim difference (fluctuating - static): min={plot_data['nlim_diff'].min():.4f}, "
          f"max={plot_data['nlim_diff'].max():.4f}, mean={plot_data['nlim_diff'].mean():.4f}")
    
    # Calculate the percentage of mutants in each quadrant
    q1 = len(plot_data[(plot_data['clim_diff'] > 0) & (plot_data['nlim_diff'] > 0)]) / len(plot_data) * 100
    q2 = len(plot_data[(plot_data['clim_diff'] < 0) & (plot_data['nlim_diff'] > 0)]) / len(plot_data) * 100
    q3 = len(plot_data[(plot_data['clim_diff'] < 0) & (plot_data['nlim_diff'] < 0)]) / len(plot_data) * 100
    q4 = len(plot_data[(plot_data['clim_diff'] > 0) & (plot_data['nlim_diff'] < 0)]) / len(plot_data) * 100
    
    print("\nPercentage of mutants in each quadrant:")
    print(f"Q1 (Clim+, Nlim+): {q1:.2f}%")
    print(f"Q2 (Clim-, Nlim+): {q2:.2f}%")
    print(f"Q3 (Clim-, Nlim-): {q3:.2f}%")
    print(f"Q4 (Clim+, Nlim-): {q4:.2f}%")
    
    # Calculate correlation
    corr, p_val = safe_correlation(plot_data['clim_diff'], plot_data['nlim_diff'])
    
    if corr is not None:
        print(f"\nCorrelation between Clim and Nlim differences: {corr:.4f}")
        if p_val is not None:
            print(f"P-value: {p_val:.6f}")
    else:
        print("\nCould not calculate correlation")
    
    # Create scatter plot
    print("Creating scatter plot...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set background colors for quadrants
    ax.axhspan(0, plot_data['nlim_diff'].max() * 1.1, xmin=0.5, xmax=1, alpha=0.1, color='green', zorder=0)
    ax.axhspan(plot_data['nlim_diff'].min() * 1.1, 0, xmin=0.5, xmax=1, alpha=0.1, color='red', zorder=0)
    ax.axhspan(0, plot_data['nlim_diff'].max() * 1.1, xmin=0, xmax=0.5, alpha=0.1, color='blue', zorder=0)
    ax.axhspan(plot_data['nlim_diff'].min() * 1.1, 0, xmin=0, xmax=0.5, alpha=0.1, color='purple', zorder=0)
    
    # Plot the data
    scatter = ax.scatter(
        plot_data['clim_diff'], 
        plot_data['nlim_diff'],
        alpha=0.7, 
        c=abs(plot_data['clim_diff']) + abs(plot_data['nlim_diff']),  # Color by total absolute difference
        cmap='viridis',
        s=30,  # Point size
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Total Absolute Difference', rotation=270, labelpad=20)
    
    # Add labels and title
    ax.set_xlabel('Clim Fluctuating - Clim Static', fontsize=14)
    ax.set_ylabel('Nlim Fluctuating - Nlim Static', fontsize=14)
    ax.set_title('Fitness Differences Between Fluctuating and Static Environments', fontsize=16)
    
    # Add quadrant labels
    ax.text(plot_data['clim_diff'].max() * 0.75, plot_data['nlim_diff'].max() * 0.75, 
            f"Q1: {q1:.1f}%\nBetter in both\nfluctuating environments", 
            ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    
    ax.text(plot_data['clim_diff'].min() * 0.75, plot_data['nlim_diff'].max() * 0.75, 
            f"Q2: {q2:.1f}%\nWorse in Clim\nBetter in Nlim", 
            ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    
    ax.text(plot_data['clim_diff'].min() * 0.75, plot_data['nlim_diff'].min() * 0.75, 
            f"Q3: {q3:.1f}%\nWorse in both\nfluctuating environments", 
            ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    
    ax.text(plot_data['clim_diff'].max() * 0.75, plot_data['nlim_diff'].min() * 0.75, 
            f"Q4: {q4:.1f}%\nBetter in Clim\nWorse in Nlim", 
            ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, zorder=1)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, zorder=1)
    
    # Add correlation text
    if corr is not None:
        ax.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.5), fontsize=10)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Plot saved as '{args.output}'")
    
    plt.show()

if __name__ == "__main__":
    main()
