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
    """Calculate component fitness from Switch data for Clim and Nlim cycles, then take absolute values."""
    # Define which timepoints correspond to which environment
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

def safe_correlation(x, y):
    """Safely calculate correlation with proper error handling."""
    # Drop any NaN or infinite values
    df = pd.DataFrame({'x': x, 'y': y})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(df) < 2:
        return None
    
    try:
        # Use scipy's pearsonr which is more robust
        corr, _ = stats.pearsonr(df['x'], df['y'])
        return corr
    except:
        # Fall back to pandas correlation
        try:
            corr = df['x'].corr(df['y'])
            return corr
        except:
            return None

def safe_polyfit(x, y, deg=1):
    """Safely perform polynomial fitting with error handling."""
    # Create a DataFrame to handle the data cleaning
    df = pd.DataFrame({'x': x, 'y': y})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(df) < 2:
        return None, None
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use scipy's linregress which is more robust for simple linear regression
            slope, intercept, _, _, _ = stats.linregress(df['x'], df['y'])
            return slope, intercept
    except:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coeffs = np.polyfit(df['x'], df['y'], deg)
                return coeffs[0], coeffs[1]  # slope, intercept
        except:
            return None, None

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
                       default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/fit_chk_calc/plots/fitness_comparison_threshold.png',
                       help='Path to save the output plot')
    parser.add_argument('--min_fitness', type=float, default=-1,
                       help='Minimum fitness value to include in analysis')
    parser.add_argument('--max_fitness', type=float, default=1,
                       help='Maximum fitness value to include in analysis')
    parser.add_argument('--threshold', type=float, default=0.2,
                       help='Minimum y-axis value (component fitness) to include in the plot')
    
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
    
    # Remove any rows that have NaN values in the plotting columns
    plot_data = plot_data.dropna(subset=['fitness_diff', 'clim_component', 'nlim_component'])
    
    # Save the complete processed data before threshold filtering
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    complete_data_output = os.path.join(output_dir, 'processed_fitness_data_complete.csv')
    plot_data.to_csv(complete_data_output, index=False)
    print(f"Complete processed data saved to {complete_data_output}")
    
    # Filter to only include mutants with component fitness > threshold
    threshold = args.threshold
    print(f"Filtering to only include mutants with component fitness > {threshold}...")
    
    pre_filter_count = len(plot_data)
    plot_data = plot_data[
        (plot_data['clim_component'] > threshold) | 
        (plot_data['nlim_component'] > threshold)
    ]
    post_filter_count = len(plot_data)
    
    print(f"Filtered out {pre_filter_count - post_filter_count} mutants with component fitness <= {threshold}")
    print(f"Final dataset contains {post_filter_count} mutants")
    
    # Save the filtered processed data
    filtered_data_output = os.path.join(output_dir, f'processed_fitness_data_threshold_{threshold}.csv')
    plot_data.to_csv(filtered_data_output, index=False)
    print(f"Threshold-filtered data saved to {filtered_data_output}")
    
    # Print some statistics on the filtered data
    print("\nStatistics for threshold-filtered data:")
    print(f"Fitness Difference (|Clim - Nlim|/2): min={plot_data['fitness_diff'].min():.4f}, "
          f"max={plot_data['fitness_diff'].max():.4f}, mean={plot_data['fitness_diff'].mean():.4f}")
    print(f"Clim Component in Switch: min={plot_data['clim_component'].min():.4f}, "
          f"max={plot_data['clim_component'].max():.4f}, mean={plot_data['clim_component'].mean():.4f}")
    print(f"Nlim Component in Switch: min={plot_data['nlim_component'].min():.4f}, "
          f"max={plot_data['nlim_component'].max():.4f}, mean={plot_data['nlim_component'].mean():.4f}")
    
    # Create scatter plot
    print("Creating scatter plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Split the data based on which component exceeds the threshold
    clim_high = plot_data[plot_data['clim_component'] > threshold]
    nlim_high = plot_data[plot_data['nlim_component'] > threshold]
    both_high = plot_data[(plot_data['clim_component'] > threshold) & (plot_data['nlim_component'] > threshold)]
    
    # Plot the data - using different colors for points that exceed threshold in both components
    if len(clim_high) > 0:
        ax.scatter(clim_high['fitness_diff'], clim_high['clim_component'], 
                alpha=0.6, label='Carbon-limited Component > 0.2', color='blue')
    
    if len(nlim_high) > 0:
        ax.scatter(nlim_high['fitness_diff'], nlim_high['nlim_component'], 
                alpha=0.6, label='Nitrogen-limited Component > 0.2', color='green', marker='^')
    
    # Add labels and title
    ax.set_xlabel('|Clim fit - Nlim fit|/2', fontsize=14)
    ax.set_ylabel(f'Absolute Component Fitness in Switch Environment (> {threshold})', fontsize=14)
    ax.set_title('Absolute Fitness in Fluctuating Environment vs. Fitness Difference in Static Environments', 
                fontsize=16)
    
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend
    ax.legend(fontsize=12)
    
    # Add reference lines
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, 
              label=f'Threshold ({threshold})')
    ax.axvline(x=0, color='grey', linestyle='-', alpha=0.3)
    
    # Set axis limits
    ax.set_xlim(0, plot_data['fitness_diff'].max() * 1.05)
    # Set y-axis limits to show only filtered data
    y_min = threshold
    y_max = max(plot_data['clim_component'].max(), plot_data['nlim_component'].max())
    padding = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # Check if we have enough data for regression lines
    if len(clim_high) >= 2:
        # Attempt to add linear regression lines
        x_clim = clim_high['fitness_diff']
        y_clim = clim_high['clim_component']
        
        # Safely try to add regression lines
        m_clim, b_clim = safe_polyfit(x_clim, y_clim)
        
        # Add regression lines and equations if calculations succeeded
        if m_clim is not None and b_clim is not None:
            # Generate points for the regression line
            x_range = np.linspace(0, x_clim.max(), 100)
            ax.plot(x_range, m_clim*x_range + b_clim, color='blue', linestyle='--', alpha=0.5)
            ax.annotate(f'Clim: y = {m_clim:.4f}x + {b_clim:.4f}', 
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        color='blue', fontsize=10)
    
    if len(nlim_high) >= 2:
        x_nlim = nlim_high['fitness_diff']
        y_nlim = nlim_high['nlim_component']
        
        m_nlim, b_nlim = safe_polyfit(x_nlim, y_nlim)
        
        if m_nlim is not None and b_nlim is not None:
            # Generate points for the regression line
            x_range = np.linspace(0, x_nlim.max(), 100)
            ax.plot(x_range, m_nlim*x_range + b_nlim, color='green', linestyle='--', alpha=0.5)
            ax.annotate(f'Nlim: y = {m_nlim:.4f}x + {b_nlim:.4f}', 
                        xy=(0.05, 0.90), xycoords='axes fraction',
                        color='green', fontsize=10)
    
    # Safely calculate correlation coefficients
    if len(clim_high) >= 2:
        clim_corr = safe_correlation(clim_high['fitness_diff'], clim_high['clim_component'])
        if clim_corr is not None:
            print(f"\nCorrelation between fitness difference and Clim component: {clim_corr:.4f}")
        else:
            print("\nCould not calculate correlation for Clim component")
    
    if len(nlim_high) >= 2:
        nlim_corr = safe_correlation(nlim_high['fitness_diff'], nlim_high['nlim_component'])
        if nlim_corr is not None:
            print(f"Correlation between fitness difference and Nlim component: {nlim_corr:.4f}")
        else:
            print("Could not calculate correlation for Nlim component")
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Plot saved as '{args.output}'")
    
    plt.show()

if __name__ == "__main__":
    main()