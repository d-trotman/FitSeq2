import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import argparse

def calculate_average_fitness(row, exclude_final=True):
    """
    Calculate average fitness across time points, excluding null values 
    and optionally the final timepoint.
    """
    # Extract all fitness columns (excluding mutant_id)
    fitness_cols = [col for col in row.index if col != 'mutant_id']
    
    # Optionally exclude the final timepoint
    if exclude_final and len(fitness_cols) > 0:
        fitness_cols = fitness_cols[:-1]
    
    # Filter out null values and calculate average
    fitness_values = [row[col] for col in fitness_cols if pd.notna(row[col])]
    
    if len(fitness_values) == 0:
        return np.nan
    
    return np.mean(fitness_values)

def process_files(clim_file, nlim_file):
    """
    Process individual mutant fitness data files.
    """
    print(f"Processing Clim file: {os.path.basename(clim_file)}")
    print(f"Processing Nlim file: {os.path.basename(nlim_file)}")
    
    try:
        # Read the CSV files
        clim_df = pd.read_csv(clim_file)
        nlim_df = pd.read_csv(nlim_file)
        
        print(f"Clim data: {len(clim_df)} rows, {len(clim_df.columns)} columns")
        print(f"Nlim data: {len(nlim_df)} rows, {len(nlim_df.columns)} columns")
        
        # Calculate average fitness for each mutant, excluding final timepoint for Clim
        clim_df['avg_fitness'] = clim_df.apply(
            lambda row: calculate_average_fitness(row, exclude_final=True), 
            axis=1
        )
        
        nlim_df['avg_fitness'] = nlim_df.apply(
            lambda row: calculate_average_fitness(row, exclude_final=False),  # Nlim already has no final timepoint
            axis=1
        )
        
        # Create dataframes with just mutant_id and average fitness
        clim_avg = clim_df[['mutant_id', 'avg_fitness']].rename(columns={'avg_fitness': 'clim_fitness'})
        nlim_avg = nlim_df[['mutant_id', 'avg_fitness']].rename(columns={'avg_fitness': 'nlim_fitness'})
        
        # Merge the dataframes on mutant_id
        merged_df = pd.merge(clim_avg, nlim_avg, on='mutant_id', how='inner')
        
        # Filter out rows with missing values
        filtered_df = merged_df.dropna()
        
        print(f"Total mutants after merging and filtering: {len(filtered_df)}")
        
        return filtered_df
        
    except Exception as e:
        print(f"Error processing files: {e}")
        raise

def create_scatter_plot(df, output_file=None):
    """
    Create a scatter plot comparing Clim and Nlim fitness values.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(
        df['nlim_fitness'], 
        df['clim_fitness'], 
        alpha=0.6, 
        s=20,
        c='purple',
        edgecolors='none'
    )
    
    # Add diagonal line (y=x)
    min_val = min(df['nlim_fitness'].min(), df['clim_fitness'].min())
    max_val = max(df['nlim_fitness'].max(), df['clim_fitness'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y = x')
    
    # Calculate and add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['nlim_fitness'], 
        df['clim_fitness']
    )
    x = np.array([min_val, max_val])
    y = intercept + slope * x
    ax.plot(x, y, 'g-', label=f'Regression line (r={r_value:.3f})')
    
    # Add labels and title
    ax.set_xlabel('Average Fitness in Nlim', fontsize=12)
    ax.set_ylabel('Average Fitness in Clim', fontsize=12)
    ax.set_title('Fitness: Clim vs Nlim', fontsize=14)
    
    # Add legend
    ax.legend()
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Calculate additional statistics
    correlation = df['nlim_fitness'].corr(df['clim_fitness'])
    clim_mean = df['clim_fitness'].mean()
    nlim_mean = df['nlim_fitness'].mean()
    
    # Calculate range for similarity threshold
    clim_range = df['clim_fitness'].max() - df['clim_fitness'].min()
    nlim_range = df['nlim_fitness'].max() - df['nlim_fitness'].min()
    range_value = max(clim_range, nlim_range)
    threshold = 0.05 * range_value  # 5% of the range
    
    # Calculate difference and count similar values
    df['difference'] = df['clim_fitness'] - df['nlim_fitness']
    similar_count = (df['difference'].abs() <= threshold).sum()
    similar_percentage = (similar_count / len(df)) * 100
    
    # Add text with statistics
    stats_text = (
        f"Correlation: {correlation:.4f}\n"
        f"Clim Mean: {clim_mean:.4f}\n"
        f"Nlim Mean: {nlim_mean:.4f}\n"
        f"Similar Values: {similar_percentage:.2f}%\n"
        f"Total Mutants: {len(df)}"
    )
    
    ax.text(
        0.05, 0.95, 
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
    )
    
    # Adjust layout and equal scaling
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    # Save the figure if an output file is specified
    if output_file:
        # Make sure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved as '{output_file}'")
    
    return fig

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Compare Clim vs Nlim fitness from individual mutant data.')
    # Use the full paths as default values instead of just the filenames
    parser.add_argument('--clim-file', type=str, 
                      default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/fit_chk_calc/Clim_rep1_individual_mutant_fitness.csv',
                      help='Path to Clim individual mutant fitness CSV file')
    parser.add_argument('--nlim-file', type=str, 
                      default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/fit_chk_calc/Nlim_rep1_individual_mutant_fitness.csv',
                      help='Path to Nlim individual mutant fitness CSV file')
    parser.add_argument('--output', type=str, 
                      default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/plots/Clim_vs_Nlim_comparison_new.png',
                      help='Output filename for the plot')
    parser.add_argument('--save-data', type=str, 
                      default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/clim_nlim_comparison_data_new.csv',
                      help='Filename to save the processed data')
    args = parser.parse_args()
    
    print("Comparing Clim vs Nlim mutant fitness data...")
    
    try:
        # Process files and get data
        df = process_files(args.clim_file, args.nlim_file)
        
        # Save processed data if requested
        if args.save_data:
            df.to_csv(args.save_data, index=False)
            print(f"Processed data saved to '{args.save_data}'")
        
        # Create and show scatter plot
        fig = create_scatter_plot(df, args.output)
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
