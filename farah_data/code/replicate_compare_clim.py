import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats

def compare_fitness_across_replicates(base_dir, output_dir, environments=["Clim"], save_plots=True):
    """
    Create panel plots comparing fitness data across replicates for each environment.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing the data files
    output_dir : str
        Directory to save the plots
    environments : list
        List of environment names to analyze
    save_plots : bool
        Whether to save the plots to files
    """
    # Create output directory if it doesn't exist
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    all_data = {}
    
    for env in environments:
        print(f"\nAnalyzing environment: {env}")
        
        # File paths
        file_paths = [
            os.path.join(base_dir, f"{env}_rep1_FitSeq2_Result.csv"),
            os.path.join(base_dir, f"{env}_rep2_FitSeq2_Result.csv"),
            os.path.join(base_dir, f"{env}_rep3_FitSeq2_Result.csv")
        ]
        
        # Debug: Print the file paths
        print("Looking for files at:")
        for fp in file_paths:
            print(f"  {fp} - Exists: {os.path.exists(fp)}")
        
        # Check if all files exist
        if not all(os.path.exists(fp) for fp in file_paths):
            print(f"Warning: Not all files exist for environment {env}. Skipping.")
            continue
        
        # Load data
        try:
            rep_data = [pd.read_csv(fp) for fp in file_paths]
        except Exception as e:
            print(f"Error loading data for environment {env}: {e}. Skipping.")
            continue
        
        # Print data information
        print(f"Number of rows in Rep1: {len(rep_data[0])}")
        print(f"Number of rows in Rep2: {len(rep_data[1])}")
        print(f"Number of rows in Rep3: {len(rep_data[2])}")
        
        # Determine the minimum number of rows across all replicates
        min_rows = min(len(df) for df in rep_data)
        
        # Create a merged dataframe using the first min_rows rows from each file
        # Note: This assumes that rows are in the same order across files
        data_dict = {'Mutant_ID': range(min_rows)}
        for i, df in enumerate(rep_data):
            data_dict[f'Rep{i+1}_Fitness'] = df['Fitness_Per_Cycle'].values[:min_rows]
        
        merged_data = pd.DataFrame(data_dict)
        all_data[env] = merged_data
        
        # Create a figure with 3 subplots (for the 3 pairwise comparisons)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Define pairs to compare
        pairs = [(0, 1), (0, 2), (1, 2)]  # Rep1 vs Rep2, Rep1 vs Rep3, Rep2 vs Rep3
        
        for ax_idx, (i, j) in enumerate(pairs):
            rep_i = i + 1
            rep_j = j + 1
            
            # Calculate correlation coefficient
            x = merged_data[f'Rep{rep_i}_Fitness']
            y = merged_data[f'Rep{rep_j}_Fitness']
            corr = stats.pearsonr(x, y)[0]
            
            # Plot the scatter plot
            axes[ax_idx].scatter(x, y, alpha=0.6)
            axes[ax_idx].set_xlabel(f'Rep{rep_i} Fitness Per Cycle', fontsize=12)
            axes[ax_idx].set_ylabel(f'Rep{rep_j} Fitness Per Cycle', fontsize=12)
            axes[ax_idx].set_title(f'Rep{rep_i} vs Rep{rep_j} (r = {corr:.3f})', fontsize=14)
            axes[ax_idx].grid(True, linestyle='--', alpha=0.7)
            
            # Add diagonal line for reference
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            axes[ax_idx].plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Make axes equal to ensure the diagonal line is at 45 degrees
            axes[ax_idx].set_aspect('equal', adjustable='box')
        
        # Add a main title
        plt.suptitle(f'Comparison of Fitness Per Cycle Between Replicates ({env} Environment)', 
                    fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        if save_plots:
            output_file = os.path.join(output_dir, f"{env}_fitness_replicates_comparison.png")
            plt.savefig(output_file, dpi=300)
            print(f"Plot saved to: {output_file}")
        
        # Show the figure
        plt.show()
    
    return all_data

# Main execution
if __name__ == "__main__":
    # Set the base directory where the data files are located
    BASE_DIR = "/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results"
    
    # Set the output directory for saving plots
    OUTPUT_DIR = "/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/plots"
    
    # List of environments to analyze
    environments = ["Clim"]
    
    # Run the analysis
    all_data = compare_fitness_across_replicates(BASE_DIR, OUTPUT_DIR, environments=environments)
    
    print("\nAnalysis complete!")