import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob
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

def process_files(data_dir):
    """
    Process individual mutant fitness data files from the specified directory.
    """
    print(f"Looking for data files in: {data_dir}")
    
    # Define file patterns for individual mutant fitness files
    clim_pattern = os.path.join(data_dir, "Clim_rep*_individual_mutant_fitness.csv")
    nlim_pattern = os.path.join(data_dir, "Nlim_rep*_individual_mutant_fitness.csv")
    switch_pattern = os.path.join(data_dir, "Switch_rep*_individual_mutant_fitness.csv")
    
    # Find files matching patterns
    clim_files = glob.glob(clim_pattern)
    nlim_files = glob.glob(nlim_pattern)
    switch_files = glob.glob(switch_pattern)
    
    # Report what files were found
    print(f"Found {len(clim_files)} Clim files: {[os.path.basename(f) for f in clim_files]}")
    print(f"Found {len(nlim_files)} Nlim files: {[os.path.basename(f) for f in nlim_files]}")
    print(f"Found {len(switch_files)} Switch files: {[os.path.basename(f) for f in switch_files]}")
    
    if not clim_files or not nlim_files or not switch_files:
        raise FileNotFoundError(f"Could not find required data files in {data_dir}")
    
    # Initialize dictionaries to store data by mutant index
    clim_data = {}
    nlim_data = {}
    switch_data = {}
    
    # Process Clim files
    for file in clim_files:
        try:
            df = pd.read_csv(file)
            print(f"Successfully read {os.path.basename(file)} with {len(df)} rows")
            
            # Calculate average fitness for each mutant, excluding final timepoint for Clim
            df['avg_fitness'] = df.apply(
                lambda row: calculate_average_fitness(row, exclude_final=True), 
                axis=1
            )
            
            # Add data to dictionary by mutant_id
            for _, row in df.iterrows():
                mutant_id = row['mutant_id']
                fitness = row['avg_fitness']
                
                if pd.notna(fitness):  # Only include non-NaN values
                    if mutant_id not in clim_data:
                        clim_data[mutant_id] = []
                    clim_data[mutant_id].append(fitness)
                    
            print(f"  Processed {len(clim_data)} unique mutants from this file")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Process Nlim files
    for file in nlim_files:
        try:
            df = pd.read_csv(file)
            print(f"Successfully read {os.path.basename(file)} with {len(df)} rows")
            
            # Calculate average fitness for each mutant (Nlim already has no final timepoint)
            df['avg_fitness'] = df.apply(
                lambda row: calculate_average_fitness(row, exclude_final=False), 
                axis=1
            )
            
            # Add data to dictionary by mutant_id
            for _, row in df.iterrows():
                mutant_id = row['mutant_id']
                fitness = row['avg_fitness']
                
                if pd.notna(fitness):  # Only include non-NaN values
                    if mutant_id not in nlim_data:
                        nlim_data[mutant_id] = []
                    nlim_data[mutant_id].append(fitness)
                    
            print(f"  Processed {len(nlim_data)} unique mutants from this file")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Process Switch files
    for file in switch_files:
        try:
            df = pd.read_csv(file)
            print(f"Successfully read {os.path.basename(file)} with {len(df)} rows")
            
            # Calculate average fitness for each mutant (check if Switch has the same format as Clim or Nlim)
            # For now, assume it's like Nlim with no final timepoint
            df['avg_fitness'] = df.apply(
                lambda row: calculate_average_fitness(row, exclude_final=False), 
                axis=1
            )
            
            # Add data to dictionary by mutant_id
            for _, row in df.iterrows():
                mutant_id = row['mutant_id']
                fitness = row['avg_fitness']
                
                if pd.notna(fitness):  # Only include non-NaN values
                    if mutant_id not in switch_data:
                        switch_data[mutant_id] = []
                    switch_data[mutant_id].append(fitness)
                    
            print(f"  Processed {len(switch_data)} unique mutants from this file")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Calculate averages and create a dataframe for plotting
    plot_data = []
    
    # Find common mutant_ids that have data in all three conditions
    common_mutants = set(clim_data.keys()).intersection(set(nlim_data.keys())).intersection(set(switch_data.keys()))
    print(f"Found {len(common_mutants)} mutants with data in all three conditions")
    
    for mutant_id in common_mutants:
        if clim_data[mutant_id] and nlim_data[mutant_id] and switch_data[mutant_id]:
            avg_clim = np.mean(clim_data[mutant_id])
            avg_nlim = np.mean(nlim_data[mutant_id])
            avg_switch = np.mean(switch_data[mutant_id])
            
            # Calculate (Clim + Nlim)/2
            combined_avg = (avg_clim + avg_nlim) / 2
            
            plot_data.append({
                'mutant_id': mutant_id,
                'clim_fitness': avg_clim,
                'nlim_fitness': avg_nlim,
                'switch_fitness': avg_switch,
                'combined_fitness': combined_avg
            })
    
    print(f"Generated data for {len(plot_data)} mutants after filtering")
    return pd.DataFrame(plot_data)

def create_scatter_plot(df):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(
        df['combined_fitness'], 
        df['switch_fitness'], 
        alpha=0.6, 
        s=20,
        c='blue',
        edgecolors='none'
    )
    
    # Add diagonal line (y=x)
    min_val = min(df['combined_fitness'].min(), df['switch_fitness'].min())
    max_val = max(df['combined_fitness'].max(), df['switch_fitness'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y = x')
    
    # Calculate and add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['combined_fitness'], 
        df['switch_fitness']
    )
    x = np.array([min_val, max_val])
    y = intercept + slope * x
    ax.plot(x, y, 'g-', label=f'Regression line (r={r_value:.3f})')
    
    # Add labels and title
    ax.set_xlabel('(Clim + Nlim)/2 Average Fitness', fontsize=12)
    ax.set_ylabel('Switch Average Fitness', fontsize=12)
    ax.set_title('Fitness: Static Average vs Switch', fontsize=14)
    
    # Add legend
    ax.legend()
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Calculate additional statistics
    correlation = df['combined_fitness'].corr(df['switch_fitness'])
    combined_mean = df['combined_fitness'].mean()
    switch_mean = df['switch_fitness'].mean()
    
    # Calculate range for similarity threshold
    combined_range = df['combined_fitness'].max() - df['combined_fitness'].min()
    switch_range = df['switch_fitness'].max() - df['switch_fitness'].min()
    range_value = max(combined_range, switch_range)
    threshold = 0.05 * range_value  # 5% of the range
    
    # Calculate difference and count similar values
    df['difference'] = df['combined_fitness'] - df['switch_fitness']
    similar_count = (df['difference'].abs() <= threshold).sum()
    similar_percentage = (similar_count / len(df)) * 100
    
    # Add text with statistics
    stats_text = (
        f"Correlation: {correlation:.4f}\n"
        f"Combined Mean: {combined_mean:.4f}\n"
        f"Switch Mean: {switch_mean:.4f}\n"
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
    
    return fig

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Analyze yeast mutant fitness data comparing static and switch conditions.')
    parser.add_argument('--data-dir', type=str, 
                      default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/fit_chk_calc/',
                      help='Directory containing the individual mutant fitness CSV files')
    parser.add_argument('--output', type=str, 
                      default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/plots/Combined_vs_Switch_new.png',
                      help='Output filename for the plot')
    parser.add_argument('--save-data', type=str, 
                      default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/fitness_combined_data_new.csv',
                      help='Filename to save the processed data')
    args = parser.parse_args()
    
    print("Comparing static (Clim+Nlim)/2 vs Switch mutant fitness data...")
    
    try:
        # Process files and get data
        df = process_files(args.data_dir)
        
        # Save processed data if requested
        if args.save_data:
            # Make sure the output directory exists
            output_dir = os.path.dirname(args.save_data)
            if output_dir and not os.path.exists(output_dir):
                print(f"Creating output directory for data: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
                
            df.to_csv(args.save_data, index=False)
            print(f"Processed data saved to '{args.save_data}'")
        
        # Create and show scatter plot
        print("Creating scatter plot...")
        fig = create_scatter_plot(df)
        
        # Make sure the output directory exists for the plot
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory for plot: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Save figure
        plt.savefig(args.output, dpi=300)
        print(f"Plot saved as '{args.output}'")
        
        # Show plot
        plt.show()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the data directory exists and contains the required CSV files.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
