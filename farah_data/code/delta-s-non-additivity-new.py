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
    
    # Initialize dictionaries to store data by mutant ID
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
            
            # Calculate average of Clim and Nlim
            combined_avg = (avg_clim + avg_nlim) / 2
            
            # Calculate delta S (Clim - Nlim)
            delta_s = avg_clim - avg_nlim
            
            # Calculate non-additivity (Switch - average of Clim and Nlim)
            non_additivity = avg_switch - combined_avg
            
            plot_data.append({
                'mutant_id': mutant_id,
                'clim_fitness': avg_clim,
                'nlim_fitness': avg_nlim,
                'switch_fitness': avg_switch,
                'combined_fitness': combined_avg,
                'delta_s': delta_s,
                'non_additivity': non_additivity
            })
    
    print(f"Generated data for {len(plot_data)} mutants after filtering")
    return pd.DataFrame(plot_data)

def create_scatter_plot(df):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(
        df['delta_s'], 
        df['non_additivity'], 
        alpha=0.6, 
        s=20,
        c='green',
        edgecolors='none'
    )
    
    # Add horizontal and vertical lines at 0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Calculate and add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['delta_s'], 
        df['non_additivity']
    )
    
    # Get the min and max for x values
    min_x = df['delta_s'].min()
    max_x = df['delta_s'].max()
    x = np.array([min_x, max_x])
    y = intercept + slope * x
    
    ax.plot(x, y, 'r-', label=f'Regression line\ny = {slope:.4f}x + {intercept:.4f}\nR² = {r_value**2:.4f}')
    
    # Add labels and title
    ax.set_xlabel('Delta S (Clim fitness - Nlim fitness)', fontsize=12)
    ax.set_ylabel('Non-additivity (Switch fitness - Average fitness)', fontsize=12)
    ax.set_title('Environmental Specialization vs Non-additivity', fontsize=14)
    
    # Add legend
    ax.legend(loc='best')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add additional statistics
    quadrant_counts = [
        ((df['delta_s'] > 0) & (df['non_additivity'] > 0)).sum(),  # Quadrant 1: positive delta_s, positive non_additivity
        ((df['delta_s'] < 0) & (df['non_additivity'] > 0)).sum(),  # Quadrant 2: negative delta_s, positive non_additivity
        ((df['delta_s'] < 0) & (df['non_additivity'] < 0)).sum(),  # Quadrant 3: negative delta_s, negative non_additivity
        ((df['delta_s'] > 0) & (df['non_additivity'] < 0)).sum()   # Quadrant 4: positive delta_s, negative non_additivity
    ]
    
    # Add text with dataset info and statistics
    stats_text = (
        f"Total mutants: {len(df)}\n"
        f"Correlation: {r_value:.4f}\n"
        f"Quadrant counts (clockwise from top-right):\n"
        f"Q1: {quadrant_counts[0]} | Q4: {quadrant_counts[3]}\n"
        f"Q2: {quadrant_counts[1]} | Q3: {quadrant_counts[2]}"
    )
    
    ax.text(
        0.02, 0.02, 
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
    )
    
    plt.tight_layout()
    
    return fig

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Analyze delta S non-additivity in yeast mutants.')
    parser.add_argument('--data-dir', type=str, 
                      default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/fit_chk_calc/',
                      help='Directory containing the individual mutant fitness CSV files')
    parser.add_argument('--output', type=str, 
                      default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/plots/DeltaS_vs_NonAdditivity_new.png',
                      help='Output filename for the plot')
    parser.add_argument('--save-data', type=str, 
                      default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/delta_s_non_additivity_data_new.csv',
                      help='Filename to save the processed data')
    args = parser.parse_args()
    
    print("Analyzing delta S non-additivity in yeast mutants...")
    
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
