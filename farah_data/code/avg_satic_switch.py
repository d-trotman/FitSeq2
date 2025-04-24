import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob

def process_files(data_dir):
    """
    Process fitness data files from the specified directory.
    """
    print(f"Looking for data files in: {data_dir}")
    
    # Define file patterns
    clim_pattern = os.path.join(data_dir, "Clim_rep*_FitSeq2_Result.csv")
    nlim_pattern = os.path.join(data_dir, "Nlim_rep*_FitSeq2_Result.csv")
    switch_pattern = os.path.join(data_dir, "Switch_rep*_FitSeq2_Result.csv")
    
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
            
            # Filter out mutants with Error_Fitness > 1
            filtered_df = df[df['Error_Fitness'] <= 0.1]
            print(f"  After filtering: {len(filtered_df)} rows")
            
            # Add data to dictionary by index
            for idx, row in filtered_df.iterrows():
                if idx not in clim_data:
                    clim_data[idx] = []
                clim_data[idx].append(row['Fitness_Per_Cycle'])
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Process Nlim files
    for file in nlim_files:
        try:
            df = pd.read_csv(file)
            print(f"Successfully read {os.path.basename(file)} with {len(df)} rows")
            
            # Filter out mutants with Error_Fitness > 1
            filtered_df = df[df['Error_Fitness'] <= 0.1]
            print(f"  After filtering: {len(filtered_df)} rows")
            
            # Add data to dictionary by index
            for idx, row in filtered_df.iterrows():
                if idx not in nlim_data:
                    nlim_data[idx] = []
                nlim_data[idx].append(row['Fitness_Per_Cycle'])
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Process Switch files
    for file in switch_files:
        try:
            df = pd.read_csv(file)
            print(f"Successfully read {os.path.basename(file)} with {len(df)} rows")
            
            # Filter out mutants with Error_Fitness > 1
            filtered_df = df[df['Error_Fitness'] <= 0.1]
            print(f"  After filtering: {len(filtered_df)} rows")
            
            # Add data to dictionary by index
            for idx, row in filtered_df.iterrows():
                if idx not in switch_data:
                    switch_data[idx] = []
                switch_data[idx].append(row['Fitness_Per_Cycle'])
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Calculate averages and create a dataframe for plotting
    plot_data = []
    
    # Find common indices that have data in all three conditions
    common_indices = set(clim_data.keys()).intersection(set(nlim_data.keys())).intersection(set(switch_data.keys()))
    print(f"Found {len(common_indices)} mutants with data in all three conditions")
    
    for idx in common_indices:
        if clim_data[idx] and nlim_data[idx] and switch_data[idx]:
            avg_clim = np.mean(clim_data[idx])
            avg_nlim = np.mean(nlim_data[idx])
            avg_switch = np.mean(switch_data[idx])
            
            # Calculate (Clim + Nlim)/2
            combined_avg = (avg_clim + avg_nlim) / 2
            
            plot_data.append({
                'mutant_index': idx,
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
    ax.set_xlabel('(Clim + Nlim)/2 Fitness', fontsize=12)
    ax.set_ylabel('Switch Fitness', fontsize=12)
    ax.set_title('Yeast Mutant Fitness: (Clim + Nlim)/2 vs Switch', fontsize=14)
    
    # Add legend
    ax.legend()
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add text with dataset info
    ax.text(
        0.02, 0.02, 
        f'Total mutants: {len(df)}\nFiltered: Error_Fitness > 1',
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom'
    )
    
    # Ensure equal scaling
    ax.set_aspect('equal')
    plt.tight_layout()
    
    return fig

def main():
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Analyze yeast mutant fitness data.')
    parser.add_argument('--data-dir', type=str, 
                        default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/',
                        help='Directory containing the CSV data files')
    parser.add_argument('--output', type=str, 
                        default='/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/plots/Combined_vs_Switch.png',
                        help='Output filename for the plot')
    args = parser.parse_args()
    
    print("Processing yeast mutant fitness data...")
    
    try:
        df = process_files(args.data_dir)
        
        print("Creating scatter plot...")
        fig = create_scatter_plot(df)
        
        # Make sure the output directory exists
        output_dir = os.path.dirname(args.output)
        if not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Save figure
        plt.savefig(args.output, dpi=300)
        print(f"Plot saved as '{args.output}'")
        
        # Show plot
        plt.show()
        
        # Also save the processed data as CSV for future reference
        data_output = os.path.join(output_dir, 'fitness_combined_data.csv')
        df.to_csv(data_output, index=False)
        print(f"Processed data saved as '{data_output}'")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the data directory exists and contains the required CSV files.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()