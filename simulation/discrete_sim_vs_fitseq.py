import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def compare_yeast_frequencies(file1_path, file2_path, timepoint=1, output_file="yeast_comparison.png"):
    """
    Compare frequencies of yeast mutants between two spreadsheets at a specific timepoint.
    
    Parameters:
    -----------
    file1_path : str
        Path to the first CSV file (estimated reads)
    file2_path : str
        Path to the second CSV file (discrete reads)
    timepoint : int
        Timepoint index to compare (default: 1)
    output_file : str
        Path to save the output scatterplot
    """
    print(f"Loading data from {file1_path} and {file2_path}...")
    
    # Load the CSV files without headers
    file1_data = pd.read_csv(file1_path, header=None)
    file2_data = pd.read_csv(file2_path, header=None)
    
    # Verify timepoint is valid
    if timepoint >= file1_data.shape[1] or timepoint < 0:
        print(f"Error: Timepoint {timepoint} is out of range. Must be between 0 and {file1_data.shape[1]-1}.")
        return
    
    print(f"Calculating frequencies for {file1_data.shape[0]} strains...")
    
    # Calculate frequencies for file 1
    file1_total = file1_data[timepoint].sum()
    file1_freq = file1_data[timepoint] / file1_total
    
    # Calculate frequencies for file 2
    file2_total = file2_data[timepoint].sum()
    file2_freq = file2_data[timepoint] / file2_total
    
    # Combine data for analysis
    combined_data = pd.DataFrame({
        'strain_index': range(len(file1_freq)),
        'file1_freq': file1_freq.values,
        'file2_freq': file2_freq.values
    })
    
    # Calculate the squared difference between frequencies
    combined_data['squared_diff'] = (combined_data['file1_freq'] - combined_data['file2_freq'])**2
    
    # Find top 10 strains with largest differences
    top_strains = combined_data.sort_values('squared_diff', ascending=False).head(10)
    
    print(f"Creating scatterplot for timepoint {timepoint}...")
    
    # Create the scatterplot
    plt.figure(figsize=(10, 8))
    
    # Plot all points
    plt.scatter(
        combined_data['file2_freq'],
        combined_data['file1_freq'],
        alpha=0.3,
        s=5,
        color='blue',
        label='All strains'
    )
    
    # Highlight top differentially abundant strains
    plt.scatter(
        top_strains['file2_freq'],
        top_strains['file1_freq'],
        color='red',
        s=50,
        alpha=0.8,
        label='Top 10 divergent strains'
    )
    
    # Add strain labels for top strains
    for _, strain in top_strains.iterrows():
        plt.annotate(
            f"Strain {strain['strain_index']}",
            (strain['file2_freq'], strain['file1_freq']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    # Add diagonal line
    max_val = max(combined_data['file1_freq'].max(), combined_data['file2_freq'].max())
    min_val = min(combined_data['file1_freq'].min(), combined_data['file2_freq'].min())
    min_val = max(min_val, 1e-10)  # To avoid log scale issues
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Set log scales for both axes
    plt.xscale('log')
    plt.yscale('log')
    
    # Add labels and title
    plt.xlabel('Discrete Reads Frequency')
    plt.ylabel('Estimated Reads Frequency')
    plt.title(f'Yeast Mutant Frequency Comparison at Timepoint {timepoint}')
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    
    # Print top strains information
    print("\nTop 10 strains with largest frequency differences:")
    for i, (_, strain) in enumerate(top_strains.iterrows(), 1):
        print(f"{i}. Strain {strain['strain_index']}: ")
        print(f"   Estimated frequency: {strain['file1_freq']:.8f}")
        print(f"   Discrete frequency: {strain['file2_freq']:.8f}")
        print(f"   Squared difference: {strain['squared_diff']:.8f}")
    
    # Return the combined data for further analysis if needed
    return combined_data

# Example usage
if __name__ == "__main__":
    import sys
    
    # Default parameters
    file1_path = "discrete_results_FitSeq2_Result_Read_Number_Estimated.csv"
    file2_path = "discrete_reads.csv"
    timepoint = 1
    output_file = "yeast_mutant_comparison.png"
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        file1_path = sys.argv[1]
    if len(sys.argv) > 2:
        file2_path = sys.argv[2]
    if len(sys.argv) > 3:
        timepoint = int(sys.argv[3])
    if len(sys.argv) > 4:
        output_file = sys.argv[4]
    
    # Run the comparison
    compare_yeast_frequencies(file1_path, file2_path, timepoint, output_file)
