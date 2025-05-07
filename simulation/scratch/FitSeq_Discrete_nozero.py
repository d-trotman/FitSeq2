import pandas as pd
import numpy as np
import os
import subprocess

def prepare_filtered_fitseq_input(input_file, output_prefix, min_reads=1):
    """
    Prepare filtered input files for FitSeq2 from simulation data
    
    Args:
        input_file: Path to simulation CSV file
        output_prefix: Prefix for output files
        min_reads: Minimum number of reads for a strain to be included
    
    Returns:
        Tuple of (reads_file, timepoints_file, filtered_count)
    """
    print(f"Preparing filtered FitSeq2 input from {input_file}")
    
    # Read the simulation data
    data = pd.read_csv(input_file)
    
    # Extract read counts columns
    read_columns = [col for col in data.columns if col.startswith('reads_t')]
    original_strain_count = len(data)
    
    # Filter out strains with zero reads at any timepoint
    nonzero_mask = (data[read_columns] >= min_reads).all(axis=1)
    filtered_data = data[nonzero_mask]
    filtered_count = original_strain_count - len(filtered_data)
    
    print(f"Filtered out {filtered_count} strains ({filtered_count/original_strain_count*100:.1f}%)")
    
    # Extract the filtered read data
    reads_data = filtered_data[read_columns].values
    
    # Create the reads CSV file
    reads_file = f"{output_prefix}_filtered_reads.csv"
    pd.DataFrame(reads_data).to_csv(reads_file, header=False, index=False)
    
    # Create timepoints file
    generations = np.arange(len(read_columns)) * 5
    total_cells = reads_data.sum(axis=0)
    
    timepoints_file = f"{output_prefix}_filtered_timepoints.csv"
    pd.DataFrame({
        'generation': generations,
        'total_cells': total_cells
    }).to_csv(timepoints_file, header=False, index=False)
    
    return reads_file, timepoints_file, filtered_count

def run_fitseq(reads_file, timepoints_file, delta_t, output_prefix, c=1.5):
    """
    Run FitSeq2 analysis with adjusted noise parameter
    """
    print(f"Running FitSeq2 analysis on {reads_file}")
    
    cmd = [
        'python3', 'FitSeq2.py',
        '-i', reads_file,
        '-t', timepoints_file,
        '-dt', str(delta_t),
        '-c', str(c),  # Slightly increased noise parameter
        '-a', 'differential_evolution',
        '-n', '10',
        '-p', '1',
        '-o', output_prefix
    ]
    
    print("Executing: " + " ".join(cmd))
    subprocess.run(cmd)

def main():
    # Process discrete simulation with filtering
    discrete_reads, discrete_timepoints, filtered_count = prepare_filtered_fitseq_input(
        'yeast_discrete_simulation.csv', 'discrete_filtered', min_reads=1)
    
    # Run FitSeq2 on filtered data with adjusted noise parameter
    run_fitseq(discrete_reads, discrete_timepoints, delta_t=5.0, 
               output_prefix='discrete_filtered_results', c=1.5)
    
    print(f"FitSeq2 analysis complete! Filtered {filtered_count} strains.")

if __name__ == "__main__":
    main()