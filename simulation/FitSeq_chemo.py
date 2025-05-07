import pandas as pd
import numpy as np
import os
import subprocess

def prepare_fitseq_input(input_file, output_prefix):
    """
    Prepare input files for FitSeq2 from simulation data
    
    Args:
        input_file: Path to simulation CSV file
        output_prefix: Prefix for output files
    
    Returns:
        Tuple of (reads_file, timepoints_file)
    """
    print(f"Preparing FitSeq2 input from {input_file}")
    
    # Read the simulation data
    data = pd.read_csv(input_file)
    
    # Extract read counts columns (reads_t0 through reads_t9)
    read_columns = [col for col in data.columns if col.startswith('reads_t')]
    reads_data = data[read_columns].values
    
    # Create the reads CSV file (each row is a strain, each column is a timepoint)
    reads_file = f"{output_prefix}_reads.csv"
    pd.DataFrame(reads_data).to_csv(reads_file, header=False, index=False)
    
    # Create timepoints file with generations and total cell numbers
    # Assuming t0, t1, etc. correspond to generations 0, 5, 10, etc.
    generations = np.arange(len(read_columns)) * 5  # Assuming 5 generations between timepoints
    total_cells = reads_data.sum(axis=0)  # Sum reads across all strains for each timepoint
    
    timepoints_file = f"{output_prefix}_timepoints.csv"
    pd.DataFrame({
        'generation': generations,
        'total_cells': total_cells
    }).to_csv(timepoints_file, header=False, index=False)
    
    return reads_file, timepoints_file

def run_fitseq(reads_file, timepoints_file, delta_t, output_prefix, c=1.0, 
               opt_algorithm='differential_evolution', max_iter=10, parallelize=1):
    """
    Run FitSeq2 analysis on prepared input files
    
    Args:
        reads_file: Path to reads CSV file
        timepoints_file: Path to timepoints CSV file
        delta_t: Number of generations between bottlenecks
        output_prefix: Prefix for output files
        c: Noise parameter (default: 1.0)
        opt_algorithm: Optimization algorithm (default: 'differential_evolution')
        max_iter: Maximum iterations (default: 10)
        parallelize: Whether to use parallelization (default: 1)
    """
    print(f"Running FitSeq2 analysis on {reads_file} and {timepoints_file}")
    
    # Build the command to run FitSeq2.py
    cmd = [
        'python3', 'FitSeq2.py',
        '-i', reads_file,
        '-t', timepoints_file,
        '-dt', str(delta_t),
        '-c', str(c),
        '-a', opt_algorithm,
        '-n', str(max_iter),
        '-p', str(parallelize),
        '-o', output_prefix
    ]
    
    # Run the command
    print("Executing: " + " ".join(cmd))
    subprocess.run(cmd)

def main():
    # Process chemostat simulation only
    chemostat_reads, chemostat_timepoints = prepare_fitseq_input(
        'yeast_chemostat_simulation.csv', 'chemostat')
    
    # Run FitSeq2 on chemostat data
    run_fitseq(chemostat_reads, chemostat_timepoints, delta_t=1.0, 
               output_prefix='chemostat_results')
    
    print("FitSeq2 analysis on chemostat data complete!")

if __name__ == "__main__":
    main()