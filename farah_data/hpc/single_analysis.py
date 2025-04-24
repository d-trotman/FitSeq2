#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import subprocess
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run FitSeq2 analysis for a specific condition and replicate')
parser.add_argument('--job-id', type=int, required=True, help='SLURM array job ID')
parser.add_argument('--cpus', type=int, default=16, help='Number of CPUs to use')
args = parser.parse_args()

# Paths to files and scripts
# IMPORTANT: Update these paths to match your HPC environment
BASE_DIR = '/home/dt2839/FitSeq2/'
BARCODE_CSV = os.path.join(BASE_DIR, "/data/barcode_counts.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "fitseq_results")
MAPPING_FILE = os.path.join(BASE_DIR, "condition_replicate_map.txt")

# Create output directories
os.makedirs(os.path.join(OUTPUT_DIR, "input"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "results"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

# Get condition and replicate from mapping file
with open(MAPPING_FILE, 'r') as f:
    lines = f.readlines()
    
if args.job_id <= len(lines):
    # Format of each line: job_id,condition,replicate
    job_mapping = lines[args.job_id - 1].strip().split(',')
    condition = job_mapping[1]
    replicate = int(job_mapping[2])
else:
    print(f"Error: Job ID {args.job_id} exceeds the number of condition-replicate pairs")
    sys.exit(1)

print(f"Processing condition: {condition}, replicate: {replicate}")

# Load barcode counts data
print(f"Loading barcode counts from {BARCODE_CSV}")
counts_df = pd.read_csv(BARCODE_CSV)

def prepare_fitseq_input(condition, replicate):
    print(f"Preparing input for {condition} replicate {replicate}")
    
    # Filter data for the given condition and replicate
    filtered_df = counts_df[(counts_df['Condition'] == condition) & 
                           (counts_df['Replicate'] == replicate)].copy()
    
    # Ensure timepoints are sorted
    filtered_df = filtered_df.sort_values(['TimePoint'])
    
    # Get barcode columns (all columns except TimePoint, Condition, Replicate)
    barcode_cols = [col for col in filtered_df.columns if col not in ['TimePoint', 'Condition', 'Replicate']]
    
    # Create count matrix (each row is a barcode, each column is a timepoint)
    count_matrix = filtered_df[barcode_cols].T  # Transpose so rows are barcodes, columns are timepoints
    
    # Create time points and cell depth file
    # Since there are no bottlenecks, we'll use the timepoints as generations
    # and the sum of counts at each timepoint as an estimate of cell depth
    timepoints = filtered_df['TimePoint'].values
    total_counts = filtered_df[barcode_cols].sum(axis=1).values
    
    # Create t_cell_depth.csv file
    t_cell_df = pd.DataFrame({
        'Timepoint': timepoints,
        'CellDepth': total_counts
    })
    
    # Save files
    count_file = f"{OUTPUT_DIR}/input/{condition}_rep{replicate}_counts.csv"
    t_cell_file = f"{OUTPUT_DIR}/input/{condition}_rep{replicate}_t_cell_depth.csv"
    
    count_matrix.to_csv(count_file, header=False, index=False)
    t_cell_df.to_csv(t_cell_file, header=False, index=False)
    
    print(f"  Count matrix shape: {count_matrix.shape}")
    print(f"  Timepoints: {timepoints}")
    
    return count_file, t_cell_file

def run_fitseq2(count_file, t_cell_file, condition, replicate, delta_t=1.0, c=0.5, 
                algorithm='differential_evolution', max_iter=10, parallelize=1, n_cpus=16):
    
    output_prefix = f"{OUTPUT_DIR}/results/{condition}_rep{replicate}"
    
    # Construct command to run FitSeq2 module
    # Since module is loaded, we can call FitSeq2.py directly
    cmd = [
        "FitSeq2.py",  # Command available after loading the module
        "-i", count_file,
        "-t", t_cell_file,
        "-dt", str(delta_t),
        "-c", str(c),
        "-a", algorithm,
        "-n", str(max_iter),
        "-p", str(parallelize),  # Keep parallelize=1 to use Pool
        "-o", output_prefix
    ]
    
    print(f"Running FitSeq2 for {condition} replicate {replicate}...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run FitSeq2
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check if FitSeq2 ran successfully
    if process.returncode == 0:
        print(f"FitSeq2 completed successfully for {condition} replicate {replicate}")
        print(process.stdout)
        return output_prefix
    else:
        print(f"Error running FitSeq2 for {condition} replicate {replicate}")
        print(f"Error: {process.stderr}")
        return None

def analyze_results(output_prefix, condition, replicate):
    # Load fitness results
    fitness_file = f"{output_prefix}_FitSeq2_Result.csv"
    if not os.path.exists(fitness_file):
        print(f"Results file not found: {fitness_file}")
        return None
    
    fitness_df = pd.read_csv(fitness_file)
    
    # Basic statistics
    mean_fitness = fitness_df['Fitness_Per_Cycle'].mean()
    median_fitness = fitness_df['Fitness_Per_Cycle'].median()
    std_fitness = fitness_df['Fitness_Per_Cycle'].std()
    
    print(f"Analysis for {condition} replicate {replicate}:")
    print(f"  Mean fitness: {mean_fitness:.6f}")
    print(f"  Median fitness: {median_fitness:.6f}")
    print(f"  Std deviation: {std_fitness:.6f}")
    
    # Create histogram of fitness values
    plt.figure(figsize=(10, 6))
    sns.histplot(fitness_df['Fitness_Per_Cycle'], kde=True)
    plt.title(f"Fitness Distribution - {condition} replicate {replicate}")
    plt.xlabel("Fitness Per Cycle")
    plt.ylabel("Count")
    plot_file = f"{OUTPUT_DIR}/plots/{condition}_rep{replicate}_fitness_histogram.png"
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Saved fitness histogram to {plot_file}")
    
    # Save summary statistics
    with open(f"{OUTPUT_DIR}/results/{condition}_rep{replicate}_summary.txt", 'w') as f:
        f.write(f"Analysis for {condition} replicate {replicate}:\n")
        f.write(f"  Mean fitness: {mean_fitness:.6f}\n")
        f.write(f"  Median fitness: {median_fitness:.6f}\n")
        f.write(f"  Std deviation: {std_fitness:.6f}\n")
    
    return fitness_df

# Main execution
try:
    # Prepare input files
    count_file, t_cell_file = prepare_fitseq_input(condition, replicate)
    
    # Run FitSeq2
    output_prefix = run_fitseq2(count_file, t_cell_file, condition, replicate, 
                               parallelize=1, n_cpus=args.cpus)
    
    if output_prefix:
        # Analyze results
        fitness_df = analyze_results(output_prefix, condition, replicate)
        
    print(f"Analysis complete for {condition} replicate {replicate}!")
    
except Exception as e:
    print(f"Error processing {condition} replicate {replicate}: {e}")
    sys.exit(1)
