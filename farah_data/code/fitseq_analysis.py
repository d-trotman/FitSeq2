#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Hardcoded paths to prevent errors
BARCODE_CSV = "/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/barcode_counts.csv"
FITSEQ_PY = "/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/code/FitSeq2.py"
OUTPUT_DIR = "/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results"

# Create output directories
os.makedirs(f"{OUTPUT_DIR}/input", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/results", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

# Load barcode counts data
print(f"Loading barcode counts from {BARCODE_CSV}")
counts_df = pd.read_csv(BARCODE_CSV)

# Display basic information about the dataset
print(f"Dataset shape: {counts_df.shape}")
print(f"Time points: {counts_df['TimePoint'].unique()}")
print(f"Conditions: {counts_df['Condition'].unique()}")
print(f"Replicates: {counts_df['Replicate'].unique()}")

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
    # For experiments without bottlenecks, use timepoints as generations
    # and the sum of counts at each timepoint as cell depth
    timepoints = filtered_df['TimePoint'].values
    total_counts = filtered_df[barcode_cols].sum(axis=1).values
    
    # Create t_cell_depth.csv file
    t_cell_df = pd.DataFrame({
        'Timepoint': timepoints,
        'CellDepth': total_counts
    })
    
    # Save files with absolute paths
    count_file = f"{OUTPUT_DIR}/input/{condition}_rep{replicate}_counts.csv"
    t_cell_file = f"{OUTPUT_DIR}/input/{condition}_rep{replicate}_t_cell_depth.csv"
    
    count_matrix.to_csv(count_file, header=False, index=False)
    t_cell_df.to_csv(t_cell_file, header=False, index=False)
    
    print(f"  Count matrix shape: {count_matrix.shape}")
    print(f"  Timepoints: {timepoints}")
    
    return count_file, t_cell_file

def run_fitseq2(count_file, t_cell_file, condition, replicate, delta_t=1.0, c=0.5, 
                algorithm='differential_evolution', max_iter=10, parallelize=1):
    
    output_prefix = f"{OUTPUT_DIR}/results/{condition}_rep{replicate}"
    
    # Construct command with full paths
    cmd = [
        "python3", FITSEQ_PY,
        "-i", count_file,
        "-t", t_cell_file,
        "-dt", str(delta_t),
        "-c", str(c),
        "-a", algorithm,
        "-n", str(max_iter),
        "-p", str(parallelize),
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
    
    return fitness_df

def process_condition(condition):
    print(f"\nProcessing condition: {condition}")
    results = {}
    
    # Get replicates for this condition
    replicates = counts_df[counts_df['Condition'] == condition]['Replicate'].unique()
    
    for rep in replicates:
        try:
            # Prepare input files
            count_file, t_cell_file = prepare_fitseq_input(condition, rep)
            
            # Run FitSeq2
            output_prefix = run_fitseq2(count_file, t_cell_file, condition, rep)
            
            if output_prefix:
                # Analyze results
                fitness_df = analyze_results(output_prefix, condition, rep)
                results[rep] = fitness_df
        except Exception as e:
            print(f"Error processing {condition} replicate {rep}: {e}")
    
    return results

def compare_conditions(all_results):
    conditions = list(all_results.keys())
    
    if len(conditions) < 2:
        print("Need at least 2 conditions to compare")
        return
    
    print("\nComparing conditions:")
    
    # Create pairs of conditions to compare
    condition_pairs = [(a, b) for i, a in enumerate(conditions) for b in conditions[i+1:]]
    
    for cond_a, cond_b in condition_pairs:
        # Skip if either condition has no results
        if not all_results[cond_a] or not all_results[cond_b]:
            print(f"Skipping comparison between {cond_a} and {cond_b} due to missing results")
            continue
        
        # Combine data from all replicates for each condition
        try:
            # Create empty lists first to avoid concatenation issues if any replicate is missing
            fitness_a_list = []
            fitness_b_list = []
            
            for rep in all_results[cond_a]:
                if all_results[cond_a][rep] is not None:
                    fitness_a_list.append(all_results[cond_a][rep]['Fitness_Per_Cycle'])
            
            for rep in all_results[cond_b]:
                if all_results[cond_b][rep] is not None:
                    fitness_b_list.append(all_results[cond_b][rep]['Fitness_Per_Cycle'])
            
            if not fitness_a_list or not fitness_b_list:
                print(f"Skipping comparison between {cond_a} and {cond_b} due to missing fitness data")
                continue
                
            fitness_a = pd.concat(fitness_a_list)
            fitness_b = pd.concat(fitness_b_list)
            
            # T-test to compare fitness distributions
            t_stat, p_val = stats.ttest_ind(fitness_a, fitness_b)
            
            print(f"Comparison between {cond_a} and {cond_b}:")
            print(f"  T-statistic: {t_stat:.4f}")
            print(f"  P-value: {p_val:.6f}")
            print(f"  {cond_a} mean: {fitness_a.mean():.6f}")
            print(f"  {cond_b} mean: {fitness_b.mean():.6f}")
            
            # Create boxplot comparison
            plt.figure(figsize=(12, 8))
            
            # Prepare data for boxplot with all replicates
            box_data = []
            box_labels = []
            
            # Add individual replicates
            for rep in sorted(all_results[cond_a].keys()):
                if all_results[cond_a][rep] is not None:
                    box_data.append(all_results[cond_a][rep]['Fitness_Per_Cycle'])
                    box_labels.append(f"{cond_a} rep{rep}")
            
            for rep in sorted(all_results[cond_b].keys()):
                if all_results[cond_b][rep] is not None:
                    box_data.append(all_results[cond_b][rep]['Fitness_Per_Cycle'])
                    box_labels.append(f"{cond_b} rep{rep}")
            
            plt.boxplot(box_data, labels=box_labels)
            plt.title(f"Fitness Comparison: {cond_a} vs {cond_b}")
            plt.ylabel("Fitness Per Cycle")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_file = f"{OUTPUT_DIR}/plots/{cond_a}_vs_{cond_b}_comparison.png"
            plt.savefig(plot_file)
            plt.close()
            
            print(f"Saved comparison plot to {plot_file}")
            
        except Exception as e:
            print(f"Error comparing {cond_a} and {cond_b}: {e}")

def main():
    # Get unique conditions
    conditions = counts_df['Condition'].unique()
    all_results = {}
    
    # Process each condition
    for condition in conditions:
        all_results[condition] = process_condition(condition)
    
    # Compare conditions
    compare_conditions(all_results)
    
    print("\nAnalysis complete!")
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()