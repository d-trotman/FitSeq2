#!/usr/bin/env python3
# Individual Mutant Fitness Calculator
# Calculates individual mutant fitness using the equation: fi+1 = fi * e^(s - s_mean)
# Rearranged to solve for s: s = ln(fi+1/fi) + s_mean

import pandas as pd
import numpy as np
import math
import argparse
import os

def calculate_mutant_fitness(counts_file, fitness_file, output_path=None):
    """
    Calculate individual mutant fitness using read counts and mean fitness values.
    
    Parameters:
    counts_file (str): Path to CSV file with read counts for each mutant at each timepoint
    fitness_file (str): Path to CSV file with mean fitness values between timepoints
    output_path (str, optional): Path to save the results CSV file or directory to save results in
    
    Returns:
    dict: Dictionary containing the fitness results and summary statistics
    """
    # Read the counts file
    counts_df = pd.read_csv(counts_file, header=None)
    
    # Read the fitness file to get mean population fitness
    fitness_df = pd.read_csv(fitness_file)
    mean_fitness_values = fitness_df['Mean_Fitness'].tolist()
    
    # Identify timepoints (columns in counts file)
    num_timepoints = counts_df.shape[1]
    timepoints = [f't{i}' for i in range(num_timepoints)]
    
    # Calculate total reads for each timepoint
    total_reads = {}
    for i in range(num_timepoints):
        total_reads[timepoints[i]] = counts_df[i].sum()
    
    # Calculate frequencies for each mutant at each timepoint
    num_mutants = counts_df.shape[0]
    frequencies = []
    
    for mutant_idx in range(num_mutants):
        mutant_freqs = {}
        for t in range(num_timepoints):
            reads = counts_df.iloc[mutant_idx, t] if not pd.isna(counts_df.iloc[mutant_idx, t]) else 0
            mutant_freqs[timepoints[t]] = reads / total_reads[timepoints[t]]
        frequencies.append(mutant_freqs)
    
    # Calculate individual fitness for each mutant between consecutive timepoints
    fitness_results = []
    
    for mutant_idx in range(num_mutants):
        mutant_fitness = {'mutant_id': mutant_idx}
        
        for t in range(num_timepoints - 1):
            timepoint1 = timepoints[t]
            timepoint2 = timepoints[t + 1]
            interval = f'{timepoint1}-{timepoint2}'
            
            freq1 = frequencies[mutant_idx][timepoint1]
            freq2 = frequencies[mutant_idx][timepoint2]
            s_mean = mean_fitness_values[t] if t < len(mean_fitness_values) else 0
            
            # Calculate s = ln(fi+1/fi) + s_mean
            if freq1 > 0 and freq2 > 0:
                s = math.log(freq2 / freq1) + s_mean
                mutant_fitness[interval] = s
            else:
                mutant_fitness[interval] = None
        
        fitness_results.append(mutant_fitness)
    
    # Calculate statistics for the first interval
    first_interval = f'{timepoints[0]}-{timepoints[1]}'
    valid_fitness_values = [result[first_interval] for result in fitness_results 
                           if result[first_interval] is not None]
    
    stats = {
        'interval': first_interval,
        'total_mutants': num_mutants,
        'valid_fitness_count': len(valid_fitness_values),
        'min_fitness': min(valid_fitness_values) if valid_fitness_values else None,
        'max_fitness': max(valid_fitness_values) if valid_fitness_values else None,
        'avg_fitness': sum(valid_fitness_values) / len(valid_fitness_values) if valid_fitness_values else None
    }
    
    # Convert fitness results to a DataFrame
    results_df = pd.DataFrame(fitness_results)
    
    # Save results to CSV if output path is specified
    if output_path:
        # Check if the output path is a directory
        if os.path.isdir(output_path):
            # If it's a directory, create a file inside it
            output_file = os.path.join(output_path, "individual_mutant_fitness.csv")
        else:
            # If it's not a directory, use it as a file path
            # Add .csv extension if not present
            if not output_path.endswith('.csv'):
                output_path = output_path + '.csv'
            output_file = output_path
            
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the results to the file
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return {
        'total_mutants': num_mutants,
        'timepoints': timepoints,
        'mean_fitness_values': mean_fitness_values,
        'fitness_results': fitness_results,
        'stats': stats,
        'results_df': results_df
    }

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Calculate individual mutant fitness from read counts and mean fitness values.')
    parser.add_argument('--counts', required=True, help='Path to counts CSV file')
    parser.add_argument('--fitness', required=True, help='Path to fitness results CSV file')
    parser.add_argument('--output', help='Path to save output CSV file')
    args = parser.parse_args()
    
    # Calculate mutant fitness
    results = calculate_mutant_fitness(args.counts, args.fitness, args.output)
    
    # Print summary statistics
    print(f"Calculated fitness for {results['total_mutants']} mutants across {len(results['timepoints'])} timepoints")
    
    print("\nMean fitness values between timepoints:")
    for i in range(len(results['timepoints']) - 1):
        if i < len(results['mean_fitness_values']):
            print(f"{results['timepoints'][i]} to {results['timepoints'][i+1]}: {results['mean_fitness_values'][i]}")
    
    print("\nFitness results for first 5 mutants:")
    for i in range(min(5, len(results['fitness_results']))):
        print(f"Mutant {results['fitness_results'][i]['mutant_id']}:", 
              {k: v for k, v in results['fitness_results'][i].items() if k != 'mutant_id'})
    
    stats = results['stats']
    print(f"\nFitness statistics between {stats['interval']}:")
    print(f"Number of mutants with valid fitness: {stats['valid_fitness_count']} out of {stats['total_mutants']}")
    
    if stats['min_fitness'] is not None:
        print(f"Min: {stats['min_fitness']:.4f}, Max: {stats['max_fitness']:.4f}, Average: {stats['avg_fitness']:.4f}")

if __name__ == "__main__":
    # If running directly, use the provided file paths
    counts_path = "/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/input/Clim_rep1_counts.csv"
    fitness_path = "/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/Clim_rep1_FitSeq2_Result.csv"
    output_path = "/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/fit_chk_calc"
    
    results = calculate_mutant_fitness(counts_path, fitness_path, output_path)
    
    # Print summary statistics
    print(f"Calculated fitness for {results['total_mutants']} mutants across {len(results['timepoints'])} timepoints")
    
    print("\nMean fitness values between timepoints:")
    for i in range(len(results['timepoints']) - 1):
        if i < len(results['mean_fitness_values']):
            print(f"{results['timepoints'][i]} to {results['timepoints'][i+1]}: {results['mean_fitness_values'][i]}")
    
    print("\nFitness results for first 5 mutants:")
    for i in range(min(5, len(results['fitness_results']))):
        print(f"Mutant {results['fitness_results'][i]['mutant_id']}:", 
              {k: v for k, v in results['fitness_results'][i].items() if k != 'mutant_id'})
    
    stats = results['stats']
    print(f"\nFitness statistics between {stats['interval']}:")
    print(f"Number of mutants with valid fitness: {stats['valid_fitness_count']} out of {stats['total_mutants']}")
    
    if stats['min_fitness'] is not None:
        print(f"Min: {stats['min_fitness']:.4f}, Max: {stats['max_fitness']:.4f}, Average: {stats['avg_fitness']:.4f}")
