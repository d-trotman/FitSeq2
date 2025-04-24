#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "fitseq_results/results")
PLOTS_DIR = os.path.join(BASE_DIR, "fitseq_results/plots")
SUMMARY_DIR = os.path.join(BASE_DIR, "fitseq_results/summary")

# Create summary directory
os.makedirs(SUMMARY_DIR, exist_ok=True)

def load_all_results():
    """Load results for all conditions and replicates"""
    all_results = {}
    
    # Get all condition-replicate combinations from result files
    result_files = glob.glob(f"{RESULTS_DIR}/*_FitSeq2_Result.csv")
    
    for file_path in result_files:
        # Extract condition and replicate from filename
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        condition = parts[0]
        replicate = int(parts[1][3:])  # Extract number from "rep1"
        
        print(f"Loading {condition} replicate {replicate}")
        
        # Load fitness data
        try:
            fitness_df = pd.read_csv(file_path)
            
            # Add to results dictionary
            if condition not in all_results:
                all_results[condition] = {}
                
            all_results[condition][replicate] = fitness_df
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_results

def compare_conditions(all_results):
    """Compare fitness distributions between conditions"""
    conditions = list(all_results.keys())
    
    if len(conditions) < 2:
        print("Need at least 2 conditions to compare")
        return
    
    print("\nComparing conditions:")
    
    # Create dataframe to store comparison results
    comparison_results = []
    
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
            
            mean_a = fitness_a.mean()
            mean_b = fitness_b.mean()
            std_a = fitness_a.std()
            std_b = fitness_b.std()
            
            print(f"Comparison between {cond_a} and {cond_b}:")
            print(f"  T-statistic: {t_stat:.4f}")
            print(f"  P-value: {p_val:.6f}")
            print(f"  {cond_a} mean: {mean_a:.6f} (± {std_a:.6f})")
            print(f"  {cond_b} mean: {mean_b:.6f} (± {std_b:.6f})")
            
            # Add to comparison results
            comparison_results.append({
                'Condition_A': cond_a,
                'Condition_B': cond_b,
                'Mean_A': mean_a,
                'Mean_B': mean_b,
                'Std_A': std_a,
                'Std_B': std_b,
                'T_statistic': t_stat,
                'P_value': p_val,
                'Significant': p_val < 0.05
            })
            
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
            plot_file = f"{PLOTS_DIR}/{cond_a}_vs_{cond_b}_comparison.png"
            plt.savefig(plot_file)
            plt.close()
            
            print(f"Saved comparison plot to {plot_file}")
            
        except Exception as e:
            print(f"Error comparing {cond_a} and {cond_b}: {e}")
    
    # Save comparison results to CSV
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df.to_csv(f"{SUMMARY_DIR}/condition_comparisons.csv", index=False)
        print(f"Saved comparison results to {SUMMARY_DIR}/condition_comparisons.csv")

def create_summary_tables(all_results):
    """Create summary tables for all conditions and replicates"""
    # Summary by replicate
    replicate_summary = []
    
    for condition in all_results:
        for replicate in all_results[condition]:
            fitness_df = all_results[condition][replicate]
            
            # Calculate summary statistics
            mean_fitness = fitness_df['Fitness_Per_Cycle'].mean()
            median_fitness = fitness_df['Fitness_Per_Cycle'].median()
            std_fitness = fitness_df['Fitness_Per_Cycle'].std()
            min_fitness = fitness_df['Fitness_Per_Cycle'].min()
            max_fitness = fitness_df['Fitness_Per_Cycle'].max()
            
            replicate_summary.append({
                'Condition': condition,
                'Replicate': replicate,
                'Mean_Fitness': mean_fitness,
                'Median_Fitness': median_fitness,
                'Std_Deviation': std_fitness,
                'Min_Fitness': min_fitness,
                'Max_Fitness': max_fitness,
                'Number_of_Barcodes': len(fitness_df)
            })
    
    # Create summary dataframe and save to CSV
    replicate_summary_df = pd.DataFrame(replicate_summary)
    replicate_summary_df.to_csv(f"{SUMMARY_DIR}/replicate_summary.csv", index=False)
    print(f"Saved replicate summary to {SUMMARY_DIR}/replicate_summary.csv")
    
    # Summary by condition (aggregating replicates)
    condition_summary = []
    
    for condition in all_results:
        # Combine fitness data from all replicates
        fitness_dfs = []
        
        for replicate in all_results[condition]:
            fitness_dfs.append(all_results[condition][replicate])
        
        if fitness_dfs:
            combined_fitness = pd.concat([df['Fitness_Per_Cycle'] for df in fitness_dfs])
            
            # Calculate summary statistics
            mean_fitness = combined_fitness.mean()
            median_fitness = combined_fitness.median()
            std_fitness = combined_fitness.std()
            min_fitness = combined_fitness.min()
            max_fitness = combined_fitness.max()
            
            condition_summary.append({
                'Condition': condition,
                'Mean_Fitness': mean_fitness,
                'Median_Fitness': median_fitness,
                'Std_Deviation': std_fitness,
                'Min_Fitness': min_fitness,
                'Max_Fitness': max_fitness,
                'Number_of_Barcodes': len(combined_fitness)
            })
    
    # Create summary dataframe and save to CSV
    condition_summary_df = pd.DataFrame(condition_summary)
    condition_summary_df.to_csv(f"{SUMMARY_DIR}/condition_summary.csv", index=False)
    print(f"Saved condition summary to {SUMMARY_DIR}/condition_summary.csv")
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    
    # Plot mean fitness with error bars
    x = range(len(condition_summary_df))
    plt.bar(x, condition_summary_df['Mean_Fitness'], yerr=condition_summary_df['Std_Deviation'])
    plt.xticks(x, condition_summary_df['Condition'], rotation=45)
    plt.ylabel('Mean Fitness')
    plt.title('Mean Fitness by Condition')
    plt.tight_layout()
    
    plt.savefig(f"{PLOTS_DIR}/condition_summary.png")
    plt.close()
    print(f"Saved condition summary plot to {PLOTS_DIR}/condition_summary.png")

def main():
    # Load results for all conditions and replicates
    all_results = load_all_results()
    
    if not all_results:
        print("No results found. Make sure all FitSeq2 jobs have completed.")
        return
    
    # Create summary tables
    create_summary_tables(all_results)
    
    # Compare conditions
    compare_conditions(all_results)
    
    print("\nPost-processing complete!")

if __name__ == "__main__":
    main()
