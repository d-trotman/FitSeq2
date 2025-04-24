import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def calculate_average_fitness(row, exclude_final=True):
    """Calculate average fitness across time points, excluding null values and optionally the final timepoint."""
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

def main():
    # Read the CSV files
    print("Reading CSV files...")
    individual_mutant_file = "/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/fit_chk_calc/Clim_rep1_individual_mutant_fitness.csv"
    fitseq2_file = "/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/Clim_rep1_FitSeq2_Result.csv"
    
    try:
        mutant_df = pd.read_csv(individual_mutant_file)
        fitseq2_df = pd.read_csv(fitseq2_file)
    except Exception as e:
        print(f"Error reading files: {e}")
        return
    
    # Check if files are loaded correctly
    print(f"Individual mutant data: {mutant_df.shape[0]} rows, {mutant_df.shape[1]} columns")
    print(f"FitSeq2 data: {fitseq2_df.shape[0]} rows, {fitseq2_df.shape[1]} columns")
    
    if mutant_df.shape[0] != fitseq2_df.shape[0]:
        print("Warning: Files have different number of rows!")
    
    # Calculate average fitness for each mutant, excluding final timepoint
    print("Calculating average fitness...")
    mutant_df['average_fitness'] = mutant_df.apply(
        lambda row: calculate_average_fitness(row, exclude_final=True), 
        axis=1
    )
    
    # Add mutant_id to FitSeq2 data if it's not already there
    # Assuming rows are in the same order in both files
    if 'mutant_id' not in fitseq2_df.columns:
        fitseq2_df['mutant_id'] = mutant_df['mutant_id']
    
    # Merge the two dataframes on mutant_id
    print("Merging datasets...")
    merged_df = pd.merge(
        mutant_df[['mutant_id', 'average_fitness']], 
        fitseq2_df[['mutant_id', 'Fitness_Per_Cycle']], 
        on='mutant_id', 
        how='inner'
    )
    
    # Rename for clarity
    merged_df = merged_df.rename(columns={'Fitness_Per_Cycle': 'fitseq2_fitness'})
    
    # Filter out rows with missing values in either fitness calculation
    print("Filtering out missing values...")
    filtered_df = merged_df.dropna(subset=['average_fitness', 'fitseq2_fitness'])
    
    print(f"Original data points: {merged_df.shape[0]}")
    print(f"Data points after filtering: {filtered_df.shape[0]}")
    
    # Calculate statistics
    print("Calculating statistics...")
    correlation = filtered_df['average_fitness'].corr(filtered_df['fitseq2_fitness'])
    avg_mean = filtered_df['average_fitness'].mean()
    fitseq2_mean = filtered_df['fitseq2_fitness'].mean()
    
    # Calculate range for similarity threshold
    avg_range = filtered_df['average_fitness'].max() - filtered_df['average_fitness'].min()
    fitseq2_range = filtered_df['fitseq2_fitness'].max() - filtered_df['fitseq2_fitness'].min()
    range_value = max(avg_range, fitseq2_range)
    threshold = 0.05 * range_value  # 5% of the range
    
    # Calculate difference and count similar values
    filtered_df['difference'] = filtered_df['average_fitness'] - filtered_df['fitseq2_fitness']
    similar_count = (filtered_df['difference'].abs() <= threshold).sum()
    similar_percentage = (similar_count / filtered_df.shape[0]) * 100
    
    # Print statistics
    print("\nAnalysis Results:")
    print(f"Correlation coefficient: {correlation:.4f}")
    print(f"Average fitness mean: {avg_mean:.4f}")
    print(f"FitSeq2 fitness mean: {fitseq2_mean:.4f}")
    print(f"Similar fitness values: {similar_count} ({similar_percentage:.2f}%)")
    
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        filtered_df['average_fitness'], 
        filtered_df['fitseq2_fitness']
    )
    
    # Create regression line
    x_min = filtered_df['average_fitness'].min()
    x_max = filtered_df['average_fitness'].max()
    x_reg = np.linspace(x_min, x_max, 100)
    y_reg = slope * x_reg + intercept
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Set style
    sns.set_style('whitegrid')
    
    # Create the scatter plot
    ax = sns.scatterplot(
        x='average_fitness', 
        y='fitseq2_fitness', 
        data=filtered_df, 
        alpha=0.5, 
        color='blue'
    )
    
    # Add regression line
    plt.plot(x_reg, y_reg, color='red', linestyle='-', linewidth=2, label=f'Regression (r={correlation:.4f})')
    
    # Add perfect correlation line (y=x)
    min_val = min(x_min, filtered_df['fitseq2_fitness'].min())
    max_val = max(x_max, filtered_df['fitseq2_fitness'].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', label='Perfect correlation')
    
    # Add labels and title
    plt.xlabel('Average Fitness (Time Points, excluding final)', fontsize=12)
    plt.ylabel('FitSeq2 Fitness Per Cycle', fontsize=12)
    plt.title('Comparison of Fitness Calculations', fontsize=14, fontweight='bold')
    
    # Add legend
    plt.legend()
    
    # Add text with statistics
    text_x = 0.05
    text_y = 0.95
    plt.text(
        text_x, text_y, 
        f"Correlation: {correlation:.4f}\n"
        f"Time Points Mean: {avg_mean:.4f}\n"
        f"FitSeq2 Mean: {fitseq2_mean:.4f}\n"
        f"Similar Values: {similar_percentage:.2f}%",
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
    )
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('fitness_comparison.png', dpi=300)
    print("Plot saved as 'fitness_comparison.png'")
    
    # Display the plot
    plt.show()
    
    # Save data to CSV for further analysis
    filtered_df.to_csv('fitness_comparison_data.csv', index=False)
    print("Data saved as 'fitness_comparison_data.csv'")
    
    return filtered_df

if __name__ == "__main__":
    result_df = main()
