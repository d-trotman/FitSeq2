import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def compare_frequencies(chemostat_sim, discrete_sim, 
                        chemostat_fitseq_reads, discrete_fitseq_reads):
    """
    Compare original frequencies with FitSeq2-predicted frequencies
    for both chemostat and discrete simulations using log scale
    
    Args:
        chemostat_sim: Path to chemostat simulation CSV
        discrete_sim: Path to discrete simulation CSV
        chemostat_fitseq_reads: Path to chemostat FitSeq2 estimated reads
        discrete_fitseq_reads: Path to discrete FitSeq2 estimated reads
    """
    # Load original data
    chemostat_data = pd.read_csv(chemostat_sim)
    discrete_data = pd.read_csv(discrete_sim)
    
    # Load FitSeq2 estimated reads
    chemostat_fitseq = pd.read_csv(chemostat_fitseq_reads, header=None)
    discrete_fitseq = pd.read_csv(discrete_fitseq_reads, header=None)
    
    # Get the number of timepoints
    num_timepoints = len([col for col in chemostat_data.columns if col.startswith('reads_t')])
    
    # Create a figure with subplots for each timepoint (excluding the first)
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    # For each timepoint (starting from t1, skipping t0)
    for idx, t in enumerate(range(1, num_timepoints)):
        # Get original frequencies
        chemostat_orig_freq = chemostat_data[f'frequency_t{t}'].values
        discrete_orig_freq = discrete_data[f'frequency_t{t}'].values
        
        # Calculate FitSeq2 frequencies (read counts divided by total)
        chemostat_fitseq_counts = chemostat_fitseq[t].values
        discrete_fitseq_counts = discrete_fitseq[t].values
        
        chemostat_fitseq_freq = chemostat_fitseq_counts / np.sum(chemostat_fitseq_counts)
        discrete_fitseq_freq = discrete_fitseq_counts / np.sum(discrete_fitseq_counts)
        
        # Replace zeros with small value for log scale
        min_nonzero = 1e-15
        chemostat_orig_freq[chemostat_orig_freq == 0] = min_nonzero
        discrete_orig_freq[discrete_orig_freq == 0] = min_nonzero
        chemostat_fitseq_freq[chemostat_fitseq_freq == 0] = min_nonzero
        discrete_fitseq_freq[discrete_fitseq_freq == 0] = min_nonzero
        
        # Plot on the appropriate subplot
        ax = axes[idx]
        
        # Plot chemostat data
        ax.scatter(chemostat_orig_freq, chemostat_fitseq_freq, 
                  alpha=0.5, label='Chemostat', color='blue', s=20)
        
        # Plot discrete data
        ax.scatter(discrete_orig_freq, discrete_fitseq_freq, 
                  alpha=0.5, label='Discrete', color='red', s=20)
        
        # Calculate regression lines
        # Chemostat
        chemo_log_x = np.log10(chemostat_orig_freq)
        chemo_log_y = np.log10(chemostat_fitseq_freq)
        chemo_slope, chemo_intercept, chemo_r, chemo_p, chemo_std_err = stats.linregress(chemo_log_x, chemo_log_y)
        
        # Discrete
        discrete_log_x = np.log10(discrete_orig_freq)
        discrete_log_y = np.log10(discrete_fitseq_freq)
        discrete_slope, discrete_intercept, discrete_r, discrete_p, discrete_std_err = stats.linregress(discrete_log_x, discrete_log_y)
        
        # Plot regression lines
        x_range = np.logspace(np.log10(min_nonzero), np.log10(1.0), 100)
        
        # Chemostat regression line
        y_chemo = 10**(chemo_slope * np.log10(x_range) + chemo_intercept)
        ax.plot(x_range, y_chemo, 'b-', linewidth=1)
        
        # Discrete regression line
        y_discrete = 10**(discrete_slope * np.log10(x_range) + discrete_intercept)
        ax.plot(x_range, y_discrete, 'r-', linewidth=1)
        
        # Add y=x line (perfect prediction) spanning the full range
        ax.plot([min_nonzero/10, 2], [min_nonzero/10, 2], 'k-', alpha=0.7, linewidth=1.5, label='y=x')
        
        # Set log scale for both axes
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Add labels and title
        ax.set_xlabel('Original Frequency')
        ax.set_ylabel('FitSeq2 Predicted Frequency')
        ax.set_title(f'Timepoint {t}\n'
                     f'Chemo: y = {10**chemo_intercept:.2e} · x^{chemo_slope:.2f}, r²={chemo_r**2:.3f}\n'
                     f'Discrete: y = {10**discrete_intercept:.2e} · x^{discrete_slope:.2f}, r²={discrete_r**2:.3f}')
        
        # Add legend (only for first subplot)
        if idx == 0:
            ax.legend()
        
        # Set equal aspect ratio for log-log plot
        ax.set_aspect('equal')
        
        # Set limits
        ax.set_xlim(min_nonzero/10, 2)
        ax.set_ylim(min_nonzero/10, 2)
            
    # Remove any unused subplots
    for i in range(num_timepoints-1, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.savefig('frequency_comparison_log.png', dpi=300)
    plt.close()
    
    print("Frequency comparison visualization saved to frequency_comparison_log.png")

def main():
    # Define file paths
    chemostat_sim = 'yeast_chemostat_simulation.csv'
    discrete_sim = 'yeast_discrete_simulation.csv'
    chemostat_fitseq_reads = 'chemostat_results_FitSeq2_Result_Read_Number_Estimated.csv'
    discrete_fitseq_reads = 'discrete_results_FitSeq2_Result_Read_Number_Estimated.csv'
    
    # Compare frequencies
    compare_frequencies(chemostat_sim, discrete_sim, 
                        chemostat_fitseq_reads, discrete_fitseq_reads)

if __name__ == "__main__":
    main()