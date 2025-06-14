import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data files
file1 = pd.read_csv("discrete_results_FitSeq2_Result_Read_Number_Estimated.csv", header=None)
file2 = pd.read_csv("discrete_reads.csv", header=None)

# Select timepoint to compare (using timepoint 1)
timepoint = 3

# Calculate frequencies
file1_freq = file1[timepoint] / file1[timepoint].sum()
file2_freq = file2[timepoint] / file2[timepoint].sum()

# Create plot
plt.figure(figsize=(10, 8))

# Plot all strains
plt.scatter(file2_freq, file1_freq, alpha=0.3, s=5, color='blue')

# Add diagonal line (equal frequency line)
max_val = max(file1_freq.max(), file2_freq.max())
min_val = max(1e-10, min(file1_freq.min(), file2_freq.min()))
plt.plot([min_val, max_val], [min_val, max_val], 'k--')

# Set log scales
plt.xscale('log')
plt.yscale('log')

# Add labels
plt.xlabel('Simulated Frequency')
plt.ylabel('FitSeq2 Frequency')
plt.title(f'Simulation vs FitSeq2 Frequency (Timepoint {timepoint})')

# Save the plot
plt.tight_layout()
plt.savefig('discrete_simfit_comparison.png.png', dpi=300)
print("Plot saved as discrete_simfit_comparison.png")
