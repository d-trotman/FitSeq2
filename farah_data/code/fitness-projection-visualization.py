import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

# Define paths to your CSV files - update these paths as needed
model_estimates_path = os.path.join("/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/results/Clim_rep1_FitSeq2_Result_Read_Number_Estimated.csv")
read_counts_path = os.path.join("/Users/dawsontrotman/Documents/GitHub/FitSeq2/farah_data/fitseq_results/input/Clim_rep1_counts.csv")

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8

def load_data(model_path, reads_path):
    """
    Load the model estimates and read counts data
    """
    try:
        # Load CSV files
        model_data = pd.read_csv(model_path)
        read_data = pd.read_csv(reads_path)
        
        print(f"Model data shape: {model_data.shape}")
        print(f"Read data shape: {read_data.shape}")
        
        # Print column names for debugging
        print("Model data columns:", model_data.columns.tolist())
        print("Read data columns:", read_data.columns.tolist())
        
        return model_data, read_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def calculate_mutant_frequency(model_data, read_data, num_mutants=50, exclude_last_timepoint=True):
    """
    Calculate the frequency of each mutant at each timepoint
    """
    # Make copies to avoid modifying original data
    model_data = model_data.copy()
    read_data = read_data.copy()
    
    # Limit to specified number of mutants
    model_data = model_data.iloc[:num_mutants]
    read_data = read_data.iloc[:num_mutants]
    
    # Get timepoint columns (all columns except any non-numeric metadata columns)
    model_timepoint_cols = [col for col in model_data.columns if pd.api.types.is_numeric_dtype(model_data[col])]
    read_timepoint_cols = [col for col in read_data.columns if pd.api.types.is_numeric_dtype(read_data[col])]
    
    print(f"Found {len(model_timepoint_cols)} timepoints in model data")
    print(f"Found {len(read_timepoint_cols)} timepoints in read data")
    
    # Exclude the last timepoint if requested and if there are enough timepoints
    if exclude_last_timepoint:
        if len(model_timepoint_cols) > 1:
            model_timepoint_cols = model_timepoint_cols[:-1]
        if len(read_timepoint_cols) > 1:
            read_timepoint_cols = read_timepoint_cols[:-1]
        print(f"Excluding last timepoint. Using {len(model_timepoint_cols)} timepoints for analysis")
    
    # Ensure we have the same number of timepoints in both datasets
    min_timepoints = min(len(model_timepoint_cols), len(read_timepoint_cols))
    if len(model_timepoint_cols) != len(read_timepoint_cols):
        print(f"Warning: Different number of timepoints in datasets. Using only the first {min_timepoints} timepoints.")
        model_timepoint_cols = model_timepoint_cols[:min_timepoints]
        read_timepoint_cols = read_timepoint_cols[:min_timepoints]
    
    # Handle missing or NaN values by replacing with 0
    model_data[model_timepoint_cols] = model_data[model_timepoint_cols].fillna(0)
    read_data[read_timepoint_cols] = read_data[read_timepoint_cols].fillna(0)
    
    # Calculate total counts for each timepoint
    model_totals = model_data[model_timepoint_cols].sum(axis=0)
    read_totals = read_data[read_timepoint_cols].sum(axis=0)
    
    # Check for zero totals and add a small epsilon to avoid division by zero
    epsilon = 1e-10
    model_totals = model_totals.replace(0, epsilon)
    read_totals = read_totals.replace(0, epsilon)
    
    # Calculate frequency for each mutant at each timepoint
    model_frequencies = model_data[model_timepoint_cols].div(model_totals, axis=1)
    read_frequencies = read_data[read_timepoint_cols].div(read_totals, axis=1)
    
    # Print statistics
    print(f"Model frequencies range: {model_frequencies.min().min()} to {model_frequencies.max().max()}")
    print(f"Read frequencies range: {read_frequencies.min().min()} to {read_frequencies.max().max()}")
    
    return model_frequencies, read_frequencies, model_timepoint_cols, read_timepoint_cols

def create_panel_plots(model_freqs, read_freqs, model_cols, read_cols, mutants_per_panel=10, panels_per_figure=5):
    """
    Create panel plots for the mutant frequencies
    
    Parameters:
    -----------
    model_freqs : DataFrame
        Model frequency data for each mutant
    read_freqs : DataFrame
        Read frequency data for each mutant
    model_cols : list
        Column names for model timepoints
    read_cols : list
        Column names for read timepoints
    mutants_per_panel : int
        Number of mutants to display in each panel
    panels_per_figure : int
        Number of panels to include in each figure
    
    Returns:
    --------
    list of Figure objects
    """
    num_mutants = len(model_freqs)
    num_panels = (num_mutants + mutants_per_panel - 1) // mutants_per_panel  # Ceiling division
    num_figures = (num_panels + panels_per_figure - 1) // panels_per_figure  # Ceiling division
    
    print(f"Creating {num_figures} figures with {panels_per_figure} panels each")
    print(f"Total panels: {num_panels}, Total mutants: {num_mutants}")
    
    # Create timepoints array for x-axis (0, 1, 2, ...)
    timepoints = np.arange(len(model_cols))
    
    # Generate distinct colors for each mutant in a panel
    color_cycle = plt.cm.tab10(np.linspace(0, 1, mutants_per_panel))
    
    figures = []
    
    # Loop through figures
    for fig_idx in range(num_figures):
        # Create figure
        fig = plt.figure(figsize=(15, 3.5 * min(panels_per_figure, num_panels - fig_idx * panels_per_figure)))
        gs = GridSpec(min(panels_per_figure, num_panels - fig_idx * panels_per_figure), 1, figure=fig, hspace=0.5)
        
        # Calculate start and end panel indices for this figure
        start_panel = fig_idx * panels_per_figure
        end_panel = min(start_panel + panels_per_figure, num_panels)
        
        # Loop through panels for this figure
        for panel_offset, panel_idx in enumerate(range(start_panel, end_panel)):
            # Determine which mutants go in this panel
            start_idx = panel_idx * mutants_per_panel
            end_idx = min(start_idx + mutants_per_panel, num_mutants)
            
            # Create subplot for this panel
            ax = fig.add_subplot(gs[panel_offset])
            
            # Plot each mutant in this panel
            for i, idx in enumerate(range(start_idx, end_idx)):
                mutant_id = idx  # Using index as mutant ID for simplicity
                color = color_cycle[i]
                
                # Get frequencies for this mutant
                model_freq_values = model_freqs.iloc[idx].values
                read_freq_values = read_freqs.iloc[idx].values
                
                # Plot lines
                model_line, = ax.plot(timepoints, model_freq_values, '-o', color=color, 
                                   label=f'Model {mutant_id}', markersize=4, linewidth=1.5)
                read_line, = ax.plot(timepoints, read_freq_values, '--*', color=color, 
                                  label=f'Read {mutant_id}', markersize=4, linewidth=1.5, alpha=0.7)
            
            # Create a better legend
            # First add legend for mutants
            handles, labels = ax.get_lines(), [f"Mutant {idx}" for idx in range(start_idx, end_idx)]
            first_legend = ax.legend(handles[::2], labels, loc='upper left', 
                                  bbox_to_anchor=(1.01, 1), title="Mutants", fontsize=8)
            ax.add_artist(first_legend)
            
            # Then add legend for data types
            ax.legend([handles[0], handles[1]], ['Model', 'Read'], 
                   loc='upper left', bbox_to_anchor=(1.01, 0.5), title="Data Type", fontsize=8)
            
            # Set labels and add grid
            ax.set_xlabel('Timepoint')
            ax.set_ylabel('Relative Frequency')
            ax.set_title(f'Mutants {start_idx}-{end_idx-1}')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Set y-axis to scientific notation if values are small
            if np.max(model_freq_values) < 0.01 or np.max(read_freq_values) < 0.01:
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                
            # Set x-ticks to integers
            ax.set_xticks(timepoints)
            ax.set_xticklabels([str(int(t)) for t in timepoints])
        
        plt.tight_layout()
        figures.append(fig)
        
    return figures

def handle_errors(model_data, read_data):
    """
    Check for potential errors in the data
    """
    if model_data is None or read_data is None:
        print("Error: One or both datasets failed to load.")
        return False
    
    if model_data.empty or read_data.empty:
        print("Error: One or both datasets are empty.")
        return False
    
    if model_data.shape[0] < 1 or read_data.shape[0] < 1:
        print("Error: Not enough mutants in one or both datasets.")
        return False
    
    numeric_model_cols = [col for col in model_data.columns if pd.api.types.is_numeric_dtype(model_data[col])]
    numeric_read_cols = [col for col in read_data.columns if pd.api.types.is_numeric_dtype(read_data[col])]
    
    if len(numeric_model_cols) < 2 or len(numeric_read_cols) < 2:
        print("Error: Not enough timepoints in one or both datasets.")
        return False
    
    return True

def main():
    """
    Main function to run the analysis
    """
    print("Loading S. cerevisiae mutant data...")
    
    # Load data
    model_data, read_data = load_data(model_estimates_path, read_counts_path)
    
    # Validate data
    if not handle_errors(model_data, read_data):
        print("Exiting due to data errors.")
        return
    
    # Calculate frequencies for the first 200 mutants
    num_mutants = 200
    print(f"Calculating frequencies for the first {num_mutants} mutants...")
    model_freqs, read_freqs, model_cols, read_cols = calculate_mutant_frequency(
        model_data, read_data, num_mutants=num_mutants
    )
    
    # Create panel plots with 10 mutants per panel, 10 panels per figure
    mutants_per_panel = 10
    panels_per_figure = 10
    print(f"Creating panel plots with {mutants_per_panel} mutants per panel, {panels_per_figure} panels per figure...")
    figures = create_panel_plots(
        model_freqs, read_freqs, model_cols, read_cols, 
        mutants_per_panel=mutants_per_panel,
        panels_per_figure=panels_per_figure
    )
    
    # Save figures
    for i, fig in enumerate(figures):
        output_path = f"mutant_fitness_panels_{i+1}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved panel plot {i+1} to {output_path}")
    
    # Also create a simplified overview with just the first 10 mutants
    print("Creating a simplified overview plot...")
    overview_fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(1, 1)
    ax = overview_fig.add_subplot(gs[0])
    
    # Create timepoints array for x-axis (0, 1, 2, ...)
    timepoints = np.arange(len(model_cols))
    
    # Generate distinct colors for the first 10 mutants
    color_cycle = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot the first 10 mutants
    for i in range(10):
        color = color_cycle[i]
        model_freq_values = model_freqs.iloc[i].values
        read_freq_values = read_freqs.iloc[i].values
        
        ax.plot(timepoints, model_freq_values, '-o', color=color, 
                label=f'Mutant {i} (Model)', markersize=5, linewidth=2)
        ax.plot(timepoints, read_freq_values, '--*', color=color, 
                label=f'Mutant {i} (Read)', markersize=5, linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Relative Frequency', fontsize=12)
    ax.set_title('Overview of First 10 Mutants', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Set x-ticks to integers
    ax.set_xticks(timepoints)
    ax.set_xticklabels([str(int(t)) for t in timepoints])
    
    # Add legend
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=10)
    
    plt.tight_layout()
    overview_path = "mutant_fitness_overview.png"
    overview_fig.savefig(overview_path, dpi=300, bbox_inches='tight')
    print(f"Saved overview plot to {overview_path}")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
