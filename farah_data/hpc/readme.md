# FitSeq2 HPC Analysis

This package contains scripts to run FitSeq2 analysis on a High-Performance Computing (HPC) cluster using SLURM.

## Files Included

1. `run_fitseq_jobs.sh` - SLURM job submission script
2. `run_single_analysis.py` - Python script to run a single condition/replicate analysis
3. `setup_fitseq.sh` - Script to set up the environment and install dependencies
4. `condition_replicate_map.txt` - Mapping file for condition-replicate combinations

## Setup Instructions

1. Copy these scripts to your HPC directory
2. Copy your `barcode_counts.csv` file to the same directory
3. Copy `FitSeq2.py` to the same directory
4. Run the setup script:

```bash
bash setup_fitseq.sh
```

## Running the Analysis

Submit the SLURM job using:

```bash
sbatch run_fitseq_jobs.sh
```

This will launch multiple job arrays, one for each condition/replicate combination.

## Customizing the Analysis

You may need to adjust the following parameters:

- In `run_fitseq_jobs.sh`:
  - `--array=1-15`: Change if you have a different number of condition/replicate pairs
  - `--cpus-per-task=16`: Adjust based on available resources
  - `--mem=32G`: Adjust memory requirements
  - `--time=24:00:00`: Adjust time limit

- In `run_single_analysis.py`:
  - Update the file paths at the top of the script if needed
  - Change FitSeq2 parameters (delta_t, c, algorithm, max_iter) if needed

## Output

The analysis will generate:

- Input files in `fitseq_results/input/`
- Result files in `fitseq_results/results/`
- Plots in `fitseq_results/plots/`
- Log files in `logs/`

## Post-Processing

After all jobs complete, you may want to run additional analysis to compare conditions. You can create a separate script for this purpose.

## Troubleshooting

Check log files in the `logs/` directory if jobs fail. Common issues include:

- Missing dependencies: Make sure all required Python packages are installed
- Memory issues: Increase the `--mem` parameter
- Time limit: Increase the `--time` parameter
- Resource availability: Check cluster availability and job queue
