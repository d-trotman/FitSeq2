#!/bin/bash
#SBATCH --job-name=FitSeq2
#SBATCH --output=logs/fitseq_%A_%a.out
#SBATCH --error=logs/fitseq_%A_%a.err
#SBATCH --array=1-15  # Adjust based on your condition-replicate pairs
#SBATCH --cpus-per-task=16  # Adjust based on available resources
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Print some diagnostic information
echo "Starting FitSeq2 job $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
echo "Using $SLURM_CPUS_PER_TASK CPUs"

# Load the FitSeq2 module
module load fitseq2/20241219

# Create output directories if they don't exist
mkdir -p results
mkdir -p logs

# Run the corresponding analysis based on array ID
python3 run_single_analysis.py --job-id=$SLURM_ARRAY_TASK_ID --cpus=$SLURM_CPUS_PER_TASK

echo "Completed FitSeq2 job $SLURM_ARRAY_TASK_ID at $(date)"
