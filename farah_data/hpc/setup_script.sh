#!/bin/bash
# Setup script for FitSeq2 analysis on HPC

# Create directory structure
mkdir -p fitseq_results/input
mkdir -p fitseq_results/results
mkdir -p fitseq_results/plots
mkdir -p logs

# Create condition_replicate_map.txt file
echo "Creating condition-replicate mapping file..."
# Format: job_id,condition,replicate
cat > condition_replicate_map.txt << 'EOF'
1,Clim,1
2,Clim,2
3,Clim,3
4,Nlim,1
5,Nlim,2
6,Nlim,3
7,PulseAS,1
8,PulseAS,2
9,PulseAS,3
10,PulseGln,1
11,PulseGln,2
12,PulseGln,3
13,Switch,1
14,Switch,2
15,Switch,3
EOF

# Check if required Python packages are available
echo "Checking for required Python packages..."
python3 -c "import pandas, numpy, matplotlib, seaborn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Some required Python packages are missing."
    echo "You may need to load a Python module or use a virtual environment."
    echo "Required packages: pandas, numpy, matplotlib, seaborn"
fi

echo "Setup complete! To run the analysis, execute:"
echo "sbatch run_fitseq_jobs.sh"
