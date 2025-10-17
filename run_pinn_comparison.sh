#!/bin/bash
#SBATCH --time 08:00:00
#SBATCH --job-name pinn-comparison
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --mem=64GB
#SBATCH --output pinn-comparison.out
#SBATCH --error pinn-comparison.err

# Print job info for debugging
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"

# Load your environment with error checking
echo "Activating Python environment..."
source /mnt/rafast/miler/python-env/pytorch/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate Python environment"
    exit 1
fi

# Verify Python and packages
echo "Python path: $(which python)"

# Change to your working directory
echo "Changing to working directory..."
cd /mnt/rafast/miler/codes/Pinns
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to change to /home/astro/miler/codes/Pinns"
    exit 1
fi

echo "Current directory: $(pwd)"
echo "Files in directory:"
ls -la test_many_pinn.py

# Run the PINN comparison script with verbose output
echo "Starting PINN comparison at: $(date)"
python -u test_many_guided_pinn.py \
    --n_train 400 \
    --n_val 200 \
    --n_test 200 \
    --epochs 300 \
    --output_dir ./pinn_guided_results

# Check exit status
if [ $? -eq 0 ]; then
    echo "PINN comparison completed successfully at $(date)"
else
    echo "ERROR: PINN comparison failed with exit code $?"
    exit 1
fi
