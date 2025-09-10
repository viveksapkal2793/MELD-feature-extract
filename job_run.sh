#!/bin/bash
#SBATCH --job-name=mae_extract      # Job name
#SBATCH --output=logs/%x_%j.out     # Output log file (%j = job ID)
#SBATCH --error=logs/%x_%j.err      # Error log file
#SBATCH --partition=gpu             # Partition (queue), check your cluster’s partition names
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=8G                   # Memory (adjust as needed)
#SBATCH --cpus-per-task=8           # Number of CPU cores
#SBATCH --time=00:10:00             # Max runtime (24 hrs here)

# Load required modules
module load python/3.10
# module load cuda/11.8   # Example, check your server’s CUDA

# Activate virtual environment
source venv/bin/activate

# Test GPU access (optional sanity check)
python gpu_check.py