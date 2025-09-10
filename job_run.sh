#!/bin/bash
#SBATCH --job-name=mae_meld_extract
#SBATCH --output=/scratch/data/bikash_rs/vivek/MELD-feature-extract/logs/%x_%j.out
#SBATCH --error=/scratch/data/bikash_rs/vivek/MELD-feature-extract/logs/%x_%j.err
#SBATCH --partition=test              # GPU partition
#SBATCH --cpus-per-task=12           # CPUs for dataloader + decoding
#SBATCH --mem-per-cpu=2G             # 2 GB per CPU core = 24 GB total
#SBATCH --time=24:00:00              # Walltime
#SBATCH --nodes=1                    # Single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH -D /scratch/data/bikash_rs/vivek/MELD-feature-extract   # Working directory (important!)

# Load modules (adjust to your cluster setup)
module load python/3.10
# module load cuda/11.8

# Activate your venv (must exist in the repo root)
source venv/bin/activate

# Sanity check GPU
# python gpu_check.py

# Run extraction
python extract_mae_embedding.py --dataset 'MELD' --video_dir "/scratch/data/bikash_rs/vivek/dataset/MELD.Raw/train_splits" --save_dir "/scratch/data/bikash_rs/vivek/dataset/Meld_feat_ext/train_mae_feat" --pretrain_model "mae_checkpoint-340" --device cpu --feature_level UTTERANCE
