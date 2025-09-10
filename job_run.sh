#!/bin/bash
#SBATCH --job-name=mae_meld_extract
#SBATCH --output=/scratch/data/bikash_rs/vivek/MELD-feature-extract/logs/%x_%j.out
#SBATCH --error=/scratch/data/bikash_rs/vivek/MELD-feature-extract/logs/%x_%j.err
#SBATCH --partition=dgx          # GPU partition
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --cpus-per-task=12       # CPUs for dataloader + decoding
#SBATCH --mem=64G                # Memory
#SBATCH --time=24:00:00          # Walltime
#SBATCH --nodes=1                # Single node

# Load modules (adjust to your cluster setup)
module load python/3.10
# module load cuda/11.8

# Activate your venv
source venv/bin/activate

# Sanity check GPU
python gpu_check.py

# Run extraction
# python extract_mae_embedding.py \
#     --dataset 'MELD' \
#     --video_dir "/scratch/data/bikash_rs/vivek/dataset/MELD.Raw/train_splits" \
#     --save_dir "/scratch/data/bikash_rs/vivek/dataset/Meld_feat_ext/train_mae_feat" \
#     --pretrain_model "mae_checkpoint-340" \
#     --device cuda \
#     --feature_level UTTERANCE
