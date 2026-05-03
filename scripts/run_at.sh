#!/bin/bash
#SBATCH --job-name=tf_at
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --partition=education_gpu
#SBATCH --output=logs/%x-%j.out

cd ~/project_cpsc4520/cpsc4520_bcw45/5420-multimodal/

# Load modules
module load miniconda

# Activate environment
source activate multimodal_env

# Run training
python -m src.train_contrastive_tf --data_path data/ --seed 42 --batch_size 256 --gene_sets_path data/kegg_2021_human.json

CHECKPOINT_DIR=$(ls -td results/checkpoints/* | head -n 1)

python -m src.evaluate --tf_dir "$CHECKPOINT_DIR"
python -m src.attention_analysis --checkpoint_dir "$CHECKPOINT_DIR"
python -m src.attribution_ablation --checkpoint_dir "$CHECKPOINT_DIR" --data_path data/ --batch_size 256