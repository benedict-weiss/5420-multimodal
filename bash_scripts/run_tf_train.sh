#!/bin/bash
#SBATCH --job-name=run_tf_train
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --partition=education_gpu
#SBATCH --output=logs/%x-%j.out

# When run directly (not via sbatch), submit one job per seed and exit.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  for seed in 13 42 77; do
    SEED=$seed sbatch "$0"
  done
  exit 0
fi

set -euo pipefail

cd ~/project_cpsc4520/cpsc4520_bcw45/5420-multimodal/

module load miniconda
source activate multimodal_env

mkdir -p logs
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
CLASSIFIER_LR="${CLASSIFIER_LR:-1e-3}"
TEMPERATURE="${TEMPERATURE:-0.07}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
EMBED_DIM="${EMBED_DIM:-128}"
CLASSIFIER_HIDDEN_DIM="${CLASSIFIER_HIDDEN_DIM:-64}"
CLASSIFIER_DROPOUT="${CLASSIFIER_DROPOUT:-0.3}"

CONTRASTIVE_EPOCHS="${CONTRASTIVE_EPOCHS:-150}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-30}"

PATIENCE="${PATIENCE:-20}"
MIN_DELTA="${MIN_DELTA:-1e-4}"
VAL_RATIO="${VAL_RATIO:-0.1}"

DATA_PATH="${DATA_PATH:-data/}"
GENE_SETS_PATH="${GENE_SETS_PATH:-data/kegg_2021_human.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/checkpoints}"
GENE_OUTPUT_DIR="${GENE_OUTPUT_DIR:-results/checkpoints}"
N_HVGS="${N_HVGS:-512}"

# ── Pathway-token transformer ─────────────────────────────────────────────────
echo ""
echo "--- [1/2] train_contrastive_tf (pathway tokens) ---"
python -m src.train_contrastive_tf \
  --data_path              "$DATA_PATH" \
  --output_dir             "$OUTPUT_DIR" \
  --seed                   "$SEED" \
  --batch_size             "$BATCH_SIZE" \
  --lr                     "$LR" \
  --temperature            "$TEMPERATURE" \
  --weight_decay           "$WEIGHT_DECAY" \
  --dim_feedforward        "$HIDDEN_DIM" \
  --embedding_dim          "$EMBED_DIM" \
  --classifier_hidden_dim  "$CLASSIFIER_HIDDEN_DIM" \
  --classifier_dropout     "$CLASSIFIER_DROPOUT" \
  --contrastive_epochs     "$CONTRASTIVE_EPOCHS" \
  --classifier_epochs      "$CLASSIFIER_EPOCHS" \
  --patience               "$PATIENCE" \
  --min_delta              "$MIN_DELTA" \
  --val_ratio              "$VAL_RATIO" \
  --gene_sets_path         "$GENE_SETS_PATH"

# ── Gene-token transformer ────────────────────────────────────────────────────
echo ""
echo "--- [2/2] train_contrastive_tf_gene (gene tokens) ---"
python -m src.train_contrastive_tf_gene \
  --data_path              "$DATA_PATH" \
  --output_dir             "$GENE_OUTPUT_DIR" \
  --seed                   "$SEED" \
  --batch_size             "$BATCH_SIZE" \
  --n_hvgs                 "$N_HVGS" \
  --lr                     "$LR" \
  --temperature            "$TEMPERATURE" \
  --weight_decay           "$WEIGHT_DECAY" \
  --dim_feedforward        "$HIDDEN_DIM" \
  --embedding_dim          "$EMBED_DIM" \
  --classifier_hidden_dim  "$CLASSIFIER_HIDDEN_DIM" \
  --classifier_dropout     "$CLASSIFIER_DROPOUT" \
  --contrastive_epochs     "$CONTRASTIVE_EPOCHS" \
  --classifier_epochs      "$CLASSIFIER_EPOCHS" \
  --patience               "$PATIENCE" \
  --min_delta              "$MIN_DELTA" \
  --val_ratio              "$VAL_RATIO"

echo ""
echo "=== Training complete ==="
echo "  Pathway outputs: $OUTPUT_DIR"
echo "  Gene outputs:    $GENE_OUTPUT_DIR"
