#!/bin/bash
#SBATCH --job-name=smoke_run_tf_train
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --partition=education_gpu
#SBATCH --output=logs/%x-%j.out
# Smoke test for run_tf_train.sh: tiny settings to verify forward pass, shapes,
# and checkpoint writing before submitting full seed-sweep jobs.

set -euo pipefail

cd ~/project_cpsc4520/cpsc4520_bcw45/5420-multimodal/

module load miniconda
source activate multimodal_env

mkdir -p logs
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SEED="${SEED:-42}"
BATCH_SIZE=64
LR="${LR:-1e-3}"
CLASSIFIER_LR="${CLASSIFIER_LR:-1e-3}"
TEMPERATURE="${TEMPERATURE:-0.07}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
EMBED_DIM="${EMBED_DIM:-128}"
CLASSIFIER_HIDDEN_DIM="${CLASSIFIER_HIDDEN_DIM:-64}"
CLASSIFIER_DROPOUT="${CLASSIFIER_DROPOUT:-0.3}"

CONTRASTIVE_EPOCHS=2
CLASSIFIER_EPOCHS=2
PATIENCE=5
VAL_RATIO=0.15
MAX_CELLS=1000

DATA_PATH="${DATA_PATH:-data/}"
GENE_SETS_PATH="${GENE_SETS_PATH:-data/kegg_2021_human.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/smoke}"
GENE_OUTPUT_DIR="${GENE_OUTPUT_DIR:-results/smoke}"
N_HVGS=64

echo "=== Smoke test: run_tf_train ==="
echo "  SEED:      $SEED"
echo "  MAX_CELLS: $MAX_CELLS"
mkdir -p "$OUTPUT_DIR" logs

# ── Pathway-token transformer ─────────────────────────────────────────────────
echo ""
echo "--- [1/2] train_contrastive_tf (pathway tokens) ---"
python -m src.train_contrastive_tf \
  --data_path              "$DATA_PATH" \
  --output_dir             "$OUTPUT_DIR" \
  --seed                   "$SEED" \
  --max_cells              $MAX_CELLS \
  --batch_size             $BATCH_SIZE \
  --lr                     "$LR" \
  --temperature            "$TEMPERATURE" \
  --weight_decay           "$WEIGHT_DECAY" \
  --dim_feedforward        "$HIDDEN_DIM" \
  --embedding_dim          "$EMBED_DIM" \
  --classifier_hidden_dim  "$CLASSIFIER_HIDDEN_DIM" \
  --classifier_dropout     "$CLASSIFIER_DROPOUT" \
  --contrastive_epochs     $CONTRASTIVE_EPOCHS \
  --classifier_epochs      $CLASSIFIER_EPOCHS \
  --patience               $PATIENCE \
  --val_ratio              $VAL_RATIO \
  --gene_sets_path         "$GENE_SETS_PATH"

CHECKPOINT_DIR=$(ls -td "$OUTPUT_DIR"/contrastive_tf_seed${SEED}_* 2>/dev/null | head -n 1)
[[ -z "$CHECKPOINT_DIR" ]] && { echo "ERROR: no pathway-tf checkpoint found under $OUTPUT_DIR" >&2; exit 1; }
echo "  checkpoint: $CHECKPOINT_DIR"

# ── Gene-token transformer ────────────────────────────────────────────────────
echo ""
echo "--- [2/2] train_contrastive_tf_gene (gene tokens) ---"
python -m src.train_contrastive_tf_gene \
  --data_path              "$DATA_PATH" \
  --output_dir             "$GENE_OUTPUT_DIR" \
  --seed                   "$SEED" \
  --max_cells              $MAX_CELLS \
  --batch_size             $BATCH_SIZE \
  --n_hvgs                 $N_HVGS \
  --lr                     "$LR" \
  --temperature            "$TEMPERATURE" \
  --weight_decay           "$WEIGHT_DECAY" \
  --dim_feedforward        "$HIDDEN_DIM" \
  --embedding_dim          "$EMBED_DIM" \
  --classifier_hidden_dim  "$CLASSIFIER_HIDDEN_DIM" \
  --classifier_dropout     "$CLASSIFIER_DROPOUT" \
  --contrastive_epochs     $CONTRASTIVE_EPOCHS \
  --classifier_epochs      $CLASSIFIER_EPOCHS \
  --patience               $PATIENCE \
  --val_ratio              $VAL_RATIO

GENE_CHECKPOINT_DIR=$(ls -td "$GENE_OUTPUT_DIR"/contrastive_tf_gene_seed${SEED}_* 2>/dev/null | head -n 1)
[[ -z "$GENE_CHECKPOINT_DIR" ]] && { echo "ERROR: no gene-tf checkpoint found under $GENE_OUTPUT_DIR" >&2; exit 1; }
echo "  checkpoint: $GENE_CHECKPOINT_DIR"

echo ""
echo "=== Smoke test passed ==="
echo "  Pathway checkpoint: $CHECKPOINT_DIR"
echo "  Gene checkpoint:    $GENE_CHECKPOINT_DIR"
