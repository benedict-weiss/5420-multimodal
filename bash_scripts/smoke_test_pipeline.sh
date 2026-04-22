#!/bin/bash
# Smoke test: runs the full pipeline end-to-end with tiny settings to catch import
# errors, shape mismatches, and checkpoint compatibility issues before submitting
# a full cluster job. Completes in ~5 minutes on CPU.
#
# Usage:
#   bash bash_scripts/smoke_test_pipeline.sh
#   DATA_PATH=data/ bash bash_scripts/smoke_test_pipeline.sh
#
# Not intended for SLURM — run locally or in an interactive session.

set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

DATA_PATH="${DATA_PATH:-data/}"
OUTPUT_DIR="${OUTPUT_DIR:-results/smoke}"
GENE_OUTPUT_DIR="${GENE_OUTPUT_DIR:-results/smoke}"
SEED=42
MAX_CELLS=1000
BATCH_SIZE=64
CONTRASTIVE_EPOCHS=2
CLASSIFIER_EPOCHS=2
PATIENCE=5
VAL_RATIO=0.15
GENE_SETS_PATH="${GENE_SETS_PATH:-data/kegg_2021_human.json}"
N_HVGS=64

echo "=== Smoke test pipeline ==="
echo "  DATA_PATH:   $DATA_PATH"
echo "  OUTPUT_DIR:  $OUTPUT_DIR"
echo "  MAX_CELLS:   $MAX_CELLS"
mkdir -p "$OUTPUT_DIR" logs

# ── Stage 1: pathway-token transformer ───────────────────────────────────────
echo ""
echo "--- [1/6] train_contrastive_tf (pathway tokens) ---"
python -m src.train_contrastive_tf \
  --data_path       "$DATA_PATH" \
  --output_dir      "$OUTPUT_DIR" \
  --seed            $SEED \
  --max_cells       $MAX_CELLS \
  --batch_size      $BATCH_SIZE \
  --contrastive_epochs $CONTRASTIVE_EPOCHS \
  --classifier_epochs  $CLASSIFIER_EPOCHS \
  --patience        $PATIENCE \
  --val_ratio       $VAL_RATIO \
  --gene_sets_path  "$GENE_SETS_PATH" \
  --cpu

CHECKPOINT_DIR=$(ls -td "$OUTPUT_DIR"/contrastive_tf_seed${SEED}_* 2>/dev/null | head -n 1)
[[ -z "$CHECKPOINT_DIR" ]] && { echo "ERROR: no pathway-tf checkpoint found"; exit 1; }
echo "  checkpoint: $CHECKPOINT_DIR"

# ── Stage 2: evaluate ────────────────────────────────────────────────────────
echo ""
echo "--- [2/6] evaluate ---"
python -m src.evaluate --tf_dir "$CHECKPOINT_DIR"

# ── Stage 3: attention analysis ──────────────────────────────────────────────
echo ""
echo "--- [3/6] attention_analysis ---"
python -m src.attention_analysis --checkpoint_dir "$CHECKPOINT_DIR"

# ── Stage 4: attention graph (raw only, global scope — fastest meaningful check)
echo ""
echo "--- [4/6] attention_graph (raw, global) ---"
python -m src.attention_graph \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --data_path      "$DATA_PATH" \
  --methods        raw \
  --scopes         global \
  --modalities     rna,protein \
  --batch_size     $BATCH_SIZE \
  --cpu

# ── Stage 5: gene-token transformer ──────────────────────────────────────────
echo ""
echo "--- [5/6] train_contrastive_tf_gene (gene tokens) ---"
python -m src.train_contrastive_tf_gene \
  --data_path      "$DATA_PATH" \
  --output_dir     "$GENE_OUTPUT_DIR" \
  --seed           $SEED \
  --max_cells      $MAX_CELLS \
  --batch_size     $BATCH_SIZE \
  --n_hvgs         $N_HVGS \
  --contrastive_epochs $CONTRASTIVE_EPOCHS \
  --classifier_epochs  $CLASSIFIER_EPOCHS \
  --patience       $PATIENCE \
  --val_ratio      $VAL_RATIO \
  --cpu

GENE_CHECKPOINT_DIR=$(ls -td "$GENE_OUTPUT_DIR"/contrastive_tf_gene_seed${SEED}_* 2>/dev/null | head -n 1)
[[ -z "$GENE_CHECKPOINT_DIR" ]] && { echo "ERROR: no gene-tf checkpoint found"; exit 1; }
echo "  checkpoint: $GENE_CHECKPOINT_DIR"

# ── Stage 6: attention graph on gene checkpoint ───────────────────────────────
echo ""
echo "--- [6/6] attention_graph on gene checkpoint (raw, global) ---"
python -m src.attention_graph \
  --checkpoint_dir "$GENE_CHECKPOINT_DIR" \
  --data_path      "$DATA_PATH" \
  --methods        raw \
  --scopes         global \
  --modalities     rna,protein \
  --batch_size     $BATCH_SIZE \
  --cpu

echo ""
echo "=== Smoke test passed ==="
echo "  Pathway checkpoint: $CHECKPOINT_DIR"
echo "  Gene checkpoint:    $GENE_CHECKPOINT_DIR"
