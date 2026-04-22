#!/bin/bash
#SBATCH --job-name=train_tf_full_pipeline
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --partition=education_gpu
#SBATCH --output=logs/%x-%j.out

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

# ── Stage 1: pathway-token transformer ───────────────────────────────────────
echo ""
echo "--- [1/7] train_contrastive_tf (pathway tokens) ---"
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

CHECKPOINT_DIR=$(ls -td "$OUTPUT_DIR"/contrastive_tf_seed${SEED}_* 2>/dev/null | head -n 1)
[[ -z "$CHECKPOINT_DIR" ]] && { echo "ERROR: no pathway-tf checkpoint found under $OUTPUT_DIR" >&2; exit 1; }
echo "  checkpoint: $CHECKPOINT_DIR"

# ── Stage 2: evaluate ────────────────────────────────────────────────────────
echo ""
echo "--- [2/7] evaluate ---"
python -m src.evaluate --tf_dir "$CHECKPOINT_DIR"

# ── Stage 3: attention analysis ──────────────────────────────────────────────
echo ""
echo "--- [3/7] attention_analysis ---"
python -m src.attention_analysis --checkpoint_dir "$CHECKPOINT_DIR"

# ── Stage 4: attribution ablation ────────────────────────────────────────────
echo ""
echo "--- [4/7] attribution_ablation ---"
python -m src.attribution_ablation \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --data_path      "$DATA_PATH" \
  --batch_size     "$BATCH_SIZE"

# ── Stage 5: attention graph (pathway checkpoint) ────────────────────────────
echo ""
echo "--- [5/7] attention_graph (pathway tokens) ---"
python -m src.attention_graph \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --data_path      "$DATA_PATH" \
  --methods        raw,rollout,grad_attn \
  --scopes         global,per_cell_type \
  --modalities     rna,protein \
  --batch_size     "$BATCH_SIZE"

# ── Stage 6: gene-token transformer ──────────────────────────────────────────
echo ""
echo "--- [6/7] train_contrastive_tf_gene (gene tokens) ---"
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

GENE_CHECKPOINT_DIR=$(ls -td "$GENE_OUTPUT_DIR"/contrastive_tf_gene_seed${SEED}_* 2>/dev/null | head -n 1)
[[ -z "$GENE_CHECKPOINT_DIR" ]] && { echo "ERROR: no gene-tf checkpoint found under $GENE_OUTPUT_DIR" >&2; exit 1; }
echo "  checkpoint: $GENE_CHECKPOINT_DIR"

# ── Stage 7: attention graph (gene checkpoint) ───────────────────────────────
echo ""
echo "--- [7/7] attention_graph (gene tokens) ---"
python -m src.attention_graph \
  --checkpoint_dir "$GENE_CHECKPOINT_DIR" \
  --data_path      "$DATA_PATH" \
  --methods        raw,rollout,grad_attn \
  --scopes         global,per_cell_type \
  --modalities     rna,protein \
  --batch_size     "$BATCH_SIZE"

echo ""
echo "=== Pipeline complete ==="
echo "  Pathway outputs: $CHECKPOINT_DIR"
echo "  Gene outputs:    $GENE_CHECKPOINT_DIR"
