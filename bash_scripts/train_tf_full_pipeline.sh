#!/bin/bash
#SBATCH --job-name=train_tf_full_pipeline
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --partition=education_gpu
#SBATCH --output=logs/%x-%j.out

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

N_HVGS="${N_HVGS:-512}"
GENE_OUTPUT_DIR="${GENE_OUTPUT_DIR:-results}"

python -m src.train_contrastive_tf \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --temperature "$TEMPERATURE" \
  --weight_decay "$WEIGHT_DECAY" \
  --dim_feedforward "$HIDDEN_DIM" \
  --embedding_dim "$EMBED_DIM" \
  --classifier_hidden_dim "$CLASSIFIER_HIDDEN_DIM" \
  --classifier_dropout "$CLASSIFIER_DROPOUT" \
  --contrastive_epochs "$CONTRASTIVE_EPOCHS" \
  --classifier_epochs "$CLASSIFIER_EPOCHS" \
  --patience "$PATIENCE" \
  --min_delta "$MIN_DELTA" \
  --val_ratio "$VAL_RATIO" \
  --gene_sets_path "$GENE_SETS_PATH"

CHECKPOINT_DIR=$(ls -td "$OUTPUT_DIR"/contrastive_tf_seed${SEED}_* 2>/dev/null | head -n 1)
if [[ -z "$CHECKPOINT_DIR" ]]; then
  echo "No checkpoint found under $OUTPUT_DIR matching seed${SEED}_*" >&2
  exit 1
fi
echo "Using checkpoint: $CHECKPOINT_DIR"

python -m src.evaluate --tf_dir "$CHECKPOINT_DIR"

python -m src.attention_analysis --checkpoint_dir "$CHECKPOINT_DIR"

python -m src.attribution_ablation \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --data_path "$DATA_PATH" \
  --batch_size "$BATCH_SIZE"

python -m src.attention_graph \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --data_path "$DATA_PATH" \
  --methods raw,rollout,grad_attn \
  --scopes global,per_cell_type \
  --modalities rna,protein \
  --batch_size "$BATCH_SIZE"

python -m src.train_contrastive_tf_gene \
  --data_path "$DATA_PATH" \
  --output_dir "$GENE_OUTPUT_DIR" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --n_hvgs "$N_HVGS" \
  --lr "$LR" \
  --temperature "$TEMPERATURE" \
  --weight_decay "$WEIGHT_DECAY" \
  --dim_feedforward "$HIDDEN_DIM" \
  --embedding_dim "$EMBED_DIM" \
  --classifier_hidden_dim "$CLASSIFIER_HIDDEN_DIM" \
  --classifier_dropout "$CLASSIFIER_DROPOUT" \
  --contrastive_epochs "$CONTRASTIVE_EPOCHS" \
  --classifier_epochs "$CLASSIFIER_EPOCHS" \
  --patience "$PATIENCE" \
  --min_delta "$MIN_DELTA" \
  --val_ratio "$VAL_RATIO"

GENE_CHECKPOINT_DIR=$(ls -td "$GENE_OUTPUT_DIR"/contrastive_tf_gene_seed${SEED}_* 2>/dev/null | head -n 1)
if [[ -z "$GENE_CHECKPOINT_DIR" ]]; then
  echo "No gene-token checkpoint found under $GENE_OUTPUT_DIR matching seed${SEED}_*" >&2
  exit 1
fi
echo "Using gene-token checkpoint: $GENE_CHECKPOINT_DIR"

python -m src.attention_graph \
  --checkpoint_dir "$GENE_CHECKPOINT_DIR" \
  --data_path "$DATA_PATH" \
  --methods raw,rollout,grad_attn \
  --scopes global,per_cell_type \
  --modalities rna,protein \
  --batch_size "$BATCH_SIZE"

echo "Pipeline complete."
echo "  Pathway outputs: $CHECKPOINT_DIR"
echo "  Gene outputs:    $GENE_CHECKPOINT_DIR"
