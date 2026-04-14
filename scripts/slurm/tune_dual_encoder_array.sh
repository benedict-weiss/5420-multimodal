#!/usr/bin/env bash
#SBATCH --job-name=dualenc-tune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

# -----------------------------------------------------------------------------
# Slurm array tuner for dual encoders.
#
# Submit examples:
#   MODEL=mlp sbatch --array=0-107 scripts/slurm/tune_dual_encoder_array.sh
#   MODEL=tf  sbatch --array=0-431 scripts/slurm/tune_dual_encoder_array.sh
#
# You can shrink/expand the search space via env vars without editing this file.
# -----------------------------------------------------------------------------

REPO_ROOT="${REPO_ROOT:-$PWD}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/checkpoints}"
MODEL="${MODEL:-mlp}"                    # mlp | tf

# Optional environment activation.
ENV_ACTIVATE="${ENV_ACTIVATE:-}"

# Global training controls.
VAL_RATIO="${VAL_RATIO:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
PATIENCE="${PATIENCE:-10}"
MIN_DELTA="${MIN_DELTA:-1e-4}"
CONTRASTIVE_EPOCHS="${CONTRASTIVE_EPOCHS:-150}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-30}"

# Search-space defaults (override as space-separated env vars).
read -r -a SEED_LIST <<< "${SEED_LIST:-13 42 77}"
read -r -a LR_LIST <<< "${LR_LIST:-3e-4 1e-3 3e-3}"
read -r -a TEMP_LIST <<< "${TEMP_LIST:-0.03 0.07 0.1}"
read -r -a BATCH_LIST <<< "${BATCH_LIST:-128 256}"

# MLP-only search dimensions.
read -r -a HIDDEN_LIST <<< "${HIDDEN_LIST:-256 512}"
read -r -a EMBED_LIST <<< "${EMBED_LIST:-128}"
read -r -a CLASSIFIER_DROPOUT_LIST <<< "${CLASSIFIER_DROPOUT_LIST:-0.2}"

# Transformer-only search dimensions.
read -r -a D_MODEL_LIST <<< "${D_MODEL_LIST:-64 128}"
read -r -a NHEAD_LIST <<< "${NHEAD_LIST:-4}"
read -r -a NUM_LAYERS_LIST <<< "${NUM_LAYERS_LIST:-2 4}"
read -r -a FF_LIST <<< "${FF_LIST:-256}"
read -r -a TF_DROPOUT_LIST <<< "${TF_DROPOUT_LIST:-0.1 0.2}"
read -r -a TF_EMBED_LIST <<< "${TF_EMBED_LIST:-128}"

mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT"

if [[ -n "$ENV_ACTIVATE" ]]; then
  eval "$ENV_ACTIVATE"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required. Submit with sbatch --array=...}"

idx_mod_pick() {
  local idx="$1"
  shift
  local arr=("$@")
  local n="${#arr[@]}"
  local pick=$(( idx % n ))
  local next=$(( idx / n ))
  echo "${arr[$pick]} $next"
}

if [[ "$MODEL" == "mlp" ]]; then
  total=$(( ${#SEED_LIST[@]} * ${#LR_LIST[@]} * ${#TEMP_LIST[@]} * ${#BATCH_LIST[@]} * ${#HIDDEN_LIST[@]} * ${#EMBED_LIST[@]} * ${#CLASSIFIER_DROPOUT_LIST[@]} ))
  if (( TASK_ID < 0 || TASK_ID >= total )); then
    echo "TASK_ID=$TASK_ID out of range for MODEL=mlp (0..$((total - 1)))." >&2
    exit 2
  fi

  idx="$TASK_ID"
  read -r seed idx <<< "$(idx_mod_pick "$idx" "${SEED_LIST[@]}")"
  read -r lr idx <<< "$(idx_mod_pick "$idx" "${LR_LIST[@]}")"
  read -r temp idx <<< "$(idx_mod_pick "$idx" "${TEMP_LIST[@]}")"
  read -r batch idx <<< "$(idx_mod_pick "$idx" "${BATCH_LIST[@]}")"
  read -r hidden idx <<< "$(idx_mod_pick "$idx" "${HIDDEN_LIST[@]}")"
  read -r embed idx <<< "$(idx_mod_pick "$idx" "${EMBED_LIST[@]}")"
  read -r clf_drop idx <<< "$(idx_mod_pick "$idx" "${CLASSIFIER_DROPOUT_LIST[@]}")"

  trial="mlp_s${seed}_lr${lr}_t${temp}_b${batch}_h${hidden}_e${embed}_cd${clf_drop}"
  out_dir="$OUT_ROOT/tune/$trial"
  mkdir -p "$out_dir"

  python src/train_contrastive_mlp.py \
    --data_path "$DATA_PATH" \
    --output_dir "$out_dir" \
    --seed "$seed" \
    --batch_size "$batch" \
    --lr "$lr" \
    --temperature "$temp" \
    --weight_decay "$WEIGHT_DECAY" \
    --hidden_dim "$hidden" \
    --embedding_dim "$embed" \
    --classifier_dropout "$clf_drop" \
    --contrastive_epochs "$CONTRASTIVE_EPOCHS" \
    --classifier_epochs "$CLASSIFIER_EPOCHS" \
    --val_ratio "$VAL_RATIO" \
    --patience "$PATIENCE" \
    --min_delta "$MIN_DELTA" \
    --split_col is_train \
    --split_test_values test iid_holdout

elif [[ "$MODEL" == "tf" ]]; then
  total=$(( ${#SEED_LIST[@]} * ${#LR_LIST[@]} * ${#TEMP_LIST[@]} * ${#BATCH_LIST[@]} * ${#D_MODEL_LIST[@]} * ${#NHEAD_LIST[@]} * ${#NUM_LAYERS_LIST[@]} * ${#FF_LIST[@]} * ${#TF_DROPOUT_LIST[@]} * ${#TF_EMBED_LIST[@]} ))
  if (( TASK_ID < 0 || TASK_ID >= total )); then
    echo "TASK_ID=$TASK_ID out of range for MODEL=tf (0..$((total - 1)))." >&2
    exit 2
  fi

  idx="$TASK_ID"
  read -r seed idx <<< "$(idx_mod_pick "$idx" "${SEED_LIST[@]}")"
  read -r lr idx <<< "$(idx_mod_pick "$idx" "${LR_LIST[@]}")"
  read -r temp idx <<< "$(idx_mod_pick "$idx" "${TEMP_LIST[@]}")"
  read -r batch idx <<< "$(idx_mod_pick "$idx" "${BATCH_LIST[@]}")"
  read -r d_model idx <<< "$(idx_mod_pick "$idx" "${D_MODEL_LIST[@]}")"
  read -r nhead idx <<< "$(idx_mod_pick "$idx" "${NHEAD_LIST[@]}")"
  read -r layers idx <<< "$(idx_mod_pick "$idx" "${NUM_LAYERS_LIST[@]}")"
  read -r ff idx <<< "$(idx_mod_pick "$idx" "${FF_LIST[@]}")"
  read -r tf_drop idx <<< "$(idx_mod_pick "$idx" "${TF_DROPOUT_LIST[@]}")"
  read -r embed idx <<< "$(idx_mod_pick "$idx" "${TF_EMBED_LIST[@]}")"

  trial="tf_s${seed}_lr${lr}_t${temp}_b${batch}_dm${d_model}_h${nhead}_L${layers}_ff${ff}_d${tf_drop}_e${embed}"
  out_dir="$OUT_ROOT/tune/$trial"
  mkdir -p "$out_dir"

  python src/train_contrastive_tf.py \
    --data_path "$DATA_PATH" \
    --output_dir "$out_dir" \
    --seed "$seed" \
    --batch_size "$batch" \
    --lr "$lr" \
    --temperature "$temp" \
    --weight_decay "$WEIGHT_DECAY" \
    --d_model "$d_model" \
    --nhead "$nhead" \
    --num_layers "$layers" \
    --dim_feedforward "$ff" \
    --dropout "$tf_drop" \
    --embedding_dim "$embed" \
    --contrastive_epochs "$CONTRASTIVE_EPOCHS" \
    --classifier_epochs "$CLASSIFIER_EPOCHS" \
    --val_ratio "$VAL_RATIO" \
    --patience "$PATIENCE" \
    --min_delta "$MIN_DELTA" \
    --no_save_attention
else
  echo "Unsupported MODEL='$MODEL'. Use MODEL=mlp or MODEL=tf." >&2
  exit 1
fi

echo "Tuning task completed for MODEL=$MODEL, TASK_ID=$TASK_ID"
