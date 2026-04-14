#!/usr/bin/env bash
#SBATCH --job-name=dualenc-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

# -----------------------------
# User-configurable variables
# -----------------------------
REPO_ROOT="${REPO_ROOT:-$PWD}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data}"
MODEL="${MODEL:-mlp}"                    # mlp | tf
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/checkpoints}"

# Optional environment activation.
# Example: ENV_ACTIVATE="source ~/miniconda3/etc/profile.d/conda.sh && conda activate multimodal"
ENV_ACTIVATE="${ENV_ACTIVATE:-}"

mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT"

if [[ -n "$ENV_ACTIVATE" ]]; then
  eval "$ENV_ACTIVATE"
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_OUT="$OUT_ROOT/smoke_${MODEL}_seed${SEED}_${RUN_TS}"
mkdir -p "$RUN_OUT"

COMMON_ARGS=(
  --data_path "$DATA_PATH"
  --seed "$SEED"
  --batch_size "$BATCH_SIZE"
  --contrastive_epochs 2
  --classifier_epochs 2
  --patience 1
  --min_delta 1e-4
  --val_ratio 0.1
  --output_dir "$RUN_OUT"
)

if [[ "$MODEL" == "mlp" ]]; then
  # Include iid_holdout in test set by default to avoid accidental leakage into train.
  python src/train_contrastive_mlp.py \
    "${COMMON_ARGS[@]}" \
    --split_col is_train \
    --split_test_values test iid_holdout
elif [[ "$MODEL" == "tf" ]]; then
  # max_cells keeps smoke tests fast on full pipelines.
  python src/train_contrastive_tf.py \
    "${COMMON_ARGS[@]}" \
    --max_cells 8000 \
    --no_save_attention
else
  echo "Unsupported MODEL='$MODEL'. Use MODEL=mlp or MODEL=tf." >&2
  exit 1
fi

echo "Smoke test finished. Outputs in: $RUN_OUT"
