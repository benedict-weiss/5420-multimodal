#!/usr/bin/env bash
#SBATCH --job-name=mlp-best-relaxed
#SBATCH --partition=gpu_devel
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_5000_ada:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

# -----------------------------------------------------------------------------
# Train best mlp config with very lax Stage A early stopping, then run evaluate.py.
#
# Submit:
#   sbatch scripts/slurm/train_mlp_best_relaxed_eval.sh
#
# Optional overrides:
#   DATA_PATH=/path/to/data sbatch scripts/slurm/train_mlp_best_relaxed_eval.sh
# -----------------------------------------------------------------------------

REPO_ROOT="${REPO_ROOT:-$PWD}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/checkpoints}"
RUN_ROOT="${RUN_ROOT:-$OUT_ROOT/final_mlp_relaxed}"
FIG_ROOT="${FIG_ROOT:-$REPO_ROOT/results/figures/final_mlp_relaxed}"

# Optional environment activation.
ENV_ACTIVATE="${ENV_ACTIVATE:-}"

# Best config from secondary sweep.
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

# Training schedule requested.
CONTRASTIVE_EPOCHS="${CONTRASTIVE_EPOCHS:-150}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-30}"

# Effectively disable Stage A early stopping.
PATIENCE="${PATIENCE:-1000}"
MIN_DELTA="${MIN_DELTA:-0.0}"
VAL_RATIO="${VAL_RATIO:-0.1}"

# Split-aware evaluation policy:
#   train split values -> training
#   test split values  -> validation/checkpointing
#   iid_holdout        -> final held-out test
SPLIT_COL="${SPLIT_COL:-is_train}"
SPLIT_VAL_VALUES="${SPLIT_VAL_VALUES:-test}"
SPLIT_TEST_VALUES="${SPLIT_TEST_VALUES:-iid_holdout}"

# Optional Stage A checkpoint criterion.
STAGE_A_SELECT_METRIC="${STAGE_A_SELECT_METRIC:-probe_accuracy}"
STAGE_A_PROBE_EVERY="${STAGE_A_PROBE_EVERY:-5}"
STAGE_A_PROBE_EPOCHS="${STAGE_A_PROBE_EPOCHS:-3}"
STAGE_A_PROBE_LR="${STAGE_A_PROBE_LR:-1e-3}"
STAGE_A_PROBE_MIN_DELTA="${STAGE_A_PROBE_MIN_DELTA:-1e-4}"

mkdir -p "$REPO_ROOT/logs" "$RUN_ROOT" "$FIG_ROOT"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ -n "$ENV_ACTIVATE" ]]; then
  eval "$ENV_ACTIVATE"
fi

echo "[train] Running best mlp config with relaxed early stopping"
python -m src.train_contrastive_mlp \
  --data_path "$DATA_PATH" \
  --output_dir "$RUN_ROOT" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --classifier_lr "$CLASSIFIER_LR" \
  --temperature "$TEMPERATURE" \
  --weight_decay "$WEIGHT_DECAY" \
  --hidden_dim "$HIDDEN_DIM" \
  --embedding_dim "$EMBED_DIM" \
  --classifier_hidden_dim "$CLASSIFIER_HIDDEN_DIM" \
  --classifier_dropout "$CLASSIFIER_DROPOUT" \
  --contrastive_epochs "$CONTRASTIVE_EPOCHS" \
  --classifier_epochs "$CLASSIFIER_EPOCHS" \
  --val_ratio "$VAL_RATIO" \
  --patience "$PATIENCE" \
  --min_delta "$MIN_DELTA" \
  --stage_a_select_metric "$STAGE_A_SELECT_METRIC" \
  --stage_a_probe_every "$STAGE_A_PROBE_EVERY" \
  --stage_a_probe_epochs "$STAGE_A_PROBE_EPOCHS" \
  --stage_a_probe_lr "$STAGE_A_PROBE_LR" \
  --stage_a_probe_min_delta "$STAGE_A_PROBE_MIN_DELTA" \
  --split_col "$SPLIT_COL" \
  --split_val_values "$SPLIT_VAL_VALUES" \
  --split_test_values "$SPLIT_TEST_VALUES"

# Resolve latest run directory generated above.
latest_run=$(find "$RUN_ROOT" -maxdepth 1 -type d -name "contrastive_mlp_seed${SEED}_*" | sort | tail -n 1)
if [[ -z "${latest_run:-}" ]]; then
  echo "[error] Could not find newly generated run directory under $RUN_ROOT"
  exit 1
fi

eval_out="$FIG_ROOT/$(basename "$latest_run")"
mkdir -p "$eval_out"

echo "[eval] Running evaluate.py for $latest_run"
python -m src.evaluate \
  --checkpoint_dir "$RUN_ROOT" \
  --mlp_dir "$latest_run" \
  --output_dir "$eval_out"

echo "[done] Run dir: $latest_run"
echo "[done] Figures: $eval_out"
