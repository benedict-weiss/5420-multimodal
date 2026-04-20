#!/usr/bin/env bash
#SBATCH --job-name=mlp-best-train-eval
#SBATCH --partition=gpu_devel
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_5000_ada:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

# -----------------------------------------------------------------------------
# Train best MLP config and run evaluate.py.
#
# Modes:
#   MODE=single     -> one confirmation run with fixed settings
#   MODE=micro_tune -> small targeted sweep around best settings
#   MODE=seed_test  -> repeat single-run setup across multiple seeds
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
RUN_ROOT="${RUN_ROOT:-$OUT_ROOT/final_mlp_overfitfix}"
FIG_ROOT="${FIG_ROOT:-$REPO_ROOT/results/figures/final_mlp_overfitfix}"

# Execution mode: single or micro_tune.
MODE="${MODE:-single}"

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

# Overfitting-aware defaults for Stage A stopping.
PATIENCE="${PATIENCE:-20}"
MIN_DELTA="${MIN_DELTA:-1e-4}"
VAL_RATIO="${VAL_RATIO:-0.1}"

# Split-aware evaluation policy:
#   split_test_values are treated as final held-out test.
#   validation defaults to a stratified split from training data.
#   predefined validation values are opt-in only.
SPLIT_COL="${SPLIT_COL:-is_train}"
SPLIT_TEST_VALUES="${SPLIT_TEST_VALUES:-iid_holdout}"
USE_PREDEFINED_VAL_SPLIT="${USE_PREDEFINED_VAL_SPLIT:-0}"
SPLIT_VAL_VALUES="${SPLIT_VAL_VALUES:-}"

# Optional Stage A checkpoint criterion.
STAGE_A_SELECT_METRIC="${STAGE_A_SELECT_METRIC:-val_loss}"
STAGE_A_INPUT_DROPOUT="${STAGE_A_INPUT_DROPOUT:-0.15}"
STAGE_A_PROBE_EVERY="${STAGE_A_PROBE_EVERY:-3}"
STAGE_A_PROBE_EPOCHS="${STAGE_A_PROBE_EPOCHS:-5}"
STAGE_A_PROBE_LR="${STAGE_A_PROBE_LR:-1e-3}"
STAGE_A_PROBE_MIN_DELTA="${STAGE_A_PROBE_MIN_DELTA:-2e-3}"
STAGE_A_PROBE_START_EPOCH="${STAGE_A_PROBE_START_EPOCH:-10}"

# Micro-tune knobs (2 x 2 x 2 = 8 runs by default).
read -r -a MICRO_LR_LIST <<< "${MICRO_LR_LIST:-3e-4 1e-3}"
read -r -a MICRO_WD_LIST <<< "${MICRO_WD_LIST:-1e-5 3e-5}"
read -r -a MICRO_PATIENCE_LIST <<< "${MICRO_PATIENCE_LIST:-8 12}"

# Seed-testing list (uses same config/metrics as MODE=single).
read -r -a SEED_LIST <<< "${SEED_LIST:-13 42 77}"

mkdir -p "$REPO_ROOT/logs" "$RUN_ROOT" "$FIG_ROOT"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ -n "$ENV_ACTIVATE" ]]; then
  eval "$ENV_ACTIVATE"
fi

run_one() {
  local run_label="$1"
  local run_seed="$2"
  local run_lr="$3"
  local run_wd="$4"
  local run_patience="$5"

  local out_dir="$RUN_ROOT/$run_label"
  local fig_dir="$FIG_ROOT/$run_label"
  mkdir -p "$out_dir" "$fig_dir"

  echo "[train] label=$run_label seed=$run_seed lr=$run_lr wd=$run_wd patience=$run_patience"
  local extra_split_args=()
  if [[ "$USE_PREDEFINED_VAL_SPLIT" == "1" && -n "$SPLIT_VAL_VALUES" ]]; then
    extra_split_args+=(--use_predefined_val_split)
    extra_split_args+=(--split_val_values "$SPLIT_VAL_VALUES")
  fi

  python -m src.train_contrastive_mlp \
    --data_path "$DATA_PATH" \
    --output_dir "$out_dir" \
    --seed "$run_seed" \
    --batch_size "$BATCH_SIZE" \
    --lr "$run_lr" \
    --classifier_lr "$CLASSIFIER_LR" \
    --temperature "$TEMPERATURE" \
    --weight_decay "$run_wd" \
    --hidden_dim "$HIDDEN_DIM" \
    --embedding_dim "$EMBED_DIM" \
    --classifier_hidden_dim "$CLASSIFIER_HIDDEN_DIM" \
    --classifier_dropout "$CLASSIFIER_DROPOUT" \
    --contrastive_epochs "$CONTRASTIVE_EPOCHS" \
    --classifier_epochs "$CLASSIFIER_EPOCHS" \
    --val_ratio "$VAL_RATIO" \
    --patience "$run_patience" \
    --min_delta "$MIN_DELTA" \
    --stage_a_select_metric "$STAGE_A_SELECT_METRIC" \
    --stage_a_input_dropout "$STAGE_A_INPUT_DROPOUT" \
    --stage_a_probe_every "$STAGE_A_PROBE_EVERY" \
    --stage_a_probe_epochs "$STAGE_A_PROBE_EPOCHS" \
    --stage_a_probe_lr "$STAGE_A_PROBE_LR" \
    --stage_a_probe_min_delta "$STAGE_A_PROBE_MIN_DELTA" \
    --stage_a_probe_start_epoch "$STAGE_A_PROBE_START_EPOCH" \
    --split_col "$SPLIT_COL" \
    --split_test_values "$SPLIT_TEST_VALUES" \
    "${extra_split_args[@]}"

  local latest_run
  latest_run=$(find "$out_dir" -maxdepth 1 -type d -name "contrastive_mlp_seed${run_seed}_*" | sort | tail -n 1)
  if [[ -z "${latest_run:-}" ]]; then
    echo "[error] Could not find newly generated run directory under $out_dir"
    exit 1
  fi

  local eval_out="$fig_dir/$(basename "$latest_run")"
  mkdir -p "$eval_out"

  echo "[eval] Running evaluate.py for $latest_run"
  python -m src.evaluate \
    --checkpoint_dir "$out_dir" \
    --mlp_dir "$latest_run" \
    --output_dir "$eval_out"

  echo "[done] Run dir: $latest_run"
  echo "[done] Figures: $eval_out"
}

if [[ "$MODE" == "single" ]]; then
  run_one "single_seed${SEED}" "$SEED" "$LR" "$WEIGHT_DECAY" "$PATIENCE"
elif [[ "$MODE" == "micro_tune" ]]; then
  total=$(( ${#MICRO_LR_LIST[@]} * ${#MICRO_WD_LIST[@]} * ${#MICRO_PATIENCE_LIST[@]} ))
  echo "[micro_tune] Running $total configurations"
  for lr_i in "${MICRO_LR_LIST[@]}"; do
    for wd_i in "${MICRO_WD_LIST[@]}"; do
      for pat_i in "${MICRO_PATIENCE_LIST[@]}"; do
        label="micro_s${SEED}_lr${lr_i}_wd${wd_i}_pat${pat_i}"
        run_one "$label" "$SEED" "$lr_i" "$wd_i" "$pat_i"
      done
    done
  done
elif [[ "$MODE" == "seed_test" ]]; then
  echo "[seed_test] Running ${#SEED_LIST[@]} seeds with single-run config"
  for seed_i in "${SEED_LIST[@]}"; do
    label="seedtest_s${seed_i}"
    run_one "$label" "$seed_i" "$LR" "$WEIGHT_DECAY" "$PATIENCE"
  done
else
  echo "[error] Unsupported MODE='$MODE'. Use MODE=single, MODE=micro_tune, or MODE=seed_test"
  exit 1
fi
