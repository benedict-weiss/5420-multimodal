#!/usr/bin/env bash
#SBATCH --job-name=dualenc-secondary
#SBATCH --partition=gpu_devel
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_5000_ada:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

# -----------------------------------------------------------------------------
# Secondary MLP tuning sweep around current best region.
#
# Submit:
#   sbatch scripts/slurm/tune_dual_encoder_secondary.sh
#
# Optional overrides:
#   DATA_PATH=/path/to/data OUT_ROOT=/path/to/out sbatch ...
# -----------------------------------------------------------------------------

REPO_ROOT="${REPO_ROOT:-$PWD}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/checkpoints}"

# Optional environment activation.
ENV_ACTIVATE="${ENV_ACTIVATE:-}"

# Global training controls.
VAL_RATIO="${VAL_RATIO:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
PATIENCE="${PATIENCE:-20}"
MIN_DELTA="${MIN_DELTA:-1e-5}"
CONTRASTIVE_EPOCHS="${CONTRASTIVE_EPOCHS:-200}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-50}"
CLASSIFIER_HIDDEN_DIM="${CLASSIFIER_HIDDEN_DIM:-64}"

# Focused search-space defaults (override as space-separated env vars).
read -r -a SEED_LIST <<< "${SEED_LIST:-13 42 77}"
read -r -a LR_LIST <<< "${LR_LIST:-1e-3}"
read -r -a CLASSIFIER_LR_LIST <<< "${CLASSIFIER_LR_LIST:-3e-4 1e-3}"
read -r -a TEMP_LIST <<< "${TEMP_LIST:-0.05 0.07 0.09}"
read -r -a BATCH_LIST <<< "${BATCH_LIST:-128}"
read -r -a HIDDEN_LIST <<< "${HIDDEN_LIST:-512}"
read -r -a EMBED_LIST <<< "${EMBED_LIST:-128 256}"
read -r -a CLASSIFIER_DROPOUT_LIST <<< "${CLASSIFIER_DROPOUT_LIST:-0.1 0.2 0.3}"

mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT"

# Ensure Python can import the local src package on HPC nodes.
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ -n "$ENV_ACTIVATE" ]]; then
  eval "$ENV_ACTIVATE"
fi

total=$(( ${#SEED_LIST[@]} * ${#LR_LIST[@]} * ${#CLASSIFIER_LR_LIST[@]} * ${#TEMP_LIST[@]} * ${#BATCH_LIST[@]} * ${#HIDDEN_LIST[@]} * ${#EMBED_LIST[@]} * ${#CLASSIFIER_DROPOUT_LIST[@]} ))
echo "[tune-secondary] Starting focused MLP sweep with $total configurations"

for seed in "${SEED_LIST[@]}"; do
  for lr in "${LR_LIST[@]}"; do
    for clf_lr in "${CLASSIFIER_LR_LIST[@]}"; do
      for temp in "${TEMP_LIST[@]}"; do
        for batch in "${BATCH_LIST[@]}"; do
          for hidden in "${HIDDEN_LIST[@]}"; do
            for embed in "${EMBED_LIST[@]}"; do
              for clf_drop in "${CLASSIFIER_DROPOUT_LIST[@]}"; do
                trial="mlp2_s${seed}_lr${lr}_clr${clf_lr}_t${temp}_b${batch}_h${hidden}_e${embed}_cd${clf_drop}"
                out_dir="$OUT_ROOT/tune/$trial"
                mkdir -p "$out_dir"

                # Skip reruns when this trial already completed once.
                if find "$out_dir" -type f -name metrics.json -print -quit | grep -q .; then
                  echo "[tune-secondary] Skip existing: $trial"
                  continue
                fi

                echo "[tune-secondary] Running: $trial"
                python -m src.train_contrastive_mlp \
                  --data_path "$DATA_PATH" \
                  --output_dir "$out_dir" \
                  --seed "$seed" \
                  --batch_size "$batch" \
                  --lr "$lr" \
                  --classifier_lr "$clf_lr" \
                  --temperature "$temp" \
                  --weight_decay "$WEIGHT_DECAY" \
                  --hidden_dim "$hidden" \
                  --embedding_dim "$embed" \
                  --classifier_hidden_dim "$CLASSIFIER_HIDDEN_DIM" \
                  --classifier_dropout "$clf_drop" \
                  --contrastive_epochs "$CONTRASTIVE_EPOCHS" \
                  --classifier_epochs "$CLASSIFIER_EPOCHS" \
                  --val_ratio "$VAL_RATIO" \
                  --patience "$PATIENCE" \
                  --min_delta "$MIN_DELTA" \
                  --split_col is_train \
                  --split_test_values test iid_holdout
              done
            done
          done
        done
      done
    done
  done
done

echo "[tune-secondary] Focused MLP sweep complete"
