#!/usr/bin/env bash
#SBATCH --job-name=mlp-debug-split
#SBATCH --partition=gpu_devel
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_5000_ada:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

# -----------------------------------------------------------------------------
# Debug MLP split quality and metric behavior without touching evaluate.py.
#
# What it does:
#   1) Prints train/test/validation split coverage before training.
#   2) Runs a short MLP training job with the current upstream split policy.
#   3) Summarizes Stage A val-loss behavior and final metrics from metrics.json.
#
# Typical submit:
#   sbatch scripts/slurm/debug_mlp_split_quality.sh
#
# Common overrides:
#   DATA_PATH=/path/to/data
#   MODE=debug
#   USE_PREDEFINED_VAL_SPLIT=1
#   SPLIT_VAL_VALUES=<value(s)>
#   SPLIT_TEST_VALUES=<value(s)>
#   TEST_DONORS=<donor ids>
# -----------------------------------------------------------------------------

REPO_ROOT="${REPO_ROOT:-$PWD}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/checkpoints}"
RUN_ROOT="${RUN_ROOT:-$OUT_ROOT/debug_mlp_split_quality}"
FIG_ROOT="${FIG_ROOT:-$REPO_ROOT/results/figures/debug_mlp_split_quality}"

ENV_ACTIVATE="${ENV_ACTIVATE:-}"
SEED="${SEED:-42}"
MODE="${MODE:-debug}"

# Short run so you can get feedback quickly.
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
CLASSIFIER_LR="${CLASSIFIER_LR:-1e-3}"
TEMPERATURE="${TEMPERATURE:-0.07}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
EMBED_DIM="${EMBED_DIM:-128}"
CLASSIFIER_HIDDEN_DIM="${CLASSIFIER_HIDDEN_DIM:-64}"
CLASSIFIER_DROPOUT="${CLASSIFIER_DROPOUT:-0.3}"

CONTRASTIVE_EPOCHS="${CONTRASTIVE_EPOCHS:-40}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-15}"
PATIENCE="${PATIENCE:-10}"
MIN_DELTA="${MIN_DELTA:-1e-4}"
VAL_RATIO="${VAL_RATIO:-0.1}"

SPLIT_COL="${SPLIT_COL:-is_train}"
SPLIT_TEST_VALUES="${SPLIT_TEST_VALUES:-iid_holdout}"
USE_PREDEFINED_VAL_SPLIT="${USE_PREDEFINED_VAL_SPLIT:-0}"
SPLIT_VAL_VALUES="${SPLIT_VAL_VALUES:-}"
TEST_DONORS="${TEST_DONORS:-}"

read -r -a TEST_DONOR_LIST <<< "$TEST_DONORS"
read -r -a SPLIT_TEST_VALUE_LIST <<< "$SPLIT_TEST_VALUES"
read -r -a SPLIT_VAL_VALUE_LIST <<< "$SPLIT_VAL_VALUES"

mkdir -p "$REPO_ROOT/logs" "$RUN_ROOT" "$FIG_ROOT"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ -n "$ENV_ACTIVATE" ]]; then
  eval "$ENV_ACTIVATE"
fi

echo "[debug] repo_root=$REPO_ROOT"
echo "[debug] data_path=$DATA_PATH"
echo "[debug] mode=$MODE seed=$SEED"
echo "[debug] split_col=$SPLIT_COL split_test_values=$SPLIT_TEST_VALUES use_predefined_val_split=$USE_PREDEFINED_VAL_SPLIT split_val_values=$SPLIT_VAL_VALUES test_donors=$TEST_DONORS"

export DATA_PATH
export SPLIT_COL
export SPLIT_TEST_VALUES
export SPLIT_VAL_VALUES
export USE_PREDEFINED_VAL_SPLIT
export TEST_DONORS
export SEED
export VAL_RATIO
export LABEL_COL="${LABEL_COL:-cell_type}"
export DONOR_COL="${DONOR_COL:-DonorNumber}"
export TEST_SIZE="${TEST_SIZE:-0.2}"

python - <<'PY'
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from src.preprocessing import get_labels, load_data, split_by_donor, split_modalities

def parse_bool_like(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    return None

def split_counts(labels):
    uniq, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, counts)}

def build_train_val_indices(labels, val_ratio, seed):
    idx = np.arange(labels.shape[0])
    if val_ratio <= 0 or idx.size < 2:
        return idx, np.array([], dtype=int)

    unique_labels, counts = np.unique(labels, return_counts=True)
    label_to_count = {int(label): int(count) for label, count in zip(unique_labels, counts)}
    eligible_labels = unique_labels[counts >= 2]

    if eligible_labels.size == 0:
        try:
            train_idx, val_idx = train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=labels)
        except ValueError:
            train_idx, val_idx = train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=None)
        return np.asarray(train_idx), np.asarray(val_idx)

    rng = np.random.default_rng(seed)
    class_indices = {}
    for label in unique_labels:
        label_idx = idx[labels == label].copy()
        rng.shuffle(label_idx)
        class_indices[int(label)] = label_idx

    target_val_size = int(round(labels.shape[0] * val_ratio))
    target_val_size = max(target_val_size, int(eligible_labels.size))
    target_val_size = min(target_val_size, labels.shape[0] - 1)

    val_counts = {int(label): 1 for label in eligible_labels}
    remaining = target_val_size - int(eligible_labels.size)

    if remaining > 0:
        eligible_counts = np.array([label_to_count[int(label)] for label in eligible_labels], dtype=float)
        capacities = eligible_counts - 1.0
        total_capacity = float(capacities.sum())
        if total_capacity > 0:
            desired = capacities / total_capacity * remaining
            extra_counts = np.floor(desired).astype(int)
            leftover = remaining - int(extra_counts.sum())
            remainders = desired - extra_counts
            if leftover > 0:
                for pos in np.argsort(-remainders):
                    if leftover <= 0:
                        break
                    if extra_counts[pos] < int(capacities[pos]):
                        extra_counts[pos] += 1
                        leftover -= 1
            if leftover > 0:
                for pos in np.argsort(-capacities):
                    if leftover <= 0:
                        break
                    available = int(capacities[pos]) - extra_counts[pos]
                    if available <= 0:
                        continue
                    take = min(available, leftover)
                    extra_counts[pos] += take
                    leftover -= take
            for pos, label in enumerate(eligible_labels):
                val_counts[int(label)] += int(extra_counts[pos])

    val_idx_parts = []
    train_idx_parts = []
    for label in unique_labels:
        label_key = int(label)
        label_idx = class_indices[label_key]
        n_val_label = min(int(val_counts.get(label_key, 0)), int(label_idx.size))
        val_idx_parts.append(label_idx[:n_val_label])
        train_idx_parts.append(label_idx[n_val_label:])

    val_idx = np.concatenate(val_idx_parts) if val_idx_parts else np.array([], dtype=int)
    train_idx = np.concatenate(train_idx_parts) if train_idx_parts else np.array([], dtype=int)

    if val_idx.size == 0 or train_idx.size == 0:
        try:
            train_idx, val_idx = train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=labels)
        except ValueError:
            train_idx, val_idx = train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=None)
    return np.asarray(train_idx), np.asarray(val_idx)

def resolve_data_file(data_path: str) -> str:
    p = Path(data_path)
    if p.is_file():
        return str(p)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"data_path does not exist: {data_path}")

    candidates = sorted(list(p.glob("*.h5ad")) + list(p.glob("*.h5ad.gz")))
    if not candidates:
        raise FileNotFoundError(f"No .h5ad or .h5ad.gz files found under {data_path}")
    if len(candidates) > 1:
        print(f"[warn] Multiple dataset files found, using: {candidates[0]}")
    return str(candidates[0])


data_path = resolve_data_file(os.environ["DATA_PATH"])
split_col = os.environ.get("SPLIT_COL", "is_train")
split_test_values = os.environ.get("SPLIT_TEST_VALUES", "iid_holdout").split()
split_val_values = os.environ.get("SPLIT_VAL_VALUES", "").split()
use_predefined_val_split = os.environ.get("USE_PREDEFINED_VAL_SPLIT", "0") == "1"
test_donors = os.environ.get("TEST_DONORS", "").split()
seed = int(os.environ.get("SEED", "42"))
val_ratio = float(os.environ.get("VAL_RATIO", "0.1"))

adata = load_data(data_path)
labels_all, label_mapping = get_labels(adata, label_col=os.environ.get("LABEL_COL", "cell_type"))

if test_donors:
    train_global_idx, test_global_idx = split_by_donor(adata, test_donors=test_donors, donor_col=os.environ.get("DONOR_COL", "DonorNumber"))
    val_global_idx = np.array([], dtype=int)
elif split_col in adata.obs.columns:
    def _mask_from_values(series, raw_values):
        parsed_column_values = [parse_bool_like(v) for v in series.values]
        column_is_bool_like = all(v is not None for v in parsed_column_values)
        parsed_values = [parse_bool_like(v) for v in raw_values]
        values_are_bool_like = all(v is not None for v in parsed_values)
        if column_is_bool_like and values_are_bool_like:
            target_bool = set(parsed_values)
            return np.array([v in target_bool for v in parsed_column_values], dtype=bool)
        if column_is_bool_like and set(str(v).strip().lower() for v in raw_values) == {"test"}:
            return np.array([v is False for v in parsed_column_values], dtype=bool)
        split_values_str = set(str(v) for v in raw_values)
        split_arr = series.astype(str).values
        return np.array([v in split_values_str for v in split_arr], dtype=bool)

    split_series = adata.obs[split_col]
    test_mask = _mask_from_values(split_series, split_test_values)
    val_mask = _mask_from_values(split_series, split_val_values) if (use_predefined_val_split and split_val_values) else np.zeros_like(test_mask)
    train_mask = ~(test_mask | val_mask)
    test_global_idx = np.where(test_mask)[0]
    train_global_idx = np.where(train_mask)[0]
    val_global_idx = np.where(val_mask)[0] if use_predefined_val_split and split_val_values else np.array([], dtype=int)
else:
    train_global_idx, test_global_idx = train_test_split(np.arange(adata.shape[0]), test_size=float(os.environ.get("TEST_SIZE", "0.2")), random_state=seed, stratify=labels_all)
    train_global_idx = np.asarray(train_global_idx)
    test_global_idx = np.asarray(test_global_idx)
    val_global_idx = np.array([], dtype=int)

print("[split] total_labels=", split_counts(labels_all))
print("[split] train_labels=", split_counts(labels_all[train_global_idx]))
print("[split] test_labels=", split_counts(labels_all[test_global_idx]))
if val_global_idx.size > 0:
    print("[split] predefined_val_labels=", split_counts(labels_all[val_global_idx]))
else:
    local_train_idx, local_val_idx = build_train_val_indices(labels_all[train_global_idx], val_ratio=val_ratio, seed=seed)
    print("[split] train_to_val_labels=", split_counts(labels_all[train_global_idx][local_val_idx]))
    print("[split] train_to_val_size=", int(local_val_idx.size))

print("[split] label_mapping_size=", len(label_mapping))
PY

echo "[debug] starting short training run"
train_args=(
    --data_path "$DATA_PATH"
    --output_dir "$RUN_ROOT"
    --seed "$SEED"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --classifier_lr "$CLASSIFIER_LR"
    --temperature "$TEMPERATURE"
    --weight_decay "$WEIGHT_DECAY"
    --hidden_dim "$HIDDEN_DIM"
    --embedding_dim "$EMBED_DIM"
    --classifier_hidden_dim "$CLASSIFIER_HIDDEN_DIM"
    --classifier_dropout "$CLASSIFIER_DROPOUT"
    --contrastive_epochs "$CONTRASTIVE_EPOCHS"
    --classifier_epochs "$CLASSIFIER_EPOCHS"
    --val_ratio "$VAL_RATIO"
    --patience "$PATIENCE"
    --min_delta "$MIN_DELTA"
    --split_col "$SPLIT_COL"
)

if [[ ${#SPLIT_TEST_VALUE_LIST[@]} -gt 0 ]]; then
    train_args+=(--split_test_values "${SPLIT_TEST_VALUE_LIST[@]}")
fi

if [[ "$USE_PREDEFINED_VAL_SPLIT" == "1" && ${#SPLIT_VAL_VALUE_LIST[@]} -gt 0 ]]; then
    train_args+=(--use_predefined_val_split)
    train_args+=(--split_val_values "${SPLIT_VAL_VALUE_LIST[@]}")
fi

if [[ ${#TEST_DONOR_LIST[@]} -gt 0 ]]; then
    train_args+=(--test_donors "${TEST_DONOR_LIST[@]}")
fi

python -m src.train_contrastive_mlp \
    "${train_args[@]}"

latest_run=$(find "$RUN_ROOT" -maxdepth 1 -type d -name "contrastive_mlp_seed${SEED}_*" | sort | tail -n 1)
if [[ -z "${latest_run:-}" ]]; then
  echo "[error] could not find a run directory under $RUN_ROOT"
  exit 1
fi

echo "[debug] summarizing metrics from $latest_run"
export LATEST_RUN="$latest_run"
export RUN_ROOT="$RUN_ROOT"
python - <<'PY'
import json
import math
import os
from pathlib import Path

latest_run = Path(os.environ["LATEST_RUN"])
metrics_path = latest_run / "metrics.json"
if not metrics_path.exists():
    raise SystemExit(f"metrics.json not found: {metrics_path}")

with open(metrics_path, encoding="utf-8") as f:
    metrics = json.load(f)

stage_a = metrics.get("stage_a_history", [])
stage_b = metrics.get("stage_b_history", [])
final_acc = metrics.get("final_accuracy")
final_auroc = metrics.get("final_macro_auroc")

def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))

print("[metrics] final_accuracy=", final_acc)
print("[metrics] final_macro_auroc=", final_auroc)
print("[metrics] stage_a_epochs=", len(stage_a))
print("[metrics] stage_b_epochs=", len(stage_b))

if stage_a:
    train_losses = [float(row["train_loss"]) for row in stage_a if finite(row.get("train_loss"))]
    val_losses = [float(row["val_loss"]) for row in stage_a if finite(row.get("val_loss"))]
    if train_losses and val_losses:
        print("[metrics] stage_a_train_loss_first_last=", train_losses[0], train_losses[-1])
        print("[metrics] stage_a_val_loss_first_last=", val_losses[0], val_losses[-1])
        print("[metrics] stage_a_val_loss_min=", min(val_losses))
        print("[metrics] stage_a_val_loss_span=", max(val_losses) - min(val_losses))

if stage_b:
    val_accs = [float(row["val_accuracy"]) for row in stage_b if finite(row.get("val_accuracy"))]
    val_aurocs = [float(row["val_macro_auroc"]) for row in stage_b if finite(row.get("val_macro_auroc"))]
    print("[metrics] stage_b_val_accuracy_best=", max(val_accs) if val_accs else None)
    print("[metrics] stage_b_val_macro_auroc_best=", max(val_aurocs) if val_aurocs else None)
PY

echo "[done] debug run complete"