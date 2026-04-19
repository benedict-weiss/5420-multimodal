"""
train_baseline.py — RNA-only baseline training (Model 1)

Implement the following:

1. Load preprocessed data:
    - RNA PCA matrix (256-d) and labels from preprocessing.py
    - Split by donor into train/test

2. Build model:
    - MLPEncoder(input_dim=256, hidden_dim=256, output_dim=128) — no L2 normalization
    - ClassificationHead(input_dim=128, n_classes=n)

3. Training loop:
    - Optimizer: Adam(lr=1e-3, weight_decay=1e-5)
    - Loss: F.cross_entropy
    - Train for 50 epochs (or until convergence)
    - Track train/val loss and accuracy per epoch

4. Evaluation:
    - Call evaluate.py functions on test set
    - Save model checkpoint to results/

5. Command-line usage:
    - python src/train_baseline.py --data_path data/ --seed 42 --epochs 50

Hyperparameters:
    - lr: 1e-3, weight_decay: 1e-5, batch_size: 256
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from src.models.classifier import ClassificationHead
from src.models.mlp_encoder import MLPEncoder
from src.preprocessing import (
    get_labels,
    load_data,
    preprocess_rna,
    split_by_donor,
    split_modalities,
)

try:
    from src.evaluate import compute_accuracy, compute_auroc
except Exception:
    compute_accuracy = None
    compute_auroc = None


def _sanitize_json(obj: object) -> object:
    """Recursively replace non-finite floats (NaN/Inf) with None for valid JSON output."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_data_file(data_path: str) -> str:
    """Resolve either a direct file path or a directory containing one h5ad file."""
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


def clone_state_dict(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, n_classes: int
) -> dict:
    """Compute accuracy + macro AUROC with safe fallbacks.

    4. Evaluation: Call evaluate.py functions on test set
    """
    # Use evaluate.py functions when available, otherwise fall back to sklearn directly
    if callable(compute_accuracy):
        overall_acc, per_class = compute_accuracy(y_true, y_pred)
    else:
        overall_acc = float(accuracy_score(y_true, y_pred))
        per_class = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    if callable(compute_auroc):
        try:
            auroc = float(compute_auroc(y_true, y_proba, n_classes))
        except Exception:
            auroc = float("nan")
    else:
        try:
            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
            auroc = float(
                roc_auc_score(
                    y_true_bin,
                    y_proba,
                    average="macro",
                    multi_class="ovr",
                )
            )
        except Exception:
            auroc = float("nan")

    return {"accuracy": overall_acc, "macro_auroc": auroc, "per_class": per_class}


def make_rna_loader(
    rna: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(np.asarray(rna, dtype=np.float32)),
        torch.as_tensor(np.asarray(labels), dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def evaluate_epoch(
    loader: DataLoader,
    encoder: MLPEncoder,
    classifier: ClassificationHead,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    encoder.eval()
    classifier.eval()

    total_loss = 0.0
    n_batches = 0
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_proba_all: list[np.ndarray] = []

    for rna, y in loader:
        rna, y = rna.to(device), y.to(device)
        logits = classifier(encoder(rna))
        loss = F.cross_entropy(logits, y)
        proba = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)

        total_loss += loss.item()
        n_batches += 1
        y_true_all.append(y.cpu().numpy())
        y_pred_all.append(pred.cpu().numpy())
        y_proba_all.append(proba.cpu().numpy())

    if n_batches == 0:
        return float("inf"), np.array([]), np.array([]), np.array([[]])

    return (
        total_loss / n_batches,
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        np.concatenate(y_proba_all),
    )


def build_train_val_indices(
    labels: np.ndarray, val_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(labels.shape[0])
    if val_ratio <= 0:
        return idx, np.array([], dtype=int)

    try:
        train_idx, val_idx = train_test_split(
            idx, test_size=val_ratio, random_state=seed, stratify=labels
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            idx, test_size=val_ratio, random_state=seed, stratify=None
        )
    return np.asarray(train_idx), np.asarray(val_idx)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # 1. Load preprocessed data: RNA PCA matrix (256-d) and labels from preprocessing.py
    dataset_file = resolve_data_file(args.data_path)
    adata = load_data(dataset_file)

    if not adata.var_names.is_unique:
        adata.var_names_make_unique()

    labels_all, label_mapping = get_labels(adata, label_col=args.label_col)
    n_classes = len(label_mapping)

    # 1. Split by donor into train/test
    if args.test_donors:
        train_global_idx, test_global_idx = split_by_donor(
            adata, test_donors=args.test_donors, donor_col=args.donor_col
        )
    elif args.split_col in adata.obs.columns:
        def _parse_bool_like(value: object) -> bool | None:
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            text = str(value).strip().lower()
            if text in {"true", "1", "yes", "y", "t"}:
                return True
            if text in {"false", "0", "no", "n", "f"}:
                return False
            return None

        raw_split_values = list(args.split_test_values)
        split_series = adata.obs[args.split_col]
        split_arr_raw = split_series.values

        parsed_column_values = [_parse_bool_like(v) for v in split_arr_raw]
        column_is_bool_like = all(v is not None for v in parsed_column_values)
        parsed_test_values = [_parse_bool_like(v) for v in raw_split_values]
        test_values_are_bool_like = all(v is not None for v in parsed_test_values)

        if column_is_bool_like and test_values_are_bool_like:
            split_values_bool = set(parsed_test_values)
            test_mask = np.array([v in split_values_bool for v in parsed_column_values], dtype=bool)
        elif column_is_bool_like and set(str(v).strip().lower() for v in raw_split_values) == {"test"}:
            test_mask = np.array([v is False for v in parsed_column_values], dtype=bool)
        else:
            split_values_str = set(str(v) for v in raw_split_values)
            split_arr = split_series.astype(str).values
            test_mask = np.array([v in split_values_str for v in split_arr], dtype=bool)

        test_global_idx = np.where(test_mask)[0]
        train_global_idx = np.where(~test_mask)[0]
        print(
            f"Using predefined split column '{args.split_col}' with test values "
            f"{sorted(str(v) for v in raw_split_values)}"
        )
    else:
        train_global_idx, test_global_idx = train_test_split(
            np.arange(adata.shape[0]),
            test_size=args.test_size,
            random_state=args.seed,
            stratify=labels_all,
        )
        train_global_idx = np.asarray(train_global_idx)
        test_global_idx = np.asarray(test_global_idx)
        print(f"Using random stratified split with test_size={args.test_size}")

    if len(test_global_idx) == 0:
        raise ValueError("Test split is empty. Adjust --test_donors or split settings.")

    # RNA-only: discard protein modality
    rna_adata, _ = split_modalities(adata)

    # Leakage-safe RNA preprocessing: fit PCA on train only, transform test with the same model
    train_rna = rna_adata[train_global_idx].copy()
    test_rna = rna_adata[test_global_idx].copy()
    train_rna_pca, pca_model, hvg_genes = preprocess_rna(
        train_rna, n_comps=args.rna_pca_dim, return_pca_model=True
    )
    test_rna_pca = preprocess_rna(
        test_rna, n_comps=args.rna_pca_dim, pca_model=pca_model, hvg_genes=hvg_genes
    )

    train_labels = labels_all[train_global_idx]
    test_labels = labels_all[test_global_idx]

    print(f"Prepared matrices: train RNA {train_rna_pca.shape}, test RNA {test_rna_pca.shape}")

    # Carve out a validation split for checkpoint selection (track train/val loss per epoch)
    train_local_idx, val_local_idx = build_train_val_indices(
        labels=train_labels, val_ratio=args.val_ratio, seed=args.seed
    )

    if len(val_local_idx) == 0:
        warnings.warn(
            "Validation split is empty; checkpoint selection will use the training split. "
            "Consider increasing --val_ratio or training set size.",
            UserWarning,
            stacklevel=2,
        )
        val_local_idx = train_local_idx

    train_loader = make_rna_loader(
        train_rna_pca[train_local_idx], train_labels[train_local_idx],
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = make_rna_loader(
        train_rna_pca[val_local_idx], train_labels[val_local_idx],
        batch_size=args.batch_size, shuffle=False,
    )
    test_loader = make_rna_loader(
        test_rna_pca, test_labels, batch_size=args.batch_size, shuffle=False,
    )

    # 2. Build model:
    #    MLPEncoder(input_dim=256, hidden_dim=256, output_dim=128) — no L2 normalization
    #    ClassificationHead(input_dim=128, n_classes=n)
    encoder = MLPEncoder(
        input_dim=train_rna_pca.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=args.embedding_dim,
        normalize_output=False,  # raw activations; no L2 norm for baseline
        dropout=args.encoder_dropout,
    ).to(device)
    classifier = ClassificationHead(
        input_dim=args.embedding_dim,   # 128 from RNA encoder only (no protein concat)
        n_classes=n_classes,
        hidden_dim=args.classifier_hidden_dim,
        dropout=args.classifier_dropout,
    ).to(device)

    # 3. Training loop:
    #    Optimizer: Adam(lr=1e-3, weight_decay=1e-5)
    #    Loss: F.cross_entropy
    #    Train for 50 epochs (or until convergence)
    #    Track train/val loss and accuracy per epoch
    optimizer = Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = -1.0
    best_encoder_state = clone_state_dict(encoder)
    best_classifier_state = clone_state_dict(classifier)
    history: list[dict] = []

    print("\n=== Training RNA-only baseline ===")
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        classifier.train()

        total_train_loss = 0.0
        n_train_batches = 0

        for rna, y in train_loader:
            rna, y = rna.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(classifier(encoder(rna)), y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            n_train_batches += 1

        train_loss = total_train_loss / max(1, n_train_batches)

        val_loss, y_val_true, y_val_pred, y_val_proba = evaluate_epoch(
            loader=val_loader, encoder=encoder, classifier=classifier, device=device
        )
        val_metrics = compute_metrics(y_val_true, y_val_pred, y_val_proba, n_classes=n_classes)

        # Track train/val loss and accuracy per epoch
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_auroc": float(val_metrics["macro_auroc"]),
        })

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = float(val_metrics["accuracy"])
            best_encoder_state = clone_state_dict(encoder)
            best_classifier_state = clone_state_dict(classifier)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_auroc={val_metrics['macro_auroc']:.4f}"
        )

    encoder.load_state_dict(best_encoder_state)
    classifier.load_state_dict(best_classifier_state)

    # 4. Evaluation: Call evaluate.py functions on test set
    final_test_loss, y_true, y_pred, y_proba = evaluate_epoch(
        loader=test_loader, encoder=encoder, classifier=classifier, device=device
    )
    final_metrics = compute_metrics(y_true, y_pred, y_proba, n_classes=n_classes)

    # 4. Save model checkpoint to results/
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"baseline_rna_seed{args.seed}_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "best_val_acc": best_val_acc,
            "final_test_loss": float(final_test_loss),
            "final_metrics": {
                "accuracy": float(final_metrics["accuracy"]),
                "macro_auroc": float(final_metrics["macro_auroc"]),
            },
            "args": vars(args),
        },
        run_dir / "checkpoint.pt",
    )

    with open(run_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in label_mapping.items()}, f, indent=2)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            _sanitize_json({
                "final_test_loss": float(final_test_loss),
                "final_accuracy": float(final_metrics["accuracy"]),
                "final_macro_auroc": float(final_metrics["macro_auroc"]),
                "history": history,
            }),
            f,
            indent=2,
        )

    # Persist PCA metadata for strict feature alignment on reuse
    with open(run_dir / "hvg_genes.json", "w", encoding="utf-8") as f:
        json.dump(hvg_genes, f)
    torch.save(pca_model, run_dir / "rna_pca_model.pt")

    print("\n=== Training complete ===")
    print(f"Run directory: {run_dir}")
    print(f"Final test loss:     {final_test_loss:.4f}")
    print(f"Final test accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final macro AUROC:   {final_metrics['macro_auroc']:.4f}")


# 5. Command-line usage:
#    python src/train_baseline.py --data_path data/ --seed 42 --epochs 50
#
# Hyperparameters:
#    lr: 1e-3, weight_decay: 1e-5, batch_size: 256
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RNA-only MLP baseline (Model 1)")

    parser.add_argument("--data_path", type=str, default="data", help="Dataset file or directory")
    parser.add_argument("--output_dir", type=str, default="results/checkpoints")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available")

    parser.add_argument("--label_col", type=str, default="cell_type")
    parser.add_argument("--donor_col", type=str, default="DonorNumber")
    parser.add_argument(
        "--test_donors",
        type=str,
        nargs="*",
        default=None,
        help="Donor IDs to hold out as test set (e.g., donor8 donor9)",
    )
    parser.add_argument("--split_col", type=str, default="is_train")
    parser.add_argument(
        "--split_test_values",
        type=str,
        nargs="*",
        default=["test"],
        help="Values in split_col treated as test when --test_donors is not set",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--rna_pca_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--classifier_hidden_dim", type=int, default=64)
    parser.add_argument("--classifier_dropout", type=float, default=0.2)
    parser.add_argument("--encoder_dropout", type=float, default=0.5)

    # Hyperparameters: lr: 1e-3, weight_decay: 1e-5, batch_size: 256
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
