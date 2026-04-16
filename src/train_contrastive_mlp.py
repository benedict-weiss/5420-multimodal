"""End-to-end two-stage training for the contrastive MLP dual-encoder model."""

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
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset import CITEseqDataset, get_dataloaders
from src.models.classifier import ClassificationHead
from src.models.contrastive_loss import CLIPLoss
from src.models.mlp_encoder import MLPEncoder
from src.preprocessing import (
    get_labels,
    load_data,
    preprocess_protein,
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, n_classes: int) -> dict:
    """Compute accuracy + macro AUROC with safe fallbacks."""
    metrics = {}

    if callable(compute_accuracy):
        overall_acc, per_class = compute_accuracy(y_true, y_pred)
    else:
        overall_acc = float(accuracy_score(y_true, y_pred))
        per_class = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # AUROC can be undefined when a split is missing classes. Compute over
    # classes present in y_true to avoid unnecessary NaN/null outputs.
    try:
        present = np.sort(np.unique(y_true).astype(int))
        if present.size < 2:
            auroc = float("nan")
        elif present.size == 2:
            pos_class = int(present[1])
            if y_proba.ndim == 2 and y_proba.shape[1] > pos_class:
                scores = y_proba[:, pos_class]
            else:
                scores = y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] > 1 else y_proba
            auroc = float(roc_auc_score(y_true, scores))
        else:
            # Remap present global labels to contiguous [0..k-1] for multiclass AUROC.
            proba_present = y_proba[:, present]
            # roc_auc_score(multi_class='ovr') expects class probabilities to
            # sum to 1 across provided classes. If some global classes are
            # absent in the current split, renormalize after subsetting.
            proba_row_sum = np.sum(proba_present, axis=1, keepdims=True)
            proba_row_sum = np.where(proba_row_sum > 0, proba_row_sum, 1.0)
            proba_present = proba_present / proba_row_sum
            remap = {c: i for i, c in enumerate(present.tolist())}
            y_true_remap = np.array([remap[int(c)] for c in y_true], dtype=int)
            auroc = float(
                roc_auc_score(
                    y_true_remap,
                    proba_present,
                    average="macro",
                    multi_class="ovr",
                )
            )
    except Exception:
        auroc = float("nan")

    metrics["accuracy"] = overall_acc
    metrics["macro_auroc"] = auroc
    metrics["per_class"] = per_class
    return metrics


def run_contrastive_epoch(
    loader: DataLoader,
    rna_encoder: MLPEncoder,
    protein_encoder: MLPEncoder,
    loss_fn: CLIPLoss,
    device: torch.device,
    optimizer: Optimizer | None = None,
    input_dropout: float = 0.0,
    noise_std: float = 0.0,
) -> float:
    train_mode = optimizer is not None
    rna_encoder.train(train_mode)
    protein_encoder.train(train_mode)

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        rna = batch["rna"].to(device)
        protein = batch["protein"].to(device)

        # Regularize Stage A by perturbing inputs only during training.
        if train_mode and input_dropout > 0.0:
            rna = F.dropout(rna, p=input_dropout, training=True)
            protein = F.dropout(protein, p=input_dropout, training=True)
        if train_mode and noise_std > 0.0:
            rna = rna + torch.randn_like(rna) * noise_std
            protein = protein + torch.randn_like(protein) * noise_std

        if train_mode:
            optimizer.zero_grad()

        z_rna = rna_encoder(rna)
        z_protein = protein_encoder(protein)
        loss = loss_fn(z_rna, z_protein)

        if train_mode:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if n_batches == 0:
        return float("inf")
    return total_loss / n_batches


@torch.no_grad()
def evaluate_classifier_epoch(
    loader: DataLoader,
    rna_encoder: MLPEncoder,
    protein_encoder: MLPEncoder,
    classifier: ClassificationHead,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    rna_encoder.eval()
    protein_encoder.eval()
    classifier.eval()

    total_loss = 0.0
    n_batches = 0
    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    for batch in loader:
        rna = batch["rna"].to(device)
        protein = batch["protein"].to(device)
        y = batch["label"].to(device)

        z = torch.cat([rna_encoder(rna), protein_encoder(protein)], dim=1)
        logits = classifier(z)
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

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_proba = np.concatenate(y_proba_all)
    return total_loss / n_batches, y_true, y_pred, y_proba


def run_stage_a_probe(
    train_loader: DataLoader,
    val_loader: DataLoader,
    rna_encoder: MLPEncoder,
    protein_encoder: MLPEncoder,
    n_classes: int,
    embedding_dim: int,
    classifier_hidden_dim: int,
    classifier_lr: float,
    weight_decay: float,
    probe_epochs: int,
    device: torch.device,
) -> float:
    """Train a lightweight classifier probe on frozen Stage A embeddings and return val accuracy."""
    rna_was_training = rna_encoder.training
    protein_was_training = protein_encoder.training
    rna_encoder.eval()
    protein_encoder.eval()

    probe = ClassificationHead(
        input_dim=embedding_dim * 2,
        n_classes=n_classes,
        hidden_dim=classifier_hidden_dim,
        dropout=0.0,
    ).to(device)
    probe_optimizer = Adam(probe.parameters(), lr=classifier_lr, weight_decay=weight_decay)

    for _ in range(max(1, probe_epochs)):
        probe.train()
        for batch in train_loader:
            rna = batch["rna"].to(device)
            protein = batch["protein"].to(device)
            y = batch["label"].to(device)

            with torch.no_grad():
                z = torch.cat([rna_encoder(rna), protein_encoder(protein)], dim=1)

            probe_optimizer.zero_grad()
            logits = probe(z)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            probe_optimizer.step()

    _, y_val_true, y_val_pred, y_val_proba = evaluate_classifier_epoch(
        loader=val_loader,
        rna_encoder=rna_encoder,
        protein_encoder=protein_encoder,
        classifier=probe,
        device=device,
    )
    probe_metrics = compute_metrics(y_val_true, y_val_pred, y_val_proba, n_classes=n_classes)

    rna_encoder.train(rna_was_training)
    protein_encoder.train(protein_was_training)
    return float(probe_metrics["accuracy"])


def build_train_val_indices(labels: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(labels.shape[0])
    if val_ratio <= 0 or idx.size < 2:
        return idx, np.array([], dtype=int)

    unique_labels, counts = np.unique(labels, return_counts=True)
    label_to_count = {int(label): int(count) for label, count in zip(unique_labels, counts)}
    eligible_labels = unique_labels[counts >= 2]

    if eligible_labels.size == 0:
        warnings.warn(
            "Could not build a class-covered validation split because every class has fewer than 2 examples. "
            "Falling back to a regular stratified split, if possible.",
            UserWarning,
            stacklevel=2,
        )
        try:
            train_idx, val_idx = train_test_split(
                idx,
                test_size=val_ratio,
                random_state=seed,
                stratify=labels,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                idx,
                test_size=val_ratio,
                random_state=seed,
                stratify=None,
            )
        return np.asarray(train_idx), np.asarray(val_idx)

    rng = np.random.default_rng(seed)
    class_indices: dict[int, np.ndarray] = {}
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
        warnings.warn(
            "Validation split could not be made class-covered; falling back to a regular stratified split.",
            UserWarning,
            stacklevel=2,
        )
        try:
            train_idx, val_idx = train_test_split(
                idx,
                test_size=val_ratio,
                random_state=seed,
                stratify=labels,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                idx,
                test_size=val_ratio,
                random_state=seed,
                stratify=None,
            )

    return np.asarray(train_idx), np.asarray(val_idx)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    dataset_file = resolve_data_file(args.data_path)
    adata = load_data(dataset_file)

    if not adata.var_names.is_unique:
        adata.var_names_make_unique()

    labels_all, label_mapping = get_labels(adata, label_col=args.label_col)
    n_classes = len(label_mapping)

    # Train/test split. The held-out test set is selected once and never reused
    # for Stage A selection or Stage B checkpointing.
    val_global_idx = np.array([], dtype=int)
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
        raw_val_values = (
            list(args.split_val_values)
            if (args.use_predefined_val_split and args.split_val_values)
            else []
        )
        split_series = adata.obs[args.split_col]
        split_arr_raw = split_series.values

        parsed_column_values = [_parse_bool_like(v) for v in split_arr_raw]
        column_is_bool_like = all(v is not None for v in parsed_column_values)
        parsed_test_values = [_parse_bool_like(v) for v in raw_split_values]
        test_values_are_bool_like = all(v is not None for v in parsed_test_values)

        def _mask_from_values(raw_values: list[object]) -> np.ndarray:
            parsed_values = [_parse_bool_like(v) for v in raw_values]
            values_are_bool_like = all(v is not None for v in parsed_values)

            if column_is_bool_like and values_are_bool_like:
                target_bool = set(parsed_values)
                return np.array([v in target_bool for v in parsed_column_values], dtype=bool)
            if column_is_bool_like and set(str(v).strip().lower() for v in raw_values) == {"test"}:
                return np.array([v is False for v in parsed_column_values], dtype=bool)

            split_values_str = set(str(v) for v in raw_values)
            split_arr = split_series.astype(str).values
            return np.array([v in split_values_str for v in split_arr], dtype=bool)

        test_mask = _mask_from_values(raw_split_values)
        val_mask = _mask_from_values(raw_val_values) if raw_val_values else np.zeros_like(test_mask)

        overlap_mask = test_mask & val_mask
        if np.any(overlap_mask):
            raise ValueError("split_test_values and split_val_values overlap. Provide disjoint split values.")

        train_mask = ~(test_mask | val_mask)
        if not np.any(train_mask):
            raise ValueError("Training split is empty after applying split_test_values/split_val_values.")

        test_global_idx = np.where(test_mask)[0]
        train_global_idx = np.where(train_mask)[0]
        if raw_val_values:
            val_global_idx = np.where(val_mask)[0]
        print(
            f"Using predefined split column '{args.split_col}' with test values "
            f"{sorted(str(v) for v in raw_split_values)}"
        )
        if raw_val_values:
            print(f"Using predefined validation values {sorted(str(v) for v in raw_val_values)}")
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

    if args.use_predefined_val_split and args.split_val_values and len(val_global_idx) == 0:
        raise ValueError("Validation split is empty. Adjust --split_val_values.")

    rna_adata, protein_adata = split_modalities(adata)

    # Leakage-safe RNA preprocessing: fit PCA on train only, transform test with same PCA + HVGs
    train_rna = rna_adata[train_global_idx].copy()
    test_rna = rna_adata[test_global_idx].copy()
    train_rna_pca, pca_model, hvg_genes = preprocess_rna(
        train_rna,
        n_comps=args.rna_pca_dim,
        return_pca_model=True,
    )
    test_rna_pca = preprocess_rna(
        test_rna,
        n_comps=args.rna_pca_dim,
        pca_model=pca_model,
        hvg_genes=hvg_genes,
    )

    # Protein preprocessing (per split)
    train_protein = preprocess_protein(protein_adata[train_global_idx].copy())
    test_protein = preprocess_protein(protein_adata[test_global_idx].copy())

    train_labels = labels_all[train_global_idx]
    test_labels = labels_all[test_global_idx]

    print(
        f"Prepared matrices: train RNA {train_rna_pca.shape}, train protein {train_protein.shape}, "
        f"test RNA {test_rna_pca.shape}, test protein {test_protein.shape}"
    )

    # Validation is always derived from training data unless an explicit
    # predefined validation split is requested.
    if len(val_global_idx) > 0:
        val_rna = rna_adata[val_global_idx].copy()
        val_rna_pca = preprocess_rna(
            val_rna,
            n_comps=args.rna_pca_dim,
            pca_model=pca_model,
            hvg_genes=hvg_genes,
        )
        val_protein = preprocess_protein(protein_adata[val_global_idx].copy())
        val_labels = labels_all[val_global_idx]

        stage_a_train_loader = DataLoader(
            CITEseqDataset(train_rna_pca, train_protein, train_labels),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        stage_a_val_loader = DataLoader(
            CITEseqDataset(val_rna_pca, val_protein, val_labels),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
        classifier_train_dataset = CITEseqDataset(train_rna_pca, train_protein, train_labels)
        classifier_val_dataset = CITEseqDataset(val_rna_pca, val_protein, val_labels)
    else:
        train_local_idx, val_local_idx = build_train_val_indices(
            labels=train_labels,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        trainval_rna = train_rna_pca
        trainval_protein = train_protein
        trainval_labels = train_labels

        if len(val_local_idx) == 0:
            warnings.warn(
                "Validation split is empty; proceeding without a held-out validation set. "
                "Stage A monitoring will use the training split. "
                "Consider increasing --val_ratio or training set size.",
                UserWarning,
                stacklevel=2,
            )
            val_local_idx = train_local_idx

        stage_a_train_loader, stage_a_val_loader = get_dataloaders(
            rna_pca=trainval_rna,
            protein_clr=trainval_protein,
            labels=trainval_labels,
            train_idx=train_local_idx,
            test_idx=val_local_idx,
            batch_size=args.batch_size,
        )
        # Mirror transformer setup: Stage B trains only on train split, never on val split.
        classifier_train_dataset = CITEseqDataset(
            trainval_rna[train_local_idx],
            trainval_protein[train_local_idx],
            trainval_labels[train_local_idx],
        )
        classifier_val_dataset = CITEseqDataset(
            trainval_rna[val_local_idx],
            trainval_protein[val_local_idx],
            trainval_labels[val_local_idx],
        )

    # Full-train + test loaders for Stage B classifier training/evaluation
    classifier_test_dataset = CITEseqDataset(test_rna_pca, test_protein, test_labels)

    classifier_train_loader = DataLoader(
        classifier_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    classifier_val_loader = DataLoader(
        classifier_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    classifier_test_loader = DataLoader(
        classifier_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Models
    rna_encoder = MLPEncoder(
        input_dim=train_rna_pca.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=args.embedding_dim,
        normalize_output=True,
    ).to(device)
    protein_encoder = MLPEncoder(
        input_dim=train_protein.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=args.embedding_dim,
        normalize_output=True,
    ).to(device)
    classifier = ClassificationHead(
        input_dim=args.embedding_dim * 2,
        n_classes=n_classes,
        hidden_dim=args.classifier_hidden_dim,
        dropout=args.classifier_dropout,
    ).to(device)
    clip_loss = CLIPLoss(temperature=args.temperature)

    # Stage A: contrastive pretraining with early stopping on validation loss
    # Original behavior used Adam with fixed LR.
    # stage_a_optimizer = Adam(
    #     list(rna_encoder.parameters()) + list(protein_encoder.parameters()),
    #     lr=args.lr,
    #     weight_decay=args.weight_decay,
    # )
    stage_a_optimizer = AdamW(
        list(rna_encoder.parameters()) + list(protein_encoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    stage_a_scheduler = CosineAnnealingLR(
        stage_a_optimizer,
        T_max=max(1, args.contrastive_epochs),
        eta_min=args.lr * args.stage_a_min_lr_ratio,
    )

    best_val = float("inf")
    best_probe_acc = -1.0
    best_epoch = -1
    patience_counter = 0
    best_rna_state = clone_state_dict(rna_encoder)
    best_protein_state = clone_state_dict(protein_encoder)

    stage_a_history = []
    print("\n=== Stage A: Contrastive pretraining ===")
    for epoch in range(1, args.contrastive_epochs + 1):
        train_loss = run_contrastive_epoch(
            loader=stage_a_train_loader,
            rna_encoder=rna_encoder,
            protein_encoder=protein_encoder,
            loss_fn=clip_loss,
            device=device,
            optimizer=stage_a_optimizer,
            input_dropout=args.stage_a_input_dropout,
            noise_std=args.stage_a_noise_std,
        )
        with torch.no_grad():
            val_loss = run_contrastive_epoch(
                loader=stage_a_val_loader,
                rna_encoder=rna_encoder,
                protein_encoder=protein_encoder,
                loss_fn=clip_loss,
                device=device,
                optimizer=None,
                input_dropout=0.0,
                noise_std=0.0,
            )

        stage_a_scheduler.step()
        current_lr = float(stage_a_optimizer.param_groups[0]["lr"])

        stage_a_history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": current_lr,
            }
        )

        improved = False
        probe_val_acc: float | None = None
        if args.stage_a_select_metric == "probe_accuracy" and epoch >= args.stage_a_probe_start_epoch:
            should_probe = (epoch == 1) or (epoch % max(1, args.stage_a_probe_every) == 0)
            if should_probe:
                probe_val_acc = run_stage_a_probe(
                    train_loader=classifier_train_loader,
                    val_loader=classifier_val_loader,
                    rna_encoder=rna_encoder,
                    protein_encoder=protein_encoder,
                    n_classes=n_classes,
                    embedding_dim=args.embedding_dim,
                    classifier_hidden_dim=args.classifier_hidden_dim,
                    classifier_lr=args.stage_a_probe_lr,
                    weight_decay=args.weight_decay,
                    probe_epochs=args.stage_a_probe_epochs,
                    device=device,
                )
                stage_a_history[-1]["probe_val_accuracy"] = float(probe_val_acc)
                improved = (probe_val_acc - best_probe_acc) > args.stage_a_probe_min_delta
                if improved:
                    best_probe_acc = float(probe_val_acc)
                    best_val = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    best_rna_state = clone_state_dict(rna_encoder)
                    best_protein_state = clone_state_dict(protein_encoder)
                else:
                    patience_counter += 1
        elif args.stage_a_select_metric == "probe_accuracy" and epoch < args.stage_a_probe_start_epoch:
            improved = (best_val - val_loss) > args.min_delta
            if improved:
                best_val = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_rna_state = clone_state_dict(rna_encoder)
                best_protein_state = clone_state_dict(protein_encoder)
            else:
                patience_counter += 1
        else:
            improved = (best_val - val_loss) > args.min_delta
            if improved:
                best_val = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_rna_state = clone_state_dict(rna_encoder)
                best_protein_state = clone_state_dict(protein_encoder)
            else:
                patience_counter += 1

        print(
            f"[Stage A] Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | best_val={best_val:.4f} | lr={current_lr:.2e}"
            + (
                f" | probe_val_acc={probe_val_acc:.4f} | best_probe={best_probe_acc:.4f}"
                if probe_val_acc is not None
                else ""
            )
        )

        if args.stage_a_select_metric == "probe_accuracy" and probe_val_acc is None:
            continue

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
            break

    rna_encoder.load_state_dict(best_rna_state)
    protein_encoder.load_state_dict(best_protein_state)
    print(f"Loaded best Stage A weights from epoch {best_epoch} (val_loss={best_val:.4f}).")

    # Stage B: freeze encoders, train classifier
    for p in rna_encoder.parameters():
        p.requires_grad = False
    for p in protein_encoder.parameters():
        p.requires_grad = False

    # Original behavior used Stage A lr for Stage B as well.
    # stage_b_optimizer = Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stage_b_optimizer = Adam(
        classifier.parameters(),
        lr=args.classifier_lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = -1.0
    best_classifier_state = clone_state_dict(classifier)
    stage_b_history = []

    print("\n=== Stage B: Classifier training ===")
    for epoch in range(1, args.classifier_epochs + 1):
        rna_encoder.eval()
        protein_encoder.eval()
        classifier.train()

        total_train_loss = 0.0
        n_train_batches = 0

        for batch in classifier_train_loader:
            rna = batch["rna"].to(device)
            protein = batch["protein"].to(device)
            y = batch["label"].to(device)

            stage_b_optimizer.zero_grad()
            with torch.no_grad():
                z_rna = rna_encoder(rna)
                z_protein = protein_encoder(protein)
            z = torch.cat([z_rna, z_protein], dim=1)
            logits = classifier(z)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            stage_b_optimizer.step()

            total_train_loss += loss.item()
            n_train_batches += 1

        train_loss = total_train_loss / max(1, n_train_batches)

        # Checkpoint selection uses val split (not test) to avoid leakage
        val_loss, y_val_true, y_val_pred, y_val_proba = evaluate_classifier_epoch(
            loader=classifier_val_loader,
            rna_encoder=rna_encoder,
            protein_encoder=protein_encoder,
            classifier=classifier,
            device=device,
        )
        val_metrics = compute_metrics(y_val_true, y_val_pred, y_val_proba, n_classes=n_classes)

        stage_b_history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_auroc": float(val_metrics["macro_auroc"]),
            }
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = float(val_metrics["accuracy"])
            best_classifier_state = clone_state_dict(classifier)

        print(
            f"[Stage B] Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_auroc={val_metrics['macro_auroc']:.4f}"
        )

    classifier.load_state_dict(best_classifier_state)

    final_test_loss, y_true, y_pred, y_proba = evaluate_classifier_epoch(
        loader=classifier_test_loader,
        rna_encoder=rna_encoder,
        protein_encoder=protein_encoder,
        classifier=classifier,
        device=device,
    )
    final_metrics = compute_metrics(y_true, y_pred, y_proba, n_classes=n_classes)

    # Save artifacts
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"contrastive_mlp_seed{args.seed}_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "stage": "contrastive_pretrain",
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val),
            "rna_encoder_state_dict": rna_encoder.state_dict(),
            "protein_encoder_state_dict": protein_encoder.state_dict(),
            "args": vars(args),
        },
        run_dir / "stage_a_best.pt",
    )
    torch.save(
        {
            "stage": "classifier_finetune",
            "rna_encoder_state_dict": rna_encoder.state_dict(),
            "protein_encoder_state_dict": protein_encoder.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "final_test_loss": float(final_test_loss),
            "final_metrics": {
                "accuracy": float(final_metrics["accuracy"]),
                "macro_auroc": float(final_metrics["macro_auroc"]),
            },
            "args": vars(args),
        },
        run_dir / "stage_b_best.pt",
    )

    with open(run_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in label_mapping.items()}, f, indent=2)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            _sanitize_json(
                {
                    "final_test_loss": float(final_test_loss),
                    "final_accuracy": float(final_metrics["accuracy"]),
                    "final_macro_auroc": float(final_metrics["macro_auroc"]),
                    "stage_a_history": stage_a_history,
                    "stage_b_history": stage_b_history,
                }
            ),
            f,
            indent=2,
        )

    # Persist PCA metadata needed for strict train/test feature alignment on reuse
    with open(run_dir / "hvg_genes.json", "w", encoding="utf-8") as f:
        json.dump(hvg_genes, f)
    torch.save(pca_model, run_dir / "rna_pca_model.pt")

    # Save test embeddings for evaluate.py (Option B)
    with torch.no_grad():
        rna_encoder.eval()
        protein_encoder.eval()
        z_rna_list, z_prot_list = [], []
        for batch in classifier_test_loader:
            z_rna_list.append(rna_encoder(batch["rna"].to(device)).cpu().numpy())
            z_prot_list.append(protein_encoder(batch["protein"].to(device)).cpu().numpy())
    z_rna_all = np.concatenate(z_rna_list, axis=0)
    z_prot_all = np.concatenate(z_prot_list, axis=0)
    np.save(run_dir / "test_embeddings.npy", np.concatenate([z_rna_all, z_prot_all], axis=1))
    np.save(run_dir / "test_rna_embeddings.npy", z_rna_all)
    np.save(run_dir / "test_protein_embeddings.npy", z_prot_all)
    np.save(run_dir / "test_labels.npy", test_labels)
    print(f"Test embeddings saved: rna {z_rna_all.shape}, protein {z_prot_all.shape}")

    print("\n=== Training complete ===")
    print(f"Run directory: {run_dir}")
    print(f"Final test loss: {final_test_loss:.4f}")
    print(f"Final test accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final macro AUROC: {final_metrics['macro_auroc']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train contrastive MLP dual-encoder")

    parser.add_argument("--data_path", type=str, default="data", help="Dataset file path or directory")
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
        help="Optional donor IDs to hold out (e.g., donor8 donor9)",
    )
    parser.add_argument("--split_col", type=str, default="is_train")
    parser.add_argument(
        "--split_test_values",
        type=str,
        nargs="*",
        default=["test"],
        help="Values in split_col treated as test when --test_donors is not provided",
    )
    parser.add_argument(
        "--split_val_values",
        type=str,
        nargs="*",
        default=None,
        help="Optional values in split_col treated as validation; leaves remaining values for training",
    )
    parser.add_argument(
        "--use_predefined_val_split",
        action="store_true",
        help="Use --split_val_values for validation instead of the default train-only stratified split",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--rna_pca_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--classifier_hidden_dim", type=int, default=64)
    parser.add_argument("--classifier_dropout", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    # Original behavior reused --lr for both stages.
    # parser.add_argument("--classifier_lr", type=float, default=1e-3)
    parser.add_argument(
        "--classifier_lr",
        type=float,
        default=1e-3,
        help="Learning rate for Stage B classifier training",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--stage_a_input_dropout",
        type=float,
        default=0.05,
        help="Input feature dropout applied during Stage A contrastive training",
    )
    parser.add_argument(
        "--stage_a_noise_std",
        type=float,
        default=0.01,
        help="Gaussian noise std added to inputs during Stage A contrastive training",
    )
    parser.add_argument(
        "--stage_a_min_lr_ratio",
        type=float,
        default=0.1,
        help="Cosine LR schedule floor for Stage A as a fraction of --lr",
    )

    parser.add_argument("--contrastive_epochs", type=int, default=150)
    parser.add_argument("--classifier_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument(
        "--stage_a_select_metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "probe_accuracy"],
        help="Checkpoint/early-stop criterion for Stage A",
    )
    parser.add_argument(
        "--stage_a_probe_every",
        type=int,
        default=5,
        help="Probe Stage A representation every N epochs when using probe_accuracy criterion",
    )
    parser.add_argument(
        "--stage_a_probe_epochs",
        type=int,
        default=3,
        help="Number of epochs to train the Stage A probe head",
    )
    parser.add_argument(
        "--stage_a_probe_lr",
        type=float,
        default=1e-3,
        help="Learning rate for Stage A probe classifier",
    )
    parser.add_argument(
        "--stage_a_probe_min_delta",
        type=float,
        default=1e-4,
        help="Minimum probe accuracy improvement to reset Stage A patience",
    )
    parser.add_argument(
        "--stage_a_probe_start_epoch",
        type=int,
        default=5,
        help="Do not use probe-based Stage A checkpointing before this epoch",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
