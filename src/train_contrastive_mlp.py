"""End-to-end two-stage training for the contrastive MLP dual-encoder model."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.optim import Adam
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
    optimizer: Adam | None = None,
) -> float:
    train_mode = optimizer is not None
    rna_encoder.train(train_mode)
    protein_encoder.train(train_mode)

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        rna = batch["rna"].to(device)
        protein = batch["protein"].to(device)

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


def build_train_val_indices(labels: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(labels.shape[0])
    if val_ratio <= 0:
        return idx, np.array([], dtype=int)

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

    # Train/test split
    if args.test_donors:
        train_global_idx, test_global_idx = split_by_donor(
            adata, test_donors=args.test_donors, donor_col=args.donor_col
        )
    elif args.split_col in adata.obs.columns:
        split_values = set(args.split_test_values)
        split_arr = adata.obs[args.split_col].astype(str).values
        test_mask = np.array([v in split_values for v in split_arr], dtype=bool)
        test_global_idx = np.where(test_mask)[0]
        train_global_idx = np.where(~test_mask)[0]
        print(
            f"Using predefined split column '{args.split_col}' with test values "
            f"{sorted(split_values)}"
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

    # Train/val split for Stage A early stopping
    train_local_idx, val_local_idx = build_train_val_indices(
        labels=train_labels,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    trainval_rna = train_rna_pca
    trainval_protein = train_protein
    trainval_labels = train_labels

    if len(val_local_idx) == 0:
        raise ValueError("Validation split is empty. Increase train set size or val_ratio.")

    stage_a_train_loader, stage_a_val_loader = get_dataloaders(
        rna_pca=trainval_rna,
        protein_clr=trainval_protein,
        labels=trainval_labels,
        train_idx=train_local_idx,
        test_idx=val_local_idx,
        batch_size=args.batch_size,
    )

    # Full-train + test loaders for Stage B classifier training/evaluation
    classifier_train_dataset = CITEseqDataset(train_rna_pca, train_protein, train_labels)
    classifier_test_dataset = CITEseqDataset(test_rna_pca, test_protein, test_labels)

    classifier_train_loader = DataLoader(
        classifier_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    stage_a_optimizer = Adam(
        list(rna_encoder.parameters()) + list(protein_encoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
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
        )
        with torch.no_grad():
            val_loss = run_contrastive_epoch(
                loader=stage_a_val_loader,
                rna_encoder=rna_encoder,
                protein_encoder=protein_encoder,
                loss_fn=clip_loss,
                device=device,
                optimizer=None,
            )

        stage_a_history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )

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
            f"val_loss={val_loss:.4f} | best_val={best_val:.4f}"
        )

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

    stage_b_optimizer = Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_test_acc = -1.0
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
        test_loss, y_true, y_pred, y_proba = evaluate_classifier_epoch(
            loader=classifier_test_loader,
            rna_encoder=rna_encoder,
            protein_encoder=protein_encoder,
            classifier=classifier,
            device=device,
        )

        test_metrics = compute_metrics(y_true, y_pred, y_proba, n_classes=n_classes)
        stage_b_history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "test_loss": float(test_loss),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_macro_auroc": float(test_metrics["macro_auroc"]),
            }
        )

        if test_metrics["accuracy"] > best_test_acc:
            best_test_acc = float(test_metrics["accuracy"])
            best_classifier_state = clone_state_dict(classifier)

        print(
            f"[Stage B] Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"test_loss={test_loss:.4f} | test_acc={test_metrics['accuracy']:.4f} | "
            f"test_auroc={test_metrics['macro_auroc']:.4f}"
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
            {
                "final_test_loss": float(final_test_loss),
                "final_accuracy": float(final_metrics["accuracy"]),
                "final_macro_auroc": float(final_metrics["macro_auroc"]),
                "stage_a_history": stage_a_history,
                "stage_b_history": stage_b_history,
            },
            f,
            indent=2,
        )

    # Persist PCA metadata needed for strict train/test feature alignment on reuse
    with open(run_dir / "hvg_genes.json", "w", encoding="utf-8") as f:
        json.dump(hvg_genes, f)
    torch.save(pca_model, run_dir / "rna_pca_model.pt")

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
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--rna_pca_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--classifier_hidden_dim", type=int, default=64)
    parser.add_argument("--classifier_dropout", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument("--contrastive_epochs", type=int, default=150)
    parser.add_argument("--classifier_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
