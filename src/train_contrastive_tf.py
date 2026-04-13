"""End-to-end two-stage training for the contrastive transformer dual-encoder model."""

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
from torch.utils.data import DataLoader

from src.dataset import CITEseqDataset, get_dataloaders
from src.models.classifier import ClassificationHead
from src.models.contrastive_loss import CLIPLoss
from src.models.transformer_encoder import TransformerEncoder
from src.preprocessing import (
    build_pathway_tokens,
    get_labels,
    load_data,
    preprocess_protein,
    split_modalities,
    split_by_donor,
)

try:
    from src.evaluate import compute_accuracy, compute_auroc
except Exception:
    compute_accuracy = None
    compute_auroc = None


def _sanitize_json(obj: object) -> object:
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
    metrics: dict = {}
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
                roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
            )
        except Exception:
            auroc = float("nan")
    metrics["accuracy"] = overall_acc
    metrics["macro_auroc"] = auroc
    metrics["per_class"] = per_class
    return metrics


def run_contrastive_epoch(
    loader: DataLoader,
    rna_encoder: TransformerEncoder,
    protein_encoder: TransformerEncoder,
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
        # "rna" key holds pathway tokens (pathway_matrix passed as rna_pca in dataloaders)
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

    return total_loss / n_batches if n_batches > 0 else float("inf")


@torch.no_grad()
def evaluate_classifier_epoch(
    loader: DataLoader,
    rna_encoder: TransformerEncoder,
    protein_encoder: TransformerEncoder,
    classifier: ClassificationHead,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    rna_encoder.train(False)
    protein_encoder.train(False)
    classifier.train(False)

    total_loss = 0.0
    n_batches = 0
    y_true_all, y_pred_all, y_proba_all = [], [], []

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

    return (
        total_loss / n_batches,
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        np.concatenate(y_proba_all),
    )


@torch.no_grad()
def extract_attention(
    loader: DataLoader,
    rna_encoder: TransformerEncoder,
    protein_encoder: TransformerEncoder,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect CLS attention weights from both encoders over the full loader.

    Returns:
        (attn_rna, attn_protein, labels)
        attn_rna:    (n_cells, n_pathway_tokens)
        attn_protein:(n_cells, n_protein_tokens)
        labels:      (n_cells,) integer labels
    """
    rna_encoder.train(False)
    protein_encoder.train(False)

    attn_rna_all, attn_protein_all, labels_all = [], [], []

    for batch in loader:
        rna = batch["rna"].to(device)
        protein = batch["protein"].to(device)

        _ = rna_encoder(rna)
        _ = protein_encoder(protein)

        a_rna = rna_encoder.get_attention_weights()
        a_protein = protein_encoder.get_attention_weights()

        if a_rna is not None:
            attn_rna_all.append(a_rna.detach().cpu().numpy())
        if a_protein is not None:
            attn_protein_all.append(a_protein.detach().cpu().numpy())
        labels_all.append(batch["label"].numpy())

    return (
        np.concatenate(attn_rna_all) if attn_rna_all else np.array([]),
        np.concatenate(attn_protein_all) if attn_protein_all else np.array([]),
        np.concatenate(labels_all),
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
    else:
        train_global_idx, test_global_idx = train_test_split(
            np.arange(adata.shape[0]),
            test_size=args.test_size,
            random_state=args.seed,
            stratify=labels_all,
        )
        train_global_idx = np.asarray(train_global_idx)
        test_global_idx = np.asarray(test_global_idx)

    if len(test_global_idx) == 0:
        raise ValueError("Test split is empty. Adjust --test_donors or --test_size.")

    rna_adata, protein_adata = split_modalities(adata)

    # Pathway tokens: built on full dataset — KEGG gene sets are global, no leakage risk.
    gene_sets = None
    if args.gene_sets_path:
        print(f"Loading gene sets from {args.gene_sets_path}...")
        with open(args.gene_sets_path, "r", encoding="utf-8") as f:
            gene_sets = json.load(f)
    print("Building pathway tokens (KEGG_2021_Human)...")
    pathway_matrix, pathway_names = build_pathway_tokens(rna_adata, gene_sets=gene_sets)
    n_pathways = pathway_matrix.shape[1]
    print(f"  Pathway matrix shape: {pathway_matrix.shape}")

    # Protein preprocessing per split
    train_protein = preprocess_protein(protein_adata[train_global_idx].copy())
    test_protein = preprocess_protein(protein_adata[test_global_idx].copy())
    n_proteins = train_protein.shape[1]

    train_pathway = pathway_matrix[train_global_idx]
    test_pathway = pathway_matrix[test_global_idx]
    train_labels = labels_all[train_global_idx]
    test_labels = labels_all[test_global_idx]

    print(
        f"Prepared: train pathway {train_pathway.shape}, "
        f"train protein {train_protein.shape}"
    )

    train_local_idx, val_local_idx = build_train_val_indices(
        labels=train_labels, val_ratio=args.val_ratio, seed=args.seed
    )
    if len(val_local_idx) == 0:
        warnings.warn(
            "Validation split is empty; using train split for Stage A monitoring.",
            UserWarning,
            stacklevel=2,
        )
        val_local_idx = train_local_idx

    # Dataloaders — pathway_matrix passed as rna_pca; stored under batch["rna"]
    stage_a_train_loader, stage_a_val_loader = get_dataloaders(
        rna_pca=train_pathway,
        protein_clr=train_protein,
        labels=train_labels,
        train_idx=train_local_idx,
        test_idx=val_local_idx,
        batch_size=args.batch_size,
    )

    classifier_train_dataset = CITEseqDataset(train_pathway, train_protein, train_labels)
    classifier_test_dataset = CITEseqDataset(test_pathway, test_protein, test_labels)
    classifier_val_dataset = CITEseqDataset(
        train_pathway[val_local_idx],
        train_protein[val_local_idx],
        train_labels[val_local_idx],
    )

    classifier_train_loader = DataLoader(
        classifier_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    classifier_val_loader = DataLoader(
        classifier_val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    classifier_test_loader = DataLoader(
        classifier_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # Models
    rna_encoder = TransformerEncoder(
        n_tokens=n_pathways,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        output_dim=args.embedding_dim,
    ).to(device)
    protein_encoder = TransformerEncoder(
        n_tokens=n_proteins,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        output_dim=args.embedding_dim,
    ).to(device)
    classifier = ClassificationHead(
        input_dim=args.embedding_dim * 2,
        n_classes=n_classes,
        hidden_dim=args.classifier_hidden_dim,
        dropout=args.classifier_dropout,
    ).to(device)
    clip_loss = CLIPLoss(temperature=args.temperature)

    # Stage A: Contrastive pretraining
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
    stage_a_history: list[dict] = []

    print("\n=== Stage A: Contrastive pretraining ===")
    for epoch in range(1, args.contrastive_epochs + 1):
        train_loss = run_contrastive_epoch(
            stage_a_train_loader, rna_encoder, protein_encoder,
            clip_loss, device, optimizer=stage_a_optimizer,
        )
        with torch.no_grad():
            val_loss = run_contrastive_epoch(
                stage_a_val_loader, rna_encoder, protein_encoder,
                clip_loss, device, optimizer=None,
            )

        stage_a_history.append({"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss)})

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
            f"[Stage A] Epoch {epoch:03d} | train={train_loss:.4f} | "
            f"val={val_loss:.4f} | best={best_val:.4f}"
        )
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    rna_encoder.load_state_dict(best_rna_state)
    protein_encoder.load_state_dict(best_protein_state)
    print(f"Loaded best Stage A weights from epoch {best_epoch}.")

    # Stage B: Classifier training
    for p in rna_encoder.parameters():
        p.requires_grad = False
    for p in protein_encoder.parameters():
        p.requires_grad = False

    stage_b_optimizer = Adam(
        classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_val_acc = -1.0
    best_classifier_state = clone_state_dict(classifier)
    stage_b_history: list[dict] = []

    print("\n=== Stage B: Classifier training ===")
    for epoch in range(1, args.classifier_epochs + 1):
        rna_encoder.train(False)
        protein_encoder.train(False)
        classifier.train(True)

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
        val_loss, y_val_true, y_val_pred, y_val_proba = evaluate_classifier_epoch(
            classifier_val_loader, rna_encoder, protein_encoder, classifier, device
        )
        val_metrics = compute_metrics(y_val_true, y_val_pred, y_val_proba, n_classes)
        stage_b_history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_auroc": float(val_metrics["macro_auroc"]),
        })

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = float(val_metrics["accuracy"])
            best_classifier_state = clone_state_dict(classifier)

        print(
            f"[Stage B] Epoch {epoch:03d} | train={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

    classifier.load_state_dict(best_classifier_state)

    final_test_loss, y_true, y_pred, y_proba = evaluate_classifier_epoch(
        classifier_test_loader, rna_encoder, protein_encoder, classifier, device
    )
    final_metrics = compute_metrics(y_true, y_pred, y_proba, n_classes)

    # Attention extraction
    attn_rna, attn_protein, attn_labels = None, None, None
    if args.save_attention:
        print("\nExtracting attention weights from test set...")
        attn_rna, attn_protein, attn_labels = extract_attention(
            classifier_test_loader, rna_encoder, protein_encoder, device
        )
        print(f"  RNA attention: {attn_rna.shape}, Protein attention: {attn_protein.shape}")

    # Save artifacts
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"contrastive_tf_seed{args.seed}_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "stage": "contrastive_pretrain",
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val),
            "n_pathways": n_pathways,
            "n_proteins": n_proteins,
            "rna_encoder_state_dict": rna_encoder.state_dict(),
            "protein_encoder_state_dict": protein_encoder.state_dict(),
            "args": vars(args),
        },
        run_dir / "stage_a_best.pt",
    )
    torch.save(
        {
            "stage": "classifier_finetune",
            "n_pathways": n_pathways,
            "n_proteins": n_proteins,
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
    with open(run_dir / "pathway_names.json", "w", encoding="utf-8") as f:
        json.dump(pathway_names, f, indent=2)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            _sanitize_json({
                "final_test_loss": float(final_test_loss),
                "final_accuracy": float(final_metrics["accuracy"]),
                "final_macro_auroc": float(final_metrics["macro_auroc"]),
                "stage_a_history": stage_a_history,
                "stage_b_history": stage_b_history,
            }),
            f, indent=2,
        )

    if args.save_attention and attn_rna is not None:
        np.save(run_dir / "tf_attention_rna.npy", attn_rna)
        np.save(run_dir / "tf_attention_protein.npy", attn_protein)
        np.save(run_dir / "tf_attention_labels.npy", attn_labels)
        print(f"Attention weights saved to {run_dir}/")

    print("\n=== Training complete ===")
    print(f"Run directory: {run_dir}")
    print(f"Final test accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final macro AUROC:   {final_metrics['macro_auroc']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train contrastive transformer dual-encoder")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--label_col", type=str, default="cell_type")
    parser.add_argument("--donor_col", type=str, default="DonorNumber")
    parser.add_argument("--test_donors", type=str, nargs="*", default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    # Transformer hyperparameters
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--classifier_hidden_dim", type=int, default=64)
    parser.add_argument("--classifier_dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--contrastive_epochs", type=int, default=150)
    parser.add_argument("--classifier_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--save_attention", action="store_true", default=True)
    parser.add_argument("--no_save_attention", dest="save_attention", action="store_false")
    parser.add_argument(
        "--gene_sets_path", type=str, default=None,
        help="Path to pre-cached KEGG gene sets JSON (e.g. data/kegg_2021_human.json). "
             "If omitted, fetches from Enrichr API at runtime.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
