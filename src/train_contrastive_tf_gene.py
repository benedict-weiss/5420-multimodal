"""
End-to-end two-stage training for the contrastive transformer dual-encoder, using
per-gene HVG tokens instead of KEGG pathway tokens.

The only substantive difference versus train_contrastive_tf.py is the RNA
tokenizer: ``build_gene_tokens`` selects the top-N HVGs (log-normalized scalar
expression) as tokens so downstream attention produces a gene×gene adjacency
directly. Checkpoint layout is kept compatible with src/attention_graph.py:
tokens are saved as ``gene_names.json``, and the encoder's token count is
persisted under the existing ``n_pathways`` key (misnomer preserved for
downstream loader compatibility).
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
import scanpy as sc
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
    build_gene_tokens,
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

    if args.max_cells is not None and args.max_cells < adata.shape[0]:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(adata.shape[0], size=args.max_cells, replace=False)
        idx.sort()
        adata = adata[idx].copy()
        print(f"Subsampled to {args.max_cells} cells for smoke test.")

    labels_all, label_mapping = get_labels(adata, label_col=args.label_col)
    n_classes = len(label_mapping)

    if args.test_donors:
        train_global_idx, test_global_idx = split_by_donor(
            adata, test_donors=args.test_donors, donor_col=args.donor_col
        )
    else:
        train_global_idx, test_global_idx = train_test_split(
            np.arange(adata.shape[0]),
            test_size=args.test_size,
            random_state=args.seed,
            stratify=None if args.max_cells is not None else labels_all,
        )
        train_global_idx = np.asarray(train_global_idx)
        test_global_idx = np.asarray(test_global_idx)

    if len(test_global_idx) == 0:
        raise ValueError("Test split is empty. Adjust --test_donors or --test_size.")

    rna_adata, protein_adata = split_modalities(adata)

    sc.pp.normalize_total(rna_adata, target_sum=1e4)
    sc.pp.log1p(rna_adata)

    # HVG selection on the FULL dataset is fine: scanpy's default "seurat"-style
    # selection uses mean/variance statistics and does not leak labels. If you
    # want stricter separation, select on train_global_idx then subset.
    print(f"Building gene tokens (top {args.n_hvgs} HVGs)...")
    gene_matrix, gene_names = build_gene_tokens(rna_adata, n_hvgs=args.n_hvgs)
    n_tokens = gene_matrix.shape[1]
    print(f"  Gene token matrix shape: {gene_matrix.shape}")

    train_protein = preprocess_protein(protein_adata[train_global_idx].copy())
    test_protein = preprocess_protein(protein_adata[test_global_idx].copy())
    n_proteins = train_protein.shape[1]

    train_rna = gene_matrix[train_global_idx]
    test_rna = gene_matrix[test_global_idx]
    train_labels = labels_all[train_global_idx]
    test_labels = labels_all[test_global_idx]

    print(
        f"Prepared: train rna {train_rna.shape}, train protein {train_protein.shape}"
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

    stage_a_train_loader, stage_a_val_loader = get_dataloaders(
        rna_pca=train_rna,
        protein_clr=train_protein,
        labels=train_labels,
        train_idx=train_local_idx,
        test_idx=val_local_idx,
        batch_size=args.batch_size,
    )

    classifier_train_dataset = CITEseqDataset(
        train_rna[train_local_idx],
        train_protein[train_local_idx],
        train_labels[train_local_idx],
    )
    classifier_test_dataset = CITEseqDataset(test_rna, test_protein, test_labels)
    classifier_val_dataset = CITEseqDataset(
        train_rna[val_local_idx],
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

    rna_encoder = TransformerEncoder(
        n_tokens=n_tokens,
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
        n_train_correct = 0
        n_train_total = 0
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
            n_train_correct += (logits.argmax(dim=1) == y).sum().item()
            n_train_total += y.size(0)

        train_loss = total_train_loss / max(1, n_train_batches)
        train_accuracy = n_train_correct / max(1, n_train_total)
        val_loss, y_val_true, y_val_pred, y_val_proba = evaluate_classifier_epoch(
            classifier_val_loader, rna_encoder, protein_encoder, classifier, device
        )
        val_metrics = compute_metrics(y_val_true, y_val_pred, y_val_proba, n_classes)
        stage_b_history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
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

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"contrastive_tf_gene_seed{args.seed}_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the ``n_pathways`` key under its existing name so the generic
    # encoder-loader in src/attention_graph.py (and attention_analysis.py)
    # continues to work without a branch on tokenization.
    torch.save(
        {
            "stage": "contrastive_pretrain",
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val),
            "n_pathways": n_tokens,
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
            "n_pathways": n_tokens,
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
        json.dump({int(v): str(k) for k, v in label_mapping.items()}, f, indent=2)
    # attention_graph.py keys off gene_names.json vs pathway_names.json to pick tokenization.
    with open(run_dir / "gene_names.json", "w", encoding="utf-8") as f:
        json.dump(gene_names, f, indent=2)
    with open(run_dir / "protein_names.json", "w", encoding="utf-8") as f:
        json.dump([str(n) for n in protein_adata.var_names], f, indent=2)
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

    with torch.no_grad():
        rna_encoder.train(False)
        protein_encoder.train(False)
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
    print(f"Final test accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final macro AUROC:   {final_metrics['macro_auroc']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train gene-token contrastive transformer dual-encoder")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--label_col", type=str, default="cell_type")
    parser.add_argument("--donor_col", type=str, default="DonorNumber")
    parser.add_argument("--test_donors", type=str, nargs="*", default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--n_hvgs", type=int, default=512,
                        help="Number of HVGs to tokenize per gene. 512 → seq_len 513 with CLS.")
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
    parser.add_argument(
        "--max_cells", type=int, default=None,
        help="Randomly subsample to this many cells after loading (for smoke tests).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
