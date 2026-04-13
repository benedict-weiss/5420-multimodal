# Contrastive Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Model 3 — a two-stage contrastive transformer with pathway/protein token encoders and attention weight extraction for biological interpretability.

**Architecture:** Two `TransformerEncoder` instances (RNA pathway tokens ~300-d, protein tokens 134-d) pretrained with CLIP-style contrastive loss (Stage A), then frozen while a `ClassificationHead` is trained on concatenated embeddings (Stage B). After Stage B, CLS attention weights are extracted and saved for downstream analysis.

**Tech Stack:** PyTorch 2.2.2, `nn.TransformerEncoderLayer` subclass for attention capture, seaborn for visualization, existing `CLIPLoss`/`ClassificationHead`/`CITEseqDataset`/`get_dataloaders`/`build_pathway_tokens` reused unchanged.

---

## File Map

| File | Action |
|------|--------|
| `src/models/transformer_encoder.py` | Implement (replace stub) |
| `tests/test_transformer_encoder.py` | Create new |
| `src/train_contrastive_tf.py` | Implement (replace stub) |
| `src/attention_analysis.py` | Implement (replace stub) |

---

## Task 1: Write failing tests for TransformerEncoder

**Files:**
- Create: `tests/test_transformer_encoder.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for TransformerEncoder — output shape, L2 norm, attention weights."""

import pytest
import torch
from src.models.transformer_encoder import TransformerEncoder


@pytest.fixture
def small_rna_enc():
    return TransformerEncoder(
        n_tokens=20, d_model=16, nhead=2, num_layers=2,
        dim_feedforward=32, dropout=0.0, output_dim=8,
    )


@pytest.fixture
def small_protein_enc():
    return TransformerEncoder(
        n_tokens=10, d_model=16, nhead=2, num_layers=2,
        dim_feedforward=32, dropout=0.0, output_dim=8,
    )


def test_output_shape_rna(small_rna_enc):
    x = torch.randn(4, 20)
    out = small_rna_enc(x)
    assert out.shape == (4, 8), f"Expected (4, 8), got {out.shape}"


def test_output_shape_protein(small_protein_enc):
    x = torch.randn(4, 10)
    out = small_protein_enc(x)
    assert out.shape == (4, 8), f"Expected (4, 8), got {out.shape}"


def test_l2_normalization(small_rna_enc):
    x = torch.randn(4, 20)
    out = small_rna_enc(x)
    norms = out.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), \
        f"Output not unit-normalized. Norms: {norms}"


def test_attention_none_before_forward():
    enc = TransformerEncoder(
        n_tokens=20, d_model=16, nhead=2, num_layers=1,
        dim_feedforward=32, dropout=0.0, output_dim=8,
    )
    assert enc.get_attention_weights() is None


def test_attention_weights_shape(small_rna_enc):
    x = torch.randn(4, 20)
    _ = small_rna_enc(x)
    attn = small_rna_enc.get_attention_weights()
    assert attn is not None
    assert attn.shape == (4, 20), f"Expected (4, 20), got {attn.shape}"


def test_attention_weights_nonnegative(small_rna_enc):
    x = torch.randn(4, 20)
    _ = small_rna_enc(x)
    attn = small_rna_enc.get_attention_weights()
    assert (attn >= 0).all(), "Attention weights contain negative values"


def test_attention_weights_sum_leq_one(small_rna_enc):
    """CLS-to-token slice sums to <= 1.0 (CLS-to-CLS weight is absorbed)."""
    x = torch.randn(4, 20)
    _ = small_rna_enc(x)
    attn = small_rna_enc.get_attention_weights()
    row_sums = attn.sum(dim=1)
    assert (row_sums <= 1.0 + 1e-5).all(), f"Row sums exceed 1.0: {row_sums}"


def test_clip_loss_smoke(small_rna_enc, small_protein_enc):
    """Two TransformerEncoders + CLIPLoss should produce a finite scalar loss."""
    from src.models.contrastive_loss import CLIPLoss
    loss_fn = CLIPLoss(temperature=0.07)
    z_rna = small_rna_enc(torch.randn(4, 20))
    z_protein = small_protein_enc(torch.randn(4, 10))
    loss = loss_fn(z_rna, z_protein)
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
```

- [ ] **Step 2: Run tests — expect ImportError**

```
pytest tests/test_transformer_encoder.py -v
```

Expected: `ImportError: cannot import name 'TransformerEncoder'`

- [ ] **Step 3: Commit the failing tests**

```
git add tests/test_transformer_encoder.py
git commit -m "test: add failing tests for TransformerEncoder"
```

---

## Task 2: Implement TransformerEncoder

**Files:**
- Modify: `src/models/transformer_encoder.py` (replace stub)

- [ ] **Step 1: Replace the stub**

The key design: `AttentionTransformerEncoderLayer` subclasses `nn.TransformerEncoderLayer`
and overrides `_sa_block` (PyTorch 2.x private method) to call `self_attn` with
`need_weights=True, average_attn_weights=False`. `enable_nested_tensor=False` on the
`nn.TransformerEncoder` disables the C++ fast path so `_sa_block` is always invoked.

```python
"""Transformer encoder with CLS token for pathway/protein token inputs."""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    TransformerEncoderLayer that captures per-head attention weights.

    Overrides _sa_block (PyTorch 2.x) to request need_weights=True from
    nn.MultiheadAttention. Everything else is inherited unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_attn_weights: Optional[torch.Tensor] = None

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        x, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        self._last_attn_weights = attn_weights.detach()  # (batch, nhead, seq_len, seq_len)
        return self.dropout1(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with learnable CLS token for token-level inputs.

    Each input is a 1-D vector of scalar token values (e.g. mean pathway expression
    or CLR protein levels). Tokens are projected to d_model, a CLS token is
    prepended, and the CLS output produces a fixed-size cell embedding.

    Args:
        n_tokens:        Number of input tokens (pathways or proteins).
        d_model:         Internal transformer dimension.
        nhead:           Number of attention heads. Must divide d_model evenly.
        num_layers:      Number of encoder layers stacked.
        dim_feedforward: FFN hidden dimension within each layer.
        dropout:         Dropout rate.
        output_dim:      Final L2-normalized embedding dimension.
    """

    def __init__(
        self,
        n_tokens: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_dim: int = 128,
    ):
        super().__init__()
        self.n_tokens = n_tokens

        self.input_proj = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoding = nn.Parameter(torch.randn(1, n_tokens + 1, d_model))

        layer = AttentionTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # ensure _sa_block is always called
        )
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token values of shape (batch, n_tokens).

        Returns:
            L2-normalized embeddings of shape (batch, output_dim).
        """
        batch = x.size(0)
        # (batch, n_tokens) -> (batch, n_tokens, 1) -> (batch, n_tokens, d_model)
        x = self.input_proj(x.unsqueeze(-1))
        # Prepend CLS: (batch, n_tokens+1, d_model)
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_encoding
        x = self.encoder(x)
        cls_out = x[:, 0, :]
        return F.normalize(self.output_proj(cls_out), p=2, dim=1)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        CLS-to-token attention from the last encoder layer, averaged over heads.

        Returns:
            Tensor (batch, n_tokens), or None if called before any forward pass.
        """
        last_layer: AttentionTransformerEncoderLayer = self.encoder.layers[-1]
        if last_layer._last_attn_weights is None:
            return None
        # (batch, nhead, seq_len, seq_len) -> mean over heads -> CLS row, skip CLS-to-CLS
        attn_avg = last_layer._last_attn_weights.mean(dim=1)  # (batch, seq_len, seq_len)
        return attn_avg[:, 0, 1:]  # (batch, n_tokens)
```

- [ ] **Step 2: Run tests — expect all to pass**

```
pytest tests/test_transformer_encoder.py -v
```

Expected: 8 tests PASSED.

- [ ] **Step 3: Commit**

```
git add src/models/transformer_encoder.py
git commit -m "feat: implement TransformerEncoder with attention weight extraction"
```

---

## Task 3: Implement train_contrastive_tf.py

**Files:**
- Modify: `src/train_contrastive_tf.py` (replace stub)

Near-copy of `train_contrastive_mlp.py`. Key differences:
- Imports: `TransformerEncoder` instead of `MLPEncoder`; add `build_pathway_tokens`; remove PCA imports
- Data prep: `build_pathway_tokens(rna_adata)` on full dataset (no leakage — KEGG gene sets are global)
- Pathway matrix passed as `rna_pca` arg in dataloaders — stored under `batch["rna"]` key, no loop changes needed
- Encoder constructors use `TransformerEncoder(n_tokens=n_pathways/n_proteins, ...)`
- After Stage B: `extract_attention()` saves `tf_attention_rna.npy`, `tf_attention_protein.npy`, `tf_attention_labels.npy`
- `parse_args()` has transformer hypers (`--d_model`, `--nhead`, `--num_layers`, `--dim_feedforward`, `--dropout`) instead of `--rna_pca_dim`/`--hidden_dim`; default `--batch_size 256`

- [ ] **Step 1: Write the full file**

Full file — paste as `src/train_contrastive_tf.py`:

```python
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
            attn_rna_all.append(a_rna.cpu().numpy())
        if a_protein is not None:
            attn_protein_all.append(a_protein.cpu().numpy())
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
    print("Building pathway tokens (KEGG_2021_Human)...")
    pathway_matrix, pathway_names = build_pathway_tokens(rna_adata)
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
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
```

- [ ] **Step 2: Import smoke test**

```
python -c "import src.train_contrastive_tf; print('Import OK')"
```

Expected: `Import OK`

- [ ] **Step 3: Commit**

```
git add src/train_contrastive_tf.py
git commit -m "feat: implement train_contrastive_tf.py two-stage transformer training"
```

---

## Task 4: Implement attention_analysis.py

**Files:**
- Modify: `src/attention_analysis.py` (replace stub)

- [ ] **Step 1: Replace stub with full implementation**

```python
"""Attention heatmaps and biological validation for the contrastive transformer."""

from __future__ import annotations
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


def aggregate_attention_by_cell_type(
    attention_weights: np.ndarray,
    labels: np.ndarray,
    label_names: dict,
) -> dict:
    """
    Compute mean per-token attention for each cell type.

    Args:
        attention_weights: (n_cells, n_tokens) from TransformerEncoder.get_attention_weights().
        labels:            (n_cells,) integer labels (matching label_mapping.json keys).
        label_names:       {int_label: str_name} from label_mapping.json.
                           Keys may be ints or int-castable strings.

    Returns:
        {cell_type_str: np.ndarray(n_tokens,)} — mean attention per token.
    """
    result: dict = {}
    for label_int, label_str in label_names.items():
        mask = labels == int(label_int)
        if mask.sum() > 0:
            result[label_str] = attention_weights[mask].mean(axis=0)
    return result


def get_top_tokens(
    attention_by_type: dict,
    token_names: list,
    top_k: int = 10,
) -> dict:
    """
    For each cell type, return the top-k most attended tokens.

    Args:
        attention_by_type: {cell_type: np.ndarray(n_tokens,)}.
        token_names:       List of token name strings (length == n_tokens).
        top_k:             Number of top tokens to return per cell type.

    Returns:
        {cell_type: [(token_name, attention_score), ...]} sorted descending by score.
    """
    result: dict = {}
    for cell_type, attn in attention_by_type.items():
        top_indices = np.argsort(attn)[::-1][:top_k]
        result[cell_type] = [(token_names[i], float(attn[i])) for i in top_indices]
    return result


def validate_against_markers(
    top_tokens_dict: dict,
    expected_markers: dict,
) -> dict:
    """
    Check how many known biological markers appear in each cell type's top-k tokens.

    Args:
        top_tokens_dict:  {cell_type: [(name, score), ...]} from get_top_tokens.
        expected_markers: {cell_type: [marker_names]}, e.g.:
                            {'HSC': ['CD34', 'CD38'],
                             'B cell': ['CD19', 'CD20'],
                             'CD4 T': ['CD3', 'CD4'],
                             'CD8 T': ['CD3', 'CD8']}
                          Only cell types present in both dicts are evaluated.

    Returns:
        {cell_type: {'found': [...], 'missing': [...], 'precision': float}}
    """
    result: dict = {}
    for cell_type, top_tokens in top_tokens_dict.items():
        if cell_type not in expected_markers:
            continue
        expected = expected_markers[cell_type]
        top_names = {name for name, _ in top_tokens}
        found = [m for m in expected if m in top_names]
        missing = [m for m in expected if m not in top_names]
        precision = len(found) / len(expected) if expected else 0.0
        result[cell_type] = {"found": found, "missing": missing, "precision": precision}
    return result


def plot_attention_heatmap(
    attention_by_type: dict,
    token_names: list,
    title: str,
    save_path: str,
    top_n: int = 20,
) -> None:
    """
    Seaborn heatmap of mean attention: rows = cell types, cols = tokens.

    When n_tokens > top_n, filters to the top_n tokens by mean attention
    across all cell types (prevents illegible x-axis for RNA ~300 pathways).

    Args:
        attention_by_type: {cell_type: np.ndarray(n_tokens,)}.
        token_names:       List of token name strings.
        title:             Plot title string.
        save_path:         Output file path (PNG recommended).
        top_n:             Maximum tokens to display (applied when n_tokens > top_n).
    """
    cell_types = list(attention_by_type.keys())
    matrix = np.array([attention_by_type[ct] for ct in cell_types])  # (n_types, n_tokens)

    if len(token_names) > top_n:
        avg_attn = matrix.mean(axis=0)
        top_indices = np.argsort(avg_attn)[::-1][:top_n]
        matrix = matrix[:, top_indices]
        token_names_plot = [token_names[i] for i in top_indices]
    else:
        token_names_plot = list(token_names)

    fig_w = max(12, len(token_names_plot) * 0.5)
    fig_h = max(5, len(cell_types) * 0.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        matrix,
        xticklabels=token_names_plot,
        yticklabels=cell_types,
        cmap="viridis",
        ax=ax,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("Token")
    ax.set_ylabel("Cell type")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_token_attention_per_cell_type(
    attention_weights: np.ndarray,
    token_names: list,
    cell_type_labels: np.ndarray,
    selected_types: list,
    save_path: str,
    top_n: int = 20,
) -> None:
    """
    Violin plot of attention distribution per token for selected cell types.

    Useful for seeing whether attention is focused (peaked) or diffuse.
    Requires pandas.

    Args:
        attention_weights: (n_cells, n_tokens) float array.
        token_names:       List of token name strings.
        cell_type_labels:  (n_cells,) string array of cell type names.
        selected_types:    Cell type names to include.
        save_path:         Output file path.
        top_n:             Show only the top_n tokens by mean attention over selected types.
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError("pandas is required for plot_token_attention_per_cell_type")
    import pandas as pd

    sel_mask = np.isin(cell_type_labels, selected_types)
    if sel_mask.sum() == 0:
        raise ValueError(f"No cells found for selected_types={selected_types}")

    avg_attn = attention_weights[sel_mask].mean(axis=0)
    top_indices = np.argsort(avg_attn)[::-1][:top_n]
    top_names = [token_names[i] for i in top_indices]

    rows = []
    for ct in selected_types:
        ct_mask = cell_type_labels == ct
        if ct_mask.sum() == 0:
            continue
        for rank, idx in enumerate(top_indices):
            for val in attention_weights[ct_mask, idx]:
                rows.append({"cell_type": ct, "token": top_names[rank], "attention": float(val)})
    df = pd.DataFrame(rows)

    fig_w = max(12, len(top_names) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    sns.violinplot(data=df, x="token", y="attention", hue="cell_type", ax=ax, cut=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_title("Attention distribution by token and cell type")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 2: Import smoke test**

```
python -c "from src.attention_analysis import aggregate_attention_by_cell_type, get_top_tokens, validate_against_markers, plot_attention_heatmap, plot_token_attention_per_cell_type; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```
git add src/attention_analysis.py
git commit -m "feat: implement attention_analysis.py for transformer interpretability"
```

---

## Task 5: Final verification

- [ ] **Step 1: Full test suite**

```
pytest tests/ -v
```

Expected: All tests passing (preprocessing suite + 8 transformer encoder tests).

- [ ] **Step 2: Import smoke test for both scripts**

```
python -c "import src.train_contrastive_tf; import src.attention_analysis; print('All imports OK')"
```

Expected: `All imports OK`

- [ ] **Step 3: Commit any fixes**

```
git add -p
git commit -m "fix: resolve issues found during final verification"
```
