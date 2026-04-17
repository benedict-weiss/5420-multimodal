"""Attention heatmaps and biological validation for the contrastive transformer."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


def _resolve_data_file(data_path: str) -> Path:
    path = Path(data_path)
    if path.is_file():
        return path
    candidates = sorted(list(path.glob("*.h5ad")) + list(path.glob("*.h5ad.gz")))
    if not candidates:
        raise FileNotFoundError(f"No .h5ad or .h5ad.gz files found under {data_path}")
    return candidates[0]


def _load_anndata(data_path: str):
    import anndata as ad

    path = _resolve_data_file(data_path)
    if path.suffix != ".gz":
        return ad.read_h5ad(path)

    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with gzip.open(path, "rb") as f_in:
            with open(tmp_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return ad.read_h5ad(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _clr_normalize(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32) + 1.0
    log_matrix = np.log(matrix)
    return log_matrix - log_matrix.mean(axis=1, keepdims=True)


def _extract_protein_attention_per_head_from_checkpoint(
    checkpoint_dir: Path,
    data_path: str,
) -> np.ndarray:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    from src.models.transformer_encoder import TransformerEncoder

    stage_a = torch.load(checkpoint_dir / "stage_a_best.pt", map_location="cpu")
    train_args = stage_a["args"]
    adata = _load_anndata(data_path)
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()

    if train_args.get("max_cells") is not None and train_args["max_cells"] < adata.shape[0]:
        rng = np.random.default_rng(train_args["seed"])
        idx = np.sort(rng.choice(adata.shape[0], size=train_args["max_cells"], replace=False))
        adata = adata[idx].copy()

    label_encoder = LabelEncoder()
    labels_all = label_encoder.fit_transform(adata.obs[train_args["label_col"]].values)

    if train_args.get("test_donors"):
        donors = adata.obs[train_args["donor_col"]].values
        test_idx = np.flatnonzero(np.isin(donors, train_args["test_donors"]))
    else:
        _, test_idx = train_test_split(
            np.arange(adata.shape[0]),
            test_size=train_args["test_size"],
            random_state=train_args["seed"],
            stratify=None if train_args.get("max_cells") is not None else labels_all,
        )
    test_idx = np.asarray(test_idx)
    if test_idx.size == 0:
        raise ValueError("Checkpoint test split resolved to zero cells during per-head extraction.")

    from src.preprocessing import preprocess_protein
    protein_adata = adata[:, adata.var["feature_types"] == "ADT"].copy()
    test_protein = preprocess_protein(protein_adata[test_idx].copy())

    protein_encoder = TransformerEncoder(
        n_tokens=stage_a["n_proteins"],
        d_model=train_args["d_model"],
        nhead=train_args["nhead"],
        num_layers=train_args["num_layers"],
        dim_feedforward=train_args["dim_feedforward"],
        dropout=train_args["dropout"],
        output_dim=train_args["embedding_dim"],
    )
    protein_encoder.load_state_dict(stage_a["protein_encoder_state_dict"])
    protein_encoder.eval()

    per_head_batches: list[np.ndarray] = []
    batch_size = int(train_args.get("batch_size", 256))
    for start in range(0, test_protein.shape[0], batch_size):
        batch = torch.tensor(test_protein[start:start + batch_size], dtype=torch.float32)
        with torch.no_grad():
            _ = protein_encoder(batch)
        per_head = protein_encoder.get_attention_weights_per_head()
        if per_head is None:
            continue
        per_head_batches.append(
            np.stack(
                [
                    per_head[layer_name].cpu().numpy()
                    for layer_name in sorted(
                        per_head,
                        key=lambda name: int(name.removeprefix("layer")),
                    )
                ],
                axis=1,
            )
        )

    if not per_head_batches:
        raise ValueError("Per-head protein attention extraction produced no batches.")

    attn = np.concatenate(per_head_batches, axis=0)
    np.save(checkpoint_dir / "tf_attention_protein_per_head.npy", attn)
    print(f"Saved: {checkpoint_dir / 'tf_attention_protein_per_head.npy'}")
    return attn


def reduce_per_head_attention(attention_per_head: np.ndarray, reduction: str) -> np.ndarray:
    """
    Reduce (cells, layers, heads, tokens) attention to (cells, tokens).

    The reduction is taken across both layers and heads so the output captures
    either the overall average behavior or the strongest single-head signal.
    """
    if attention_per_head.ndim != 4:
        raise ValueError(
            f"Expected per-head attention with 4 dims, got shape {attention_per_head.shape}"
        )
    if reduction == "mean":
        return attention_per_head.mean(axis=(1, 2))
    if reduction == "max":
        return attention_per_head.max(axis=(1, 2))
    raise ValueError(f"Unsupported reduction: {reduction}")


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


def resolve_marker_alias(marker_name: str, token_names: list[str]) -> Optional[str]:
    """
    Map a canonical marker name to the concrete token name used by the ADT panel.

    Prefers exact matches, then common panel suffix forms like CD4-1.
    Returns None when no plausible token exists.
    """
    if marker_name in token_names:
        return marker_name

    prefix_matches = [name for name in token_names if name.startswith(f"{marker_name}-")]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        return sorted(prefix_matches)[0]

    return None


def validate_against_markers(
    top_tokens_dict: dict,
    expected_markers: dict,
    token_names: Optional[list[str]] = None,
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
        token_names:      Optional full token list. When provided, canonical marker
                          names are resolved to concrete panel aliases like CD4-1
                          before scoring.

    Returns:
        {cell_type: {'found': [...], 'missing': [...], 'recall': float}}
    """
    result: dict = {}
    for cell_type, top_tokens in top_tokens_dict.items():
        if cell_type not in expected_markers:
            continue
        expected = expected_markers[cell_type]
        top_names = {name for name, _ in top_tokens}
        found = []
        missing = []
        for marker in expected:
            resolved = resolve_marker_alias(marker, token_names) if token_names is not None else marker
            if resolved is None:
                missing.append(marker)
                continue
            if resolved in top_names:
                found.append(marker)
            else:
                missing.append(marker)
        recall = len(found) / len(expected) if expected else 0.0
        result[cell_type] = {"found": found, "missing": missing, "recall": recall}
    return result


def compute_marker_ranks(
    attention_by_type: dict,
    expected_markers: dict,
    token_names: list[str],
) -> dict:
    """
    For each (cell type, expected marker), report rank (1 = most attended).

    Returns: {cell_type: {marker: {"resolved": str|None, "rank": int|None,
                                    "n_tokens": int, "percentile": float|None}}}
    Percentile is (n_tokens - rank + 1) / n_tokens; 1.0 means top, 0.0 means bottom.
    """
    result: dict = {}
    for cell_type, attn in attention_by_type.items():
        if cell_type not in expected_markers:
            continue
        order = np.argsort(attn)[::-1]
        rank_of_idx = {int(idx): r + 1 for r, idx in enumerate(order)}
        n = len(token_names)
        entries: dict = {}
        for marker in expected_markers[cell_type]:
            resolved = resolve_marker_alias(marker, token_names)
            if resolved is None or resolved not in token_names:
                entries[marker] = {"resolved": None, "rank": None,
                                   "n_tokens": n, "percentile": None}
                continue
            idx = token_names.index(resolved)
            rank = rank_of_idx[idx]
            entries[marker] = {
                "resolved": resolved,
                "rank": rank,
                "n_tokens": n,
                "percentile": (n - rank + 1) / n,
            }
        result[cell_type] = entries
    return result


def compute_specificity_scores(attention_by_type: dict) -> dict:
    """
    Z-score attention per token across cell types.

    High z means the cell type attends this token unusually much vs other types —
    a better proxy for 'marker' than raw attention, which is biased toward
    globally-interesting tokens like CD45.

    Returns same shape as input: {cell_type: np.ndarray(n_tokens,)}.
    """
    cell_types = list(attention_by_type.keys())
    matrix = np.array([attention_by_type[ct] for ct in cell_types])  # (types, tokens)
    mu = matrix.mean(axis=0, keepdims=True)
    sigma = matrix.std(axis=0, keepdims=True) + 1e-8
    z = (matrix - mu) / sigma
    return {ct: z[i] for i, ct in enumerate(cell_types)}


def best_rank_across_heads(
    attention_per_head: np.ndarray,   # (cells, layers, heads, tokens)
    labels: np.ndarray,
    label_names: dict,
    expected_markers: dict,
    token_names: list[str],
) -> dict:
    """
    For each (cell_type, marker), find the (layer, head) whose cell-type-mean
    attention gives that marker the best (lowest) rank, and report it.

    Returns:
      {cell_type: {marker: {"resolved": str|None,
                             "best_rank": int|None,
                             "best_layer": int|None,
                             "best_head": int|None,
                             "n_tokens": int}}}
    """
    if attention_per_head.ndim != 4:
        raise ValueError(f"Expected 4-d per-head attn, got {attention_per_head.shape}")
    n_cells, n_layers, n_heads, n_tokens = attention_per_head.shape
    result: dict = {}
    for label_int, label_str in label_names.items():
        if label_str not in expected_markers:
            continue
        mask = labels == int(label_int)
        if mask.sum() == 0:
            continue
        # (layers, heads, tokens) mean attention for this cell type
        mean_attn = attention_per_head[mask].mean(axis=0)
        entries: dict = {}
        for marker in expected_markers[label_str]:
            resolved = resolve_marker_alias(marker, token_names)
            if resolved is None or resolved not in token_names:
                entries[marker] = {"resolved": None, "best_rank": None,
                                   "best_layer": None, "best_head": None,
                                   "n_tokens": n_tokens}
                continue
            tok_idx = token_names.index(resolved)
            best_rank = None
            best_lh = (None, None)
            for layer in range(n_layers):
                for head in range(n_heads):
                    row = mean_attn[layer, head]
                    # rank of tok_idx: 1 + count of tokens with strictly greater attn
                    rank = 1 + int((row > row[tok_idx]).sum())
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_lh = (layer, head)
            entries[marker] = {
                "resolved": resolved, "best_rank": best_rank,
                "best_layer": best_lh[0], "best_head": best_lh[1],
                "n_tokens": n_tokens,
            }
        result[label_str] = entries
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
    if len(cell_types) == 0:
        raise ValueError("attention_by_type dict is empty. No cell types to plot.")
    matrix = np.array([attention_by_type[ct] for ct in cell_types])  # (n_types, n_tokens)
    if matrix.shape[1] != len(token_names):
        raise ValueError(
            f"token_names length {len(token_names)} != attention matrix width {matrix.shape[1]}"
        )

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
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_celltype_top_heatmap(
    attention_by_type: dict,
    token_names: list,
    title: str,
    save_path: str,
    top_k_per_row: int = 5,
) -> None:
    """
    Heatmap of attention restricted to the union of each cell type's top-K tokens.

    Unlike plot_attention_heatmap (which filters by GLOBAL mean attention and
    hides cell-type-specific markers), this surfaces tokens that are strong
    within any single row — e.g. CD25 for T reg, CD161 for MAIT.

    Columns are sorted by descending max-per-column so the strongest per-row
    peaks cluster on the left.
    """
    cell_types = list(attention_by_type.keys())
    if len(cell_types) == 0:
        raise ValueError("attention_by_type dict is empty.")
    matrix = np.array([attention_by_type[ct] for ct in cell_types])
    if matrix.shape[1] != len(token_names):
        raise ValueError(
            f"token_names length {len(token_names)} != attention matrix width {matrix.shape[1]}"
        )

    union: set = set()
    for row in matrix:
        union.update(np.argsort(row)[::-1][:top_k_per_row].tolist())
    union_idx = np.array(sorted(union, key=lambda i: -matrix[:, i].max()))
    sub = matrix[:, union_idx]
    sub_names = [token_names[i] for i in union_idx]

    fig_w = max(12, len(sub_names) * 0.35)
    fig_h = max(5, len(cell_types) * 0.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        sub,
        xticklabels=sub_names,
        yticklabels=cell_types,
        cmap="viridis",
        ax=ax,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("Token")
    ax.set_ylabel("Cell type")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
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

    if attention_weights.shape[1] != len(token_names):
        raise ValueError(
            f"token_names length {len(token_names)} != attention matrix width {attention_weights.shape[1]}"
        )
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
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.set_title("Attention distribution by token and cell type")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _compose_rollout(attn_layers: list) -> np.ndarray:
    """
    Attention rollout (Abnar & Zuidema 2020) across layers.

    attn_layers: list of (batch, seq_len, seq_len) head-averaged attention per layer,
                 in forward order (layer 0 first).
    Returns (batch, seq_len, seq_len) rollout matrix.
    """
    batch, S, _ = attn_layers[0].shape
    rollout = np.broadcast_to(np.eye(S), (batch, S, S)).copy()
    I = np.eye(S)
    for A in attn_layers:
        A_hat = 0.5 * A + 0.5 * I
        A_hat = A_hat / A_hat.sum(-1, keepdims=True).clip(1e-8)
        rollout = np.einsum("bij,bjk->bik", A_hat, rollout)
    return rollout


def extract_rollout_attributions(
    encoder,
    data: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Run encoder over data and return CLS attention rollout attribution.

    Returns (N, n_tokens) array — each row is the rollout importance score
    of each input token for that cell's CLS representation.
    """
    encoder.eval()
    all_rollout: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, data.shape[0], batch_size):
            batch = torch.tensor(
                data[start : start + batch_size], dtype=torch.float32, device=device
            )
            _ = encoder(batch)
            full_attn = encoder.get_full_attention_per_layer()
            if full_attn is None:
                continue
            layers = sorted(full_attn, key=lambda k: int(k.removeprefix("layer")))
            attn_layers = [
                full_attn[l].detach().cpu().float().numpy().mean(axis=1)  # (b, S, S)
                for l in layers
            ]
            rollout = _compose_rollout(attn_layers)  # (b, S, S)
            all_rollout.append(rollout[:, 0, 1:])    # CLS row, drop CLS-to-CLS
    return np.concatenate(all_rollout, axis=0) if all_rollout else np.array([])


def extract_gradient_attention_attributions(
    protein_encoder,
    classifier,
    protein_data: np.ndarray,
    rna_embeddings: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Chefer et al. (CVPR 2021) gradient-weighted attention rollout.

    Weights each layer's attention map by ReLU(gradient of correct-class logit
    w.r.t. that attention map), then composes across layers. More reliable than
    raw attention because it reflects causal influence on the prediction.

    Uses register_hook (not retain_grad) because attention weights are non-leaf
    intermediate tensors — retain_grad() is unreliable for these in PyTorch.

    Returns (N, n_proteins) attribution array.
    """
    protein_encoder.eval()
    classifier.eval()
    protein_encoder.set_retain_attn_grad(True)

    all_grad_attn: list[np.ndarray] = []
    N = protein_data.shape[0]
    n_grad_captured = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        # requires_grad=True creates computation graph through frozen encoder
        p = torch.tensor(
            protein_data[start:end], dtype=torch.float32, device=device, requires_grad=True
        )
        r = torch.tensor(rna_embeddings[start:end], dtype=torch.float32, device=device)
        y = torch.tensor(labels[start:end], dtype=torch.long, device=device)

        z_p = protein_encoder(p)

        # Register hooks AFTER forward (attention tensors now exist) but BEFORE backward.
        # register_hook fires during backward for any tensor with a grad_fn,
        # unlike retain_grad() which fails silently for non-leaf tensors.
        full_attn = protein_encoder.get_full_attention_per_layer()
        if full_attn is None:
            all_grad_attn.append(np.zeros((end - start, protein_data.shape[1])))
            classifier(torch.cat([r, z_p], dim=1))  # consume graph
            continue

        captured_grads: dict[str, torch.Tensor] = {}
        hooks = []
        for l_name, A in full_attn.items():
            if A.grad_fn is not None:
                def _hook(grad, name=l_name):
                    captured_grads[name] = grad.detach()
                hooks.append(A.register_hook(_hook))

        logits = classifier(torch.cat([r, z_p], dim=1))
        target_logits = logits[torch.arange(end - start, device=device), y]
        target_logits.sum().backward()

        for h in hooks:
            h.remove()

        n_grad_captured += len(captured_grads)

        layers = sorted(full_attn, key=lambda k: int(k.removeprefix("layer")))
        b = end - start
        S = full_attn[layers[0]].shape[-1]
        rollout = np.broadcast_to(np.eye(S), (b, S, S)).copy()
        I = np.eye(S)

        for l in layers:
            A_np = full_attn[l].detach().cpu().float().numpy()  # (b, nhead, S, S)
            G = captured_grads.get(l)
            if G is not None:
                G_np = G.cpu().float().numpy()
                weighted = (np.maximum(G_np, 0) * A_np).mean(axis=1)  # (b, S, S)
            else:
                # Gradient not captured: fall back to plain attention for this layer
                weighted = A_np.mean(axis=1)
            A_hat = weighted + I
            A_hat = A_hat / A_hat.sum(-1, keepdims=True).clip(1e-8)
            rollout = np.einsum("bij,bjk->bik", A_hat, rollout)

        all_grad_attn.append(rollout[:, 0, 1:])  # (b, n_tokens)

    protein_encoder.set_retain_attn_grad(False)
    if N > 0:
        expected = (N // batch_size + (1 if N % batch_size else 0)) * len(
            protein_encoder.get_full_attention_per_layer() or {}
        )
        print(
            f"[grad-attn] Captured gradients for {n_grad_captured} / {expected} "
            "layer×batch combinations."
        )
    return np.concatenate(all_grad_attn, axis=0) if all_grad_attn else np.array([])


def _extract_advanced_attributions_from_checkpoint(
    checkpoint_dir: Path,
    data_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the contrastive_tf model from a checkpoint, reconstruct the test split,
    and return (rollout, grad_attn) arrays of shape (N, n_proteins).
    Both are aligned with tf_attention_labels.npy in the same checkpoint directory.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    from src.models.classifier import ClassificationHead
    from src.models.transformer_encoder import TransformerEncoder

    stage_a = torch.load(checkpoint_dir / "stage_a_best.pt", map_location="cpu")
    stage_b = torch.load(checkpoint_dir / "stage_b_best.pt", map_location="cpu")
    args = stage_a["args"]

    adata = _load_anndata(data_path)
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    if args.get("max_cells") is not None and args["max_cells"] < adata.shape[0]:
        rng = np.random.default_rng(args["seed"])
        idx = np.sort(rng.choice(adata.shape[0], size=args["max_cells"], replace=False))
        adata = adata[idx].copy()

    label_encoder = LabelEncoder()
    labels_all = label_encoder.fit_transform(adata.obs[args["label_col"]].values)

    if args.get("test_donors"):
        donors = adata.obs[args["donor_col"]].values
        test_idx = np.flatnonzero(np.isin(donors, args["test_donors"]))
    else:
        _, test_idx = train_test_split(
            np.arange(adata.shape[0]),
            test_size=args["test_size"],
            random_state=args["seed"],
            stratify=None if args.get("max_cells") is not None else labels_all,
        )
    test_idx = np.asarray(test_idx)

    from src.preprocessing import preprocess_protein
    protein_adata = adata[:, adata.var["feature_types"] == "ADT"].copy()
    test_protein = preprocess_protein(protein_adata[test_idx].copy())

    # Use saved labels (from training) to ensure correct-class gradient direction
    test_labels = np.load(checkpoint_dir / "tf_attention_labels.npy")
    assert test_labels.shape[0] == test_protein.shape[0], (
        f"Saved labels N={test_labels.shape[0]} vs reconstructed protein N={test_protein.shape[0]}. "
        "Test split mismatch — verify --data_path matches the training run."
    )

    n_classes = stage_b.get("n_classes") or stage_b["classifier_state_dict"]["net.3.weight"].shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    protein_encoder = TransformerEncoder(
        n_tokens=stage_a["n_proteins"],
        d_model=args["d_model"], nhead=args["nhead"],
        num_layers=args["num_layers"], dim_feedforward=args["dim_feedforward"],
        dropout=args["dropout"], output_dim=args["embedding_dim"],
    ).to(device).eval()
    protein_encoder.load_state_dict(stage_a["protein_encoder_state_dict"])

    classifier = ClassificationHead(
        input_dim=2 * args["embedding_dim"],
        n_classes=n_classes,
        hidden_dim=args.get("classifier_hidden_dim", 64),
        dropout=args.get("classifier_dropout", 0.2),
    ).to(device).eval()
    classifier.load_state_dict(stage_b["classifier_state_dict"])

    rna_emb = np.load(checkpoint_dir / "test_rna_embeddings.npy")

    print("Computing attention rollout (protein)...")
    rollout = extract_rollout_attributions(protein_encoder, test_protein, device)

    print("Computing gradient × attention (protein)...")
    grad_attn = extract_gradient_attention_attributions(
        protein_encoder, classifier, test_protein, rna_emb, test_labels, device
    )

    return rollout, grad_attn


# Known biological markers for validation. Keys must match BMMC cell_type labels
# in adata.obs['cell_type']. Markers must exist in the ADT panel (134 proteins;
# CD34 is notably absent, so HSC is validated via CD38 only).
_DEFAULT_MARKERS = {
    "HSC": ["CD38", "CD45RA"],
    "NK": ["CD56", "CD16", "CD335"],
    "Transitional B": ["CD24", "CD38", "IgM", "CD19"],
    "pDC": ["CD123", "CD303", "CD304"],
    "CD14+ Mono": ["CD14", "HLA-DR", "CD11b", "CD64"],
    "CD16+ Mono": ["CD16", "CX3CR1", "HLA-DR"],
    "CD4+ T naive": ["CD3", "CD4", "CD45RA", "CD62L"],
    "CD4+ T activated": ["CD3", "CD4", "CD45RO", "CD69"],
    "CD8+ T naive": ["CD3", "CD8", "CD45RA", "CD62L"],
    "Naive CD20+ B IGKC+": ["CD19", "CD20", "IgM", "IgD", "CD21"],
    "Naive CD20+ B IGKC-": ["CD19", "CD20", "IgM", "IgD", "CD21"],
    "MAIT": ["CD3", "CD8", "CD161", "TCRVa7.2"],
    "T reg": ["CD3", "CD4", "CD25", "CD127"],
    "Plasma cell IGKC+": ["CD38", "CD27"],
    "Plasma cell IGKC-": ["CD38", "CD27"],
    "cDC2": ["CD11c", "HLA-DR", "CD1c"],
    "Erythroblast": ["CD71"],
    "Normoblast": ["CD71"],
    "Reticulocyte": ["CD71"],
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyze transformer attention weights from a contrastive_tf checkpoint dir"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to a contrastive_tf run directory containing .npy attention files",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to save plots (defaults to --checkpoint_dir)",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Top-k tokens for marker validation")
    parser.add_argument("--top_n_heatmap", type=int, default=20, help="Max tokens in heatmap")
    parser.add_argument(
        "--top_k_per_row", type=int, default=5,
        help="Per-row top-K for union heatmap (protein only)",
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Fallback: path to .h5ad(.gz) to infer protein names "
             "if protein_names.json is absent in --checkpoint_dir",
    )
    parser.add_argument(
        "--head_reduction", type=str, default="max", choices=["mean", "max"],
        help="How to reduce saved per-head attention across layers/heads for the new plots",
    )
    args = parser.parse_args(argv)

    ckpt = Path(args.checkpoint_dir)
    out = Path(args.output_dir) if args.output_dir else ckpt
    out.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    attn_rna = np.load(ckpt / "tf_attention_rna.npy")
    attn_protein = np.load(ckpt / "tf_attention_protein.npy")
    labels = np.load(ckpt / "tf_attention_labels.npy")
    protein_per_head_path = ckpt / "tf_attention_protein_per_head.npy"
    if protein_per_head_path.exists():
        attn_protein_per_head = np.load(protein_per_head_path)
    elif args.data_path is not None:
        print("Per-head protein attention missing; re-extracting from stage_a_best.pt...")
        attn_protein_per_head = _extract_protein_attention_per_head_from_checkpoint(
            ckpt, args.data_path
        )
    else:
        attn_protein_per_head = None

    with open(ckpt / "label_mapping.json", encoding="utf-8") as f:
        raw = json.load(f)
    # label_mapping.json is saved as {int_idx: str_name}; JSON keys are always strings
    label_names = {int(k): v for k, v in raw.items()}

    with open(ckpt / "pathway_names.json", encoding="utf-8") as f:
        pathway_names: list[str] = json.load(f)

    n_proteins = attn_protein.shape[1]
    protein_names_path = ckpt / "protein_names.json"
    if protein_names_path.exists():
        with open(protein_names_path, encoding="utf-8") as f:
            protein_names = json.load(f)
        if len(protein_names) != n_proteins:
            raise ValueError(
                f"protein_names.json has {len(protein_names)} entries but "
                f"attention has {n_proteins} tokens"
            )
    elif args.data_path is not None:
        adata = _load_anndata(args.data_path)
        if not adata.var_names.is_unique:
            adata.var_names_make_unique()
        protein_adata = adata[:, adata.var["feature_types"] == "ADT"].copy()
        protein_names = [str(n) for n in protein_adata.var_names]
        if len(protein_names) != n_proteins:
            raise ValueError(
                f"Inferred {len(protein_names)} protein names from {args.data_path} "
                f"but attention has {n_proteins} tokens"
            )
        # Persist for future runs on this checkpoint
        with open(protein_names_path, "w", encoding="utf-8") as f:
            json.dump(protein_names, f, indent=2)
        print(f"Inferred and saved: {protein_names_path}")
    else:
        print("[warn] protein_names.json missing and --data_path not provided; "
              "falling back to generic names (marker validation will fail).")
        protein_names = [f"protein_{i}" for i in range(n_proteins)]

    print(f"Loaded: {attn_rna.shape[0]} cells, {attn_rna.shape[1]} pathways, "
          f"{n_proteins} proteins, {len(label_names)} cell types")

    # Aggregate by cell type
    attn_by_type_rna = aggregate_attention_by_cell_type(attn_rna, labels, label_names)
    attn_by_type_prot = aggregate_attention_by_cell_type(attn_protein, labels, label_names)
    attn_by_type_prot_per_head = None
    if attn_protein_per_head is not None:
        attn_protein_reduced = reduce_per_head_attention(
            attn_protein_per_head, reduction=args.head_reduction
        )
        attn_by_type_prot_per_head = aggregate_attention_by_cell_type(
            attn_protein_reduced, labels, label_names
        )

    # Heatmaps
    plot_attention_heatmap(
        attn_by_type_rna, pathway_names,
        title="RNA pathway attention by cell type",
        save_path=str(out / "attention_heatmap_rna.png"),
        top_n=args.top_n_heatmap,
    )
    print(f"Saved: {out / 'attention_heatmap_rna.png'}")

    specificity_rna = compute_specificity_scores(attn_by_type_rna)
    plot_per_celltype_top_heatmap(
        specificity_rna, pathway_names,
        title="RNA pathway attention SPECIFICITY (z-scored across cell types, per-row top-K)",
        save_path=str(out / "attention_heatmap_rna_specificity.png"),
        top_k_per_row=args.top_k_per_row,
    )
    print(f"Saved: {out / 'attention_heatmap_rna_specificity.png'}")
    plot_per_celltype_top_heatmap(
        attn_by_type_rna, pathway_names,
        title=f"RNA pathway attention by cell type (per-row top-{args.top_k_per_row}, union)",
        save_path=str(out / "attention_heatmap_rna_per_row.png"),
        top_k_per_row=args.top_k_per_row,
    )
    print(f"Saved: {out / 'attention_heatmap_rna_per_row.png'}")

    specificity = compute_specificity_scores(attn_by_type_prot)
    plot_per_celltype_top_heatmap(
        specificity, protein_names,
        title="Protein attention SPECIFICITY (z-scored across cell types, per-row top-K)",
        save_path=str(out / "attention_heatmap_protein_specificity.png"),
        top_k_per_row=args.top_k_per_row,
    )
    print(f"Saved: {out / 'attention_heatmap_protein_specificity.png'}")

    plot_attention_heatmap(
        attn_by_type_prot, protein_names,
        title="Protein attention by cell type",
        save_path=str(out / "attention_heatmap_protein.png"),
        top_n=args.top_n_heatmap,
    )
    print(f"Saved: {out / 'attention_heatmap_protein.png'}")

    plot_per_celltype_top_heatmap(
        attn_by_type_prot, protein_names,
        title=f"Protein attention by cell type (per-row top-{args.top_k_per_row}, union)",
        save_path=str(out / "attention_heatmap_protein_per_row.png"),
        top_k_per_row=args.top_k_per_row,
    )
    print(f"Saved: {out / 'attention_heatmap_protein_per_row.png'}")

    # Marker validation on protein tokens
    top_tokens = get_top_tokens(attn_by_type_prot, protein_names, top_k=args.top_k)
    validation = validate_against_markers(top_tokens, _DEFAULT_MARKERS, token_names=protein_names)
    print("\n=== Marker validation (protein) ===")
    for ct, res in validation.items():
        print(f"  {ct}: recall={res['recall']:.2f}  found={res['found']}  missing={res['missing']}")

    ranks = compute_marker_ranks(attn_by_type_prot, _DEFAULT_MARKERS, protein_names)
    top_tokens_spec = get_top_tokens(specificity, protein_names, top_k=args.top_k)
    validation_spec = validate_against_markers(
        top_tokens_spec, _DEFAULT_MARKERS, token_names=protein_names
    )
    ranks_spec = compute_marker_ranks(specificity, _DEFAULT_MARKERS, protein_names)

    enriched = {
        ct: {
            "top_10_recall": validation.get(ct, {}).get("recall"),
            "top_10_recall_specificity": validation_spec.get(ct, {}).get("recall"),
            "marker_ranks_raw": ranks.get(ct, {}),
            "marker_ranks_specificity": ranks_spec.get(ct, {}),
        }
        for ct in _DEFAULT_MARKERS
    }

    # Save validation results
    val_path = out / "marker_validation.json"
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)
    print(f"Saved: {val_path}")

    if attn_by_type_prot_per_head is not None:
        suffix = f"per_head_{args.head_reduction}"
        plot_per_celltype_top_heatmap(
            attn_by_type_prot_per_head, protein_names,
            title=(
                "Protein attention by cell type "
                f"({args.head_reduction} over per-head CLS attention)"
            ),
            save_path=str(out / f"attention_heatmap_protein_{suffix}.png"),
            top_k_per_row=args.top_k_per_row,
        )
        print(f"Saved: {out / f'attention_heatmap_protein_{suffix}.png'}")

        top_tokens_per_head = get_top_tokens(
            attn_by_type_prot_per_head, protein_names, top_k=args.top_k
        )
        validation_per_head = validate_against_markers(
            top_tokens_per_head, _DEFAULT_MARKERS, token_names=protein_names
        )
        print(f"\n=== Marker validation (protein, {args.head_reduction} over per-head) ===")
        for ct, res in validation_per_head.items():
            print(
                f"  {ct}: recall={res['recall']:.2f}  "
                f"found={res['found']}  missing={res['missing']}"
            )

        val_per_head_path = out / f"marker_validation_{suffix}.json"
        with open(val_per_head_path, "w", encoding="utf-8") as f:
            json.dump(validation_per_head, f, indent=2)
        print(f"Saved: {val_per_head_path}")

        best_rank = best_rank_across_heads(
            attn_protein_per_head, labels, label_names,
            _DEFAULT_MARKERS, protein_names,
        )
        best_rank_path = out / "marker_best_rank_per_head.json"
        with open(best_rank_path, "w", encoding="utf-8") as f:
            json.dump(best_rank, f, indent=2)
        print(f"Saved: {best_rank_path}")
        print("\n=== Best (layer, head) per marker ===")
        for ct, markers in best_rank.items():
            for m, info in markers.items():
                if info["best_rank"] is None:
                    continue
                print(f"  {ct} / {m} ({info['resolved']}): "
                      f"rank={info['best_rank']}/{info['n_tokens']} "
                      f"@layer{info['best_layer']} head{info['best_head']}")
    else:
        print(
            "[warn] Per-head protein attention unavailable; skipping per-head-aware "
            "heatmap and marker validation."
        )

    # Advanced attribution methods: rollout + gradient×attention
    if args.data_path is not None:
        print("\n=== Advanced attribution methods ===")
        try:
            rollout, grad_attn = _extract_advanced_attributions_from_checkpoint(
                ckpt, args.data_path
            )
            np.save(ckpt / "tf_attention_protein_rollout.npy", rollout)
            np.save(ckpt / "tf_attention_protein_grad_attn.npy", grad_attn)

            attn_by_type_rollout = aggregate_attention_by_cell_type(rollout, labels, label_names)
            attn_by_type_grad = aggregate_attention_by_cell_type(grad_attn, labels, label_names)

            for method_name, attn_by_type in [
                ("rollout", attn_by_type_rollout),
                ("grad_attn", attn_by_type_grad),
            ]:
                spec = compute_specificity_scores(attn_by_type)

                plot_per_celltype_top_heatmap(
                    spec, protein_names,
                    title=f"Protein {method_name} SPECIFICITY (z-scored across cell types)",
                    save_path=str(out / f"attention_heatmap_protein_{method_name}_specificity.png"),
                    top_k_per_row=args.top_k_per_row,
                )
                print(f"Saved: {out / f'attention_heatmap_protein_{method_name}_specificity.png'}")

                top_t = get_top_tokens(attn_by_type, protein_names, top_k=args.top_k)
                val = validate_against_markers(top_t, _DEFAULT_MARKERS, token_names=protein_names)
                rnks = compute_marker_ranks(attn_by_type, _DEFAULT_MARKERS, protein_names)

                print(f"\n=== Marker validation ({method_name}) ===")
                for ct, res in val.items():
                    print(
                        f"  {ct}: recall={res['recall']:.2f}  "
                        f"found={res['found']}  missing={res['missing']}"
                    )

                val_path = out / f"marker_validation_{method_name}.json"
                with open(val_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            ct: {
                                "top_k_recall": val.get(ct, {}).get("recall"),
                                "found": val.get(ct, {}).get("found"),
                                "missing": val.get(ct, {}).get("missing"),
                                "marker_ranks": rnks.get(ct, {}),
                            }
                            for ct in _DEFAULT_MARKERS
                        },
                        f, indent=2,
                    )
                print(f"Saved: {val_path}")
        except Exception as exc:
            import traceback
            print(f"[warn] Advanced attribution failed: {exc}")
            traceback.print_exc()
    else:
        print(
            "\n[info] Skipping rollout + gradient×attention (pass --data_path to enable)."
        )

    ablation_path = ckpt / "ablation_logit_drop_per_type.npy"
    order_path = ckpt / "ablation_per_type_order.json"
    if ablation_path.exists() and order_path.exists():
        abl = np.load(ablation_path)
        with open(order_path) as f:
            order = json.load(f)
        abl_by_type = {ct: abl[i] for i, ct in enumerate(order)}
        abl_ranks = compute_marker_ranks(abl_by_type, _DEFAULT_MARKERS, protein_names)
        abl_path = out / "marker_ranks_ablation.json"
        with open(abl_path, "w") as f:
            json.dump(abl_ranks, f, indent=2)
        print(f"Saved: {abl_path}")
        plot_per_celltype_top_heatmap(
            abl_by_type, protein_names,
            title="Per-protein logit drop by cell type (token ablation)",
            save_path=str(out / "attention_heatmap_protein_ablation.png"),
            top_k_per_row=args.top_k_per_row,
        )
        print(f"Saved: {out / 'attention_heatmap_protein_ablation.png'}")


if __name__ == "__main__":
    main()
