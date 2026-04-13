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
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_title("Attention distribution by token and cell type")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
