"""Visualization utilities for protein token ablation results.

Generates:
1. A row-normalized heatmap of top ablation proteins across cell types.
2. Horizontal bar charts for selected cell types showing top-k ablation drops.

Inputs are the saved artifacts from ``src.attribution_ablation``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_ablation_artifacts(checkpoint_dir: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Load per-cell-type ablation matrix and metadata from a checkpoint directory."""
    matrix = np.load(checkpoint_dir / "ablation_logit_drop_per_type.npy")
    with open(checkpoint_dir / "ablation_per_type_order.json") as f:
        cell_types = json.load(f)
    with open(checkpoint_dir / "protein_names.json") as f:
        protein_names = json.load(f)
    return matrix, cell_types, protein_names


def select_top_proteins(
    matrix: np.ndarray,
    protein_names: list[str],
    top_n: int,
) -> tuple[np.ndarray, list[str]]:
    """Keep proteins with the strongest cell-type-specific ablation signal."""
    if matrix.shape[1] != len(protein_names):
        raise ValueError(
            f"Protein count mismatch: matrix has {matrix.shape[1]} columns, "
            f"but {len(protein_names)} names were provided."
        )

    # Rank by strongest positive effect in any cell type; break ties by mean effect.
    max_drop = matrix.max(axis=0)
    mean_drop = matrix.mean(axis=0)
    top_indices = np.lexsort((-mean_drop, -max_drop))[:top_n]
    top_indices = np.array(sorted(top_indices, key=lambda idx: (-max_drop[idx], -mean_drop[idx])))
    return matrix[:, top_indices], [protein_names[idx] for idx in top_indices]


def row_zscore(matrix: np.ndarray) -> np.ndarray:
    """Standardize each cell-type row to highlight relative marker preference."""
    mu = matrix.mean(axis=1, keepdims=True)
    sigma = matrix.std(axis=1, keepdims=True) + 1e-8
    return (matrix - mu) / sigma


def plot_ablation_heatmap(
    matrix: np.ndarray,
    cell_types: list[str],
    protein_names: list[str],
    save_path: Path,
    title: str,
) -> None:
    """Plot a row-normalized heatmap of ablation scores."""
    fig_w = max(12, len(protein_names) * 0.42)
    fig_h = max(8, len(cell_types) * 0.28)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        row_zscore(matrix),
        xticklabels=protein_names,
        yticklabels=cell_types,
        cmap="RdBu_r",
        center=0.0,
        cbar_kws={"label": "Row z-score of mean logit drop"},
        ax=ax,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("Protein token")
    ax.set_ylabel("Cell type")
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_top_bars(
    matrix: np.ndarray,
    cell_types: list[str],
    protein_names: list[str],
    selected_cell_types: list[str],
    save_path: Path,
    top_k: int,
) -> None:
    """Plot top-k ablation proteins for selected cell types as small multiples."""
    valid_types = [cell_type for cell_type in selected_cell_types if cell_type in cell_types]
    if not valid_types:
        raise ValueError("No selected cell types were found in ablation_per_type_order.json")

    n_panels = len(valid_types)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, max(3.8 * nrows, 4.5)),
        squeeze=False,
    )

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    cell_to_idx = {cell_type: idx for idx, cell_type in enumerate(cell_types)}
    for ax, cell_type in zip(axes.flat, valid_types):
        row = matrix[cell_to_idx[cell_type]]
        top_idx = np.argsort(row)[::-1][:top_k]
        values = row[top_idx][::-1]
        labels = [protein_names[idx] for idx in top_idx][::-1]
        colors = ["#b2182b" if value >= 0 else "#2166ac" for value in values]
        ax.barh(labels, values, color=colors)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(cell_type)
        ax.set_xlabel("Mean correct-class logit drop")
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Top ablation drivers by representative cell type", y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def choose_default_cell_types(cell_types: list[str]) -> list[str]:
    """Prefer representative immune and lineage states when present."""
    preferred = [
        "NK",
        "pDC",
        "CD8+ T naive",
        "CD14+ Mono",
        "Transitional B",
        "Proerythroblast",
    ]
    chosen = [cell_type for cell_type in preferred if cell_type in cell_types]
    if len(chosen) >= min(6, len(cell_types)):
        return chosen
    for cell_type in cell_types:
        if cell_type not in chosen:
            chosen.append(cell_type)
        if len(chosen) == min(6, len(cell_types)):
            break
    return chosen


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot ablation result summaries")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--top_n_heatmap", type=int, default=30)
    parser.add_argument("--top_k_bars", type=int, default=8)
    parser.add_argument(
        "--cell_types",
        nargs="*",
        default=None,
        help="Optional list of cell types for the bar chart panel",
    )
    args = parser.parse_args(argv)

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix, cell_types, protein_names = load_ablation_artifacts(checkpoint_dir)
    heatmap_matrix, heatmap_proteins = select_top_proteins(
        matrix=matrix,
        protein_names=protein_names,
        top_n=min(args.top_n_heatmap, matrix.shape[1]),
    )

    heatmap_path = output_dir / "ablation_heatmap_top_proteins.png"
    plot_ablation_heatmap(
        matrix=heatmap_matrix,
        cell_types=cell_types,
        protein_names=heatmap_proteins,
        save_path=heatmap_path,
        title=f"Ablation heatmap: top {len(heatmap_proteins)} protein drivers",
    )

    selected_cell_types = args.cell_types or choose_default_cell_types(cell_types)
    bars_path = output_dir / "ablation_top_bars_selected_cell_types.png"
    plot_top_bars(
        matrix=matrix,
        cell_types=cell_types,
        protein_names=protein_names,
        selected_cell_types=selected_cell_types,
        save_path=bars_path,
        top_k=min(args.top_k_bars, matrix.shape[1]),
    )

    summary = {
        "heatmap_path": str(heatmap_path),
        "bar_chart_path": str(bars_path),
        "selected_cell_types": selected_cell_types,
        "top_n_heatmap": min(args.top_n_heatmap, matrix.shape[1]),
        "top_k_bars": min(args.top_k_bars, matrix.shape[1]),
    }
    with open(output_dir / "ablation_visualization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
