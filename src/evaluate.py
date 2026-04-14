"""evaluate.py — Evaluation metrics and paper figure generator for CITE-seq models."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy as _scipy_entropy
from scipy.stats import ranksums
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "baseline": "#4C72B0",
    "mlp": "#DD8452",
    "tf": "#55A868",
}
MODEL_DISPLAY_NAMES = {
    "baseline": "RNA Baseline",
    "mlp": "Contrastive MLP",
    "tf": "Contrastive TF",
}

# ── Metric Functions ──────────────────────────────────────────────────────────


def compute_auroc(y_true: np.ndarray, y_pred_proba: np.ndarray, n_classes: int) -> float:
    """Macro-averaged AUC-ROC (one-vs-rest). n_classes kept for API compatibility."""
    n_unique = len(np.unique(y_true))
    if n_unique == 2:
        # Binary case: roc_auc_score expects 1-D scores (positive class proba)
        scores = y_pred_proba[:, 1] if y_pred_proba.ndim == 2 else y_pred_proba
        return float(roc_auc_score(y_true, scores))
    return float(roc_auc_score(y_true, y_pred_proba, average="macro", multi_class="ovr"))


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict]:
    """Overall accuracy + per-class classification report as dict."""
    overall = float(accuracy_score(y_true, y_pred))
    per_class = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return overall, per_class


def compute_asw(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette width normalized to [0, 1]. Subsamples to 10k if larger."""
    if len(embeddings) > 10_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(embeddings), size=10_000, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
    raw = silhouette_score(embeddings, labels)
    return float((raw + 1) / 2)


def compute_recall_at_k(
    z_rna: np.ndarray,
    z_protein: np.ndarray,
    k_values: list[int] | None = None,
) -> dict:
    """
    Cross-modal retrieval recall@k. Both inputs must be L2-normalized.
    For each RNA embedding, checks if the paired protein embedding (same index)
    is within the top-k most similar protein embeddings by cosine similarity.
    """
    if k_values is None:
        k_values = [10, 20, 30, 40, 50]
    sim = z_rna @ z_protein.T  # (n, n) — cosine sim for L2-normed inputs
    results: dict = {}
    for k in k_values:
        top_k = np.argsort(sim, axis=1)[:, -k:]  # (n, k) highest-similarity indices
        hits = np.array([i in top_k[i] for i in range(len(z_rna))])
        results[k] = float(hits.mean())
    return results


def plot_phate(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """PHATE scatter plot colored by cell type. Subsamples to 20k if larger."""
    import phate  # optional dep; imported lazily to avoid hard dependency at module load

    if len(embeddings) > 20_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(embeddings), size=20_000, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    phate_op = phate.PHATE(n_components=2, n_jobs=1, verbose=False)
    coords = phate_op.fit_transform(embeddings)

    unique_labels = sorted(set(labels))
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(unique_labels), 1))
    label_to_color = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 7))
    for lbl in unique_labels:
        mask = np.array(labels) == lbl
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[label_to_color[lbl]], label=str(lbl), alpha=0.4, s=4,
        )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6, markerscale=3)
    ax.set_title(title)
    ax.set_xlabel("PHATE 1")
    ax.set_ylabel("PHATE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def compute_batch_entropy(
    embeddings: np.ndarray,
    batch_labels: np.ndarray,
    n_neighbors: int = 50,
) -> float:
    """
    Mean entropy of batch label distribution over k-NN neighborhoods.
    High entropy → good batch mixing (no batch leakage in embeddings).
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine", n_jobs=-1)
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    indices = indices[:, 1:]  # exclude self

    unique_batches = np.unique(batch_labels)
    batch_to_idx = {b: i for i, b in enumerate(unique_batches)}

    entropies: list[float] = []
    for i in range(len(embeddings)):
        neighbor_batch_ids = np.array([batch_to_idx[b] for b in batch_labels[indices[i]]])
        counts = np.bincount(neighbor_batch_ids, minlength=len(unique_batches)).astype(float)
        if counts.sum() > 0:
            entropies.append(float(_scipy_entropy(counts / counts.sum())))
    return float(np.mean(entropies)) if entropies else 0.0


def run_significance_test(
    scores_model1: list[float],
    scores_model2: list[float],
) -> float:
    """Wilcoxon rank-sum test p-value between two lists of metric scores."""
    return float(ranksums(scores_model1, scores_model2).pvalue)
