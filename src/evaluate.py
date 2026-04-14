"""evaluate.py — Evaluation metrics and paper figure generator for CITE-seq models."""

from __future__ import annotations

import argparse
import json
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


# ── Internal helpers ──────────────────────────────────────────────────────────


def _find_latest_run(checkpoint_dir: Path, prefix: str) -> Path | None:
    candidates = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    return max(candidates, key=lambda d: d.stat().st_mtime) if candidates else None


def _load_run(run_dir: Path) -> dict:
    """Load metrics.json and available .npy / .json artifacts from a run directory."""
    data: dict = {"dir": run_dir}

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            data["metrics"] = json.load(f)

    for key, fname in [
        ("test_embeddings", "test_embeddings.npy"),
        ("test_labels", "test_labels.npy"),
        ("test_rna_embeddings", "test_rna_embeddings.npy"),
        ("test_protein_embeddings", "test_protein_embeddings.npy"),
    ]:
        p = run_dir / fname
        if p.exists():
            data[key] = np.load(p)

    label_path = run_dir / "label_mapping.json"
    if label_path.exists():
        with open(label_path, encoding="utf-8") as f:
            data["label_mapping"] = json.load(f)

    return data


def _int_labels_to_strings(label_ints: np.ndarray, label_mapping: dict) -> np.ndarray:
    """Convert integer label array to string names using label_mapping.json.

    Handles both {str_name: int_idx} (baseline/mlp format) and
    {int_idx: str_name} (tf format) by inspecting value types.
    """
    if not label_mapping:
        return label_ints.astype(str)
    first_val = next(iter(label_mapping.values()))
    if isinstance(first_val, int) or (isinstance(first_val, str) and first_val.lstrip("-").isdigit()):
        # {str_name: int_idx} format — values are integers
        int_to_name = {int(v): str(k) for k, v in label_mapping.items()}
    else:
        # {int_idx: str_name} format — values are cell type strings
        int_to_name = {int(k): str(v) for k, v in label_mapping.items()}
    return np.array([int_to_name.get(int(l), str(l)) for l in label_ints])


# ── Figure A: model comparison bar chart ─────────────────────────────────────


def _fig_model_comparison(runs: dict, output_dir: Path) -> None:
    metric_keys = ["final_accuracy", "final_macro_auroc"]
    metric_labels = ["Accuracy", "Macro AUROC"]
    model_keys = [k for k in ["baseline", "mlp", "tf"] if k in runs]

    x = np.arange(len(metric_keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, mkey in enumerate(model_keys):
        metrics = runs[mkey].get("metrics", {})
        vals = [float(metrics.get(mk) or 0.0) for mk in metric_keys]
        bars = ax.bar(
            x + i * width, vals, width,
            label=MODEL_DISPLAY_NAMES[mkey], color=MODEL_COLORS[mkey],
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x + width * (len(model_keys) - 1) / 2)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Accuracy and Macro AUROC")
    ax.legend()
    plt.tight_layout()
    save_path = output_dir / "fig_model_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Figure B: PHATE plots ─────────────────────────────────────────────────────


def _fig_phate(runs: dict, output_dir: Path) -> None:
    for mkey in ["baseline", "mlp", "tf"]:
        if mkey not in runs:
            continue
        run = runs[mkey]
        if "test_embeddings" not in run or "test_labels" not in run:
            print(f"[warn] Skipping PHATE for {mkey}: test_embeddings.npy not found")
            continue
        labels = _int_labels_to_strings(run["test_labels"], run.get("label_mapping", {}))
        plot_phate(
            run["test_embeddings"], labels,
            title=f"PHATE — {MODEL_DISPLAY_NAMES[mkey]}",
            save_path=str(output_dir / f"fig_phate_{mkey}.png"),
        )


# ── Figure C: training curves ─────────────────────────────────────────────────


def _fig_training_curves(runs: dict, output_dir: Path) -> None:
    model_keys = [k for k in ["baseline", "mlp", "tf"] if k in runs]
    n_models = len(model_keys)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 4 * n_models), squeeze=False)

    for row, mkey in enumerate(model_keys):
        ax = axes[row][0]
        metrics = runs[mkey].get("metrics", {})

        if mkey == "baseline":
            history = metrics.get("history", [])
            if history:
                epochs = [h["epoch"] for h in history]
                ax.plot(epochs, [h["train_loss"] for h in history], label="train", color="tab:blue")
                ax.plot(epochs, [h["val_loss"] for h in history], label="val", color="tab:orange")
        else:
            stage_a = metrics.get("stage_a_history", [])
            stage_b = metrics.get("stage_b_history", [])
            if stage_a:
                ea = [h["epoch"] for h in stage_a]
                ax.plot(ea, [h["train_loss"] for h in stage_a], label="Stage A train", color="tab:blue")
                ax.plot(ea, [h["val_loss"] for h in stage_a], label="Stage A val",
                        color="tab:blue", linestyle="--")
            if stage_b:
                offset = stage_a[-1]["epoch"] if stage_a else 0
                eb = [offset + h["epoch"] for h in stage_b]
                ax.plot(eb, [h["train_loss"] for h in stage_b], label="Stage B train", color="tab:orange")
                ax.plot(eb, [h["val_loss"] for h in stage_b], label="Stage B val",
                        color="tab:orange", linestyle="--")
                if stage_a:
                    ax.axvline(offset, color="gray", linestyle=":", alpha=0.7, label="Stage A→B")

        ax.set_title(f"{MODEL_DISPLAY_NAMES[mkey]} — Training Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)

    plt.tight_layout()
    save_path = output_dir / "fig_training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Figure D: Recall@k ────────────────────────────────────────────────────────


def _fig_recall_at_k(runs: dict, output_dir: Path) -> None:
    k_values = [10, 20, 30, 40, 50]
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False

    for mkey in ["mlp", "tf"]:
        if mkey not in runs:
            continue
        run = runs[mkey]
        if "test_rna_embeddings" not in run or "test_protein_embeddings" not in run:
            print(f"[warn] Skipping Recall@k for {mkey}: cross-modal embeddings not found")
            continue
        recall = compute_recall_at_k(
            run["test_rna_embeddings"], run["test_protein_embeddings"], k_values=k_values
        )
        ax.plot(
            k_values, [recall[k] for k in k_values],
            marker="o", label=MODEL_DISPLAY_NAMES[mkey], color=MODEL_COLORS[mkey],
        )
        plotted = True

    if not plotted:
        print("[warn] No cross-modal embeddings found; skipping fig_recall_at_k.png")
        plt.close(fig)
        return

    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title("Cross-Modal Retrieval: Recall@k")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    save_path = output_dir / "fig_recall_at_k.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Figure E: ASW bar chart ───────────────────────────────────────────────────


def _fig_asw(runs: dict, output_dir: Path) -> None:
    asw_scores: dict[str, float] = {}
    for mkey in ["baseline", "mlp", "tf"]:
        if mkey not in runs:
            continue
        run = runs[mkey]
        if "test_embeddings" not in run or "test_labels" not in run:
            print(f"[warn] Skipping ASW for {mkey}: test_embeddings.npy not found")
            continue
        asw_scores[mkey] = compute_asw(run["test_embeddings"], run["test_labels"])
        print(f"  ASW ({MODEL_DISPLAY_NAMES[mkey]}): {asw_scores[mkey]:.4f}")

    if not asw_scores:
        print("[warn] No embeddings found; skipping fig_asw.png")
        return

    sorted_models = sorted(asw_scores, key=asw_scores.__getitem__, reverse=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(
        [MODEL_DISPLAY_NAMES[m] for m in sorted_models],
        [asw_scores[m] for m in sorted_models],
        color=[MODEL_COLORS[m] for m in sorted_models],
    )
    for bar, mkey in zip(bars, sorted_models):
        val = asw_scores[mkey]
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Normalized ASW")
    ax.set_title("Average Silhouette Width (normalized to [0,1])")
    plt.tight_layout()
    save_path = output_dir / "fig_asw.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper figures from trained model checkpoints"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="results/checkpoints",
        help="Parent directory containing model run subdirectories",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/figures",
        help="Directory to write output figures",
    )
    parser.add_argument("--baseline_dir", type=str, default=None,
                        help="Pin a specific baseline run directory (overrides auto-discovery)")
    parser.add_argument("--mlp_dir", type=str, default=None,
                        help="Pin a specific contrastive_mlp run directory")
    parser.add_argument("--tf_dir", type=str, default=None,
                        help="Pin a specific contrastive_tf run directory")
    args = parser.parse_args(argv)

    ckpt_root = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overrides = {"baseline": args.baseline_dir, "mlp": args.mlp_dir, "tf": args.tf_dir}
    prefixes = {
        "baseline": "baseline_rna_",
        "mlp": "contrastive_mlp_",
        "tf": "contrastive_tf_",
    }

    run_dirs: dict[str, Path] = {}
    for mkey, override in overrides.items():
        if override:
            p = Path(override)
            if not p.is_dir():
                print(f"[error] --{mkey}_dir does not exist: {p}")
                return
            run_dirs[mkey] = p
        elif ckpt_root.exists():
            found = _find_latest_run(ckpt_root, prefixes[mkey])
            if found:
                run_dirs[mkey] = found

    if not run_dirs:
        print(f"No model runs found under {ckpt_root}. "
              "Pass --checkpoint_dir or --baseline_dir / --mlp_dir / --tf_dir.")
        return

    print("\n=== Selected run directories ===")
    for mkey, d in run_dirs.items():
        print(f"  {MODEL_DISPLAY_NAMES[mkey]}: {d}")

    runs = {mkey: _load_run(d) for mkey, d in run_dirs.items()}

    print("\n=== Generating figures ===")
    _fig_model_comparison(runs, output_dir)
    _fig_training_curves(runs, output_dir)
    _fig_phate(runs, output_dir)
    _fig_recall_at_k(runs, output_dir)
    _fig_asw(runs, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
