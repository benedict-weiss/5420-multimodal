"""evaluate_averaged.py — Generate mean±std figures across multiple seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── NeurIPS style ──────────────────────────────────────────────────────────────
# Single column: 3.25 in  |  Double column: 6.75 in
# Body font: 10 pt → figure labels ≥ 8 pt at final print size
# Lines ≥ 1.5 pt, DPI 300, no top/right spines, PDF output

NEURIPS_RC = {
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.framealpha": 0.7,
    "legend.edgecolor": "0.8",
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

COL1 = 3.25   # single-column width (inches)
COL2 = 6.75   # double-column width (inches)

from src.evaluate import (
    ALL_MODEL_KEYS,
    MODEL_COLORS,
    MODEL_DISPLAY_NAMES,
    SINGLE_STAGE_MODELS,
    _load_run,
    compute_asw,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _find_seed_run(checkpoint_dir: Path, prefix: str, seed: int) -> Optional[Path]:
    """Return the run directory matching <prefix><seed>_ (latest mtime if >1)."""
    tag = f"seed{seed}_"
    candidates = [
        d for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.startswith(prefix) and tag in d.name
    ]
    return max(candidates, key=lambda d: d.stat().st_mtime) if candidates else None


def _pad_series(arrays: list[list[float]]) -> np.ndarray:
    """Pad lists to max length with NaN; return (n_seeds, max_len) array."""
    max_len = max(len(a) for a in arrays)
    out = np.full((len(arrays), max_len), np.nan)
    for i, a in enumerate(arrays):
        out[i, : len(a)] = a
    return out


def _mean_std(arrays: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (epochs, mean, std) with NaN-safe averaging over seeds."""
    mat = _pad_series(arrays)
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    epochs = np.arange(1, mat.shape[1] + 1)
    return epochs, mean, std


def _plot_band(ax, epochs, mean, std, color, label, linestyle="-", alpha=0.15):
    """Plot mean line with ±1 std shaded band."""
    valid = np.isfinite(mean)
    ax.plot(epochs[valid], mean[valid], color=color, linestyle=linestyle,
            label=label, linewidth=1.5)
    ax.fill_between(
        epochs[valid],
        (mean - std)[valid],
        (mean + std)[valid],
        color=color, alpha=alpha,
    )


def _save(fig, path_stem: Path) -> None:
    """Save as both PDF (vector) and PNG."""
    for ext in (".pdf", ".png"):
        fig.savefig(path_stem.with_suffix(ext))
    print(f"Saved: {path_stem.with_suffix('.pdf')} + .png")


# ── Figure: averaged model comparison ─────────────────────────────────────────


def _fig_model_comparison_avg(all_runs: dict[str, list[dict]], output_dir: Path) -> None:
    metric_keys = ["final_accuracy", "final_macro_auroc"]
    metric_labels = ["Accuracy", "Macro AUROC"]
    model_keys = [k for k in ALL_MODEL_KEYS if all_runs[k]]

    x = np.arange(len(metric_keys))
    width = 0.8 / max(len(model_keys), 1)

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots(figsize=(COL2, 2.2))

        for i, mkey in enumerate(model_keys):
            means, stds = [], []
            for mk in metric_keys:
                vals = [
                    float(r["metrics"].get(mk) or 0.0)
                    for r in all_runs[mkey]
                    if "metrics" in r
                ]
                means.append(np.mean(vals))
                stds.append(np.std(vals))

            bars = ax.bar(
                x + i * width, means, width, yerr=stds, capsize=2,
                label=MODEL_DISPLAY_NAMES[mkey], color=MODEL_COLORS[mkey],
                error_kw={"linewidth": 0.8},
            )
            for bar, m, s in zip(bars, means, stds):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.01,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=6,
                )

        ax.set_xticks(x + width * (len(model_keys) - 1) / 2)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.legend(ncol=2, loc="lower right")
        fig.tight_layout()
        _save(fig, output_dir / "fig_model_comparison_avg")
        plt.close(fig)


# ── Figure: averaged training curves ──────────────────────────────────────────


def _fig_training_curves_avg(all_runs: dict[str, list[dict]], output_dir: Path) -> None:
    model_keys = [k for k in ALL_MODEL_KEYS if all_runs[k]]
    n = len(model_keys)

    with plt.rc_context(NEURIPS_RC):
        fig, axes = plt.subplots(n, 1, figsize=(COL2, 2.0 * n), squeeze=False)

        for row, mkey in enumerate(model_keys):
            ax = axes[row][0]
            runs = all_runs[mkey]

            if mkey in SINGLE_STAGE_MODELS:
                train_lists, val_lists = [], []
                for r in runs:
                    hist = r.get("metrics", {}).get("history", [])
                    if hist:
                        train_lists.append([h["train_loss"] for h in hist])
                        val_lists.append([h["val_loss"] for h in hist])
                if train_lists:
                    ep, m, s = _mean_std(train_lists)
                    _plot_band(ax, ep, m, s, "tab:blue", "train")
                    ep, m, s = _mean_std(val_lists)
                    _plot_band(ax, ep, m, s, "tab:orange", "val", linestyle="--")
                test_losses = [
                    float(r["metrics"]["final_test_loss"])
                    for r in runs
                    if "metrics" in r and "final_test_loss" in r["metrics"]
                ]
                if test_losses:
                    ax.axhline(np.mean(test_losses), color="tab:red", linestyle=":",
                               linewidth=1.0, label=f"test ({np.mean(test_losses):.3f})")
            else:
                sa_train, sa_val, sb_train, sb_val = [], [], [], []
                a_lengths = []
                for r in runs:
                    metrics = r.get("metrics", {})
                    sa = metrics.get("stage_a_history", [])
                    sb = metrics.get("stage_b_history", [])
                    if sa:
                        sa_train.append([h["train_loss"] for h in sa])
                        sa_val.append([h["val_loss"] for h in sa])
                        a_lengths.append(len(sa))
                    if sb:
                        sb_train.append([h["train_loss"] for h in sb])
                        sb_val.append([h["val_loss"] for h in sb])

                if sa_train:
                    ep, m, s = _mean_std(sa_train)
                    _plot_band(ax, ep, m, s, "tab:blue", "Stage A train")
                    ep, m, s = _mean_std(sa_val)
                    _plot_band(ax, ep, m, s, "tab:blue", "Stage A val", linestyle="--")
                    mean_a_len = int(np.mean(a_lengths))
                    ax.axvline(mean_a_len, color="gray", linestyle=":", linewidth=0.8,
                               alpha=0.7, label="A→B")
                if sb_train:
                    mean_a_len = int(np.mean(a_lengths)) if a_lengths else 0
                    ep_raw, m, s = _mean_std(sb_train)
                    _plot_band(ax, ep_raw + mean_a_len, m, s, "tab:orange", "Stage B train")
                    ep_raw, m, s = _mean_std(sb_val)
                    _plot_band(ax, ep_raw + mean_a_len, m, s, "tab:orange", "Stage B val",
                               linestyle="--")
                test_losses = [
                    float(r["metrics"]["final_test_loss"])
                    for r in runs
                    if "metrics" in r and "final_test_loss" in r["metrics"]
                ]
                if test_losses:
                    ax.axhline(np.mean(test_losses), color="tab:red", linestyle=":",
                               linewidth=1.0, label=f"test ({np.mean(test_losses):.3f})")

            ax.set_ylabel("Loss")
            ax.set_title(MODEL_DISPLAY_NAMES[mkey], pad=3)
            ax.legend(ncol=2)
            if row == n - 1:
                ax.set_xlabel("Epoch")

        fig.tight_layout()
        _save(fig, output_dir / "fig_training_curves_avg")
        plt.close(fig)


# ── Figure: averaged accuracy curves ──────────────────────────────────────────


def _fig_accuracy_curves_avg(all_runs: dict[str, list[dict]], output_dir: Path) -> None:
    model_keys = [k for k in ALL_MODEL_KEYS if all_runs[k]]
    n = len(model_keys)

    with plt.rc_context(NEURIPS_RC):
        fig, axes = plt.subplots(n, 1, figsize=(COL2, 2.0 * n), squeeze=False)

        for row, mkey in enumerate(model_keys):
            ax = axes[row][0]
            runs = all_runs[mkey]

            if mkey in SINGLE_STAGE_MODELS:
                val_lists, train_lists = [], []
                for r in runs:
                    hist = r.get("metrics", {}).get("history", [])
                    if hist:
                        val_lists.append([h["val_accuracy"] for h in hist])
                        if "train_accuracy" in hist[0]:
                            train_lists.append([h["train_accuracy"] for h in hist])
                if train_lists:
                    ep, m, s = _mean_std(train_lists)
                    _plot_band(ax, ep, m, s, "tab:blue", "train")
                if val_lists:
                    ep, m, s = _mean_std(val_lists)
                    _plot_band(ax, ep, m, s, "tab:orange", "val", linestyle="--")
                test_accs = [
                    float(r["metrics"]["final_accuracy"])
                    for r in runs
                    if "metrics" in r and "final_accuracy" in r["metrics"]
                ]
                if test_accs:
                    ax.axhline(np.mean(test_accs), color="tab:red", linestyle=":",
                               linewidth=1.0, label=f"test ({np.mean(test_accs):.3f})")
            else:
                sb_val, sb_train = [], []
                a_lengths = []
                for r in runs:
                    metrics = r.get("metrics", {})
                    sa = metrics.get("stage_a_history", [])
                    sb = metrics.get("stage_b_history", [])
                    if sa:
                        a_lengths.append(len(sa))
                    if sb:
                        sb_val.append([h["val_accuracy"] for h in sb])
                        if "train_accuracy" in sb[0]:
                            sb_train.append([h["train_accuracy"] for h in sb])

                mean_a_len = int(np.mean(a_lengths)) if a_lengths else 0
                if sb_train:
                    ep_raw, m, s = _mean_std(sb_train)
                    _plot_band(ax, ep_raw + mean_a_len, m, s, "tab:blue", "Stage B train")
                if sb_val:
                    ep_raw, m, s = _mean_std(sb_val)
                    _plot_band(ax, ep_raw + mean_a_len, m, s, "tab:orange", "Stage B val",
                               linestyle="--")
                test_accs = [
                    float(r["metrics"]["final_accuracy"])
                    for r in runs
                    if "metrics" in r and "final_accuracy" in r["metrics"]
                ]
                if test_accs:
                    ax.axhline(np.mean(test_accs), color="tab:red", linestyle=":",
                               linewidth=1.0, label=f"test ({np.mean(test_accs):.3f})")

            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax.set_ylabel("Accuracy")
            ax.set_title(MODEL_DISPLAY_NAMES[mkey], pad=3)
            ax.legend(ncol=2)
            if row == n - 1:
                ax.set_xlabel("Epoch")

        fig.tight_layout()
        _save(fig, output_dir / "fig_accuracy_curves_avg")
        plt.close(fig)


# ── Figure: averaged ASW ───────────────────────────────────────────────────────


def _fig_asw_avg(all_runs: dict[str, list[dict]], output_dir: Path) -> None:
    asw_means: dict[str, float] = {}
    asw_stds: dict[str, float] = {}

    for mkey in ALL_MODEL_KEYS:
        runs = all_runs[mkey]
        scores = []
        for r in runs:
            if "test_embeddings" not in r or "test_labels" not in r:
                continue
            scores.append(compute_asw(r["test_embeddings"], r["test_labels"]))
        if not scores:
            print(f"[warn] Skipping ASW for {mkey}: no embeddings found")
            continue
        asw_means[mkey] = float(np.mean(scores))
        asw_stds[mkey] = float(np.std(scores))
        print(f"  ASW ({MODEL_DISPLAY_NAMES[mkey]}): {asw_means[mkey]:.4f} ± {asw_stds[mkey]:.4f}")

    if not asw_means:
        print("[warn] No ASW data; skipping fig_asw_avg")
        return

    sorted_models = sorted(asw_means, key=asw_means.__getitem__, reverse=True)

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots(figsize=(COL1, 0.45 * len(sorted_models) + 0.6))
        bars = ax.barh(
            [MODEL_DISPLAY_NAMES[m] for m in sorted_models],
            [asw_means[m] for m in sorted_models],
            xerr=[asw_stds[m] for m in sorted_models],
            color=[MODEL_COLORS[m] for m in sorted_models],
            capsize=2, error_kw={"linewidth": 0.8}, height=0.55,
        )
        for i, mkey in enumerate(sorted_models):
            m, s = asw_means[mkey], asw_stds[mkey]
            ax.text(m + s + 0.008, i, f"{m:.3f}", va="center", fontsize=6)
        ax.set_xlim(0, max(asw_means.values()) * 1.25)
        ax.set_xlabel("Normalized ASW")
        fig.tight_layout()
        _save(fig, output_dir / "fig_asw_avg")
        plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate mean±std figures across seeds from trained model checkpoints"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="results/checkpoints",
        help="Parent directory containing model run subdirectories",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/figures/averaged",
        help="Directory to write output figures",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[13, 42, 77],
        help="Seed values to aggregate (default: 13 42 77)",
    )
    parser.add_argument(
        "--exclude", type=str, nargs="*", default=["tf_gene:77"],
        metavar="MODEL:SEED",
        help="Exclude specific model+seed combinations (default: tf_gene:77 — failed run)",
    )
    args = parser.parse_args(argv)

    # Parse exclusions into a set of (model_key, seed) tuples
    exclusions: set[tuple[str, int]] = set()
    for item in args.exclude:
        mkey, seed_str = item.split(":")
        exclusions.add((mkey, int(seed_str)))
    if exclusions:
        print(f"[info] Excluding: {', '.join(f'{m}:seed{s}' for m, s in sorted(exclusions))}")

    ckpt_root = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefixes = {
        "baseline": "baseline_rna_seed",
        "baseline_protein": "baseline_protein_seed",
        "mlp": "contrastive_mlp_seed",
        "tf": "contrastive_tf_seed",
        "tf_gene": "contrastive_tf_gene_seed",
    }

    # Build per-model seed lists, applying exclusions
    model_seeds = {
        mkey: [s for s in args.seeds if (mkey, s) not in exclusions]
        for mkey in prefixes
    }

    print(f"\n=== Loading runs ===")
    all_runs: dict[str, list[dict]] = {k: [] for k in ALL_MODEL_KEYS}
    for mkey, prefix in prefixes.items():
        seeds_for_model = model_seeds[mkey]
        for seed in seeds_for_model:
            d = _find_seed_run(ckpt_root, prefix, seed)
            if d is None:
                print(f"[warn] No run found for {mkey} seed{seed} — skipping")
                continue
            print(f"  {MODEL_DISPLAY_NAMES[mkey]} seed{seed}: {d.name}")
            all_runs[mkey].append(_load_run(d))

    if not any(all_runs.values()):
        print(f"No runs found under {ckpt_root}")
        return

    print(f"\n=== Generating averaged figures ===")
    _fig_model_comparison_avg(all_runs, output_dir)
    _fig_training_curves_avg(all_runs, output_dir)
    _fig_accuracy_curves_avg(all_runs, output_dir)
    _fig_asw_avg(all_runs, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
