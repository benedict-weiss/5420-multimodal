"""evaluate_averaged.py — Generate mean±std figures across multiple seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def _plot_band(ax, epochs, mean, std, color, label, linestyle="-", alpha=0.2):
    """Plot mean line with ±1 std shaded band."""
    valid = np.isfinite(mean)
    ax.plot(epochs[valid], mean[valid], color=color, linestyle=linestyle, label=label)
    ax.fill_between(
        epochs[valid],
        (mean - std)[valid],
        (mean + std)[valid],
        color=color, alpha=alpha,
    )


# ── Figure: averaged model comparison ─────────────────────────────────────────


def _fig_model_comparison_avg(all_runs: dict[str, list[dict]], output_dir: Path) -> None:
    metric_keys = ["final_accuracy", "final_macro_auroc"]
    metric_labels = ["Accuracy", "Macro AUROC"]
    model_keys = [k for k in ALL_MODEL_KEYS if all_runs[k]]

    x = np.arange(len(metric_keys))
    width = 0.8 / max(len(model_keys), 1)
    fig, ax = plt.subplots(figsize=(9, 5))

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
            x + i * width, means, width, yerr=stds, capsize=4,
            label=MODEL_DISPLAY_NAMES[mkey], color=MODEL_COLORS[mkey],
        )
        for bar, m, s in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.012,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x + width * (len(model_keys) - 1) / 2)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Mean ± Std Across Seeds")
    ax.legend()
    plt.tight_layout()
    save_path = output_dir / "fig_model_comparison_avg.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Figure: averaged training curves ──────────────────────────────────────────


def _fig_training_curves_avg(all_runs: dict[str, list[dict]], output_dir: Path) -> None:
    model_keys = [k for k in ALL_MODEL_KEYS if all_runs[k]]
    fig, axes = plt.subplots(len(model_keys), 1, figsize=(10, 4 * len(model_keys)), squeeze=False)

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
                ax.axhline(
                    np.mean(test_losses), color="tab:red", linestyle=":",
                    label=f"test mean ({np.mean(test_losses):.4f})",
                )
        else:
            # Stage A
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
                ax.axvline(mean_a_len, color="gray", linestyle=":", alpha=0.7, label="Stage A→B")

            if sb_train:
                mean_a_len = int(np.mean(a_lengths)) if a_lengths else 0
                ep_raw, m, s = _mean_std(sb_train)
                ep = ep_raw + mean_a_len
                _plot_band(ax, ep, m, s, "tab:orange", "Stage B train")
                ep_raw, m, s = _mean_std(sb_val)
                ep = ep_raw + mean_a_len
                _plot_band(ax, ep, m, s, "tab:orange", "Stage B val", linestyle="--")

            # test loss hline
            test_losses = [
                float(r["metrics"]["final_test_loss"])
                for r in runs
                if "metrics" in r and "final_test_loss" in r["metrics"]
            ]
            if test_losses:
                ax.axhline(
                    np.mean(test_losses), color="tab:red", linestyle=":",
                    label=f"test mean ({np.mean(test_losses):.4f})",
                )

        ax.set_title(f"{MODEL_DISPLAY_NAMES[mkey]} — Training Curves (mean ± std)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)

    plt.tight_layout()
    save_path = output_dir / "fig_training_curves_avg.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Figure: averaged accuracy curves ──────────────────────────────────────────


def _fig_accuracy_curves_avg(all_runs: dict[str, list[dict]], output_dir: Path) -> None:
    model_keys = [k for k in ALL_MODEL_KEYS if all_runs[k]]
    fig, axes = plt.subplots(len(model_keys), 1, figsize=(10, 4 * len(model_keys)), squeeze=False)

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
                ax.axhline(
                    np.mean(test_accs), color="tab:red", linestyle=":",
                    label=f"test mean ({np.mean(test_accs):.4f})",
                )
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
                _plot_band(ax, ep_raw + mean_a_len, m, s, "tab:orange", "Stage B val", linestyle="--")

            test_accs = [
                float(r["metrics"]["final_accuracy"])
                for r in runs
                if "metrics" in r and "final_accuracy" in r["metrics"]
            ]
            if test_accs:
                ax.axhline(
                    np.mean(test_accs), color="tab:red", linestyle=":",
                    label=f"test mean ({np.mean(test_accs):.4f})",
                )

        ax.set_ylim(0, 1.05)
        ax.set_title(f"{MODEL_DISPLAY_NAMES[mkey]} — Accuracy Curves (mean ± std)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8)

    plt.tight_layout()
    save_path = output_dir / "fig_accuracy_curves_avg.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


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
        print("[warn] No ASW data; skipping fig_asw_avg.png")
        return

    sorted_models = sorted(asw_means, key=asw_means.__getitem__, reverse=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(
        [MODEL_DISPLAY_NAMES[m] for m in sorted_models],
        [asw_means[m] for m in sorted_models],
        xerr=[asw_stds[m] for m in sorted_models],
        color=[MODEL_COLORS[m] for m in sorted_models],
        capsize=4,
    )
    for mkey in sorted_models:
        m, s = asw_means[mkey], asw_stds[mkey]
        ax.text(m + s + 0.005, sorted_models.index(mkey),
                f"{m:.3f}±{s:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Normalized ASW")
    ax.set_title("Average Silhouette Width — Mean ± Std Across Seeds")
    plt.tight_layout()
    save_path = output_dir / "fig_asw_avg.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


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
