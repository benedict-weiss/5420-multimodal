# evaluate.py Design Spec
_Date: 2026-04-14_

## Overview

Implement `src/evaluate.py` — the shared metrics library and paper-figure generator for the
multimodal CITE-seq project. The file serves two roles:

1. **Metric functions** imported by all three train scripts at training time.
2. **CLI entry point** that discovers trained checkpoints and generates publication figures A–E.

---

## Part 1 — Metric Functions (7 stubs)

All functions are pure (no side effects, no I/O). Signatures are fixed by the existing stub and
the `try`-imports in `train_baseline.py` and `train_contrastive_mlp.py`.

### `compute_auroc(y_true, y_pred_proba, n_classes) -> float`
- `sklearn.metrics.roc_auc_score(y_true, y_pred_proba, average='macro', multi_class='ovr')`
- `n_classes` kept in signature for compatibility; sklearn infers it from the data.
- Returns macro-averaged AUC-ROC as a float.

### `compute_accuracy(y_true, y_pred) -> tuple[float, dict]`
- `overall = sklearn.metrics.accuracy_score(y_true, y_pred)`
- `per_class = sklearn.metrics.classification_report(y_true, y_pred, output_dict=True, zero_division=0)`
- Returns `(overall_accuracy, per_class_dict)`.

### `compute_asw(embeddings, labels) -> float`
- Subsample to 10k cells if `n > 10_000` (silhouette is O(n²)).
- `raw = sklearn.metrics.silhouette_score(embeddings, labels)`
- Normalize: `(raw + 1) / 2` → range [0, 1].
- Returns normalized ASW as float.

### `compute_recall_at_k(z_rna, z_protein, k_values=[10,20,30,40,50]) -> dict`
- Both inputs are L2-normalized (contrastive encoder outputs).
- Use cosine similarity matrix: `sim = z_rna @ z_protein.T` (shape n×n).
- For each row i, check if i is in `top-k` column indices of row i.
- Returns `{k: mean_recall for k in k_values}`.

### `plot_phate(embeddings, labels, title, save_path)`
- Subsample to 20k cells if `n > 20_000`.
- `phate_op = phate.PHATE(n_components=2, n_jobs=1, verbose=False)`
- `coords = phate_op.fit_transform(embeddings)`
- Scatter plot: one color per unique label, legend outside right, alpha=0.4, s=4.
- Save to `save_path` as PNG at dpi=150.

### `compute_batch_entropy(embeddings, batch_labels, n_neighbors=50) -> float`
- Build k-NN graph (cosine) with sklearn `NearestNeighbors`.
- For each cell, find `n_neighbors` neighbors; compute entropy of batch label distribution.
- Entropy: `scipy.stats.entropy(counts / counts.sum())`.
- Return mean entropy across all cells.

### `run_significance_test(scores_model1, scores_model2) -> float`
- `scipy.stats.ranksums(scores_model1, scores_model2)`
- Returns p-value (`.pvalue` attribute).

---

## Part 2 — Train Script Changes (Option B: save embeddings)

Each train script saves additional `.npy` artifacts at the end of its `main()`, using the
best-checkpoint model in eval mode on the held-out test set.

### Files saved (per run directory)

| File | Models | Content |
|---|---|---|
| `test_embeddings.npy` | all 3 | Classifier input: shape `(n_test, emb_dim)` |
| `test_labels.npy` | all 3 | Integer labels: shape `(n_test,)` |
| `test_rna_embeddings.npy` | contrastive_mlp, contrastive_tf | L2-normed RNA encoder output |
| `test_protein_embeddings.npy` | contrastive_mlp, contrastive_tf | L2-normed protein encoder output |

**Baseline** — `test_embeddings` = encoder output (128-d, no L2 norm).  
**Contrastive models** — `test_embeddings` = `np.concatenate([z_rna, z_protein], axis=1)` (256-d).

Extraction is done with `torch.no_grad()` after `encoder.eval()` / `classifier.eval()`.

---

## Part 3 — `evaluate.py main()` — Paper Figure Generator

### CLI

```
python src/evaluate.py \
    --checkpoint_dir results/checkpoints \
    --output_dir results/figures
```

Optional: `--baseline_dir`, `--mlp_dir`, `--tf_dir` to pin specific run directories instead of
auto-discovery.

### Auto-discovery

Scan `checkpoint_dir` for subdirectories matching:
- `baseline_rna_*` → baseline model
- `contrastive_mlp_*` → MLP model  
- `contrastive_tf_*` → transformer model

If multiple matches exist, pick the **most recently modified** directory.
Print which directories were selected.

### Figure A — Model Comparison Bar Chart (`fig_model_comparison.png`)

- Source: `metrics.json` from each run (`final_accuracy`, `final_macro_auroc`)
- Grouped bar chart: x-axis = metric (Accuracy, Macro AUROC), groups = models
- 3 bars per metric group, consistent colors across all figures
- Annotate each bar with its numeric value
- y-axis range [0, 1]

### Figure B — PHATE Plots (`fig_phate_baseline.png`, `fig_phate_mlp.png`, `fig_phate_tf.png`)

- Source: `test_embeddings.npy` + `test_labels.npy` + `label_mapping.json`
- One PNG per model (allows easy inclusion as separate panels in paper)
- Call `plot_phate(embeddings, label_strings, title, save_path)`
- Convert integer labels → string names via `label_mapping.json`
- Consistent color palette across all three plots (use tab20 or a fixed dict)

### Figure C — Training Curves (`fig_training_curves.png`)

- Source: `metrics.json` history arrays from each run
- 3-row subplot (one per model), each showing train_loss and val_loss vs epoch
- For contrastive models: Stage A and Stage B plotted as separate line groups with a divider
- x-axis = epoch, y-axis = loss, legend per subplot

### Figure D — Recall@k (`fig_recall_at_k.png`)

- Source: `test_rna_embeddings.npy` + `test_protein_embeddings.npy` for MLP and TF runs
- Call `compute_recall_at_k(z_rna, z_protein, k_values=[10,20,30,40,50])`
- Line plot: x = k, y = recall, one line per contrastive model
- Baseline is excluded (no cross-modal embeddings)

### Figure E — ASW Bar Chart (`fig_asw.png`)

- Source: `test_embeddings.npy` + `test_labels.npy` for all 3 models
- Call `compute_asw(embeddings, labels)` per model
- Horizontal bar chart, sorted descending
- x-axis range [0, 1] (normalized ASW)

---

## Style Conventions

- `matplotlib.use("Agg")` at import (non-interactive, consistent with `attention_analysis.py`)
- All figures: `dpi=150`, `bbox_inches="tight"`
- Model colors: `{"baseline": "#4C72B0", "mlp": "#DD8452", "tf": "#55A868"}` — used consistently
- Figure titles include model name for standalone readability
- Print each saved file path after writing

---

## File Layout

Single file `src/evaluate.py`. No new modules. Follows `attention_analysis.py` conventions:
- `from __future__ import annotations`
- `matplotlib.use("Agg")` before pyplot import
- `main(argv=None)` + `if __name__ == "__main__": main()`
- argparse CLI at bottom

---

## What Is NOT in scope

- Reloading model weights (evaluate.py never imports torch models)
- Confusion matrices (not requested)
- Batch entropy figure (function implemented but not plotted — kept for train script use)
- `run_significance_test` (function implemented but not called in main — available for notebook use)
