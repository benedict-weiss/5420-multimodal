# Plan: Per-head attention extraction (fix CD56/CD16 dilution on NK)

## Context

User flagged: on NK cells, canonical markers **CD56 and CD16 should be very bright**, but they don't show up in the attention heatmap. Investigation revealed the root cause is **head averaging**, not that the model ignores these markers.

Measured on real NK cells through the trained protein encoder:

| Reduction | CD56 rank | CD16 rank | Notable |
|-----------|-----------|-----------|---------|
| **Mean over heads, last layer** (current) | 11/134 | 20/134 | Attention entropy 4.66 ≈ uniform 4.90 |
| **Layer 0, head 0** | 38 | **2** (attn=0.155) | |
| **Layer 0, head 2** | **6** | 46 | Top: CD48, CD45RA, CD31, CD7 |
| **Layer 1, head 3** | 17 | **3** (attn=0.059) | Top: CD45, CD57, CD16, CD45RA |

Individual heads do attend to canonical markers. Averaging 4 heads — three of which distribute attention elsewhere — washes the signal out. `get_attention_weights` in `src/models/transformer_encoder.py:123-135` hard-codes `mean(dim=1)` over heads and picks only the last layer.

Goal: surface per-head attention so CD56/CD16 (and other head-specific markers) become visible, without breaking existing artifacts.

## Approach

Save per-head attention arrays for both layers at training time, and add analysis functions that compute a **max-over-heads** reduction (keeps the strongest signal from any single head) alongside the existing mean reduction. Also provide a way to re-extract per-head attention from an already-trained checkpoint so we don't have to retrain.

### 1. `src/models/transformer_encoder.py`

Add a new method on `TransformerEncoder`:

```python
def get_attention_weights_per_head(self) -> Optional[dict]:
    """Returns {'layer0': (batch, nhead, n_tokens), 'layer1': (batch, nhead, n_tokens)}
    with CLS-row attention per head per layer. None if no forward pass run yet."""
```

Leave existing `get_attention_weights()` unchanged (backward-compat with the saved `tf_attention_protein.npy` artifacts other tooling may read).

### 2. `src/train_contrastive_tf.py`

Where `tf_attention_protein.npy` / `tf_attention_rna.npy` are currently saved, also save:
- `tf_attention_protein_per_head.npy` — shape `(n_cells, n_layers, n_heads, n_proteins)`
- `tf_attention_rna_per_head.npy` — same layout for RNA pathways

Keep the existing mean-reduced arrays (no regression).

### 3. `src/attention_analysis.py`

Add:

- **Loader fallback**: if `tf_attention_protein_per_head.npy` is absent **and** `--data_path` provided, re-extract per-head attention from the checkpoint's `stage_a_best.pt` (load encoder, run a single forward pass over test split protein data, capture per-head weights, save the npy). Then proceed normally. This rescues the existing `20260414_194858` checkpoint without retraining.
- **New reduction option**: `--head_reduction {mean,max}` (default `max` for the new per-head-aware plots; `mean` preserves legacy).
- **New plot call in `main()`**: a second per-cell-type top-K heatmap using `max`-over-heads reduction, saved as `attention_heatmap_protein_per_head_max.png`.
- **New marker validation pass** on the max-reduced per-head attention, saved as `marker_validation_per_head_max.json`, so we can compare recall against the existing mean-based `marker_validation.json`.

Keep all existing outputs so the fix is additive.

## Files modified

- `src/models/transformer_encoder.py` — add `get_attention_weights_per_head()`, ~15 lines.
- `src/train_contrastive_tf.py` — extract + save per-head arrays at the same point the current arrays are saved, ~15 lines.
- `src/attention_analysis.py` — loader fallback with on-demand re-extraction, `--head_reduction` flag, extra plot + validation call, ~80 lines.

No changes to `train_contrastive_mlp.py`, `train_baseline.py`, `preprocessing.py`, or `dataset.py`.

## Verification

1. Re-run on existing checkpoint (triggers on-demand per-head re-extraction):
   ```bash
   python -m src.attention_analysis \
     --checkpoint_dir results/checkpoints/contrastive_tf_seed42_20260414_194858 \
     --data_path data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad \
     --head_reduction max
   ```
2. Confirm new artifacts exist:
   - `tf_attention_protein_per_head.npy` (shape `(n_cells, 2, 4, 134)`)
   - `attention_heatmap_protein_per_head_max.png`
   - `marker_validation_per_head_max.json`
3. Confirm NK recall improves:
   ```bash
   python -c "import json; d=json.load(open('results/checkpoints/contrastive_tf_seed42_20260414_194858/marker_validation_per_head_max.json')); print(d['NK'])"
   ```
   Expect `CD16` in `found` at minimum (rank #3 under max reduction).
4. Spot-check other cell types: T reg should find CD25, MAIT should find CD161, pDC should find CD123 — same markers as before, now with clearer attention peaks.
5. Run a single-epoch smoke of `train_contrastive_tf.py` to confirm per-head arrays are produced by fresh runs:
   ```bash
   python -m src.train_contrastive_tf --data_path data/ --epochs 1 --max_cells 2000
   ```
   Check the new run dir contains both `tf_attention_protein.npy` and `tf_attention_protein_per_head.npy`.
6. Confirm existing `attention_heatmap_protein.png`, `attention_heatmap_protein_per_row.png`, and `marker_validation.json` outputs still regenerate and are byte-identical to previous runs (no regression on legacy path).
