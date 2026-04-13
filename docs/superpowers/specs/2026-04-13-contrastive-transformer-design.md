# Contrastive Transformer (Model 3) — Design Spec

**Date:** 2026-04-13  
**Status:** Approved

## Overview

Model 3 extends the dual-encoder contrastive MLP (Model 2) by replacing the MLP encoders with transformer encoders operating on pathway-level tokens (RNA) and protein tokens (134-d). The key addition over Model 2 is attention weight extraction, which enables post-hoc biological interpretability: which pathways and proteins drive each cell type's classification.

Three files need to be implemented; all other infrastructure (CLIPLoss, ClassificationHead, CITEseqDataset, get_dataloaders, preprocessing functions) is reused unchanged.

---

## 1. `src/models/transformer_encoder.py`

### `AttentionTransformerEncoderLayer`

Minimal subclass of `nn.TransformerEncoderLayer`. Overrides `forward` to call `self.self_attn` with `need_weights=True, average_attn_weights=False`, storing per-head attention in `self._last_attn_weights`. The rest of the forward pass (residual connection, layer norm, feedforward block) reuses the parent implementation by calling `super().forward()` after substituting the attention output.

**Why subclass rather than hook:** A `register_forward_hook` on the `self_attn` module cannot change the kwargs that PyTorch's `TransformerEncoderLayer` passes internally (`need_weights=False` by default). Subclassing cleanly overrides the call without any version-specific patching.

### `TransformerEncoder`

```
input_proj:    Linear(1, d_model)           # scalar token → d_model
cls_token:     Parameter(1, 1, d_model)     # learnable
pos_encoding:  Parameter(1, n_tokens+1, d_model)  # learnable, +1 for CLS
encoder:       nn.TransformerEncoder(AttentionTransformerEncoderLayer × num_layers)
output_proj:   Linear(d_model, output_dim)
```

**Forward pass:**
1. `x` shape `(batch, n_tokens)` → reshape to `(batch, n_tokens, 1)`
2. Project: `x = input_proj(x)` → `(batch, n_tokens, d_model)`
3. Prepend CLS: `cat([cls_token.expand(batch, -1, -1), x], dim=1)` → `(batch, n_tokens+1, d_model)`
4. Add positional encoding
5. Pass through transformer encoder
6. Extract CLS output: `x[:, 0, :]` → `(batch, d_model)`
7. Project and L2-normalize: `F.normalize(output_proj(x), p=2, dim=1)` → `(batch, output_dim)`

**`get_attention_weights()`:**
- Reads `_last_attn_weights` from the final `AttentionTransformerEncoderLayer`
- Averages across heads: `attn.mean(dim=1)` → `(batch, n_tokens+1, n_tokens+1)`
- Returns CLS row, skipping CLS-to-CLS: `[:, 0, 1:]` → `(batch, n_tokens)`
- Returns `None` if called before any forward pass

**Default hyperparameters (from spec):**
- `d_model=64`, `nhead=4`, `num_layers=2`, `dim_feedforward=256`, `dropout=0.1`, `output_dim=128`

---

## 2. `src/train_contrastive_tf.py`

Near-copy of `train_contrastive_mlp.py` with these specific differences:

| Aspect | MLP version | Transformer version |
|--------|-------------|---------------------|
| RNA preprocessing | `preprocess_rna()` → PCA (256-d) | `build_pathway_tokens()` → pathway matrix (~300-d) |
| RNA encoder | `MLPEncoder(input_dim=256)` | `TransformerEncoder(n_tokens=n_pathways)` |
| Protein encoder | `MLPEncoder(input_dim=134)` | `TransformerEncoder(n_tokens=134)` |
| Batch key for RNA | `batch["rna"]` | `batch["pathway"]` |
| Leakage prevention | Fit PCA on train only, transform test | None needed — KEGG gene sets are global, not fit to data |
| Post-training step | — | Extract attention weights from test set, save as `.npy` |

**Leakage note:** `build_pathway_tokens` uses global KEGG_2021_Human gene sets (not derived from expression data), so pathway tokens can be built on the full dataset before splitting — no train/test leakage risk unlike PCA.

**Attention saving (after Stage B):**
- Run full test set through both encoders in eval mode
- Collect `rna_encoder.get_attention_weights()` and `protein_encoder.get_attention_weights()` per batch
- Concatenate across batches → `(n_test_cells, n_tokens)`
- Save to `results/tf_attention_rna.npy`, `results/tf_attention_protein.npy`, and corresponding labels `results/tf_attention_labels.npy`
- Controlled by `--save_attention` flag (default: True)

**CLI:**
```
python src/train_contrastive_tf.py --data_path data/ --seed 42 --batch_size 256
```

All other structure identical to MLP version: same `run_contrastive_epoch`, `evaluate_classifier_epoch`, early stopping (patience=10, min_delta=1e-4), Stage A (150 epochs) / Stage B (50 epochs), checkpointing, metrics.

---

## 3. `src/attention_analysis.py`

Pure library, no CLI. Five functions:

**`aggregate_attention_by_cell_type(attention_weights, labels, label_names) -> dict`**
- Groups `(n_cells, n_tokens)` array by integer label, computes per-cell-type mean attention
- Returns `{cell_type_str: np.ndarray(n_tokens,)}`

**`get_top_tokens(attention_by_type, token_names, top_k=10) -> dict`**
- For each cell type, argsort descending, return top-k `[(token_name, score), ...]`
- Returns `{cell_type: [(name, score), ...]}`

**`validate_against_markers(top_tokens_dict, expected_markers) -> dict`**
- Checks overlap between top-k token names and known biological markers
- Example markers: `{'HSC': ['CD34', 'CD38'], 'B cell': ['CD19', 'CD20'], 'CD4 T': ['CD3', 'CD4'], ...}`
- Returns `{cell_type: {'found': [...], 'missing': [...], 'precision': float}}`

**`plot_attention_heatmap(attention_by_type, token_names, title, save_path)`**
- Seaborn heatmap: rows = cell types, cols = token names
- RNA (~300 pathway tokens): filter to top-N per cell type before plotting (default top_n=20)
- Protein (134 tokens): show all, or configurable top_n

**`plot_token_attention_per_cell_type(attention_weights, token_names, cell_type_labels, selected_types, save_path)`**
- Violin or box plot of attention distribution across tokens for selected cell types
- Shows whether attention is focused (peaked) or diffuse

---

## 4. `tests/test_transformer_encoder.py`

Minimal test suite (no real data required — all synthetic tensors):

1. **Output shape** — `(batch, output_dim)` for RNA-like (n_tokens=300) and protein-like (n_tokens=134) inputs
2. **L2 normalization** — output vectors have unit norm (within float tolerance)
3. **Attention shape** — `get_attention_weights()` returns `(batch, n_tokens)` after a forward pass
4. **Attention validity** — weights are non-negative; each row sums ≤ 1.0 (softmax outputs from attention)
5. **Smoke test** — two encoders + `CLIPLoss` produces a finite scalar loss

---

## File Summary

| File | Action |
|------|--------|
| `src/models/transformer_encoder.py` | Implement (replace stub) |
| `src/train_contrastive_tf.py` | Implement (replace stub) |
| `src/attention_analysis.py` | Implement (replace stub) |
| `tests/test_transformer_encoder.py` | Create new |
| All other files | No changes |
