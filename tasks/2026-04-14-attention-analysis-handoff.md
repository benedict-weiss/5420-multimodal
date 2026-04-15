# Attention Analysis Handoff

## What changed

- `src/models/transformer_encoder.py`
  - Added `get_attention_weights_per_head()`
  - Existing `get_attention_weights()` is unchanged and still returns last-layer mean-over-heads CLS attention

- `src/train_contrastive_tf.py`
  - `extract_attention()` now collects both legacy mean-reduced attention and raw per-head attention
  - Fresh runs now save:
    - `tf_attention_rna_per_head.npy`
    - `tf_attention_protein_per_head.npy`

- `src/attention_analysis.py`
  - Added fallback re-extraction for old checkpoints missing `tf_attention_protein_per_head.npy`
  - Added `--head_reduction {mean,max}`
  - Added per-head-aware heatmap output
  - Added per-head-aware marker validation output
  - Added alias-aware marker validation so canonical markers resolve to panel names like `CD4-1`, `CD14-1`, and `CD38-1`

## Latest checkpoint assessed

- Checkpoint dir:
  `results/checkpoints/contrastive_tf_seed42_20260414_194858`
- Metrics from `metrics.json`:
  - accuracy `0.8560`
  - macro AUROC `0.9967`
  - test loss `0.3701`

## Main interpretation

- The checkpoint itself looks strong.
- The original biology mismatch was driven largely by the attention summary, not obviously by a failed model.
- Protein attention contains biologically meaningful signal in individual heads that was getting washed out by the old last-layer mean-over-heads reduction.
- RNA pathway attention is still intrinsically noisy to interpret because pathway tokens are broad overlapping averages with KEGG labels that are not clean lineage markers.

## Current artifacts in the latest checkpoint

- Existing legacy artifacts:
  - `tf_attention_rna.npy`
  - `tf_attention_protein.npy`
  - `marker_validation.json`
  - `attention_heatmap_protein.png`
  - `attention_heatmap_protein_per_row.png`

- New per-head artifacts:
  - `tf_attention_protein_per_head.npy`
    - verified shape: `(18053, 2, 4, 134)`
  - `attention_heatmap_protein_per_head_max.png`
  - `attention_heatmap_protein_per_head_mean.png`
  - `marker_validation_per_head_max.json`
  - `marker_validation_per_head_mean.json`

## Important biological findings

- Raw per-head inspection showed canonical markers in specialized heads:
  - NK: `CD16` reaches rank 1 in one layer-0 head; `CD56` reaches rank 4 in another head
  - pDC: `CD123` is strong; `CD303` appears in some heads but is weaker in the top-10 summary
  - CD8 naive: `CD8` is strongly recovered in per-head analysis

- Alias-aware per-head max validation improved false negatives caused by panel naming:
  - `HSC -> CD38`
  - `Plasma cell IGKC+ -> CD38`
  - `Plasma cell IGKC- -> CD38`
  - `NK -> CD16`
  - `pDC -> CD123`
  - `CD8+ T naive -> CD8`

- Current alias-aware `marker_validation_per_head_max.json` summary:
  - mean recall `0.281`
  - still weak or missing in top-10 summary:
    - `CD56` for NK
    - `CD303` for pDC
    - `CD3` for several T-cell states
    - several B-cell markers

## Commands used

- Rebuild per-head attention for the existing checkpoint and regenerate outputs:

```bash
python -m src.attention_analysis \
  --checkpoint_dir results/checkpoints/contrastive_tf_seed42_20260414_194858 \
  --data_path data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad \
  --head_reduction max
```

- Regenerate per-head mean outputs:

```bash
python -m src.attention_analysis \
  --checkpoint_dir results/checkpoints/contrastive_tf_seed42_20260414_194858 \
  --head_reduction mean
```

## Recommended next steps

1. Add resolved token names to the validation JSON so outputs show mappings like `CD38 -> CD38-1`.
2. Add rank-based marker reporting instead of top-10-only recall.
3. If stronger attribution is needed, add token ablation or gradient-based attribution instead of relying on attention alone.
