# CLAUDE.md — Multimodal CITE-seq Cell State Prediction

## Goal
Classify immune cell types from paired RNA + protein (CITE-seq) data using contrastive learning. Compare: RNA-only baseline → contrastive MLP → contrastive transformer. Transformer adds interpretability via attention weights.

## Dataset
- **File**: `GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz` from GEO GSE194122
- **Download**: `wget 'ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz'`
- **Load**: `anndata.read_h5ad()` → single AnnData object
- **Size**: 90,261 cells, 12 donors, 4 sites
- **Modalities concatenated in one matrix**: split on `adata.var['feature_types']` — `'GEX'` (13,953 genes) and `'ADT'` (134 proteins)
- **Labels**: cell type annotations in `adata.obs` (exact column name TBD — inspect after loading)
- **Batch info**: donor and site IDs in `adata.obs`
- **Train/test split**: hold out 2-3 donors. Secondary split: hold out 1 site.

## Preprocessing

### RNA
1. Separate: `adata[:, adata.var['feature_types'] == 'GEX']`
2. `scanpy.pp.normalize_total(target_sum=1e4)` → `log1p` → `highly_variable_genes(n_top_genes=4000)` → subset → `scale(max_value=10)` → `pca(n_comps=256)`
3. Model input: `adata.obsm['X_pca']` — shape (90261, 256)

### Protein
1. Separate: `adata[:, adata.var['feature_types'] == 'ADT']`
2. CLR normalize with `muon.prot.pp.clr()`
3. Model input: `adata.X` — shape (90261, 134), no PCA needed

### Pathway tokenization (transformer only)
1. Load KEGG gene sets via `gseapy.get_library('KEGG_2021_Human')`
2. For each pathway with ≥5 measured genes in HVG list, average log-normalized expression across member genes
3. Result: (90261, ~300) matrix. Store pathway names for attention interpretation.
4. Reshape to (batch, n_pathways, 1) for transformer input — each token is a scalar

## Three Models

### Model 1: RNA-Only Baseline
- Input: RNA PCA (256-d)
- Encoder: Linear(256,256)→BN→ReLU→Linear(256,128)→BN→ReLU
- Head: Linear(128,64)→ReLU→Dropout(0.2)→Linear(64,n_classes)
- Loss: cross-entropy
- Purpose: control

### Model 2: Contrastive MLP
- Two MLP encoders, same architecture as Model 1 encoder but separate instances
  - RNA encoder: input 256 (PCA), output 128, L2-normalized
  - Protein encoder: input 134 (CLR), output 128, L2-normalized
- Contrastive loss: CLIP-style symmetric cross-entropy on cosine similarity matrix. Temperature=0.07. Positive pairs = same cell across modalities (diagonal). Negatives = all other cells in batch.
- Classification head: on concatenated [z_rna; z_protein] (256-d) → Linear(256,64)→ReLU→Dropout(0.2)→Linear(64,n_classes)
- Two-stage training:
  - Stage A: contrastive loss only on encoders, 150 epochs, early stopping (patience=10, min_delta=1e-4)
  - Stage B: freeze encoders, train classifier head with cross-entropy, 50 epochs

### Model 3: Contrastive Transformer
- Same contrastive framework and classification head as Model 2
- Replace MLP encoders with transformer encoders:
  - **RNA encoder**: ~300 pathway tokens → Linear(1,64) projection → learnable positional encoding → prepend learnable CLS token → 2 TransformerEncoderLayers (d_model=64, nhead=4, dim_ff=256, dropout=0.1, gelu) → CLS output → Linear(64,128) → L2 normalize
  - **Protein encoder**: 134 protein tokens → same architecture as RNA encoder but n_tokens=134
- Attention extraction: hook on last layer's self_attn, average across heads, take CLS row (row 0), skip column 0. Gives per-cell attention over tokens.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Temperature | 0.07 |
| LR | 1e-3 (Adam) |
| Weight decay | 1e-5 |
| Batch size | 256-512 |
| Contrastive epochs | 150 (early stop patience=10) |
| Classifier epochs | 50 |
| Embedding dim | 128 |
| MLP hidden dim | 256 |
| Transformer d_model | 64 |
| Transformer heads | 4 |
| Transformer layers | 2 |
| Transformer dim_ff | 256 |
| Dropout | 0.1 (transformer), 0.2 (classifier) |
| PCA components | 256 |
| HVGs | 4000 |
| Min genes per pathway | 5 |

## Evaluation

- AUC-ROC (macro one-vs-rest)
- Accuracy (overall + per-class)
- Average Silhouette Width on embeddings (normalize: (ASW+1)/2)
- Recall@k on contrastive embeddings (k=10,20,30,40,50)
- PHATE colored by cell type (all three models)
- Attention heatmaps: cell type × pathway tokens, cell type × protein tokens
- Top-10 attended tokens per cell type vs known markers (CD34→HSC, CD19→B cell, CD3/CD4/CD8→T cell, CD14→monocyte, CD56→NK)
- Wilcoxon rank-sum test across 5+ random seeds for significance
- Batch entropy to verify no batch effect leakage

## File Structure

```
project/
├── data/
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── models/
│   │   ├── mlp_encoder.py
│   │   ├── transformer_encoder.py
│   │   ├── contrastive_loss.py
│   │   └── classifier.py
│   ├── train_baseline.py
│   ├── train_contrastive_mlp.py
│   ├── train_contrastive_tf.py
│   ├── evaluate.py
│   └── attention_analysis.py
├── notebooks/
├── results/
├── CLAUDE.md
└── requirements.txt
```

## Dependencies
anndata, scanpy, muon, gseapy, torch, scikit-learn, phate, matplotlib, seaborn

## Hardware
GPU recommended for transformer (≥8GB VRAM). MLP trains fine on CPU. Dataset fits ~4GB RAM.
