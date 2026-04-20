# Multimodal CITE-seq Cell State Prediction

Contrastive learning on paired single-cell RNA + surface protein (CITE-seq) data to classify immune cell types. Compares three models: RNA-only baseline, contrastive MLP, and contrastive transformer with interpretable attention.

## Dataset

[NeurIPS 2021 BMMC benchmark](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122) — 90,261 bone marrow mononuclear cells from 12 donors across 4 sites. Two modalities: RNA (13,953 genes) and surface protein (134 ADT markers).

### Download

```bash
mkdir -p data
curl -o data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz \
  'ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz'
```

## Models

| Model | Input | Encoder | Loss |
|-------|-------|---------|------|
| RNA-Only Baseline | RNA PCA (256-d) | MLP | Cross-entropy |
| Contrastive MLP | RNA PCA + Protein CLR | 2x MLP | CLIP contrastive + Cross-entropy |
| Contrastive Transformer | RNA pathway tokens + Protein tokens | 2x Transformer | CLIP contrastive + Cross-entropy |

The contrastive models use two-stage training: (A) contrastive pretraining to align RNA and protein embeddings, then (B) classifier training on frozen embeddings.

The transformer model adds interpretability — attention weights reveal which pathways and proteins drive classification for each cell type.

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── data/                          # Dataset (not tracked by git)
├── src/
│   ├── preprocessing.py           # Data loading, normalization, PCA, pathway tokenization
│   ├── dataset.py                 # PyTorch Dataset and DataLoader
│   ├── models/
│   │   ├── mlp_encoder.py         # MLP encoder
│   │   ├── transformer_encoder.py # Transformer encoder with CLS token
│   │   ├── contrastive_loss.py    # CLIP-style contrastive loss
│   │   └── classifier.py         # Classification head
│   ├── train_baseline.py          # Model 1: RNA-only
│   ├── train_contrastive_mlp.py   # Model 2: contrastive MLP
│   ├── train_contrastive_tf.py    # Model 3: contrastive transformer
│   ├── evaluate.py                # Metrics + paper figure generator (PHATE, ASW, Recall@k, AUROC)
│   └── attention_analysis.py      # Attention heatmaps and marker validation
├── tests/
│   ├── conftest.py                # Shared fixtures and dataset constants
│   ├── inspect_real_data.py       # One-time script to discover dataset column names
│   ├── test_preprocessing.py      # 53 unit + integration tests for preprocessing
│   └── test_evaluate.py           # 17 unit tests for metric functions
├── results/
│   ├── checkpoints/               # Model checkpoints and saved embeddings (per run)
│   └── figures/                   # Generated paper figures
├── docs/
│   └── superpowers/specs/         # Design specs
├── implementation.md              # Detailed architecture spec
├── biology.md                     # Biological context and expected results
├── requirements.txt               # Python dependencies (pinned where needed)
├── pytest.ini                     # Test configuration
└── README.md
```

## Training

```bash
# Model 1: RNA-only baseline (~5-10 min, CPU ok)
python -m src.train_baseline --data_path data/ --seed 42

# Model 2: Contrastive MLP (~30-45 min, GPU recommended)
python -m src.train_contrastive_mlp --data_path data/ --seed 42 --batch_size 512

# Model 3: Contrastive transformer (~60-90 min, GPU required)
python -m src.train_contrastive_tf --data_path data/ --seed 42 --batch_size 256
```

Each run saves a timestamped checkpoint directory under `results/checkpoints/` containing model weights, metrics, and test-set embeddings.

## Evaluation

Generate all paper figures from trained checkpoints:

```bash
# Auto-discovers latest run of each model type
python -m src.evaluate --checkpoint_dir results/checkpoints --output_dir results/figures

# Or pin specific runs
python -m src.evaluate \
  --baseline_dir results/checkpoints/baseline_rna_seed42_<timestamp> \
  --mlp_dir results/checkpoints/contrastive_mlp_seed42_<timestamp> \
  --tf_dir results/checkpoints/contrastive_tf_seed42_<timestamp> \
  --output_dir results/figures
```

Produces five figures:

| Figure | Description |
|--------|-------------|
| `fig_model_comparison.png` | Accuracy and Macro AUROC bar chart across all 3 models |
| `fig_training_curves.png` | Train/val loss over epochs (Stage A + B for contrastive models) |
| `fig_phate_{baseline,mlp,tf}.png` | PHATE 2D embedding colored by cell type |
| `fig_recall_at_k.png` | Cross-modal retrieval Recall@k for contrastive models |
| `fig_asw.png` | Normalized average silhouette width per model |

## Attention Analysis

```bash
# Visualize transformer attention weights and validate against known markers
python -m src.attention_analysis \
  --checkpoint_dir results/checkpoints/contrastive_tf_seed42_<timestamp>
```

Produces attention heatmaps (RNA pathways and protein markers per cell type) and a `marker_validation.json` reporting recall of known markers (CD3→T, CD19→B, CD34→HSC, etc.).

## Ablation Visualization

```bash
# Plot protein ablation summaries from a checkpoint with ablation artifacts
python -m src.ablation_visualization \
  --checkpoint_dir results/checkpoints/contrastive_tf_seed42_<timestamp>
```

Produces:

| File | Description |
|------|-------------|
| `ablation_heatmap_top_proteins.png` | Row-normalized heatmap of the top ablation proteins across cell types |
| `ablation_top_bars_selected_cell_types.png` | Small-multiple bar charts of top-k protein drivers for representative cell types |
| `ablation_visualization_summary.json` | Metadata for the generated figure set |

## Testing

```bash
# All tests (70 total)
python -m pytest tests/ -v

# Preprocessing tests only (53 tests)
python -m pytest tests/test_preprocessing.py -v

# Evaluate metric function tests (17 tests)
python -m pytest tests/test_evaluate.py -v
```

Tests cover: modality splitting, RNA/protein preprocessing, pathway tokenization, label encoding, donor-based splitting, PCA leakage prevention, full pipeline integration, and all 7 evaluation metric functions.

## Dependencies

anndata, scanpy, muon, gseapy, torch, scikit-learn, phate, matplotlib, seaborn, pytest

**Note**: `numpy<2` and `setuptools<78` are pinned in `requirements.txt` for compatibility with muon and gseapy.
