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
│   ├── evaluate.py                # Metrics (AUC-ROC, accuracy, ASW, Recall@k, PHATE)
│   └── attention_analysis.py      # Attention heatmaps and marker validation
├── tests/
│   ├── conftest.py                # Shared fixtures and dataset constants
│   ├── inspect_real_data.py       # One-time script to discover dataset column names
│   └── test_preprocessing.py      # 53 unit + integration tests for preprocessing
├── notebooks/                     # Analysis notebooks
├── results/
│   ├── figures/                   # Generated plots
│   └── metrics/                   # Saved metric outputs
├── implementation.md              # Detailed architecture spec
├── biology.md                     # Biological context and expected results
├── requirements.txt               # Python dependencies (pinned where needed)
├── pytest.ini                     # Test configuration
├── README.md
└── .gitignore
```

## Usage

```bash
# Train baseline
python src/train_baseline.py --data_path data/ --seed 42

# Train contrastive MLP
python src/train_contrastive_mlp.py --data_path data/ --seed 42 --batch_size 512

# Train contrastive transformer
python src/train_contrastive_tf.py --data_path data/ --seed 42 --batch_size 256
```

## Testing

```bash
# Run all preprocessing tests
pytest tests/test_preprocessing.py -v

# Run integration tests only
pytest tests/test_preprocessing.py -v -m integration
```

58 tests cover: modality splitting, RNA/protein preprocessing, pathway tokenization, label encoding, donor-based splitting, PCA data leakage prevention, and full pipeline integration.

## Dependencies

anndata, scanpy, muon, gseapy, torch, scikit-learn, phate, matplotlib, seaborn, pytest

**Note**: `numpy<2` and `setuptools<78` are pinned in `requirements.txt` for compatibility with muon and gseapy.
