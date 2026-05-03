# Multimodal CITE-seq Cell State Prediction

Contrastive learning on paired single-cell RNA + surface protein (CITE-seq) data to classify immune cell types. Compares five models: RNA-only and protein-only baselines, a contrastive MLP, and two contrastive transformers — one using KEGG pathway tokens, one using individual gene tokens.

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
| RNA Baseline | RNA PCA (256-d) | Linear | Cross-entropy |
| Protein Baseline | Protein CLR (134-d) | Linear | Cross-entropy |
| Contrastive MLP | RNA PCA + Protein CLR | 2× MLP | CLIP contrastive + Cross-entropy |
| Contrastive TF (Pathway) | KEGG pathway tokens + Protein tokens | 2× Transformer | CLIP contrastive + Cross-entropy |
| Contrastive TF (Gene) | Gene tokens + Protein tokens | 2× Transformer | CLIP contrastive + Cross-entropy |

Contrastive models use two-stage training: (A) contrastive pretraining to align RNA and protein embeddings, then (B) classifier training on frozen embeddings. The transformer models add interpretability — attention weights reveal which pathways/genes and proteins drive classification for each cell type.

## Results (seed 13)

| Model | Accuracy | Macro AUROC |
|-------|----------|-------------|
| RNA Baseline | 73.2% | 0.987 |
| Protein Baseline | 75.3% | 0.992 |
| Contrastive MLP | 88.1% | 0.997 |
| Contrastive TF (Pathway) | 84.9% | 0.996 |
| Contrastive TF (Gene) | 85.7% | 0.996 |

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
│   ├── train_baseline_rna.py      # Model 1: RNA-only baseline
│   ├── train_baseline_protein.py  # Model 2: protein-only baseline
│   ├── train_contrastive_mlp.py   # Model 3: contrastive MLP
│   ├── train_contrastive_tf.py    # Model 4: contrastive transformer (pathway tokens)
│   ├── train_contrastive_tf_gene.py # Model 5: contrastive transformer (gene tokens)
│   ├── evaluate.py                # Metrics + paper figure generator (PHATE, ASW, AUROC)
│   └── attention_analysis.py      # Attention heatmaps and marker validation
├── tests/
│   ├── conftest.py                # Shared fixtures and dataset constants
│   ├── test_preprocessing.py      # 53 unit + integration tests for preprocessing
│   └── test_evaluate.py           # 17 unit tests for metric functions
├── scripts/                       # SLURM training and evaluation scripts
├── results/
│   ├── checkpoints/               # Model checkpoints and saved embeddings (per run)
│   └── figures/                   # Generated paper figures (organized by seed)
├── requirements.txt               # Python dependencies (pinned where needed)
├── pytest.ini                     # Test configuration
└── README.md
```

## Training

```bash
# Baselines (~5-10 min each, CPU ok)
python -m src.train_baseline_rna --data_path data/ --seed 42
python -m src.train_baseline_protein --data_path data/ --seed 42

# Contrastive MLP (~30-45 min, GPU recommended)
python -m src.train_contrastive_mlp --data_path data/ --seed 42 --batch_size 512

# Contrastive transformers (~60-90 min each, GPU required)
python -m src.train_contrastive_tf --data_path data/ --seed 42 --batch_size 256
python -m src.train_contrastive_tf_gene --data_path data/ --seed 42 --batch_size 256
```

On HPC (runs pathway + gene transformers in sequence):

```bash
SEED=42 sbatch scripts/run_tf_train.sh
```

Each run saves a timestamped checkpoint directory under `results/checkpoints/` containing model weights, metrics, and test-set embeddings.

## Evaluation

Generate all paper figures from trained checkpoints:

```bash
# Pin specific runs (recommended)
python -m src.evaluate \
  --baseline_dir        results/checkpoints/baseline_rna_seed42_<timestamp> \
  --protein_baseline_dir results/checkpoints/baseline_protein_seed42_<timestamp> \
  --mlp_dir             results/checkpoints/contrastive_mlp_seed42_<timestamp> \
  --tf_dir              results/checkpoints/contrastive_tf_seed42_<timestamp> \
  --tf_gene_dir         results/checkpoints/contrastive_tf_gene_seed42_<timestamp> \
  --output_dir          results/figures/seed42

# Or auto-discover latest run of each type
python -m src.evaluate --checkpoint_dir results/checkpoints --output_dir results/figures
```

Produces:

| Figure | Description |
|--------|-------------|
| `fig_model_comparison.png` | Accuracy and Macro AUROC bar chart across all 5 models |
| `fig_training_curves.png` | Train/val loss over epochs with final test loss line |
| `fig_accuracy_curves.png` | Train/val accuracy over epochs (Stage B) |
| `fig_phate_{baseline,baseline_protein,mlp,tf,tf_gene}.png` | PHATE 2D embedding colored by cell type |
| `fig_asw.png` | Normalized average silhouette width per model |

## Attention Analysis

```bash
python -m src.attention_analysis \
  --checkpoint_dir results/checkpoints/contrastive_tf_seed42_<timestamp>
```

Produces attention heatmaps (RNA pathways and protein markers per cell type) and a `marker_validation.json` reporting recall of known markers (CD3→T cells, CD19→B cells, CD34→HSCs, etc.).

## Testing

```bash
# All tests (70 total, ~20s)
python -m pytest tests/ -v

# Preprocessing only (53 tests)
python -m pytest tests/test_preprocessing.py -v

# Evaluate metric functions (17 tests)
python -m pytest tests/test_evaluate.py -v
```

## Dependencies

anndata, scanpy, muon, gseapy, torch, scikit-learn, phate, matplotlib, seaborn, pytest

**Note**: `numpy<2` and `setuptools<78` are pinned in `requirements.txt` for compatibility with muon and gseapy.
