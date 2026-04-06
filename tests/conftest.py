"""
Fixtures for preprocessing tests.

Column name constants discovered from real dataset via inspect_real_data.py
"""
import pytest
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

# Real column names from GSE194122 dataset
REAL_LABEL_COL = "cell_type"
REAL_DONOR_COL = "DonorNumber"
REAL_SITE_COL = "Site"
N_REAL_CELL_TYPES = 45
N_REAL_DONORS = 9


@pytest.fixture
def small_adata():
    """
    Synthetic AnnData mimicking real dataset structure.
    500 cells, 300 GEX + 10 ADT features.
    obs: cell_type (5 types), DonorNumber (4 donors), Site (2 sites)
    """
    rng = np.random.default_rng(42)
    n_cells = 500
    n_gex = 300
    n_adt = 10

    # Poisson-distributed counts (realistic sparsity)
    X_gex = sp.csr_matrix(rng.poisson(0.3, size=(n_cells, n_gex)).astype(np.float32))
    X_adt = sp.csr_matrix(rng.poisson(5.0, size=(n_cells, n_adt)).astype(np.float32))
    X = sp.hstack([X_gex, X_adt])

    # Feature metadata
    gene_names = [f"GENE{i}" for i in range(n_gex)]
    protein_names = [f"PROT{i}" for i in range(n_adt)]
    var = pd.DataFrame(
        {"feature_types": ["GEX"] * n_gex + ["ADT"] * n_adt},
        index=gene_names + protein_names
    )

    # Cell metadata: 5 cell types, 4 donors, 2 sites
    cell_types = ["B_cell", "CD4_T", "CD8_T", "Monocyte", "NK"]
    donors = ["donor1", "donor2", "donor3", "donor4"]
    sites = ["site1", "site2"]

    obs = pd.DataFrame(
        {
            REAL_LABEL_COL: pd.Categorical([cell_types[i % 5] for i in range(n_cells)]),
            REAL_DONOR_COL: pd.Categorical([donors[i % 4] for i in range(n_cells)]),
            REAL_SITE_COL: pd.Categorical([sites[i % 2] for i in range(n_cells)]),
        },
        index=[f"cell{i}" for i in range(n_cells)]
    )

    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def rna_adata(small_adata):
    """RNA modality from small_adata."""
    return small_adata[:, small_adata.var["feature_types"] == "GEX"].copy()


@pytest.fixture
def protein_adata(small_adata):
    """Protein modality from small_adata."""
    return small_adata[:, small_adata.var["feature_types"] == "ADT"].copy()


@pytest.fixture
def log_normalized_rna(rna_adata):
    """RNA after normalize_total + log1p, before HVG."""
    import scanpy
    adata = rna_adata.copy()
    scanpy.pp.normalize_total(adata, target_sum=1e4)
    scanpy.pp.log1p(adata)
    return adata


@pytest.fixture
def mock_gene_sets():
    """
    Minimal KEGG-like gene set dict for testing.
    Designed to test min_genes filtering.
    """
    return {
        "Pathway_Big": [f"GENE{i}" for i in range(10)],  # 10 genes → passes
        "Pathway_Small": [f"GENE{i}" for i in range(3)],  # 3 genes → filtered
        "Pathway_NoOverlap": ["UNKNOWN1", "UNKNOWN2"],  # 0 overlap → filtered
        "Pathway_ExactMin": [f"GENE{i}" for i in range(5)],  # 5 genes → passes (boundary)
    }


@pytest.fixture
def tiny_fixture_for_pathway_values():
    """
    Tiny AnnData for manual pathway value verification.
    2 cells, 5 genes, log-normalized.
    """
    rng = np.random.default_rng(123)
    n_cells = 2
    n_genes = 5

    # Fixed counts for reproducibility
    X = sp.csr_matrix(np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0],
         [5.0, 4.0, 3.0, 2.0, 1.0]], dtype=np.float32
    ))

    var = pd.DataFrame(
        {"feature_types": ["GEX"] * n_genes},
        index=[f"GENE{i}" for i in range(n_genes)]
    )

    obs = pd.DataFrame(
        {REAL_LABEL_COL: pd.Categorical(["A", "B"])},
        index=["cell0", "cell1"]
    )

    return ad.AnnData(X=X, obs=obs, var=var)
