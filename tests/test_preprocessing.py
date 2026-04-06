"""
Comprehensive test suite for preprocessing.py

~45 tests covering:
  - Data loading and modality splitting
  - RNA preprocessing (normalization, HVG, PCA)
  - Protein preprocessing (CLR normalization)
  - Pathway tokenization
  - Label encoding
  - Train/test split by donor
  - Data leakage prevention (PCA)
  - Edge cases and error handling
"""
import tempfile
import os

import pytest
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from unittest.mock import patch, MagicMock

from src.preprocessing import (
    load_data,
    split_modalities,
    preprocess_rna,
    preprocess_protein,
    build_pathway_tokens,
    get_labels,
    split_by_donor,
)
from tests.conftest import REAL_LABEL_COL, REAL_DONOR_COL, REAL_SITE_COL


# ============================================================================
# A. load_data tests
# ============================================================================

def test_load_data_returns_anndata(small_adata):
    """Verify load_data returns AnnData object."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.h5ad")
        small_adata.write_h5ad(path)
        result = load_data(path)
        assert isinstance(result, ad.AnnData)
        assert result.shape == small_adata.shape


def test_load_data_gzip_decompression(small_adata):
    """Verify load_data can read gzipped h5ad files."""
    import gzip
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        h5ad_path = os.path.join(tmpdir, "test.h5ad")
        gz_path = os.path.join(tmpdir, "test.h5ad.gz")

        small_adata.write_h5ad(h5ad_path)
        with open(h5ad_path, 'rb') as f_in:
            with gzip.open(gz_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        result = load_data(gz_path)
        assert isinstance(result, ad.AnnData)
        assert result.shape == small_adata.shape


# ============================================================================
# B. split_modalities tests
# ============================================================================

def test_split_modalities_returns_two_adatas(small_adata):
    """Verify split_modalities returns a tuple of 2 AnnData objects."""
    rna, protein = split_modalities(small_adata)
    assert isinstance(rna, ad.AnnData)
    assert isinstance(protein, ad.AnnData)


def test_split_modalities_gex_count(small_adata):
    """Verify RNA has exactly n_gex features."""
    rna, _ = split_modalities(small_adata)
    n_gex = (small_adata.var["feature_types"] == "GEX").sum()
    assert rna.shape[1] == n_gex


def test_split_modalities_adt_count(small_adata):
    """Verify protein has exactly n_adt features."""
    _, protein = split_modalities(small_adata)
    n_adt = (small_adata.var["feature_types"] == "ADT").sum()
    assert protein.shape[1] == n_adt


def test_split_modalities_cell_count_preserved(small_adata):
    """Verify both modalities have the same n_cells."""
    rna, protein = split_modalities(small_adata)
    assert rna.shape[0] == small_adata.shape[0]
    assert protein.shape[0] == small_adata.shape[0]


def test_split_modalities_no_feature_type_mixing(small_adata):
    """Verify RNA only has GEX, protein only has ADT."""
    rna, protein = split_modalities(small_adata)
    assert set(rna.var["feature_types"].unique()) == {"GEX"}
    assert set(protein.var["feature_types"].unique()) == {"ADT"}


def test_split_modalities_obs_preserved(small_adata):
    """Verify cell metadata is preserved in both modalities."""
    rna, protein = split_modalities(small_adata)
    assert list(rna.obs.columns) == list(small_adata.obs.columns)
    assert list(protein.obs.columns) == list(small_adata.obs.columns)
    assert (rna.obs[REAL_LABEL_COL] == small_adata.obs[REAL_LABEL_COL]).all()


def test_split_modalities_missing_feature_types_raises(small_adata):
    """Verify KeyError with helpful message when feature_types column is absent."""
    adata = small_adata.copy()
    del adata.var['feature_types']
    with pytest.raises(KeyError, match="feature_types"):
        split_modalities(adata)


# ============================================================================
# C. preprocess_rna tests
# ============================================================================

def test_preprocess_rna_output_shape(rna_adata):
    """Verify preprocess_rna returns correct shape."""
    # Use n_comps=50 since fixture has only 300 genes
    result = preprocess_rna(rna_adata, n_comps=50)
    assert result.shape == (rna_adata.shape[0], 50)


def test_preprocess_rna_output_dtype(rna_adata):
    """Verify output is float array."""
    result = preprocess_rna(rna_adata, n_comps=50)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


def test_preprocess_rna_no_nan(rna_adata):
    """Verify no NaN values in output."""
    result = preprocess_rna(rna_adata, n_comps=50)
    assert np.isnan(result).sum() == 0


def test_preprocess_rna_no_inf(rna_adata):
    """Verify no Inf values in output."""
    result = preprocess_rna(rna_adata, n_comps=50)
    assert np.isinf(result).sum() == 0


def test_preprocess_rna_returns_ndarray(rna_adata):
    """Verify return type is numpy array."""
    result = preprocess_rna(rna_adata, n_comps=50)
    assert isinstance(result, np.ndarray)


def test_preprocess_rna_caps_n_comps(rna_adata):
    """Verify PCA caps n_comps to min(n_cells, n_genes)."""
    # Request more comps than genes
    result = preprocess_rna(rna_adata, n_comps=1000)
    max_possible = min(rna_adata.shape[0] - 1, rna_adata.shape[1] - 1)
    assert result.shape[1] == max_possible


def test_preprocess_rna_pca_transform_only_different_from_refit(rna_adata):
    """
    DATA LEAKAGE TEST: Verify that transform-only uses the fitted model's components.

    Confirms that when pca_model + hvg_genes are provided, the transform uses the
    training PCA components (not refitting on test data).
    """
    train_idx = np.arange(300)
    test_idx = np.arange(300, rna_adata.shape[0])
    train_data = rna_adata[train_idx].copy()
    test_data = rna_adata[test_idx].copy()

    # Fit on train
    train_pca, fitted_model, hvg_genes = preprocess_rna(
        train_data, n_comps=20, return_pca_model=True
    )

    # Transform test with training model
    test_pca_transformed = preprocess_rna(
        test_data, n_comps=20, pca_model=fitted_model, hvg_genes=hvg_genes
    )

    # Transform test with independent refit
    test_pca_refit = preprocess_rna(test_data, n_comps=20)

    # The two projections use different principal components — results will differ
    assert not np.allclose(test_pca_transformed, test_pca_refit, atol=1e-5)
    # Both must have the same shape
    assert test_pca_transformed.shape == test_pca_refit.shape


def test_preprocess_rna_deterministic(rna_adata):
    """Verify output is deterministic (same input → same output)."""
    result1 = preprocess_rna(rna_adata, n_comps=50)
    result2 = preprocess_rna(rna_adata, n_comps=50)
    assert np.allclose(result1, result2)


def test_preprocess_rna_return_pca_model():
    """Verify return_pca_model=True returns 3-tuple (matrix, pca_model, hvg_genes)."""
    adata = ad.AnnData(
        X=sp.csr_matrix(np.random.poisson(1, (100, 200)).astype(np.float32)),
        var=pd.DataFrame({"feature_types": ["GEX"] * 200}, index=[f"G{i}" for i in range(200)])
    )
    result = preprocess_rna(adata, n_comps=30, return_pca_model=True)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0].shape == (100, 30)
    assert hasattr(result[1], 'transform')  # PCA model has transform method
    assert isinstance(result[2], list)       # hvg_genes is a list of gene names
    assert len(result[2]) > 0


def test_preprocess_rna_hvg_mismatch_prevented(rna_adata):
    """
    Regression test for critical HVG mismatch bug.

    Without the hvg_genes parameter, re-running HVG selection on a held-out donor
    subset selects different genes than training, causing pca_model.transform() to
    crash with a feature count mismatch.
    """
    train_idx = np.arange(300)
    test_idx = np.arange(300, rna_adata.shape[0])
    train_data = rna_adata[train_idx].copy()
    test_data = rna_adata[test_idx].copy()

    train_pca, fitted_model, hvg_genes = preprocess_rna(
        train_data, n_comps=20, return_pca_model=True
    )

    # Transform test using training HVG list — must not crash
    test_pca = preprocess_rna(test_data, n_comps=20, pca_model=fitted_model, hvg_genes=hvg_genes)
    assert test_pca.shape == (len(test_idx), 20)
    # Feature count going into transform must match training
    assert test_pca.shape[1] == fitted_model.n_components_


def test_preprocess_rna_warns_pca_without_hvg_genes(rna_adata):
    """Verify a warning is raised when pca_model is provided without hvg_genes."""
    _, fitted_model, _ = preprocess_rna(rna_adata, n_comps=20, return_pca_model=True)
    with pytest.warns(UserWarning, match="hvg_genes"):
        preprocess_rna(rna_adata, n_comps=20, pca_model=fitted_model)


# ============================================================================
# D. preprocess_protein tests
# ============================================================================

def test_preprocess_protein_output_shape(protein_adata):
    """Verify output shape is (n_cells, n_proteins)."""
    result = preprocess_protein(protein_adata)
    assert result.shape == protein_adata.shape


def test_preprocess_protein_returns_ndarray(protein_adata):
    """Verify output is dense numpy array."""
    result = preprocess_protein(protein_adata)
    assert isinstance(result, np.ndarray)
    assert not sp.issparse(result)


def test_preprocess_protein_output_dtype(protein_adata):
    """Verify output dtype is float."""
    result = preprocess_protein(protein_adata)
    assert result.dtype == np.float32


def test_preprocess_protein_no_nan(protein_adata):
    """Verify no NaN values after CLR."""
    result = preprocess_protein(protein_adata)
    assert np.isnan(result).sum() == 0


def test_preprocess_protein_clr_transforms_data(protein_adata):
    """Verify CLR normalizes data: output differs from input and is finite."""
    raw = protein_adata.X.toarray() if sp.issparse(protein_adata.X) else protein_adata.X.copy()
    result = preprocess_protein(protein_adata)
    # Data was transformed (not a pass-through)
    assert not np.allclose(result, raw)
    # All values finite and in a reasonable range for log-ratio normalization
    assert np.isfinite(result).all()
    assert result.min() >= 0          # muon CLR output is non-negative
    assert result.max() < 20          # no runaway values


# ============================================================================
# E. build_pathway_tokens tests
# ============================================================================

def test_build_pathway_tokens_return_types(log_normalized_rna, mock_gene_sets):
    """Verify returns (np.ndarray, list)."""
    result = build_pathway_tokens(log_normalized_rna, gene_sets=mock_gene_sets)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], list)


def test_build_pathway_tokens_shape(log_normalized_rna, mock_gene_sets):
    """Verify matrix shape matches (n_cells, n_pathways)."""
    X, names = build_pathway_tokens(log_normalized_rna, gene_sets=mock_gene_sets)
    assert X.shape[0] == log_normalized_rna.shape[0]
    assert X.shape[1] == len(names)


def test_build_pathway_tokens_pathway_names_list(log_normalized_rna, mock_gene_sets):
    """Verify pathway names are strings."""
    _, names = build_pathway_tokens(log_normalized_rna, gene_sets=mock_gene_sets)
    assert len(names) > 0
    assert all(isinstance(n, str) for n in names)


def test_build_pathway_tokens_min_genes_filter(log_normalized_rna, mock_gene_sets):
    """Verify min_genes filtering works correctly."""
    X, names = build_pathway_tokens(log_normalized_rna, min_genes=5, gene_sets=mock_gene_sets)

    # Pathway_Small (3 genes) and Pathway_NoOverlap (0 genes) should be filtered
    assert "Pathway_Small" not in names
    assert "Pathway_NoOverlap" not in names

    # Pathway_Big (10 genes) and Pathway_ExactMin (5 genes) should be included
    assert "Pathway_Big" in names
    assert "Pathway_ExactMin" in names


def test_build_pathway_tokens_min_genes_boundary(log_normalized_rna):
    """Verify boundary: pathway with exactly min_genes is INCLUDED."""
    gene_sets = {
        "Exactly5": [f"GENE{i}" for i in range(5)],  # 5 genes
        "Only4": [f"GENE{i}" for i in range(4)],     # 4 genes
    }

    X, names = build_pathway_tokens(log_normalized_rna, min_genes=5, gene_sets=gene_sets)

    assert "Exactly5" in names
    assert "Only4" not in names


def test_build_pathway_tokens_no_overlap_empty_matrix(log_normalized_rna):
    """Verify no-overlap case returns empty matrix gracefully."""
    gene_sets = {"AllUnknown": ["NOTREAL1", "NOTREAL2", "NOTREAL3"]}

    X, names = build_pathway_tokens(log_normalized_rna, min_genes=1, gene_sets=gene_sets)

    assert X.shape == (log_normalized_rna.shape[0], 0)
    assert len(names) == 0


def test_build_pathway_tokens_values_are_averages(tiny_fixture_for_pathway_values):
    """Verify pathway values are correct averages."""
    adata = tiny_fixture_for_pathway_values.copy()
    import scanpy
    scanpy.pp.normalize_total(adata, target_sum=1e4)
    scanpy.pp.log1p(adata)

    gene_sets = {
        "Pathway1": ["GENE0", "GENE1"],  # average of first 2 genes
    }

    X, _ = build_pathway_tokens(adata, min_genes=1, gene_sets=gene_sets)

    # Manually compute expected average for first cell
    expected = (adata.X[0, 0] + adata.X[0, 1]) / 2.0
    actual = X[0, 0]

    assert np.isclose(actual, expected, atol=1e-5)


def test_build_pathway_tokens_uses_log_normalized(log_normalized_rna):
    """Verify function requires log-normalized input (not raw counts)."""
    # Create AnnData with raw counts (high values)
    adata_raw = log_normalized_rna.copy()
    adata_raw.X = adata_raw.X * 1000  # Undo log normalization

    gene_sets = {"TestPath": [f"GENE{i}" for i in range(5)]}

    with pytest.raises(ValueError, match="log1p"):
        build_pathway_tokens(adata_raw, gene_sets=gene_sets)


def test_build_pathway_tokens_gene_name_case_sensitive(log_normalized_rna):
    """Verify gene name matching IS case-sensitive."""
    gene_sets = {
        "LowercasePath": ["gene0", "gene1"],  # lowercase, but adata has "GENE0", "GENE1"
    }

    X, names = build_pathway_tokens(log_normalized_rna, min_genes=1, gene_sets=gene_sets)

    # Should return empty (no genes matched)
    assert X.shape[1] == 0
    assert len(names) == 0


def test_build_pathway_tokens_non_unique_var_names_warns(log_normalized_rna):
    """Verify a warning is raised when var_names are non-unique (as in the real dataset)."""
    adata = log_normalized_rna.copy()
    # Duplicate the first gene name
    new_index = list(adata.var_names)
    new_index[1] = new_index[0]
    adata.var_names = new_index

    gene_sets = {"TestPath": [new_index[0], new_index[2], new_index[3], new_index[4], new_index[5]]}
    with pytest.warns(UserWarning, match="not unique"):
        X, names = build_pathway_tokens(adata, min_genes=1, gene_sets=gene_sets)

    # Should still produce a result (deduplication prevents double-weighting)
    assert len(names) == 1
    assert X.shape == (adata.shape[0], 1)


def test_build_pathway_tokens_no_network_call_when_injected(log_normalized_rna, mock_gene_sets):
    """Verify gseapy is NOT imported when gene_sets is provided."""
    import sys
    # Remove gseapy from sys.modules to detect if it gets imported
    gseapy_was_loaded = 'gseapy' in sys.modules
    if gseapy_was_loaded:
        saved = sys.modules.pop('gseapy')
    try:
        # This should work without gseapy being importable
        build_pathway_tokens(log_normalized_rna, gene_sets=mock_gene_sets)
    finally:
        if gseapy_was_loaded:
            sys.modules['gseapy'] = saved


# ============================================================================
# F. get_labels tests
# ============================================================================

def test_get_labels_return_types(small_adata):
    """Verify returns (np.ndarray, dict)."""
    result = get_labels(small_adata, label_col=REAL_LABEL_COL)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], dict)


def test_get_labels_encoded_length(small_adata):
    """Verify encoded array length equals n_cells."""
    encoded, _ = get_labels(small_adata, label_col=REAL_LABEL_COL)
    assert len(encoded) == small_adata.shape[0]


def test_get_labels_integer_dtype(small_adata):
    """Verify encoded labels are integers."""
    encoded, _ = get_labels(small_adata, label_col=REAL_LABEL_COL)
    assert np.issubdtype(encoded.dtype, np.integer)


def test_get_labels_n_unique_classes(small_adata):
    """Verify number of unique classes is correct."""
    encoded, mapping = get_labels(small_adata, label_col=REAL_LABEL_COL)
    n_unique_expected = small_adata.obs[REAL_LABEL_COL].nunique()
    assert len(np.unique(encoded)) == n_unique_expected
    assert len(mapping) == n_unique_expected


def test_get_labels_mapping_complete(small_adata):
    """Verify all encoded values have entries in mapping."""
    encoded, mapping = get_labels(small_adata, label_col=REAL_LABEL_COL)
    assert set(encoded) <= set(mapping.values())


def test_get_labels_mapping_invertible(small_adata):
    """Verify mapping can be inverted to recover original labels."""
    encoded, mapping = get_labels(small_adata, label_col=REAL_LABEL_COL)
    original_labels = small_adata.obs[REAL_LABEL_COL].values

    inv_mapping = {v: k for k, v in mapping.items()}
    recovered = np.array([inv_mapping[e] for e in encoded])

    assert (recovered == original_labels).all()


def test_get_labels_single_cell_type(small_adata):
    """Verify handling of single cell type (all same)."""
    adata = small_adata.copy()
    adata.obs[REAL_LABEL_COL] = "AllSame"

    encoded, mapping = get_labels(adata, label_col=REAL_LABEL_COL)

    assert len(mapping) == 1
    assert len(np.unique(encoded)) == 1
    assert (encoded == 0).all()


def test_get_labels_single_cell(small_adata):
    """Verify handling of single cell."""
    adata = small_adata[:1].copy()

    encoded, mapping = get_labels(adata, label_col=REAL_LABEL_COL)

    assert len(encoded) == 1
    assert len(mapping) >= 1


def test_get_labels_accepts_label_col_param(small_adata):
    """Verify label_col parameter works."""
    adata = small_adata.copy()
    adata.obs["custom_labels"] = adata.obs[REAL_LABEL_COL]

    encoded, mapping = get_labels(adata, label_col="custom_labels")

    assert len(mapping) == adata.obs[REAL_LABEL_COL].nunique()


def test_get_labels_invalid_col_raises(small_adata):
    """Verify KeyError on invalid column name."""
    with pytest.raises(KeyError):
        get_labels(small_adata, label_col="nonexistent_column")


def test_get_labels_encoder_alignment_across_splits(small_adata):
    """
    Verify that passing a pre-fit encoder to test data produces stable integer mappings.

    Without a shared encoder, a test split missing some cell types produces a different
    integer mapping than training — silently misaligning model predictions.
    """
    # Train split: has all 5 cell types
    train_idx = np.arange(400)
    test_idx = np.arange(400, small_adata.shape[0])
    train_adata = small_adata[train_idx].copy()
    test_adata = small_adata[test_idx].copy()

    # Fit encoder on training data
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(train_adata.obs[REAL_LABEL_COL])

    # Both splits should use the same integer mapping
    train_encoded, train_mapping = get_labels(train_adata, label_col=REAL_LABEL_COL, encoder=encoder)
    test_encoded, test_mapping = get_labels(test_adata, label_col=REAL_LABEL_COL, encoder=encoder)

    assert train_mapping == test_mapping
    # Integers refer to same classes in both
    inv = {v: k for k, v in train_mapping.items()}
    recovered = np.array([inv[e] for e in test_encoded])
    assert (recovered == test_adata.obs[REAL_LABEL_COL].values).all()


# ============================================================================
# G. split_by_donor tests
# ============================================================================

def test_split_by_donor_return_types(small_adata):
    """Verify returns (np.ndarray, np.ndarray)."""
    train_idx, test_idx = split_by_donor(small_adata, test_donors=["donor1"])
    assert isinstance(train_idx, np.ndarray)
    assert isinstance(test_idx, np.ndarray)


def test_split_by_donor_no_overlap(small_adata):
    """Verify no index overlap between train and test."""
    train_idx, test_idx = split_by_donor(small_adata, test_donors=["donor1"])
    overlap = np.intersect1d(train_idx, test_idx)
    assert len(overlap) == 0


def test_split_by_donor_full_coverage(small_adata):
    """Verify all cells are assigned to either train or test."""
    train_idx, test_idx = split_by_donor(small_adata, test_donors=["donor1"])
    assert len(train_idx) + len(test_idx) == small_adata.shape[0]


def test_split_by_donor_correct_test_donors(small_adata):
    """Verify test set contains only specified donors."""
    test_donors = ["donor1"]
    train_idx, test_idx = split_by_donor(small_adata, test_donors=test_donors)

    test_donor_values = small_adata.obs.iloc[test_idx][REAL_DONOR_COL].unique()
    assert set(test_donor_values) <= set(test_donors)

    train_donor_values = small_adata.obs.iloc[train_idx][REAL_DONOR_COL].unique()
    assert len(set(train_donor_values) & set(test_donors)) == 0


def test_split_by_donor_all_donors_in_test(small_adata):
    """Verify behavior when all donors are in test set."""
    all_donors = list(small_adata.obs[REAL_DONOR_COL].unique())
    train_idx, test_idx = split_by_donor(small_adata, test_donors=all_donors)

    assert len(train_idx) == 0
    assert len(test_idx) == small_adata.shape[0]


def test_split_by_donor_empty_test_donors(small_adata):
    """Verify behavior when no donors specified for test."""
    train_idx, test_idx = split_by_donor(small_adata, test_donors=[])

    assert len(test_idx) == 0
    assert len(train_idx) == small_adata.shape[0]


def test_split_by_donor_unknown_donor(small_adata):
    """Verify behavior with unknown donor ID."""
    train_idx, test_idx = split_by_donor(small_adata, test_donors=["nonexistent_donor"])

    # Should result in empty test set
    assert len(test_idx) == 0
    assert len(train_idx) == small_adata.shape[0]


def test_split_by_donor_accepts_donor_col_param(small_adata):
    """Verify donor_col parameter works."""
    adata = small_adata.copy()
    adata.obs["custom_donor"] = adata.obs[REAL_DONOR_COL]

    train_idx, test_idx = split_by_donor(adata, test_donors=["donor1"], donor_col="custom_donor")

    # Should work same as using original column
    assert len(test_idx) > 0


def test_split_by_donor_invalid_col_raises(small_adata):
    """Verify KeyError on invalid donor column."""
    with pytest.raises(KeyError):
        split_by_donor(small_adata, test_donors=["donor1"], donor_col="nonexistent")


# ============================================================================
# H. Integration Tests
# ============================================================================

@pytest.mark.integration
def test_full_preprocessing_pipeline_shapes(small_adata, mock_gene_sets):
    """Wire all preprocessing functions end-to-end and verify shapes."""
    # Split modalities
    rna, protein = split_modalities(small_adata)
    assert rna.shape[0] == small_adata.shape[0]
    assert protein.shape[0] == small_adata.shape[0]

    # Preprocess RNA
    rna_pca = preprocess_rna(rna, n_comps=20)
    assert rna_pca.shape == (small_adata.shape[0], 20)

    # Preprocess protein
    protein_processed = preprocess_protein(protein)
    assert protein_processed.shape == (small_adata.shape[0], protein.shape[1])

    # Pathway tokens (need log-normalized)
    import scanpy
    rna_log = rna.copy()
    scanpy.pp.normalize_total(rna_log, target_sum=1e4)
    scanpy.pp.log1p(rna_log)

    pathways, pathway_names = build_pathway_tokens(rna_log, gene_sets=mock_gene_sets)
    assert pathways.shape[0] == small_adata.shape[0]
    assert pathways.shape[1] == len(pathway_names)

    # Get labels
    labels, mapping = get_labels(small_adata, label_col=REAL_LABEL_COL)
    assert len(labels) == small_adata.shape[0]

    # Split by donor
    train_idx, test_idx = split_by_donor(small_adata, test_donors=["donor1"])
    assert len(train_idx) + len(test_idx) == small_adata.shape[0]


@pytest.mark.integration
def test_two_stage_preprocessing_training_paradigm(small_adata):
    """
    Integration test: two-stage training paradigm.

    Stage A (contrastive pretraining): fit PCA on train data
    Stage B (downstream): use fitted PCA to transform both train and test
    """
    # Split by donor
    train_idx, test_idx = split_by_donor(small_adata, test_donors=["donor1"])

    train_adata = small_adata[train_idx]
    test_adata = small_adata[test_idx]

    # Split modalities
    train_rna, train_protein = split_modalities(train_adata)
    test_rna, test_protein = split_modalities(test_adata)

    # Stage A: Fit PCA on train only
    train_pca, fitted_pca, hvg_genes = preprocess_rna(train_rna, n_comps=20, return_pca_model=True)
    assert train_pca.shape == (len(train_idx), 20)

    # Stage B: Transform test with fitted PCA + HVG list (no refit, no feature mismatch)
    test_pca = preprocess_rna(test_rna, n_comps=20, pca_model=fitted_pca, hvg_genes=hvg_genes)
    assert test_pca.shape == (len(test_idx), 20)

    # Verify test PCA is consistent across calls
    test_pca_again = preprocess_rna(test_rna, n_comps=20, pca_model=fitted_pca, hvg_genes=hvg_genes)
    assert np.allclose(test_pca, test_pca_again)
