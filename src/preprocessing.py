"""
preprocessing.py — Data loading, modality split, normalization, PCA, pathway tokenization

Functions:
  load_data(path: str) -> AnnData
  split_modalities(adata: AnnData) -> tuple[AnnData, AnnData]
  preprocess_rna(rna_adata: AnnData, n_comps=256, pca_model=None, return_pca_model=False) -> np.ndarray | tuple
  preprocess_protein(protein_adata: AnnData) -> np.ndarray
  build_pathway_tokens(rna_adata: AnnData, min_genes=5, gene_sets=None) -> tuple[np.ndarray, list[str]]
  build_gene_tokens(rna_adata: AnnData, n_hvgs=512, hvg_genes=None) -> tuple[np.ndarray, list[str]]
  get_labels(adata: AnnData, label_col: str = "cell_type") -> tuple[np.ndarray, dict]
  split_by_donor(adata: AnnData, test_donors: list[str], donor_col: str = "DonorNumber") -> tuple[np.ndarray, np.ndarray]
"""
import gzip
import shutil
import tempfile
import os

import numpy as np
import pandas as pd
import anndata
import scanpy
import muon
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


def load_data(path: str) -> anndata.AnnData:
    """
    Load h5ad file (gzip-compressed or not), print shape and metadata.

    Args:
        path: Path to .h5ad or .h5ad.gz file

    Returns:
        AnnData object with all cells and features
    """
    print(f"Loading dataset from {path}...")

    # Handle gzip decompression
    if path.endswith('.gz'):
        with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with gzip.open(path, 'rb') as f_in:
                with open(tmp_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            adata = anndata.read_h5ad(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        adata = anndata.read_h5ad(path)

    print(f"  Shape: {adata.shape}")
    print(f"  var columns: {list(adata.var.columns)}")
    print(f"  obs columns: {list(adata.obs.columns)[:5]}... (total {len(adata.obs.columns)})")
    print(f"  feature_types: {adata.var['feature_types'].value_counts().to_dict()}")

    return adata


def split_modalities(adata: anndata.AnnData) -> tuple[anndata.AnnData, anndata.AnnData]:
    """
    Split full AnnData into RNA and protein modalities.

    Args:
        adata: Full AnnData with both 'GEX' and 'ADT' in var['feature_types']

    Returns:
        (rna_adata, protein_adata) tuple
    """
    if 'feature_types' not in adata.var.columns:
        raise KeyError(
            f"'feature_types' column not found in adata.var. "
            f"Available columns: {list(adata.var.columns)}"
        )

    rna_adata = adata[:, adata.var['feature_types'] == 'GEX'].copy()
    protein_adata = adata[:, adata.var['feature_types'] == 'ADT'].copy()

    print(f"Split modalities: RNA {rna_adata.shape}, Protein {protein_adata.shape}")

    return rna_adata, protein_adata


def preprocess_rna(
    rna_adata: anndata.AnnData,
    n_comps: int = 256,
    pca_model: PCA = None,
    return_pca_model: bool = False,
    hvg_genes: list = None,
) -> np.ndarray | tuple[np.ndarray, PCA, list]:
    """
    Normalize, log-transform, select HVGs, scale, and apply PCA to RNA data.

    Args:
        rna_adata: RNA AnnData object
        n_comps: Number of PCA components (capped by min(n_cells, n_genes) at fit time)
        pca_model: Pre-fit sklearn PCA model. If None, fits a new model (training path).
                   If provided, transforms without refitting (inference path).
        return_pca_model: If True, return (matrix, pca_model, hvg_genes) 3-tuple.
        hvg_genes: Ordered list of HVG gene names from training. Must be provided when
                   pca_model is provided, to ensure the same feature set is used.
                   If None and pca_model is provided, HVGs will be re-selected on the
                   current data (wrong — will likely cause a feature mismatch crash).

    Returns:
        If return_pca_model=False: np.ndarray of shape (n_cells, n_comps)
        If return_pca_model=True: (np.ndarray, PCA model, hvg_gene_list)
    """
    import warnings

    if pca_model is not None and hvg_genes is None:
        warnings.warn(
            "pca_model provided without hvg_genes. HVG selection will be re-run on this "
            "data, which may select different genes than training and cause a feature "
            "mismatch error. Pass hvg_genes from the training call to avoid this.",
            UserWarning,
        )

    adata = rna_adata.copy()

    # Normalize and log-transform
    scanpy.pp.normalize_total(adata, target_sum=1e4)
    scanpy.pp.log1p(adata)

    # HVG subsetting: use training gene list on inference path to avoid feature mismatch
    if hvg_genes is not None:
        genes_present = [g for g in hvg_genes if g in adata.var_names]
        adata = adata[:, genes_present]
    else:
        scanpy.pp.highly_variable_genes(adata, n_top_genes=4000)
        adata = adata[:, adata.var.highly_variable]

    selected_hvg_genes = list(adata.var_names)

    # Scale
    scanpy.pp.scale(adata, max_value=10)

    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    # PCA: cap n_comps only at fit time; at transform time the model already knows its shape
    if pca_model is None:
        max_comps = min(adata.shape[0] - 1, adata.shape[1] - 1)
        actual_n_comps = min(n_comps, max_comps)
        pca_model = PCA(n_components=actual_n_comps, random_state=0)
        X_pca = pca_model.fit_transform(X)
    else:
        X_pca = pca_model.transform(X)

    print(f"RNA PCA output shape: {X_pca.shape}")

    if return_pca_model:
        return X_pca.astype(np.float32), pca_model, selected_hvg_genes
    else:
        return X_pca.astype(np.float32)


def preprocess_protein(protein_adata: anndata.AnnData) -> np.ndarray:
    """
    Apply CLR normalization to protein data.

    Args:
        protein_adata: Protein AnnData object

    Returns:
        Dense np.ndarray of shape (n_cells, n_proteins)
    """
    adata = protein_adata.copy()

    # Convert to dense if sparse (needed for scalar addition)
    if hasattr(adata.X, 'toarray'):
        adata.X = adata.X.toarray()

    # Add pseudocount of 1 (appropriate for integer ADT count data) to avoid log(0) in CLR
    adata.X = adata.X + 1.0

    # CLR normalization (muon.prot.pp.clr modifies adata in-place)
    muon.prot.pp.clr(adata)

    # Convert to dense if sparse
    X_protein = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    print(f"Protein CLR output shape: {X_protein.shape}")

    return X_protein.astype(np.float32)


def build_pathway_tokens(
    rna_adata: anndata.AnnData,
    min_genes: int = 5,
    gene_sets: dict = None
) -> tuple[np.ndarray, list]:
    """
    Build pathway-level tokens by averaging gene expression within each pathway.

    IMPORTANT: Call AFTER normalize_total + log1p, but BEFORE HVG subsetting.

    Args:
        rna_adata: RNA AnnData (must have log-normalized counts)
        min_genes: Minimum number of genes per pathway to include
        gene_sets: Dict of {pathway_name: [gene_names]} (if None, loads KEGG_2021_Human)

    Returns:
        (pathway_matrix, pathway_names) tuple where:
          - pathway_matrix: np.ndarray shape (n_cells, n_pathways)
          - pathway_names: list of pathway name strings
    """
    import warnings

    # Load gene sets if not provided
    if gene_sets is None:
        import gseapy
        print("Loading KEGG_2021_Human gene sets...")
        # gseapy 0.10.x has no get_library(); use gsea_gmt_parser which downloads from Enrichr
        if hasattr(gseapy, 'get_library'):
            gene_sets = gseapy.get_library('KEGG_2021_Human')
        else:
            gene_sets = gseapy.parser.gsea_gmt_parser('KEGG_2021_Human')

    adata = rna_adata.copy()

    # Handle non-unique var_names — duplicates would silently inflate pathway averages
    # and cause anndata to raise InvalidIndexError when subsetting by gene name.
    if not adata.var_names.is_unique:
        warnings.warn(
            "rna_adata.var_names are not unique. Making them unique before pathway averaging. "
            "Call .var_names_make_unique() on your AnnData before preprocessing to avoid this.",
            UserWarning,
        )
        adata.var_names_make_unique()

    # Ensure counts are log-normalized.
    # log1p(1e4) ≈ 9.21; threshold of 20 also accommodates CPM (log1p(1e6) ≈ 13.8).
    x_max = adata.X.max()
    if hasattr(x_max, 'item'):
        x_max = x_max.item()
    if x_max > 20:
        raise ValueError(
            f"Input data max value is {x_max:.1f}, which suggests raw or un-logged counts. "
            "build_pathway_tokens must be called AFTER normalize_total + log1p."
        )

    pathways_data = []
    pathway_names = []

    for pathway_name, genes in gene_sets.items():
        # Find genes in this pathway that are measured in the data.
        # Deduplicate to avoid double-weighting when var_names are non-unique.
        seen: set = set()
        genes_in_data = []
        for g in genes:
            if g in adata.var_names and g not in seen:
                genes_in_data.append(g)
                seen.add(g)

        # Keep pathway if it has enough genes
        if len(genes_in_data) >= min_genes:
            # Average expression across genes in this pathway
            pathway_expr = adata[:, genes_in_data].X
            if hasattr(pathway_expr, 'toarray'):
                pathway_expr = pathway_expr.toarray()
            pathway_avg = pathway_expr.mean(axis=1)

            pathways_data.append(pathway_avg)
            pathway_names.append(pathway_name)

    if len(pathways_data) == 0:
        # No pathways passed min_genes filter
        X_pathways = np.zeros((adata.shape[0], 0), dtype=np.float32)
    else:
        X_pathways = np.column_stack(pathways_data).astype(np.float32)

    print(f"Pathway tokens output shape: {X_pathways.shape}")
    print(f"  {len(pathway_names)} pathways included (>= {min_genes} genes)")

    return X_pathways, pathway_names


def build_gene_tokens(
    rna_adata: anndata.AnnData,
    n_hvgs: int = 512,
    hvg_genes: list = None,
) -> tuple[np.ndarray, list]:
    """
    Build per-gene scalar tokens from the top-N HVGs for direct gene×gene attention.

    IMPORTANT: Call AFTER normalize_total + log1p. Output is the log-normalized
    expression matrix subset to selected HVGs (no PCA, no rescale), so each token
    carries a direct gene identity.

    Args:
        rna_adata: RNA AnnData with log-normalized counts (post normalize_total + log1p).
        n_hvgs:    Number of top highly-variable genes to retain as tokens. Used only
                   when hvg_genes is None (training path).
        hvg_genes: Ordered gene list from training. When provided, skip HVG selection
                   and subset to these genes — use on inference paths to match the
                   training feature set.

    Returns:
        (gene_matrix, gene_names) where gene_matrix is (n_cells, n_genes_kept) float32.
    """
    import warnings

    adata = rna_adata.copy()

    if not adata.var_names.is_unique:
        warnings.warn(
            "rna_adata.var_names are not unique. Making them unique before gene tokenization. "
            "Call .var_names_make_unique() on your AnnData before preprocessing to avoid this.",
            UserWarning,
        )
        adata.var_names_make_unique()

    x_max = adata.X.max()
    if hasattr(x_max, "item"):
        x_max = x_max.item()
    if x_max > 20:
        raise ValueError(
            f"Input data max value is {x_max:.1f}, which suggests raw or un-logged counts. "
            "build_gene_tokens must be called AFTER normalize_total + log1p."
        )

    if hvg_genes is not None:
        genes_present = [g for g in hvg_genes if g in adata.var_names]
        adata = adata[:, genes_present]
    else:
        scanpy.pp.highly_variable_genes(adata, n_top_genes=n_hvgs)
        adata = adata[:, adata.var.highly_variable]

    selected_genes = list(adata.var_names)

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X = np.asarray(X, dtype=np.float32)

    print(f"Gene tokens output shape: {X.shape}")
    print(f"  {len(selected_genes)} HVG tokens retained")

    return X, selected_genes


def get_labels(
    adata: anndata.AnnData,
    label_col: str = "cell_type",
    encoder: LabelEncoder = None,
) -> tuple[np.ndarray, dict]:
    """
    Extract and encode cell type labels.

    Args:
        adata: AnnData object
        label_col: Column name in adata.obs containing cell type labels
        encoder: Pre-fit LabelEncoder from training. If None, fits a new encoder.
                 Pass the encoder from the training call to ensure consistent integer
                 mappings when encoding test data that may be missing some cell types.

    Returns:
        (encoded_labels, label_mapping_dict) tuple where:
          - encoded_labels: np.ndarray of integer-encoded labels
          - label_mapping_dict: {original_label: int_code} mapping
    """
    if label_col not in adata.obs.columns:
        raise KeyError(f"Column '{label_col}' not found in adata.obs. Available: {list(adata.obs.columns)}")

    labels = adata.obs[label_col].values

    if encoder is None:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(labels)
    else:
        encoded = encoder.transform(labels)

    # Create mapping
    label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    print(f"Cell type labels: {len(encoder.classes_)} classes")
    print(f"  {label_mapping}")

    return encoded, label_mapping


def split_by_donor(
    adata: anndata.AnnData,
    test_donors: list,
    donor_col: str = "DonorNumber"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split cells into train and test sets based on donor ID.

    Args:
        adata: AnnData object
        test_donors: List of donor IDs to hold out for test set
        donor_col: Column name in adata.obs containing donor IDs

    Returns:
        (train_indices, test_indices) tuple of integer indices
    """
    if donor_col not in adata.obs.columns:
        raise KeyError(f"Column '{donor_col}' not found in adata.obs. Available: {list(adata.obs.columns)}")

    donors = adata.obs[donor_col].values

    # Create index arrays
    test_mask = np.isin(donors, test_donors)
    test_indices = np.where(test_mask)[0]
    train_indices = np.where(~test_mask)[0]

    print(f"Train/test split by {donor_col}:")
    print(f"  Train: {len(train_indices)} cells")
    print(f"  Test: {len(test_indices)} cells")
    print(f"  Test donors: {test_donors}")

    return train_indices, test_indices
