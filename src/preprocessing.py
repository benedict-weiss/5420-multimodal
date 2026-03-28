"""
preprocessing.py — Data loading, modality split, normalization, PCA, pathway tokenization

Implement the following functions:

load_data(path: str) -> AnnData
    - Load the h5ad file with anndata.read_h5ad()
    - Print shape and inspect adata.obs columns to identify cell type label column
    - Print adata.var['feature_types'].value_counts() to confirm GEX/ADT split

split_modalities(adata: AnnData) -> tuple[AnnData, AnnData]
    - RNA: adata[:, adata.var['feature_types'] == 'GEX']
    - Protein: adata[:, adata.var['feature_types'] == 'ADT']
    - Return (rna_adata, protein_adata)

preprocess_rna(rna_adata: AnnData) -> np.ndarray
    - scanpy.pp.normalize_total(rna_adata, target_sum=1e4)
    - scanpy.pp.log1p(rna_adata)
    - scanpy.pp.highly_variable_genes(rna_adata, n_top_genes=4000)
    - rna_adata = rna_adata[:, rna_adata.var.highly_variable]
    - scanpy.pp.scale(rna_adata, max_value=10)
    - scanpy.tl.pca(rna_adata, n_comps=256)
    - Return rna_adata.obsm['X_pca'] — shape (n_cells, 256)

preprocess_protein(protein_adata: AnnData) -> np.ndarray
    - muon.prot.pp.clr(protein_adata)
    - Return protein_adata.X as dense array — shape (n_cells, 134)

build_pathway_tokens(rna_adata: AnnData, min_genes: int = 5) -> tuple[np.ndarray, list[str]]
    - Call AFTER normalize_total + log1p but BEFORE HVG subsetting and scaling
    - Load gene sets: gseapy.get_library('KEGG_2021_Human')
    - For each pathway, find intersection of pathway genes with rna_adata.var_names
    - Keep pathways with >= min_genes measured genes
    - For each kept pathway, average the log-normalized expression across member genes per cell
    - Return (pathway_matrix, pathway_names):
        - pathway_matrix: np.ndarray shape (n_cells, n_pathways)
        - pathway_names: list of pathway name strings (for attention interpretation)
    - NOTE: gene name format must match between KEGG library and adata.var_names

get_labels(adata: AnnData) -> tuple[np.ndarray, dict]
    - Extract cell type labels from adata.obs (column name TBD — inspect after loading)
    - Encode as integers with sklearn.preprocessing.LabelEncoder
    - Return (encoded_labels, label_mapping_dict)

split_by_donor(adata: AnnData, test_donors: list[str]) -> tuple[np.ndarray, np.ndarray]
    - Return (train_indices, test_indices) based on donor column in adata.obs
    - Hold out 2-3 donors for test set
"""
