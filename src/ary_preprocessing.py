"""Implement the following functions:

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
import anndata as an
import scanpy as sc
import muon
import numpy as np
import gseapy
from sklearn.preprocessing import LabelEncoder

## Function definitions to load and preprocess the data from a raw h5ad file

def load_data(path: str): # loads in the data 
    multidata = an.read_h5ad(path)
    print("Shape:", multidata.shape,"\n") # expected: (num_cells, num_features )
    print("obs columns:", multidata.obs.columns.tolist()) #
    print("\nFeature types:\n", multidata.var['feature_types'].value_counts())
    return multidata

# Loads data and prints pertinent information for data loading checks

# Divide data into RNA and protein transcription data
def split_modalities(multidata: an.AnnData) -> tuple[an.AnnData, an.AnnData]:
    rna_adata = multidata[:, multidata.var['feature_types'] == 'GEX'].copy() 
    protein_adata = multidata[:, multidata.var['feature_types'] == 'ADT'].copy()
    return rna_adata, protein_adata # returns the respective subsects


def preprocess_rna(rna_adata: an.AnnData):
    sc.pp.normalize_total(rna_adata, target_sum=1e4) # normalize to find proportions of expression
    sc.pp.log1p(rna_adata) # compresses scale and normalizes relative differences
    sc.pp.highly_variable_genes(rna_adata, n_top_genes=4000) # we're gleaning the top 4000 most variable genes
    rna_adata = rna_adata[:, rna_adata.var.highly_variable].copy() # we only care about these highly variable genes in our analysis
    sc.pp.scale(rna_adata, max_value=10) # scaling based on size; want each gene to partipate equally to PCA
    sc.tl.pca(rna_adata, n_comps=256) # performs PCA, only want top 256 variable genes 
    return (rna_adata.obsm['X_pca']) # expected shape = (n_cells, 256)



def preprocess_protein(protein_adata: an.AnnData) -> np.ndarray:
    muon.prot.pp.clr(protein_adata) # normalizes across protein expression within that cell
    X = protein_adata.X # need array
    if hasattr(X, 'toarray'):  # checks if sparse
        X = X.toarray() # if so, assigns to dense
    return X  # expected shape = (n_cells, 134)

# Groups RNA data based on defined biological pathways 
def build_pathway_tokens(
    rna_adata: an.AnnData, # takes in RNA
    min_genes: int = 5 # arbitrarily thresholding for pathways with 5+ genes 
) -> tuple[np.ndarray, list[str]]: # returns a string of pathway tokens alongside each RNA profile with a matching token
    gene_sets = gseapy.get_library('KEGG_2021_Human') # gets gene list from KEGG library

    var_names = set(rna_adata.var_names) 
    pathway_matrix = [] 
    pathway_names = []

    
    X = rna_adata.X # Outputs dense expression matrix
    if hasattr(X, 'toarray'):
        X = X.toarray()

    gene_index = {gene: i for i, gene in enumerate(rna_adata.var_names)} # checks for the corresponding variables/tokens

    for pathway_name, pathway_genes in gene_sets.items():
        # Loops through each gene in every pathway, then checks if measured
        measured_genes = [g for g in pathway_genes if g in var_names] 

        # Discards if not enough genes to constitute as meaningful pathway information
        if len(measured_genes) < min_genes: 
            continue

        # Get column indices for this pathway's genes        
        indices = [gene_index[g] for g in measured_genes] 

        # Average log-normalized expression across member genes per cell
        pathway_avg = X[:, indices].mean(axis=1) # Takes mean of the values of each cell's pathway columns (averaging across a single cell)
        pathway_matrix.append(pathway_avg) # filling in with these averages
        pathway_names.append(pathway_name) # filling in corresponding names

    # Coalesceses these pathways into a single matrix; expected size = (n_cells, n_pathways)
    pathway_matrix = np.stack(pathway_matrix, axis=1)  
    return pathway_matrix, pathway_names # returns the normalized genes for each matrix and their respective var_name

 # Encodes cell type labels into integer representations
def get_labels(multidata: an.AnnData) -> tuple[np.ndarray, dict]:

    label_col = None # Checking that there is data
    for candidate in ['cell_type', 'celltype', 'cell_type_l1', 'ct']: 
        if candidate in multidata.obs.columns: # finding and saving the correct column names
            label_col = candidate
            break

    if label_col is None: # Raises error if could not be detected 
        print("Available obs columns:", multidata.obs.columns.tolist()) 
        raise ValueError("Unable to locate cell type column — check output above and manually set label_col.")

    le = LabelEncoder()
    encoded = le.fit_transform(multidata.obs[label_col]) # converts to integer representations
    label_mapping = {i: name for i, name in enumerate(le.classes_)} # does this for every cell type column
    return encoded, label_mapping # returns these translations (integer reps) and their corresponding mappings of each recording


def split_by_donor(
    multidata: an.AnnData, # takes in the h5ad data
    test_donors: list[str] # deals with the list of donors
) -> tuple[np.ndarray, np.ndarray]:

    donor_series = multidata.obs['DonorID'] # extracts DonorID for every cell

    # Sorts based on whether it is labelled as a test or training datapoint
    test_mask = donor_series.isin(test_donors) # specifically assigns testing to our indicated donors
    train_mask = ~test_mask

    # Returns indices for each class
    train_indices = np.where(train_mask)[0] 
    test_indices = np.where(test_mask)[0]

    return train_indices, test_indices
