"""
Run once: python tests/inspect_real_data.py

Loads the real dataset and prints all obs/var column names and metadata.
Output is used to populate conftest.py constants.
"""
import anndata

DATA_PATH = "data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz"


def main():
    import gzip
    import shutil
    import tempfile
    import os

    print("Loading dataset...")
    # If gzipped, decompress to temp file
    if DATA_PATH.endswith('.gz'):
        with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with gzip.open(DATA_PATH, 'rb') as f_in:
                with open(tmp_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            adata = anndata.read_h5ad(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        adata = anndata.read_h5ad(DATA_PATH)

    print(f"\nDataset shape: {adata.shape}")
    print(f"Data type: {type(adata.X)}\n")

    print("=" * 80)
    print("VAR (Features) Metadata")
    print("=" * 80)
    print(f"var columns: {list(adata.var.columns)}")
    print(f"\nfeature_types value counts:")
    print(adata.var['feature_types'].value_counts().to_dict())

    print("\n" + "=" * 80)
    print("OBS (Cell) Metadata")
    print("=" * 80)
    print(f"obs columns ({len(adata.obs.columns)}):")
    for col in adata.obs.columns:
        dtype = adata.obs[col].dtype
        n_unique = adata.obs[col].nunique()
        print(f"\n  {col!r:40s}  dtype={dtype}  n_unique={n_unique}")
        if n_unique <= 30:
            counts = adata.obs[col].value_counts().sort_index()
            for val, cnt in counts.items():
                print(f"    {val!r}: {cnt}")


if __name__ == "__main__":
    main()
