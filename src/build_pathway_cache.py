"""Build and save the pathway matrix cache used by train_contrastive_tf.py.

Run once before training:
    python -m src.build_pathway_cache --data_path data/ --gene_sets_path data/kegg_2021_human.json

Outputs (beside the data file):
    pathway_matrix_cache.npy   -- float32 array (n_cells, n_pathways)
    pathway_names_cache.json   -- list of pathway name strings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scanpy as sc

from src.preprocessing import build_pathway_tokens, load_data, split_modalities


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-build pathway token matrix cache.")
    parser.add_argument("--data_path", default="data/", help="Path to data dir or .h5ad[.gz] file")
    parser.add_argument("--gene_sets_path", default="data/kegg_2021_human.json")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cache exists")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    data_dir = data_path if data_path.is_dir() else data_path.parent
    cache_mat = data_dir / "pathway_matrix_cache.npy"
    cache_names = data_dir / "pathway_names_cache.json"

    if cache_mat.exists() and cache_names.exists() and not args.force:
        mat = np.load(str(cache_mat))
        print(f"Cache already exists: {cache_mat} {mat.shape} — use --force to rebuild.")
        return

    # Resolve dataset file
    if data_path.is_file():
        dataset_file = str(data_path)
    else:
        candidates = sorted(list(data_path.glob("*.h5ad")) + list(data_path.glob("*.h5ad.gz")))
        if not candidates:
            raise FileNotFoundError(f"No .h5ad or .h5ad.gz files found under {data_path}")
        dataset_file = str(candidates[0])

    print(f"Loading dataset: {dataset_file}")
    adata = load_data(dataset_file)
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    print(f"  {adata.shape[0]} cells, {adata.shape[1]} features")

    rna_adata, _ = split_modalities(adata)
    sc.pp.normalize_total(rna_adata, target_sum=1e4)
    sc.pp.log1p(rna_adata)

    print(f"Loading gene sets from {args.gene_sets_path}...")
    with open(args.gene_sets_path, "r", encoding="utf-8") as f:
        gene_sets = json.load(f)

    print("Building pathway tokens...")
    pathway_matrix, pathway_names = build_pathway_tokens(rna_adata, gene_sets=gene_sets)

    np.save(str(cache_mat), pathway_matrix)
    with open(cache_names, "w", encoding="utf-8") as f:
        json.dump(pathway_names, f)

    print(f"Saved: {cache_mat} {pathway_matrix.shape}")
    print(f"Saved: {cache_names} ({len(pathway_names)} pathways)")


if __name__ == "__main__":
    main()
