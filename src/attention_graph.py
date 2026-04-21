"""
attention_graph.py — token×token attention adjacency → graph clustering → enrichment.

For contrastive-transformer encoders (src/models/transformer_encoder.py). Builds a
symmetric token×token adjacency from a trained encoder using one of three edge
weightings, clusters the resulting weighted graph, and runs enrichment analysis
per cluster.

Aggregation methods (all compose the full (S, S) map, then drop CLS row/col):
  - raw       : mean over layers and heads of the raw attention weights.
  - rollout   : Abnar & Zuidema (2020). Per-layer head-averaged A, propagated as
                A_hat = 0.5*A + 0.5*I (normalized), matrix-multiplied across layers.
  - grad_attn : Chefer et al. (CVPR 2021). Each layer's attention is weighted by
                ReLU(dL/dA) per head before averaging, then rolled out. Requires a
                classifier and counterpart-modality embeddings to form a loss.

Clustering: Leiden (leidenalg + python-igraph). Falls back to sklearn spectral
clustering when the igraph stack is not importable.

Enrichment: gseapy.enrichr. Token-level inputs are either gene symbols (gene
tokenization) or pathway names (pathway tokenization — the union of KEGG genes
backing the pathways in a cluster is submitted).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


# ---------------------------------------------------------------------------
# Checkpoint / data loading (mirrors src/attention_analysis.py conventions)
# ---------------------------------------------------------------------------


def _resolve_data_file(data_path: str) -> Path:
    path = Path(data_path)
    if path.is_file():
        return path
    candidates = sorted(list(path.glob("*.h5ad")) + list(path.glob("*.h5ad.gz")))
    if not candidates:
        raise FileNotFoundError(f"No .h5ad or .h5ad.gz files found under {data_path}")
    return candidates[0]


def _load_anndata(data_path: str):
    import anndata as ad

    path = _resolve_data_file(data_path)
    if path.suffix != ".gz":
        return ad.read_h5ad(path)
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with gzip.open(path, "rb") as f_in:
            with open(tmp_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return ad.read_h5ad(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Adjacency construction
# ---------------------------------------------------------------------------


def _head_average_layer_maps(full_attn: dict) -> list[np.ndarray]:
    """Return per-layer (batch, S, S) head-averaged attention, sorted by layer idx."""
    layer_names = sorted(full_attn, key=lambda k: int(k.removeprefix("layer")))
    return [
        full_attn[name].detach().cpu().float().numpy().mean(axis=1)  # (b, S, S)
        for name in layer_names
    ]


def _rollout_from_layer_maps(layer_maps: list[np.ndarray]) -> np.ndarray:
    """Abnar & Zuidema (2020) rollout. layer_maps are head-averaged (b, S, S)."""
    batch, S, _ = layer_maps[0].shape
    rollout = np.broadcast_to(np.eye(S, dtype=np.float32), (batch, S, S)).copy()
    I = np.eye(S, dtype=np.float32)
    for A in layer_maps:
        A_hat = 0.5 * A + 0.5 * I
        A_hat = A_hat / A_hat.sum(-1, keepdims=True).clip(1e-8)
        rollout = np.einsum("bij,bjk->bik", A_hat, rollout)
    return rollout


def _raw_full_from_layer_maps(layer_maps: list[np.ndarray]) -> np.ndarray:
    """Raw aggregation: mean over layers of head-averaged attention. (b, S, S)."""
    return np.stack(layer_maps, axis=0).mean(axis=0)


def _grad_rollout_batch(
    encoder,
    classifier,
    x_batch: torch.Tensor,
    other_emb_batch: torch.Tensor,
    y_batch: torch.Tensor,
    encoder_first_in_concat: bool,
) -> np.ndarray:
    """
    Chefer gradient-weighted rollout for one batch. Returns (b, S, S).

    The encoder output is concatenated with ``other_emb_batch`` (the frozen
    counterpart-modality embedding) in the order used at training time, then
    passed to the classifier. We backprop the correct-class logit and weight each
    layer's per-head attention by ReLU(dL/dA) before rolling out.
    """
    encoder.eval()
    classifier.eval()
    encoder.set_retain_attn_grad(True)

    x = x_batch.clone().detach().requires_grad_(True)
    z = encoder(x)
    full_attn = encoder.get_full_attention_per_layer()
    if full_attn is None:
        encoder.set_retain_attn_grad(False)
        b, S = x.size(0), x.size(1) + 1
        return np.broadcast_to(np.eye(S, dtype=np.float32), (b, S, S)).copy()

    captured_grads: dict[str, torch.Tensor] = {}
    hooks = []
    for l_name, A in full_attn.items():
        if A.grad_fn is not None:
            def _hook(grad, name=l_name):
                captured_grads[name] = grad.detach()
            hooks.append(A.register_hook(_hook))

    if encoder_first_in_concat:
        cat = torch.cat([z, other_emb_batch], dim=1)
    else:
        cat = torch.cat([other_emb_batch, z], dim=1)
    logits = classifier(cat)
    target = logits[torch.arange(x.size(0), device=logits.device), y_batch]
    target.sum().backward()

    for h in hooks:
        h.remove()
    encoder.set_retain_attn_grad(False)

    layer_names = sorted(full_attn, key=lambda k: int(k.removeprefix("layer")))
    b = x.size(0)
    S = full_attn[layer_names[0]].shape[-1]
    rollout = np.broadcast_to(np.eye(S, dtype=np.float32), (b, S, S)).copy()
    I = np.eye(S, dtype=np.float32)
    for l in layer_names:
        A_np = full_attn[l].detach().cpu().float().numpy()  # (b, nhead, S, S)
        G = captured_grads.get(l)
        if G is not None:
            G_np = G.cpu().float().numpy()
            weighted = (np.maximum(G_np, 0) * A_np).mean(axis=1)  # (b, S, S)
        else:
            weighted = A_np.mean(axis=1)
        A_hat = weighted + I
        A_hat = A_hat / A_hat.sum(-1, keepdims=True).clip(1e-8)
        rollout = np.einsum("bij,bjk->bik", A_hat, rollout)
    return rollout  # (b, S, S) — still includes CLS


def _aggregate_streaming(
    per_batch_fn,
    data: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    n_tokens_no_cls: int,
) -> dict[str, np.ndarray]:
    """
    Iterate over ``data`` in batches, compute per-batch (b, S, S), strip CLS, and
    accumulate per-label sums + a global sum. Returns {"global": A, label: A, ...}
    where A has shape (n_tokens_no_cls, n_tokens_no_cls).

    per_batch_fn: callable(x_batch_tensor, y_batch_tensor, start, end) -> (b, S, S) ndarray
                  (already sliced to the batch; CLS still present).
    """
    S = n_tokens_no_cls
    global_sum = np.zeros((S, S), dtype=np.float64)
    global_count = 0
    per_label_sum: dict[int, np.ndarray] = {}
    per_label_count: dict[int, int] = {}

    N = data.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = torch.tensor(data[start:end], dtype=torch.float32, device=device)
        y = torch.tensor(labels[start:end], dtype=torch.long, device=device)
        A = per_batch_fn(x, y, start, end)  # (b, S+1, S+1), CLS included
        A = A[:, 1:, 1:]  # drop CLS row/col → (b, S, S)
        global_sum += A.sum(axis=0)
        global_count += A.shape[0]
        for i, label_int in enumerate(labels[start:end]):
            li = int(label_int)
            if li not in per_label_sum:
                per_label_sum[li] = np.zeros((S, S), dtype=np.float64)
                per_label_count[li] = 0
            per_label_sum[li] += A[i]
            per_label_count[li] += 1

    result: dict[str, np.ndarray] = {}
    if global_count > 0:
        result["global"] = (global_sum / global_count).astype(np.float32)
    for li, s in per_label_sum.items():
        c = per_label_count[li]
        if c > 0:
            result[f"cell_type_{li}"] = (s / c).astype(np.float32)
    return result


def compute_adjacency(
    encoder,
    data: np.ndarray,
    labels: np.ndarray,
    method: str,
    device: torch.device,
    batch_size: int = 256,
    classifier=None,
    other_embeddings: Optional[np.ndarray] = None,
    encoder_first_in_concat: bool = True,
) -> dict[str, np.ndarray]:
    """
    Compute token×token adjacency aggregated globally and per cell type.

    Args:
        encoder:          AttentionTransformerEncoder (rna or protein).
        data:             (N, n_tokens) float32 input features.
        labels:           (N,) integer labels; used to bucket per-cell-type scopes.
        method:           One of {"raw", "rollout", "grad_attn"}.
        classifier:       Required when method == "grad_attn".
        other_embeddings: Required when method == "grad_attn"; (N, d) frozen
                          counterpart-modality embedding.
        encoder_first_in_concat: Concat order when forming classifier input.
                                 Training does [z_rna, z_protein] so the RNA
                                 encoder uses True and protein uses False.

    Returns:
        dict mapping {"global": A, "cell_type_<int>": A, ...} with A shape
        (n_tokens, n_tokens), CLS stripped, **unsymmetrized**. Post-processing
        (symmetrize, zero diagonal, sparsify) is the caller's choice.
    """
    if method not in {"raw", "rollout", "grad_attn"}:
        raise ValueError(f"Unknown method: {method!r}")
    if method == "grad_attn" and (classifier is None or other_embeddings is None):
        raise ValueError("method='grad_attn' requires classifier and other_embeddings")

    n_tokens = data.shape[1]

    def _raw_batch(x, _y, _s, _e):
        encoder.eval()
        with torch.no_grad():
            _ = encoder(x)
            full_attn = encoder.get_full_attention_per_layer()
        if full_attn is None:
            b = x.size(0)
            return np.broadcast_to(
                np.eye(n_tokens + 1, dtype=np.float32), (b, n_tokens + 1, n_tokens + 1)
            ).copy()
        return _raw_full_from_layer_maps(_head_average_layer_maps(full_attn))

    def _rollout_batch(x, _y, _s, _e):
        encoder.eval()
        with torch.no_grad():
            _ = encoder(x)
            full_attn = encoder.get_full_attention_per_layer()
        if full_attn is None:
            b = x.size(0)
            return np.broadcast_to(
                np.eye(n_tokens + 1, dtype=np.float32), (b, n_tokens + 1, n_tokens + 1)
            ).copy()
        return _rollout_from_layer_maps(_head_average_layer_maps(full_attn))

    def _grad_batch(x, y, s, e):
        other_t = torch.tensor(
            other_embeddings[s:e], dtype=torch.float32, device=device
        )
        return _grad_rollout_batch(
            encoder, classifier, x, other_t, y, encoder_first_in_concat
        )

    per_batch_fn = {"raw": _raw_batch, "rollout": _rollout_batch, "grad_attn": _grad_batch}[method]
    # grad_attn needs smaller batches to stay in GPU memory with the full graph.
    effective_bs = batch_size if method != "grad_attn" else max(8, batch_size // 4)
    return _aggregate_streaming(
        per_batch_fn, data, labels, device, effective_bs, n_tokens_no_cls=n_tokens
    )


# ---------------------------------------------------------------------------
# Post-processing + clustering
# ---------------------------------------------------------------------------


def postprocess_adjacency(
    A: np.ndarray,
    symmetrize: bool = True,
    zero_diag: bool = True,
    keep_top_pct: Optional[float] = 5.0,
) -> np.ndarray:
    """
    Symmetrize, zero the diagonal, and optionally sparsify by keeping only the
    top ``keep_top_pct`` percent of off-diagonal edges (by weight).
    """
    A = A.astype(np.float32, copy=True)
    if symmetrize:
        A = 0.5 * (A + A.T)
    if zero_diag:
        np.fill_diagonal(A, 0.0)
    if keep_top_pct is not None and keep_top_pct > 0:
        n = A.shape[0]
        upper = A[np.triu_indices(n, k=1)]
        if upper.size > 0:
            thresh = np.percentile(upper, 100 - keep_top_pct)
            mask = A >= thresh
            A = np.where(mask, A, 0.0)
            A = 0.5 * (A + A.T)  # re-symmetrize after masking
    return A


def cluster_adjacency(
    A: np.ndarray,
    method: str = "leiden",
    resolution: float = 1.0,
    n_clusters: int = 8,
    random_state: int = 0,
) -> np.ndarray:
    """
    Cluster a weighted symmetric adjacency. Returns cluster label per node.

    method='leiden' requires leidenalg + python-igraph. Falls back to spectral
    clustering (sklearn, ``n_clusters`` set explicitly) when unavailable.
    """
    n = A.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    if method == "leiden":
        try:
            import igraph as ig
            import leidenalg as la
        except ImportError:
            warnings.warn(
                "leidenalg/python-igraph not available — falling back to spectral clustering.",
                UserWarning,
            )
            method = "spectral"
        else:
            sources, targets = np.triu_indices(n, k=1)
            weights = A[sources, targets]
            keep = weights > 0
            g = ig.Graph(n=n, edges=list(zip(sources[keep], targets[keep])))
            g.es["weight"] = weights[keep].tolist()
            if len(g.es) == 0:
                return np.zeros(n, dtype=int)
            partition = la.find_partition(
                g,
                la.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=resolution,
                seed=random_state,
            )
            labels = np.zeros(n, dtype=int)
            for cid, members in enumerate(partition):
                for m in members:
                    labels[m] = cid
            return labels

    if method == "spectral":
        from sklearn.cluster import SpectralClustering

        k = min(n_clusters, max(2, n - 1))
        affinity = A.copy()
        affinity[affinity < 0] = 0
        if affinity.sum() == 0:
            return np.zeros(n, dtype=int)
        sc = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            random_state=random_state,
            assign_labels="kmeans",
        )
        return sc.fit_predict(affinity)

    raise ValueError(f"Unknown clustering method: {method!r}")


def compute_cluster_coherence(A: np.ndarray, labels: np.ndarray) -> dict:
    """
    Ratio of mean within-cluster to mean between-cluster edge weight. Higher
    indicates tighter modular structure. Diagonal is excluded.
    """
    n = A.shape[0]
    within_sum, within_count = 0.0, 0
    between_sum, between_count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                within_sum += A[i, j]
                within_count += 1
            else:
                between_sum += A[i, j]
                between_count += 1
    within_mean = within_sum / within_count if within_count else 0.0
    between_mean = between_sum / between_count if between_count else 0.0
    ratio = (within_mean / between_mean) if between_mean > 0 else float("inf")
    return {
        "within_mean": float(within_mean),
        "between_mean": float(between_mean),
        "ratio": float(ratio),
        "n_clusters": int(len(np.unique(labels))),
    }


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def _expand_pathway_cluster_to_genes(
    pathway_names_in_cluster: list[str],
    gene_sets: dict,
) -> list[str]:
    """Union of KEGG genes backing a cluster of pathway tokens."""
    genes: set = set()
    for pw in pathway_names_in_cluster:
        for g in gene_sets.get(pw, []):
            genes.add(g)
    return sorted(genes)


def enrichment_per_cluster(
    cluster_labels: np.ndarray,
    token_names: list[str],
    token_kind: str = "gene",
    gene_sets: Optional[dict] = None,
    libraries: tuple = ("GO_Biological_Process_2023", "Reactome_2022"),
    min_cluster_size: int = 3,
) -> dict:
    """
    Run gseapy.enrichr on each cluster's gene list.

    Args:
        token_kind: "gene" (tokens ARE gene symbols) or "pathway" (tokens are
                    KEGG pathway names; expand to their member genes first).
        gene_sets:  Required when token_kind == "pathway". {pathway_name: [genes]}.

    Returns:
        {cluster_id: {"genes": [...], "library": {term, adj_p, overlap, ...}}}
    """
    if token_kind not in {"gene", "pathway"}:
        raise ValueError(f"Unknown token_kind: {token_kind!r}")
    if token_kind == "pathway" and gene_sets is None:
        raise ValueError("token_kind='pathway' requires gene_sets for expansion")

    import gseapy

    result: dict = {}
    for cid in np.unique(cluster_labels):
        members = [token_names[i] for i in np.where(cluster_labels == cid)[0]]
        if len(members) < min_cluster_size:
            result[int(cid)] = {"genes": members, "skipped": "cluster too small"}
            continue
        if token_kind == "pathway":
            genes = _expand_pathway_cluster_to_genes(members, gene_sets)
        else:
            genes = members
        if len(genes) < 3:
            result[int(cid)] = {"genes": genes, "skipped": "fewer than 3 genes"}
            continue

        per_lib: dict = {}
        for lib in libraries:
            try:
                enr = gseapy.enrichr(
                    gene_list=genes,
                    gene_sets=lib,
                    outdir=None,
                    no_plot=True,
                    verbose=False,
                )
                df = enr.results if hasattr(enr, "results") else None
                if df is not None and len(df) > 0:
                    top = df.sort_values("Adjusted P-value").head(10)
                    per_lib[lib] = top[
                        ["Term", "Adjusted P-value", "Overlap", "Odds Ratio", "Combined Score"]
                    ].to_dict(orient="records")
                else:
                    per_lib[lib] = []
            except Exception as e:
                per_lib[lib] = {"error": str(e)}

        result[int(cid)] = {
            "genes": genes,
            "members": members,
            "n_members": len(members),
            "n_genes": len(genes),
            "enrichment": per_lib,
        }
    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_clustered_heatmap(
    A: np.ndarray,
    cluster_labels: np.ndarray,
    token_names: list[str],
    out_path: Path,
    title: str = "",
    max_ticks: int = 80,
) -> None:
    """Reorder tokens by cluster and draw a heatmap with cluster boundaries."""
    order = np.argsort(cluster_labels, kind="stable")
    A_ord = A[np.ix_(order, order)]
    names_ord = [token_names[i] for i in order]
    clusters_ord = cluster_labels[order]

    fig, ax = plt.subplots(figsize=(10, 9))
    vmax = np.percentile(A_ord[A_ord > 0], 99) if (A_ord > 0).any() else 1.0
    sns.heatmap(A_ord, ax=ax, cmap="viridis", cbar=True, vmin=0, vmax=vmax, square=True)

    boundaries = np.where(np.diff(clusters_ord) != 0)[0] + 1
    for b in boundaries:
        ax.axhline(b, color="white", lw=0.8)
        ax.axvline(b, color="white", lw=0.8)

    n = len(names_ord)
    if n <= max_ticks:
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_xticklabels(names_ord, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_yticklabels(names_ord, rotation=0, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_enrichment(
    enrichment: dict,
    out_path: Path,
    library: str,
    top_k: int = 5,
) -> None:
    """Horizontal bar chart of top terms per cluster for a single library."""
    clusters = [c for c, v in enrichment.items() if isinstance(v, dict) and "enrichment" in v]
    if not clusters:
        return
    n = len(clusters)
    fig, axes = plt.subplots(n, 1, figsize=(9, max(2, 1.2 * n)), squeeze=False)
    for ax, cid in zip(axes[:, 0], clusters):
        rows = enrichment[cid]["enrichment"].get(library, [])
        if not isinstance(rows, list) or not rows:
            ax.text(0.5, 0.5, "no hits", ha="center", va="center")
            ax.set_axis_off()
            continue
        rows = rows[:top_k]
        terms = [r["Term"] for r in rows][::-1]
        pvals = [-np.log10(max(float(r["Adjusted P-value"]), 1e-30)) for r in rows][::-1]
        ax.barh(terms, pvals, color="steelblue")
        ax.set_xlabel(f"-log10(adj p)  [cluster {cid}]")
        ax.tick_params(axis="y", labelsize=7)
    axes[0, 0].set_title(f"Top enrichment per cluster ({library})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Checkpoint orchestration
# ---------------------------------------------------------------------------


def _reconstruct_test_inputs(checkpoint_dir: Path, data_path: str) -> dict:
    """
    Rebuild the test-set inputs used during training so we can run the encoders
    on them. Returns a dict with ``rna_input``, ``protein_input``, ``labels``,
    ``rna_token_names``, ``protein_token_names``, plus loaded state dicts and
    the training args.

    Supports two tokenization variants written by the training scripts:
      - pathway_names.json + token_kind="pathway"  (train_contrastive_tf.py)
      - gene_names.json    + token_kind="gene"     (train_contrastive_tf_gene.py)
    """
    import scanpy as sc
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    from src.preprocessing import (
        build_gene_tokens,
        build_pathway_tokens,
        preprocess_protein,
    )

    stage_b = torch.load(checkpoint_dir / "stage_b_best.pt", map_location="cpu")
    train_args = stage_b["args"]

    adata = _load_anndata(data_path)
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    if train_args.get("max_cells") is not None and train_args["max_cells"] < adata.shape[0]:
        rng = np.random.default_rng(train_args["seed"])
        idx = np.sort(rng.choice(adata.shape[0], size=train_args["max_cells"], replace=False))
        adata = adata[idx].copy()

    label_encoder = LabelEncoder()
    labels_all = label_encoder.fit_transform(adata.obs[train_args["label_col"]].values)

    if train_args.get("test_donors"):
        donors = adata.obs[train_args["donor_col"]].values
        test_idx = np.flatnonzero(np.isin(donors, train_args["test_donors"]))
    else:
        _, test_idx = train_test_split(
            np.arange(adata.shape[0]),
            test_size=train_args["test_size"],
            random_state=train_args["seed"],
            stratify=None if train_args.get("max_cells") is not None else labels_all,
        )
    test_idx = np.asarray(test_idx)
    if test_idx.size == 0:
        raise ValueError("Test split resolved to zero cells.")

    rna_adata = adata[:, adata.var["feature_types"] == "GEX"].copy()
    protein_adata = adata[:, adata.var["feature_types"] == "ADT"].copy()

    sc.pp.normalize_total(rna_adata, target_sum=1e4)
    sc.pp.log1p(rna_adata)

    # Detect tokenization mode from artifact files
    gene_names_path = checkpoint_dir / "gene_names.json"
    pathway_names_path = checkpoint_dir / "pathway_names.json"
    if gene_names_path.exists():
        with open(gene_names_path, "r", encoding="utf-8") as f:
            rna_token_names = json.load(f)
        rna_matrix, _ = build_gene_tokens(rna_adata, hvg_genes=rna_token_names)
        rna_kind = "gene"
        gene_sets_used: Optional[dict] = None
    elif pathway_names_path.exists():
        with open(pathway_names_path, "r", encoding="utf-8") as f:
            rna_token_names = json.load(f)
        gene_sets_used = None
        if train_args.get("gene_sets_path"):
            with open(train_args["gene_sets_path"], "r", encoding="utf-8") as f:
                gene_sets_used = json.load(f)
        rna_matrix, built_names = build_pathway_tokens(rna_adata, gene_sets=gene_sets_used)
        # Verify we rebuilt the same pathway set; if not, bail rather than silently misalign.
        if built_names != rna_token_names:
            raise RuntimeError(
                "Reconstructed pathway set does not match pathway_names.json. "
                "Pass --gene_sets_path pointing to the same gene-set JSON used at train time."
            )
        rna_kind = "pathway"
    else:
        raise FileNotFoundError(
            f"Neither gene_names.json nor pathway_names.json in {checkpoint_dir}"
        )

    test_rna = rna_matrix[test_idx]
    test_protein = preprocess_protein(protein_adata[test_idx].copy())
    test_labels = labels_all[test_idx]

    with open(checkpoint_dir / "protein_names.json", "r", encoding="utf-8") as f:
        protein_token_names = json.load(f)

    return {
        "train_args": train_args,
        "stage_b": stage_b,
        "rna_input": test_rna,
        "protein_input": test_protein,
        "labels": test_labels,
        "rna_token_names": rna_token_names,
        "protein_token_names": protein_token_names,
        "rna_kind": rna_kind,
        "rna_gene_sets": gene_sets_used,
        "label_encoder": label_encoder,
    }


def _load_encoders(checkpoint_dir: Path, ctx: dict, device: torch.device):
    from src.models.classifier import ClassificationHead
    from src.models.transformer_encoder import TransformerEncoder

    args = ctx["train_args"]
    stage_b = ctx["stage_b"]

    rna_encoder = TransformerEncoder(
        n_tokens=stage_b["n_pathways"],  # used as generic "n_rna_tokens" — name preserved for ckpt compat
        d_model=args["d_model"],
        nhead=args["nhead"],
        num_layers=args["num_layers"],
        dim_feedforward=args["dim_feedforward"],
        dropout=args["dropout"],
        output_dim=args["embedding_dim"],
    ).to(device)
    protein_encoder = TransformerEncoder(
        n_tokens=stage_b["n_proteins"],
        d_model=args["d_model"],
        nhead=args["nhead"],
        num_layers=args["num_layers"],
        dim_feedforward=args["dim_feedforward"],
        dropout=args["dropout"],
        output_dim=args["embedding_dim"],
    ).to(device)
    classifier = ClassificationHead(
        input_dim=args["embedding_dim"] * 2,
        n_classes=len(set(ctx["labels"].tolist())),
        hidden_dim=args["classifier_hidden_dim"],
        dropout=args["classifier_dropout"],
    ).to(device)

    rna_encoder.load_state_dict(stage_b["rna_encoder_state_dict"])
    protein_encoder.load_state_dict(stage_b["protein_encoder_state_dict"])
    classifier.load_state_dict(stage_b["classifier_state_dict"])
    return rna_encoder, protein_encoder, classifier


def run_pipeline(
    checkpoint_dir: Path,
    data_path: str,
    methods: list[str],
    scopes: list[str],
    modalities: list[str],
    resolution: float,
    keep_top_pct: Optional[float],
    min_cluster_size: int,
    cluster_method: str,
    n_spectral_clusters: int,
    batch_size: int,
    enrichment_libraries: tuple,
    cell_types_filter: Optional[list[str]],
    out_root: Path,
    device: torch.device,
) -> None:
    """Glue: reconstruct test set, load encoders, run every (method, scope, modality)."""
    ctx = _reconstruct_test_inputs(checkpoint_dir, data_path)
    rna_encoder, protein_encoder, classifier = _load_encoders(checkpoint_dir, ctx, device)
    labels = ctx["labels"]

    # Precompute counterpart embeddings once (needed for grad_attn)
    rna_encoder.eval()
    protein_encoder.eval()
    with torch.no_grad():
        z_rna_all = []
        z_prot_all = []
        for start in range(0, ctx["rna_input"].shape[0], batch_size):
            end = start + batch_size
            rb = torch.tensor(ctx["rna_input"][start:end], dtype=torch.float32, device=device)
            pb = torch.tensor(ctx["protein_input"][start:end], dtype=torch.float32, device=device)
            z_rna_all.append(rna_encoder(rb).cpu().numpy())
            z_prot_all.append(protein_encoder(pb).cpu().numpy())
    z_rna = np.concatenate(z_rna_all, axis=0)
    z_prot = np.concatenate(z_prot_all, axis=0)

    # Load label names for per-cell-type naming
    with open(checkpoint_dir / "label_mapping.json", "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    int_to_name = {int(k): str(v) for k, v in label_mapping.items()}

    for modality in modalities:
        if modality == "rna":
            enc = rna_encoder
            data = ctx["rna_input"]
            token_names = ctx["rna_token_names"]
            token_kind = ctx["rna_kind"]
            gene_sets = ctx["rna_gene_sets"]
            other_emb = z_prot
            encoder_first = True
        elif modality == "protein":
            enc = protein_encoder
            data = ctx["protein_input"]
            token_names = ctx["protein_token_names"]
            token_kind = "gene"  # protein names are gene-like symbols; enrichr treats them as gene symbols
            gene_sets = None
            other_emb = z_rna
            encoder_first = False
        else:
            raise ValueError(f"Unknown modality: {modality}")

        for method in methods:
            print(f"\n=== {modality} / {method} ===")
            adj = compute_adjacency(
                encoder=enc,
                data=data,
                labels=labels,
                method=method,
                device=device,
                batch_size=batch_size,
                classifier=classifier if method == "grad_attn" else None,
                other_embeddings=other_emb if method == "grad_attn" else None,
                encoder_first_in_concat=encoder_first,
            )

            for scope in scopes:
                if scope == "global":
                    scope_keys = ["global"]
                elif scope == "per_cell_type":
                    scope_keys = [k for k in adj if k.startswith("cell_type_")]
                    if cell_types_filter:
                        scope_keys = [
                            k for k in scope_keys
                            if int_to_name.get(int(k.removeprefix("cell_type_")), "") in cell_types_filter
                        ]
                else:
                    raise ValueError(f"Unknown scope: {scope}")

                for key in scope_keys:
                    if key not in adj:
                        continue
                    raw_A = adj[key]
                    A = postprocess_adjacency(raw_A, keep_top_pct=keep_top_pct)
                    if not np.isfinite(A).all() or A.sum() == 0:
                        print(f"  [skip] {key}: empty/degenerate adjacency")
                        continue

                    labels_c = cluster_adjacency(
                        A,
                        method=cluster_method,
                        resolution=resolution,
                        n_clusters=n_spectral_clusters,
                    )
                    coherence = compute_cluster_coherence(A, labels_c)
                    enr = enrichment_per_cluster(
                        cluster_labels=labels_c,
                        token_names=token_names,
                        token_kind=token_kind,
                        gene_sets=gene_sets,
                        libraries=enrichment_libraries,
                        min_cluster_size=min_cluster_size,
                    )

                    if key == "global":
                        sub = out_root / f"{modality}__{method}__global"
                    else:
                        ct_name = int_to_name.get(int(key.removeprefix("cell_type_")), key)
                        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in ct_name)
                        sub = out_root / f"{modality}__{method}__per_cell_type" / safe
                    sub.mkdir(parents=True, exist_ok=True)

                    np.save(sub / "adjacency.npy", A.astype(np.float32))
                    with open(sub / "clusters.json", "w", encoding="utf-8") as f:
                        json.dump(
                            {token_names[i]: int(labels_c[i]) for i in range(len(token_names))},
                            f, indent=2,
                        )
                    with open(sub / "coherence.json", "w", encoding="utf-8") as f:
                        json.dump(coherence, f, indent=2)
                    with open(sub / "enrichment.json", "w", encoding="utf-8") as f:
                        json.dump(enr, f, indent=2, default=str)
                    plot_clustered_heatmap(
                        A, labels_c, token_names, sub / "heatmap.png",
                        title=f"{modality} {method} {key}  (ratio={coherence['ratio']:.2f})",
                    )
                    for lib in enrichment_libraries:
                        safe_lib = lib.replace("/", "_")
                        plot_top_enrichment(enr, sub / f"top_enrichment__{safe_lib}.png", library=lib)

                    print(
                        f"  {key}: {coherence['n_clusters']} clusters, "
                        f"within/between={coherence['ratio']:.2f} → {sub}"
                    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Attention adjacency graph clustering + enrichment for contrastive-tf checkpoints"
    )
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--methods", type=str, default="raw,rollout,grad_attn")
    p.add_argument("--scopes", type=str, default="global,per_cell_type")
    p.add_argument("--modalities", type=str, default="rna,protein")
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--keep_top_pct", type=float, default=5.0,
                   help="Keep only the top-N pct of off-diagonal edges. 0 or negative disables.")
    p.add_argument("--min_cluster_size", type=int, default=3)
    p.add_argument("--cluster_method", type=str, default="leiden", choices=["leiden", "spectral"])
    p.add_argument("--n_spectral_clusters", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument(
        "--enrichment_libraries",
        type=str,
        default="GO_Biological_Process_2023,Reactome_2022",
    )
    p.add_argument("--cell_types", type=str, nargs="*", default=None,
                   help="Optional filter for per-cell-type scope (cell-type string names).")
    p.add_argument("--out_subdir", type=str, default="attention_graph")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    ckpt = Path(args.checkpoint_dir)
    out_root = ckpt / args.out_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    keep_top_pct = args.keep_top_pct if args.keep_top_pct > 0 else None

    run_pipeline(
        checkpoint_dir=ckpt,
        data_path=args.data_path,
        methods=[m.strip() for m in args.methods.split(",") if m.strip()],
        scopes=[s.strip() for s in args.scopes.split(",") if s.strip()],
        modalities=[m.strip() for m in args.modalities.split(",") if m.strip()],
        resolution=args.resolution,
        keep_top_pct=keep_top_pct,
        min_cluster_size=args.min_cluster_size,
        cluster_method=args.cluster_method,
        n_spectral_clusters=args.n_spectral_clusters,
        batch_size=args.batch_size,
        enrichment_libraries=tuple(
            lib.strip() for lib in args.enrichment_libraries.split(",") if lib.strip()
        ),
        cell_types_filter=args.cell_types,
        out_root=out_root,
        device=device,
    )


if __name__ == "__main__":
    main()
