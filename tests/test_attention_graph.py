"""
Unit tests for src/attention_graph.py.

These tests exercise the adjacency-construction, postprocessing, clustering, and
enrichment-dispatch code paths without requiring a trained checkpoint or network
access. Enrichment against live gseapy.enrichr is stubbed via monkeypatch.
"""

import numpy as np
import pytest
import torch

from src.attention_graph import (
    _expand_pathway_cluster_to_genes,
    _head_average_layer_maps,
    _raw_full_from_layer_maps,
    _rollout_from_layer_maps,
    cluster_adjacency,
    compute_adjacency,
    compute_cluster_coherence,
    enrichment_per_cluster,
    postprocess_adjacency,
)
from src.models.transformer_encoder import TransformerEncoder


# ---------------------------------------------------------------------------
# Postprocessing math
# ---------------------------------------------------------------------------


def test_postprocess_symmetrizes_and_zeros_diag():
    A = np.array([[5.0, 1.0, 2.0], [3.0, 5.0, 4.0], [2.0, 4.0, 5.0]], dtype=np.float32)
    out = postprocess_adjacency(A, keep_top_pct=None)
    assert np.allclose(out, out.T)
    assert np.all(np.diag(out) == 0)
    # Off-diagonal symmetrized: (1+3)/2 = 2 at (0,1)
    assert out[0, 1] == pytest.approx(2.0)


def test_postprocess_top_pct_zeros_low_weight_edges():
    rng = np.random.default_rng(0)
    A = rng.uniform(size=(10, 10)).astype(np.float32)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    out = postprocess_adjacency(A, keep_top_pct=20.0, symmetrize=False, zero_diag=False)
    # Exactly the top 20% of upper-triangle values should survive (counted in upper tri)
    n = 10
    upper = out[np.triu_indices(n, k=1)]
    nonzero_upper = (upper > 0).sum()
    n_upper = n * (n - 1) // 2
    # Top 20% of 45 upper-tri values = 9; allow some slack from ties
    assert nonzero_upper <= int(np.ceil(0.25 * n_upper))
    assert nonzero_upper >= int(np.floor(0.15 * n_upper))


# ---------------------------------------------------------------------------
# Cluster coherence
# ---------------------------------------------------------------------------


def test_coherence_block_diagonal_has_high_ratio():
    # Two tight blocks of 3 nodes each, near-zero between them
    A = np.zeros((6, 6), dtype=np.float32)
    A[:3, :3] = 1.0
    A[3:, 3:] = 1.0
    np.fill_diagonal(A, 0.0)
    labels = np.array([0, 0, 0, 1, 1, 1])
    c = compute_cluster_coherence(A, labels)
    assert c["within_mean"] == pytest.approx(1.0)
    assert c["between_mean"] == pytest.approx(0.0)
    assert c["ratio"] == float("inf")
    assert c["n_clusters"] == 2


def test_coherence_uniform_graph_has_ratio_one():
    A = np.ones((6, 6), dtype=np.float32)
    np.fill_diagonal(A, 0.0)
    labels = np.array([0, 0, 0, 1, 1, 1])
    c = compute_cluster_coherence(A, labels)
    assert c["ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def test_spectral_clustering_recovers_blocks():
    A = np.zeros((6, 6), dtype=np.float32)
    A[:3, :3] = 1.0
    A[3:, 3:] = 1.0
    np.fill_diagonal(A, 0.0)
    labels = cluster_adjacency(A, method="spectral", n_clusters=2, random_state=0)
    # Ensure nodes 0-2 share a label and 3-5 share a label (labels may be flipped)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_clustering_returns_single_label_on_disconnected_graph():
    A = np.zeros((5, 5), dtype=np.float32)
    labels = cluster_adjacency(A, method="spectral", n_clusters=3)
    assert labels.shape == (5,)
    assert (labels == labels[0]).all()


# ---------------------------------------------------------------------------
# Aggregation primitives
# ---------------------------------------------------------------------------


def test_raw_full_mean_over_layers():
    L0 = np.ones((2, 3, 3), dtype=np.float32)
    L1 = np.full((2, 3, 3), 3.0, dtype=np.float32)
    out = _raw_full_from_layer_maps([L0, L1])
    assert out.shape == (2, 3, 3)
    assert np.allclose(out, 2.0)


def test_rollout_is_row_stochastic_and_preserves_shape():
    rng = np.random.default_rng(0)
    L0 = rng.uniform(size=(2, 4, 4)).astype(np.float32)
    L0 /= L0.sum(-1, keepdims=True)
    L1 = rng.uniform(size=(2, 4, 4)).astype(np.float32)
    L1 /= L1.sum(-1, keepdims=True)
    out = _rollout_from_layer_maps([L0, L1])
    assert out.shape == (2, 4, 4)
    # Rows should sum to ~1 (rollout matrices are row-stochastic after normalization)
    row_sums = out.sum(-1)
    assert np.allclose(row_sums, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# End-to-end adjacency extraction on a tiny random encoder
# ---------------------------------------------------------------------------


def _tiny_encoder(n_tokens: int = 8):
    return TransformerEncoder(
        n_tokens=n_tokens,
        d_model=16,
        nhead=2,
        num_layers=2,
        dim_feedforward=32,
        dropout=0.0,
        output_dim=8,
    )


@pytest.mark.parametrize("method", ["raw", "rollout"])
def test_compute_adjacency_shape_and_keys(method):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    enc = _tiny_encoder(n_tokens=8)
    data = rng.standard_normal((12, 8)).astype(np.float32)
    labels = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)

    adj = compute_adjacency(
        encoder=enc,
        data=data,
        labels=labels,
        method=method,
        device=torch.device("cpu"),
        batch_size=4,
    )
    assert "global" in adj
    assert adj["global"].shape == (8, 8)  # CLS stripped from the 9 tokens of the seq
    for lab in np.unique(labels):
        key = f"cell_type_{int(lab)}"
        assert key in adj
        assert adj[key].shape == (8, 8)


def test_head_average_preserves_per_layer_batch_shape():
    # Exercise the attention capture round-trip: run the encoder, then verify the
    # helper returns (b, S, S) per layer with the correct seq length.
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    enc = _tiny_encoder(n_tokens=6)
    x = torch.tensor(rng.standard_normal((3, 6)).astype(np.float32))
    enc.eval()
    with torch.no_grad():
        _ = enc(x)
    full_attn = enc.get_full_attention_per_layer()
    assert full_attn is not None
    per_layer = _head_average_layer_maps(full_attn)
    assert len(per_layer) == 2  # num_layers
    for m in per_layer:
        assert m.shape == (3, 7, 7)  # batch=3, seq=n_tokens+1


# ---------------------------------------------------------------------------
# Enrichment dispatch (stub enrichr to avoid network)
# ---------------------------------------------------------------------------


def test_pathway_expansion_unions_genes():
    gene_sets = {
        "PathA": ["GENE1", "GENE2", "GENE3"],
        "PathB": ["GENE2", "GENE4"],
        "PathC": ["GENE5"],
    }
    out = _expand_pathway_cluster_to_genes(["PathA", "PathB"], gene_sets)
    assert out == sorted(["GENE1", "GENE2", "GENE3", "GENE4"])


class _StubEnrichResult:
    def __init__(self, df):
        self.results = df


def test_enrichment_skips_small_clusters_and_dispatches_per_library(monkeypatch):
    import pandas as pd

    calls = []

    def fake_enrichr(gene_list, gene_sets, **kwargs):
        calls.append({"library": gene_sets, "n_genes": len(gene_list)})
        df = pd.DataFrame({
            "Term": [f"{gene_sets}_hit1"],
            "Adjusted P-value": [0.001],
            "Overlap": ["2/10"],
            "Odds Ratio": [5.0],
            "Combined Score": [20.0],
        })
        return _StubEnrichResult(df)

    import gseapy
    monkeypatch.setattr(gseapy, "enrichr", fake_enrichr)

    cluster_labels = np.array([0, 0, 0, 1, 1])  # cluster 1 below min_cluster_size=3
    token_names = ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"]
    out = enrichment_per_cluster(
        cluster_labels=cluster_labels,
        token_names=token_names,
        token_kind="gene",
        libraries=("LibX", "LibY"),
        min_cluster_size=3,
    )
    # Cluster 0 should have enrichment for both libraries; cluster 1 skipped
    assert out[0]["enrichment"]["LibX"][0]["Term"] == "LibX_hit1"
    assert out[0]["enrichment"]["LibY"][0]["Term"] == "LibY_hit1"
    assert "skipped" in out[1]
    # enrichr dispatched once per (valid_cluster, library)
    assert len(calls) == 2


def test_enrichment_pathway_kind_requires_gene_sets():
    with pytest.raises(ValueError, match="gene_sets"):
        enrichment_per_cluster(
            cluster_labels=np.array([0, 0, 0]),
            token_names=["PathA", "PathB", "PathC"],
            token_kind="pathway",
            gene_sets=None,
        )


def test_enrichment_pathway_kind_expands_before_enrichr(monkeypatch):
    import pandas as pd

    seen_gene_lists = []

    def fake_enrichr(gene_list, gene_sets, **kwargs):
        seen_gene_lists.append(sorted(gene_list))
        df = pd.DataFrame({
            "Term": ["hit"], "Adjusted P-value": [0.01],
            "Overlap": ["3/10"], "Odds Ratio": [2.0], "Combined Score": [10.0],
        })
        return _StubEnrichResult(df)

    import gseapy
    monkeypatch.setattr(gseapy, "enrichr", fake_enrichr)

    gs = {"P1": ["G1", "G2"], "P2": ["G2", "G3"], "P3": ["G4", "G5", "G6"]}
    labels = np.array([0, 0, 0])
    token_names = ["P1", "P2", "P3"]
    out = enrichment_per_cluster(
        cluster_labels=labels,
        token_names=token_names,
        token_kind="pathway",
        gene_sets=gs,
        libraries=("LibX",),
        min_cluster_size=3,
    )
    assert seen_gene_lists == [["G1", "G2", "G3", "G4", "G5", "G6"]]
    assert out[0]["n_genes"] == 6
    assert out[0]["n_members"] == 3
