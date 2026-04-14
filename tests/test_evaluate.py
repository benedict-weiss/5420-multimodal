"""Unit tests for src/evaluate.py metric functions."""
import numpy as np
import pytest
from src.evaluate import (
    compute_accuracy,
    compute_asw,
    compute_auroc,
    compute_batch_entropy,
    compute_recall_at_k,
    run_significance_test,
)


def test_compute_auroc_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])
    assert compute_auroc(y_true, y_proba, n_classes=2) == pytest.approx(1.0)


def test_compute_auroc_multiclass():
    y_true = np.repeat([0, 1, 2], 10)
    y_proba = np.zeros((30, 3))
    y_proba[np.arange(30), y_true] = 1.0
    assert compute_auroc(y_true, y_proba, n_classes=3) == pytest.approx(1.0)


def test_compute_accuracy_overall():
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 1])
    overall, per_class = compute_accuracy(y_true, y_pred)
    assert overall == pytest.approx(0.75)
    assert isinstance(per_class, dict)
    assert "macro avg" in per_class


def test_compute_accuracy_returns_tuple():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    result = compute_accuracy(y_true, y_pred)
    assert isinstance(result, tuple) and len(result) == 2


def test_compute_asw_normalized_range():
    rng = np.random.default_rng(42)
    emb = np.vstack([rng.normal(0, 0.1, (50, 8)), rng.normal(10, 0.1, (50, 8))])
    labels = np.array([0] * 50 + [1] * 50)
    asw = compute_asw(emb, labels)
    assert 0.0 <= asw <= 1.0
    assert asw > 0.9  # well-separated clusters → close to 1


def test_compute_asw_subsamples_large():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(15_000, 4))
    labels = (rng.random(15_000) > 0.5).astype(int)
    asw = compute_asw(emb, labels)  # must not OOM or error
    assert 0.0 <= asw <= 1.0


def test_compute_recall_at_k_perfect():
    n = 50
    z = np.eye(n)
    result = compute_recall_at_k(z, z, k_values=[1, 5, 10])
    assert result[1] == pytest.approx(1.0)
    assert result[5] == pytest.approx(1.0)


def test_compute_recall_at_k_random_baseline():
    rng = np.random.default_rng(0)
    n = 200
    z_rna = rng.normal(size=(n, 32))
    z_rna /= np.linalg.norm(z_rna, axis=1, keepdims=True)
    z_protein = rng.normal(size=(n, 32))
    z_protein /= np.linalg.norm(z_protein, axis=1, keepdims=True)
    result = compute_recall_at_k(z_rna, z_protein, k_values=[10, 50])
    # Random pairs: recall@k ≈ k/n, allow ±10pp
    assert abs(result[10] - 10 / 200) < 0.10
    assert abs(result[50] - 50 / 200) < 0.10


def test_compute_batch_entropy_mixed():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(100, 8))
    batch_labels = np.array([0, 1] * 50)
    ent = compute_batch_entropy(emb, batch_labels, n_neighbors=10)
    assert ent > 0.0


def test_compute_batch_entropy_homogeneous():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(100, 8))
    batch_labels = np.zeros(100, dtype=int)  # all same batch
    ent = compute_batch_entropy(emb, batch_labels, n_neighbors=10)
    assert ent == pytest.approx(0.0, abs=1e-6)


def test_run_significance_test_significant():
    p = run_significance_test([0.9, 0.91, 0.92, 0.89, 0.93], [0.5, 0.51, 0.52, 0.49, 0.53])
    assert p < 0.05


def test_run_significance_test_identical():
    scores = [0.9, 0.91, 0.92, 0.89, 0.93]
    p = run_significance_test(scores, scores)
    assert p == pytest.approx(1.0)
