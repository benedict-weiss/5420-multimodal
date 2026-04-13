"""Tests for TransformerEncoder — output shape, L2 norm, attention weights."""

import pytest
import torch
from src.models.transformer_encoder import TransformerEncoder


@pytest.fixture
def small_rna_enc():
    return TransformerEncoder(
        n_tokens=20, d_model=16, nhead=2, num_layers=2,
        dim_feedforward=32, dropout=0.0, output_dim=8,
    )


@pytest.fixture
def small_protein_enc():
    return TransformerEncoder(
        n_tokens=10, d_model=16, nhead=2, num_layers=2,
        dim_feedforward=32, dropout=0.0, output_dim=8,
    )


def test_output_shape_rna(small_rna_enc):
    x = torch.randn(4, 20)
    out = small_rna_enc(x)
    assert out.shape == (4, 8), f"Expected (4, 8), got {out.shape}"


def test_output_shape_protein(small_protein_enc):
    x = torch.randn(4, 10)
    out = small_protein_enc(x)
    assert out.shape == (4, 8), f"Expected (4, 8), got {out.shape}"


def test_l2_normalization(small_rna_enc):
    x = torch.randn(4, 20)
    out = small_rna_enc(x)
    norms = out.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), \
        f"Output not unit-normalized. Norms: {norms}"


def test_attention_none_before_forward():
    enc = TransformerEncoder(
        n_tokens=20, d_model=16, nhead=2, num_layers=1,
        dim_feedforward=32, dropout=0.0, output_dim=8,
    )
    assert enc.get_attention_weights() is None


def test_attention_weights_shape(small_rna_enc):
    x = torch.randn(4, 20)
    _ = small_rna_enc(x)
    attn = small_rna_enc.get_attention_weights()
    assert attn is not None
    # Shape is (batch, n_tokens): CLS row sliced to exclude CLS-to-CLS entry
    n_tokens = 20
    assert attn.shape == (4, n_tokens), f"Expected (4, {n_tokens}), got {attn.shape}"


def test_attention_weights_nonnegative(small_rna_enc):
    x = torch.randn(4, 20)
    _ = small_rna_enc(x)
    attn = small_rna_enc.get_attention_weights()
    assert (attn >= 0).all(), "Attention weights contain negative values"


def test_attention_weights_sum_leq_one(small_rna_enc):
    """CLS-to-token slice sums to (0, 1.0] — CLS-to-CLS weight is the missing portion."""
    x = torch.randn(4, 20)
    _ = small_rna_enc(x)
    attn = small_rna_enc.get_attention_weights()
    row_sums = attn.sum(dim=1)
    assert (row_sums > 0).all(), f"Row sums are zero or negative: {row_sums}"
    assert (row_sums <= 1.0 + 1e-5).all(), f"Row sums exceed 1.0: {row_sums}"


def test_attention_weights_under_no_grad_eval(small_rna_enc):
    """Attention extraction must work in inference mode (model.eval + no_grad)."""
    small_rna_enc.eval()
    x = torch.randn(4, 20)
    with torch.no_grad():
        _ = small_rna_enc(x)
    attn = small_rna_enc.get_attention_weights()
    assert attn is not None, "get_attention_weights() returned None under eval+no_grad"
    assert attn.shape == (4, 20)


def test_clip_loss_smoke(small_rna_enc, small_protein_enc):
    """Two TransformerEncoders + CLIPLoss should produce a finite scalar loss."""
    from src.models.contrastive_loss import CLIPLoss
    loss_fn = CLIPLoss(temperature=0.07)
    z_rna = small_rna_enc(torch.randn(4, 20))
    z_protein = small_protein_enc(torch.randn(4, 10))
    loss = loss_fn(z_rna, z_protein)
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
