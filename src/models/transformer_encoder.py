"""Transformer encoder with CLS token for pathway/protein token inputs."""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    TransformerEncoderLayer that captures per-head attention weights.

    Overrides _sa_block (PyTorch 2.x) to call self_attn with need_weights=True
    and average_attn_weights=False. Everything else is inherited unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_attn_weights: Optional[torch.Tensor] = None
        self._retain_attn_grad: bool = False

    def set_retain_attn_grad(self, value: bool) -> None:
        self._retain_attn_grad = value

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Temporarily set training=True to prevent PyTorch's per-layer C++ fast path
        # (torch._transformer_encoder_layer_fwd), which bypasses _sa_block and silently
        # drops attention weight capture when the model is in inference mode under no_grad.
        was_training = self.training
        self.training = True
        try:
            out = super().forward(src, src_mask, src_key_padding_mask, is_causal)
        finally:
            self.training = was_training
        return out

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        x, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        if self._retain_attn_grad:
            self._last_attn_weights = attn_weights  # keep in graph (no detach, hooks fire on backward)
        else:
            self._last_attn_weights = attn_weights.detach()  # (batch, nhead, seq_len, seq_len)
        return self.dropout1(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with learnable CLS token for token-level inputs.

    Each input is a 1-D vector of scalar token values (e.g. mean pathway expression
    or CLR protein levels). Tokens are projected to d_model, a CLS token is
    prepended, and the CLS output produces a fixed-size cell embedding.

    Args:
        n_tokens:        Number of input tokens (pathways or proteins).
        d_model:         Internal transformer dimension.
        nhead:           Number of attention heads. Must divide d_model evenly.
        num_layers:      Number of encoder layers stacked.
        dim_feedforward: FFN hidden dimension within each layer.
        dropout:         Dropout rate.
        output_dim:      Final L2-normalized embedding dimension.
    """

    def __init__(
        self,
        n_tokens: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_dim: int = 128,
    ):
        super().__init__()
        self.n_tokens = n_tokens

        self.input_proj = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoding = nn.Parameter(torch.randn(1, n_tokens + 1, d_model))

        layer = AttentionTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # ensure _sa_block is always called
        )
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token values of shape (batch, n_tokens).

        Returns:
            L2-normalized embeddings of shape (batch, output_dim).
        """
        batch = x.size(0)
        # (batch, n_tokens) -> (batch, n_tokens, 1) -> (batch, n_tokens, d_model)
        x = self.input_proj(x.unsqueeze(-1))
        # Prepend CLS: (batch, n_tokens+1, d_model)
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_encoding
        x = self.encoder(x)
        cls_out = x[:, 0, :]
        return F.normalize(self.output_proj(cls_out), p=2, dim=1)

    def set_retain_attn_grad(self, value: bool) -> None:
        for layer in self.encoder.layers:
            if isinstance(layer, AttentionTransformerEncoderLayer):
                layer.set_retain_attn_grad(value)

    def get_full_attention_per_layer(self) -> Optional[dict[str, torch.Tensor]]:
        """
        Full (batch, nhead, seq_len, seq_len) attention maps per layer.
        When set_retain_attn_grad(True) was called before the forward pass,
        tensors remain in the computation graph so .grad is populated after backward().
        """
        result: dict[str, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.encoder.layers):
            if layer._last_attn_weights is not None:
                result[f"layer{layer_idx}"] = layer._last_attn_weights
        return result or None

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        CLS-to-token attention from the last encoder layer, averaged over heads.

        Returns:
            Tensor (batch, n_tokens), or None if called before any forward pass.
        """
        last_layer: AttentionTransformerEncoderLayer = self.encoder.layers[-1]
        if last_layer._last_attn_weights is None:
            return None
        # (batch, nhead, seq_len, seq_len) -> mean over heads -> CLS row, skip CLS-to-CLS
        attn_avg = last_layer._last_attn_weights.mean(dim=1)  # (batch, seq_len, seq_len)
        return attn_avg[:, 0, 1:]  # (batch, n_tokens)

    def get_attention_weights_per_head(self) -> Optional[dict[str, torch.Tensor]]:
        """
        CLS-to-token attention from every encoder layer, preserving individual heads.

        Returns:
            Dict like {"layer0": (batch, nhead, n_tokens), ...}, or None if no
            layer has cached attention weights from a forward pass yet.
        """
        result: dict[str, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.encoder.layers):
            if layer._last_attn_weights is None:
                continue
            result[f"layer{layer_idx}"] = layer._last_attn_weights[:, :, 0, 1:].detach()
        return result or None
