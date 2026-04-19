"""MLP encoder for RNA/protein branches in the dual-encoder setup."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """
    Two-layer MLP encoder with optional L2-normalized output.

    Architecture:
      Linear(input_dim, hidden_dim) -> BatchNorm1d -> ReLU -> Dropout(p)
      Linear(hidden_dim, output_dim) -> BatchNorm1d -> ReLU -> Dropout(p)

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer width.
        output_dim: Embedding dimension.
        normalize_output: If True, apply L2 normalization across feature dim.
            Set False for baseline training where raw encoder activations are desired.
        dropout: Dropout probability applied after each ReLU. Default 0.0 (disabled)
            preserves the original behavior used by the contrastive models.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        normalize_output: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.normalize_output = normalize_output
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        ])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features into output_dim embeddings."""
        z = self.net(x)
        if self.normalize_output:
            z = F.normalize(z, p=2, dim=1)
        return z
