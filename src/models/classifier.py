"""Classification head for cell type prediction from learned embeddings."""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Two-layer MLP head that outputs raw logits.

    Intended use in the dual-encoder pipeline:
      input = concat([z_rna, z_protein], dim=1)
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits of shape (batch_size, n_classes)."""
        return self.net(x)
