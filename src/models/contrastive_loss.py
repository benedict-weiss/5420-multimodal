"""CLIP-style symmetric contrastive loss for paired RNA/protein embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    """Symmetric cross-entropy over pairwise similarity logits."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature

    def forward(self, z_rna: torch.Tensor, z_protein: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_rna: RNA embeddings of shape (batch_size, dim), expected L2-normalized.
            z_protein: Protein embeddings of shape (batch_size, dim), expected L2-normalized.

        Returns:
            Scalar symmetric contrastive loss.
        """
        if z_rna.ndim != 2 or z_protein.ndim != 2:
            raise ValueError("z_rna and z_protein must be 2D tensors: (batch_size, dim)")
        if z_rna.shape != z_protein.shape:
            raise ValueError(
                f"z_rna and z_protein must have the same shape, got {z_rna.shape} and {z_protein.shape}"
            )

        batch_size = z_rna.size(0)
        logits = torch.matmul(z_rna, z_protein.T) / self.temperature
        labels = torch.arange(batch_size, device=logits.device)

        loss_rna = F.cross_entropy(logits, labels)
        loss_protein = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_rna + loss_protein)
