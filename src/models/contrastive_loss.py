"""
contrastive_loss.py — CLIP-style symmetric cross-entropy contrastive loss

Implement the following:

class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07):
        - self.temperature = temperature

    def forward(self, z_rna, z_protein) -> torch.Tensor:
        - Both inputs are L2-normalized embeddings: shape (batch, 128)
        - Compute cosine similarity matrix: logits = (z_rna @ z_protein.T) / self.temperature
            - Shape: (batch, batch)
        - Labels = torch.arange(batch_size) — diagonal entries are positive pairs
          (cell i's RNA should match cell i's protein)
        - loss_rna = F.cross_entropy(logits, labels)  # RNA->Protein direction
        - loss_protein = F.cross_entropy(logits.T, labels)  # Protein->RNA direction
        - Return (loss_rna + loss_protein) / 2  # symmetric loss

Notes:
    - Inputs MUST be L2-normalized (handled by encoders)
    - Batch size matters: larger batches = more negatives = better contrastive signal
    - Temperature 0.07 sharpens the distribution — lower = harder negatives
    - Labels should be on same device as logits
"""
