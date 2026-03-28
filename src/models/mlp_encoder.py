"""
mlp_encoder.py — MLP encoder for both modalities

Implement the following:

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        - Linear(input_dim, hidden_dim) -> BatchNorm1d -> ReLU
        - Linear(hidden_dim, output_dim) -> BatchNorm1d -> ReLU

    def forward(self, x) -> torch.Tensor:
        - Pass through layers
        - L2 normalize output: F.normalize(x, p=2, dim=1)
        - Return shape (batch, output_dim)

Usage:
    - RNA encoder: MLPEncoder(input_dim=256)   # 256 PCA dims
    - Protein encoder: MLPEncoder(input_dim=134)  # 134 CLR-normalized proteins
    - Baseline uses same architecture but without L2 normalization (add a flag or subclass)
"""
