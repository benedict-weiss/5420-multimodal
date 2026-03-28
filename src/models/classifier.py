"""
classifier.py — Classification head for cell type prediction

Implement the following:

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=64, dropout=0.2):
        - Linear(input_dim, hidden_dim) -> ReLU -> Dropout(dropout) -> Linear(hidden_dim, n_classes)

    def forward(self, x) -> torch.Tensor:
        - Return logits shape (batch, n_classes)
        - Do NOT apply softmax — use with F.cross_entropy which expects raw logits

Usage:
    - Baseline (Model 1): ClassificationHead(input_dim=128, n_classes=n)
        - Input is MLP encoder output (128-d)
    - Contrastive models (Models 2 & 3): ClassificationHead(input_dim=256, n_classes=n)
        - Input is concatenated [z_rna; z_protein] (128 + 128 = 256-d)
"""
