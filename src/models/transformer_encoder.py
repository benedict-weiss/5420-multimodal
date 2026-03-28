"""
transformer_encoder.py — Transformer encoder with CLS token for token-level inputs

Implement the following:

class TransformerEncoder(nn.Module):
    def __init__(self, n_tokens, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1, output_dim=128):
        - self.input_proj = nn.Linear(1, d_model)  # each token is a scalar -> project to d_model
        - self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # learnable CLS token
        - self.pos_encoding = nn.Parameter(torch.randn(1, n_tokens + 1, d_model))  # +1 for CLS
        - self.encoder = nn.TransformerEncoder(
              nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                        activation='gelu', batch_first=True),
              num_layers=num_layers
          )
        - self.output_proj = nn.Linear(d_model, output_dim)
        - self._attention_weights = None  # store for extraction

    def forward(self, x) -> torch.Tensor:
        - x shape: (batch, n_tokens) -> reshape to (batch, n_tokens, 1)
        - Project: x = self.input_proj(x)  # (batch, n_tokens, d_model)
        - Prepend CLS: cls = self.cls_token.expand(batch, -1, -1); x = cat([cls, x], dim=1)
        - Add positional encoding: x = x + self.pos_encoding
        - Pass through transformer encoder
        - Take CLS output: x = x[:, 0, :]  # (batch, d_model)
        - Project and normalize: F.normalize(self.output_proj(x), p=2, dim=1)
        - Return shape (batch, output_dim)

    def get_attention_weights(self):
        - Return self._attention_weights

Attention extraction:
    - Register a forward hook on the last TransformerEncoderLayer's self_attn module
    - In the hook, capture the attention weights (second return value of MultiheadAttention)
    - Average across heads: attn.mean(dim=1)  # (batch, seq_len, seq_len)
    - Take CLS row: attn[:, 0, 1:]  # skip CLS-to-CLS, get CLS-to-tokens
    - Store in self._attention_weights
    - NOTE: nn.MultiheadAttention must be called with need_weights=True and average_attn_weights=False
      to get per-head weights. You may need to modify the TransformerEncoderLayer or use a custom one.

Usage:
    - RNA encoder: TransformerEncoder(n_tokens=~300)   # pathway tokens
    - Protein encoder: TransformerEncoder(n_tokens=134)  # protein tokens
"""
