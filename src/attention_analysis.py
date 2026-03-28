"""
attention_analysis.py — Attention heatmaps and biological validation

Implement the following functions:

aggregate_attention_by_cell_type(attention_weights, labels, label_names) -> dict:
    - attention_weights: (n_cells, n_tokens) from transformer encoder
    - Group by cell type, compute mean attention per token per cell type
    - Return dict: {cell_type: np.ndarray of shape (n_tokens,)}

plot_attention_heatmap(attention_by_type, token_names, title, save_path):
    - Rows = cell types, columns = token names (pathways or proteins)
    - Use seaborn.heatmap with appropriate colormap
    - For protein encoder: 134 tokens (may need to show top-N per cell type)
    - For RNA encoder: ~300 pathway tokens (definitely need to filter to top-N)
    - Save to save_path

get_top_tokens(attention_by_type, token_names, top_k=10) -> dict:
    - For each cell type, return top-k most attended tokens
    - Return dict: {cell_type: [(token_name, attention_score), ...]}

validate_against_markers(top_tokens_dict, expected_markers) -> dict:
    - expected_markers: dict from biology.md, e.g.:
        {'HSC': ['CD34', 'CD38', 'CD90', 'CD117'],
         'B cell': ['CD19', 'CD20', 'CD79a'],
         'CD4 T cell': ['CD3', 'CD4', 'CD45RA', 'CD45RO'], ...}
    - For each cell type, check how many expected markers appear in top-k attended tokens
    - Return dict: {cell_type: {'found': [...], 'missing': [...], 'precision': float}}

plot_token_attention_per_cell_type(attention_weights, token_names, cell_type_labels,
                                   selected_types, save_path):
    - For selected cell types, plot distribution of attention across all tokens
    - Box plot or violin plot showing attention variability per token
    - Useful for seeing if attention is focused or diffuse
"""
