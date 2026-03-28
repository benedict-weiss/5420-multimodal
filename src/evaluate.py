"""
evaluate.py — Evaluation metrics for all models

Implement the following functions:

compute_auroc(y_true, y_pred_proba, n_classes) -> float:
    - sklearn.metrics.roc_auc_score with average='macro', multi_class='ovr'
    - y_pred_proba: softmax probabilities shape (n_samples, n_classes)
    - Return macro-averaged AUC-ROC

compute_accuracy(y_true, y_pred) -> tuple[float, dict]:
    - Overall accuracy: sklearn.metrics.accuracy_score
    - Per-class accuracy: sklearn.metrics.classification_report (as dict)
    - Return (overall_accuracy, per_class_dict)

compute_asw(embeddings, labels) -> float:
    - sklearn.metrics.silhouette_score(embeddings, labels)
    - Normalize to [0,1]: (asw + 1) / 2
    - Use a random subsample if > 10k cells (silhouette is O(n^2))

compute_recall_at_k(z_rna, z_protein, k_values=[10, 20, 30, 40, 50]) -> dict:
    - For each RNA embedding, find k nearest neighbors in protein embedding space (cosine)
    - Check if the true match (same cell index) is in the top-k
    - Return dict: {k: recall_value for k in k_values}

plot_phate(embeddings, labels, title, save_path):
    - Fit PHATE on embeddings: phate_op = phate.PHATE(n_components=2); coords = phate_op.fit_transform(embeddings)
    - Scatter plot colored by cell type labels
    - Save to save_path as PNG
    - Use a random subsample if > 20k cells for speed

compute_batch_entropy(embeddings, batch_labels, n_neighbors=50) -> float:
    - For each cell, find n_neighbors nearest neighbors
    - Compute entropy of batch label distribution among neighbors
    - Average across all cells
    - High entropy = good mixing (no batch effect leakage)

run_significance_test(scores_model1, scores_model2) -> float:
    - scipy.stats.ranksums (Wilcoxon rank-sum test)
    - Input: lists of metric values across random seeds
    - Return p-value
"""
