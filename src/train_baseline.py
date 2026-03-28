"""
train_baseline.py — RNA-only baseline training (Model 1)

Implement the following:

1. Load preprocessed data:
    - RNA PCA matrix (256-d) and labels from preprocessing.py
    - Split by donor into train/test

2. Build model:
    - MLPEncoder(input_dim=256, hidden_dim=256, output_dim=128) — no L2 normalization
    - ClassificationHead(input_dim=128, n_classes=n)

3. Training loop:
    - Optimizer: Adam(lr=1e-3, weight_decay=1e-5)
    - Loss: F.cross_entropy
    - Train for 50 epochs (or until convergence)
    - Track train/val loss and accuracy per epoch

4. Evaluation:
    - Call evaluate.py functions on test set
    - Save model checkpoint to results/

5. Command-line usage:
    - python src/train_baseline.py --data_path data/ --seed 42 --epochs 50

Hyperparameters:
    - lr: 1e-3, weight_decay: 1e-5, batch_size: 256
"""
