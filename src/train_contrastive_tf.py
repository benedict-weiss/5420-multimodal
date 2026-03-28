"""
train_contrastive_tf.py — Contrastive transformer training, two-stage (Model 3)

Implement the following:

1. Load preprocessed data:
    - Pathway tokens (~300-d) for RNA, protein CLR (134-d), labels
    - Build pathway tokens via preprocessing.build_pathway_tokens()
    - Split by donor into train/test

2. Build model:
    - rna_encoder = TransformerEncoder(n_tokens=n_pathways, d_model=64, nhead=4,
                                       num_layers=2, dim_feedforward=256, dropout=0.1, output_dim=128)
    - protein_encoder = TransformerEncoder(n_tokens=134, d_model=64, nhead=4,
                                           num_layers=2, dim_feedforward=256, dropout=0.1, output_dim=128)
    - classifier = ClassificationHead(input_dim=256, n_classes=n)
    - contrastive_loss = CLIPLoss(temperature=0.07)

3. Stage A — Contrastive pretraining:
    - Same as MLP version but with transformer encoders
    - Max 150 epochs, early stopping (patience=10, min_delta=1e-4)
    - GPU recommended — move models and data to device

4. Stage B — Classifier training:
    - Freeze transformer encoders
    - Train classifier on concatenated [z_rna; z_protein]
    - 50 epochs

5. Attention extraction (after training):
    - For each cell in test set, run forward pass and capture attention weights
    - Use TransformerEncoder.get_attention_weights()
    - Save raw attention matrices for attention_analysis.py

6. Evaluation:
    - Call evaluate.py functions on test set
    - Save checkpoints and attention weights to results/

7. Command-line usage:
    - python src/train_contrastive_tf.py --data_path data/ --seed 42 --batch_size 256

Hyperparameters:
    - temperature: 0.07, lr: 1e-3, weight_decay: 1e-5, batch_size: 256
    - d_model: 64, nhead: 4, num_layers: 2, dim_ff: 256, dropout: 0.1
    - contrastive_epochs: 150, classifier_epochs: 50
"""
