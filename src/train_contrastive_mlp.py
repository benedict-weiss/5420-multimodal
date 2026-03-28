"""
train_contrastive_mlp.py — Contrastive MLP training, two-stage (Model 2)

Implement the following:

1. Load preprocessed data:
    - RNA PCA (256-d), protein CLR (134-d), labels
    - Split by donor into train/test

2. Build model:
    - rna_encoder = MLPEncoder(input_dim=256, output_dim=128)  # with L2 norm
    - protein_encoder = MLPEncoder(input_dim=134, output_dim=128)  # with L2 norm
    - classifier = ClassificationHead(input_dim=256, n_classes=n)  # input is concat [z_rna; z_protein]
    - contrastive_loss = CLIPLoss(temperature=0.07)

3. Stage A — Contrastive pretraining:
    - Train rna_encoder and protein_encoder only
    - Optimizer: Adam(lr=1e-3, weight_decay=1e-5)
    - Loss: CLIPLoss on (z_rna, z_protein)
    - Max 150 epochs with early stopping (patience=10, min_delta=1e-4 on val loss)
    - Log contrastive loss per epoch

4. Stage B — Classifier training:
    - Freeze rna_encoder and protein_encoder (requires_grad=False)
    - Train classifier head only
    - Generate embeddings: z_rna = rna_encoder(rna_pca), z_protein = protein_encoder(protein_clr)
    - Concatenate: z = torch.cat([z_rna, z_protein], dim=1)  # (batch, 256)
    - Loss: F.cross_entropy
    - Train for 50 epochs

5. Evaluation:
    - Call evaluate.py functions on test set
    - Save both encoder checkpoints and classifier checkpoint

6. Command-line usage:
    - python src/train_contrastive_mlp.py --data_path data/ --seed 42 --batch_size 512

Hyperparameters:
    - temperature: 0.07, lr: 1e-3, weight_decay: 1e-5, batch_size: 256-512
    - contrastive_epochs: 150, classifier_epochs: 50
"""
