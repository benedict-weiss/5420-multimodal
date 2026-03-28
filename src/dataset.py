"""
dataset.py — PyTorch Dataset and DataLoader classes

Implement the following:

class CITEseqDataset(torch.utils.data.Dataset):
    def __init__(self, rna_pca, protein_clr, labels, pathway_tokens=None):
        - Store as torch.FloatTensor for features, torch.LongTensor for labels
        - pathway_tokens is optional (only needed for transformer model)

    def __len__(self):
        - Return number of cells

    def __getitem__(self, idx):
        - Return dict with keys: 'rna', 'protein', 'label'
        - If pathway_tokens provided, also include 'pathway' key

def get_dataloaders(rna_pca, protein_clr, labels, train_idx, test_idx,
                    batch_size=256, pathway_tokens=None) -> tuple[DataLoader, DataLoader]:
    - Create CITEseqDataset for train and test splits
    - Train loader: shuffle=True, drop_last=True (important for contrastive loss — need full batches)
    - Test loader: shuffle=False, drop_last=False
    - Return (train_loader, test_loader)
"""
