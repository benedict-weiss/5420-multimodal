"""Dataset and dataloader helpers for multimodal CITE-seq training."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CITEseqDataset(Dataset):
    """Paired RNA/protein dataset with optional pathway tokens."""

    def __init__(self, rna_pca, protein_clr, labels, pathway_tokens=None):
        rna = np.asarray(rna_pca, dtype=np.float32)
        protein = np.asarray(protein_clr, dtype=np.float32)
        y = np.asarray(labels)

        if rna.shape[0] != protein.shape[0] or rna.shape[0] != y.shape[0]:
            raise ValueError(
                "rna_pca, protein_clr, and labels must have matching first dimension"
            )

        self.rna = torch.as_tensor(rna, dtype=torch.float32)
        self.protein = torch.as_tensor(protein, dtype=torch.float32)
        self.labels = torch.as_tensor(y, dtype=torch.long)

        self.pathway = None
        if pathway_tokens is not None:
            pathway = np.asarray(pathway_tokens, dtype=np.float32)
            if pathway.shape[0] != rna.shape[0]:
                raise ValueError("pathway_tokens must match number of cells")
            self.pathway = torch.as_tensor(pathway, dtype=torch.float32)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        sample = {
            "rna": self.rna[idx],
            "protein": self.protein[idx],
            "label": self.labels[idx],
        }
        if self.pathway is not None:
            sample["pathway"] = self.pathway[idx]
        return sample


def get_dataloaders(
    rna_pca,
    protein_clr,
    labels,
    train_idx,
    test_idx,
    batch_size=256,
    pathway_tokens=None,
):
    """Build train/test dataloaders from index splits."""
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)

    train_pathway = None if pathway_tokens is None else np.asarray(pathway_tokens)[train_idx]
    test_pathway = None if pathway_tokens is None else np.asarray(pathway_tokens)[test_idx]

    train_dataset = CITEseqDataset(
        rna_pca=np.asarray(rna_pca)[train_idx],
        protein_clr=np.asarray(protein_clr)[train_idx],
        labels=np.asarray(labels)[train_idx],
        pathway_tokens=train_pathway,
    )
    test_dataset = CITEseqDataset(
        rna_pca=np.asarray(rna_pca)[test_idx],
        protein_clr=np.asarray(protein_clr)[test_idx],
        labels=np.asarray(labels)[test_idx],
        pathway_tokens=test_pathway,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, test_loader
