"""Causal marker attribution via single-token ablation on the frozen model.

For each protein token, zero its CLR-normalized input value, run the full
frozen protein encoder + classifier (using the real RNA embeddings for the
same cells), and record the drop in the correct-class logit.

Output: per-cell-type mean logit drop per protein token.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch


def _resolve_data_file(data_path: str) -> Path:
    path = Path(data_path)
    if path.is_file():
        return path
    candidates = sorted(list(path.glob("*.h5ad")) + list(path.glob("*.h5ad.gz")))
    if not candidates:
        raise FileNotFoundError(f"No .h5ad under {data_path}")
    return candidates[0]


def _load_anndata(data_path: str):
    import anndata as ad
    path = _resolve_data_file(data_path)
    if path.suffix != ".gz":
        return ad.read_h5ad(path)
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with gzip.open(path, "rb") as f_in, open(tmp_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return ad.read_h5ad(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _clr_normalize(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32) + 1.0
    log_matrix = np.log(matrix)
    return log_matrix - log_matrix.mean(axis=1, keepdims=True)


def run_ablation(
    checkpoint_dir: Path,
    data_path: str,
    batch_size: int = 256,
    device: str | None = None,
) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    from src.models.classifier import ClassificationHead
    from src.models.transformer_encoder import TransformerEncoder

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"[ablation] device={device}", flush=True)
    stage_a = torch.load(checkpoint_dir / "stage_a_best.pt", map_location=device)
    stage_b = torch.load(checkpoint_dir / "stage_b_best.pt", map_location=device)
    args = stage_a["args"]

    adata = _load_anndata(data_path)
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    if args.get("max_cells") is not None and args["max_cells"] < adata.shape[0]:
        rng = np.random.default_rng(args["seed"])
        idx = np.sort(rng.choice(adata.shape[0], size=args["max_cells"], replace=False))
        adata = adata[idx].copy()

    label_encoder = LabelEncoder()
    labels_all = label_encoder.fit_transform(adata.obs[args["label_col"]].values)
    if args.get("test_donors"):
        donors = adata.obs[args["donor_col"]].values
        test_idx = np.flatnonzero(np.isin(donors, args["test_donors"]))
    else:
        _, test_idx = train_test_split(
            np.arange(adata.shape[0]),
            test_size=args["test_size"],
            random_state=args["seed"],
            stratify=None if args.get("max_cells") is not None else labels_all,
        )
    test_idx = np.asarray(test_idx)
    test_labels = labels_all[test_idx]

    protein_adata = adata[:, adata.var["feature_types"] == "ADT"].copy()
    test_protein = _clr_normalize(protein_adata[test_idx].X)  # (N, P)

    protein_encoder = TransformerEncoder(
        n_tokens=stage_a["n_proteins"],
        d_model=args["d_model"], nhead=args["nhead"],
        num_layers=args["num_layers"], dim_feedforward=args["dim_feedforward"],
        dropout=args["dropout"], output_dim=args["embedding_dim"],
    ).to(device).eval()
    protein_encoder.load_state_dict(stage_a["protein_encoder_state_dict"])

    n_classes = stage_b.get("n_classes")
    if n_classes is None:
        n_classes = stage_b["classifier_state_dict"]["net.3.weight"].shape[0]
    stage_b["n_classes"] = n_classes

    rna_emb = np.load(checkpoint_dir / "test_rna_embeddings.npy")
    assert rna_emb.shape[0] == test_protein.shape[0], \
        f"RNA emb N={rna_emb.shape[0]} vs protein N={test_protein.shape[0]}"

    classifier = ClassificationHead(
        input_dim=2 * args["embedding_dim"],
        n_classes=stage_b["n_classes"],
        hidden_dim=args.get("classifier_hidden_dim", 64),
        dropout=args.get("classifier_dropout", 0.2),
    ).to(device).eval()
    classifier.load_state_dict(stage_b["classifier_state_dict"])

    N, P = test_protein.shape
    print(f"[ablation] N_test={N}  n_proteins={P}  n_classes={stage_b['n_classes']}", flush=True)
    baseline_logits = np.zeros((N, stage_b["n_classes"]), dtype=np.float32)
    import time
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            p = torch.tensor(test_protein[start:end], dtype=torch.float32, device=device)
            r = torch.tensor(rna_emb[start:end], dtype=torch.float32, device=device)
            zp = protein_encoder(p)
            logits = classifier(torch.cat([r, zp], dim=1))
            baseline_logits[start:end] = logits.cpu().numpy()
    print(f"[ablation] baseline pass done in {time.time() - t0:.1f}s", flush=True)

    drops = np.zeros((N, P), dtype=np.float32)
    correct_logit_baseline = baseline_logits[np.arange(N), test_labels]
    # Column means: replacing a token with its mean is a neutral ablation in CLR space
    # (zeroing sets it to the geometric mean, which is non-neutral)
    col_means = test_protein.mean(axis=0)  # (P,)

    with torch.no_grad():
        for tok in range(P):
            ablated = test_protein.copy()
            ablated[:, tok] = col_means[tok]
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                p = torch.tensor(ablated[start:end], dtype=torch.float32, device=device)
                r = torch.tensor(rna_emb[start:end], dtype=torch.float32, device=device)
                zp = protein_encoder(p)
                logits = classifier(torch.cat([r, zp], dim=1)).cpu().numpy()
                drops[start:end, tok] = (
                    correct_logit_baseline[start:end] - logits[np.arange(end - start), test_labels[start:end]]
                )
            elapsed = time.time() - t0
            eta = elapsed / (tok + 1) * (P - tok - 1)
            print(f"  ablated token {tok + 1}/{P}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    with open(checkpoint_dir / "label_mapping.json") as f:
        raw = json.load(f)
    label_names = {int(k): v for k, v in raw.items()}

    per_type: dict = {}
    for li, ln in label_names.items():
        mask = test_labels == li
        if mask.sum() == 0:
            continue
        per_type[ln] = drops[mask].mean(axis=0)

    np.save(checkpoint_dir / "ablation_logit_drop_per_cell.npy", drops)
    per_type_matrix = np.array([per_type[k] for k in sorted(per_type.keys())])
    np.save(checkpoint_dir / "ablation_logit_drop_per_type.npy", per_type_matrix)
    with open(checkpoint_dir / "ablation_per_type_order.json", "w") as f:
        json.dump(sorted(per_type.keys()), f, indent=2)
    print(f"Saved: ablation_logit_drop_per_cell.npy  shape={drops.shape}")
    print(f"Saved: ablation_logit_drop_per_type.npy  shape={per_type_matrix.shape}")

    return per_type


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Token ablation attribution")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args(argv)

    ckpt = Path(args.checkpoint_dir)
    per_type = run_ablation(ckpt, args.data_path, batch_size=args.batch_size)

    with open(ckpt / "protein_names.json") as f:
        protein_names = json.load(f)

    print("\n=== Top markers by logit drop (ablation) ===")
    for ct in sorted(per_type.keys()):
        top = np.argsort(per_type[ct])[::-1][: args.top_k]
        entries = [f"{protein_names[i]}({per_type[ct][i]:+.3f})" for i in top]
        print(f"  {ct}: {entries}")


if __name__ == "__main__":
    main()
