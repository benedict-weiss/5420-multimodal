# Attention Analysis Improvements — Implementation Plan

> **For agentic workers:** Implement task-by-task. Steps use checkbox syntax.

**Goal:** Make transformer attention analysis biologically meaningful by (a) replacing brittle top-k recall with rank + specificity scoring, (b) doing per-(layer,head) best-rank marker search instead of a reduce-then-score pipeline, (c) fixing the BMMC subtype marker panel, and (d) adding causal attribution via token ablation on the frozen classifier.

**Architecture:**
- Task 1–3 modify `src/attention_analysis.py` only — pure interpretation changes on saved attention arrays.
- Task 4 adds a new `src/attribution_ablation.py` that loads `stage_a_best.pt` + `stage_b_best.pt`, zeros one protein token at a time, and records per-cell-type logit drops. This is independent from attention.
- Task 5 regenerates outputs on the latest checkpoint for human inspection.

**Tech Stack:** Python, NumPy, PyTorch, sklearn, matplotlib/seaborn. Existing test style is pytest on preprocessing; interpretation code is validated by hand-inspection of outputs, not pytest.

---

## File inventory

| File | Action | Responsibility |
|------|--------|----------------|
| `src/attention_analysis.py` | modify | Rank/specificity-based scoring, per-head best-rank search, updated marker panel, enriched JSON outputs |
| `src/attribution_ablation.py` | create | Token-ablation attribution on frozen encoders + classifier |
| `tasks/2026-04-15-attention-improvements-plan.md` | create (this file) | The plan |
| `tasks/2026-04-15-review.md` | create at end | Summary of outputs |

The latest checkpoint used for validation throughout:
`results/checkpoints/contrastive_tf_seed42_20260414_194858/`

Data: `data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz`

---

## Task 1: Marker panel — BMMC subtype-appropriate markers

**Files:**
- Modify: `src/attention_analysis.py` (`_DEFAULT_MARKERS` dict)

**Rationale:** Current panel assigns CD19/CD20 (pan-B markers) as expected for Transitional B and Naive B subtypes — these don't discriminate between B subsets. Plasma cells are marked only by CD38 (too broad: HSC also strong). Using subtype-appropriate markers matches BMMC cell-type ontology and the Seurat/Azimuth BM reference.

- [ ] **Step 1: Replace `_DEFAULT_MARKERS`**

Replace the dict literal in `src/attention_analysis.py` with:

```python
_DEFAULT_MARKERS = {
    "HSC": ["CD38", "CD45RA"],
    "NK": ["CD56", "CD16", "CD335"],
    "Transitional B": ["CD24", "CD38", "IgM", "CD19"],
    "pDC": ["CD123", "CD303", "CD304"],
    "CD14+ Mono": ["CD14", "HLA-DR", "CD11b", "CD64"],
    "CD16+ Mono": ["CD16", "CX3CR1", "HLA-DR"],
    "CD4+ T naive": ["CD3", "CD4", "CD45RA", "CD62L"],
    "CD4+ T activated": ["CD3", "CD4", "CD45RO", "CD69"],
    "CD8+ T naive": ["CD3", "CD8", "CD45RA", "CD62L"],
    "Naive CD20+ B IGKC+": ["CD19", "CD20", "IgM", "IgD", "CD21"],
    "Naive CD20+ B IGKC-": ["CD19", "CD20", "IgM", "IgD", "CD21"],
    "MAIT": ["CD3", "CD8", "CD161", "TCRVa7.2"],
    "T reg": ["CD3", "CD4", "CD25", "CD127"],
    "Plasma cell IGKC+": ["CD38", "CD27"],
    "Plasma cell IGKC-": ["CD38", "CD27"],
    "cDC2": ["CD11c", "HLA-DR", "CD1c"],
    "Erythroblast": ["CD71", "CD235a"],
    "Normoblast": ["CD71"],
    "Reticulocyte": ["CD71"],
}
```

Note: `CD127` is the IL-7R subunit (low on Treg, high on Tconv) — validates as "expected top marker" by magnitude regardless of direction, since Treg cells have *distinctively low* CD127 and the attention should still key on this channel.

- [ ] **Step 2: Verify alias resolution covers these markers**

Run: `python -c "from src.attention_analysis import resolve_marker_alias; import json; names=json.load(open('results/checkpoints/contrastive_tf_seed42_20260414_194858/protein_names.json')); [print(m, '->', resolve_marker_alias(m, names)) for m in ['CD4','CD14','CD38','CD24','CD19','CD21','CD27','CD11b','CD69','CD235a','TCRVa7.2','CX3CR1']]"`

Expected: every marker either resolves to an exact or `-1`-suffixed token, or `None`. Note which ones are `None` — those markers are genuinely absent from the ADT panel and should be removed from the dict. Update the dict to drop any `None`-resolving markers.

- [ ] **Step 3: Commit**

```bash
git add src/attention_analysis.py
git commit -m "fix(attention): BMMC subtype-specific marker panel for validation"
```

---

## Task 2: Rank-based marker reporting + specificity score

**Files:**
- Modify: `src/attention_analysis.py` (add two helpers, update main flow)

**Rationale:** Top-10 recall throws away rank information (rank 11 == rank 134 == 0). A marker's *rank* among 134 proteins and its *specificity* (z-score across cell types) are biologically interpretable.

- [ ] **Step 1: Add `compute_marker_ranks` function**

Add below `validate_against_markers`:

```python
def compute_marker_ranks(
    attention_by_type: dict,
    expected_markers: dict,
    token_names: list[str],
) -> dict:
    """
    For each (cell type, expected marker), report rank (1 = most attended).

    Returns: {cell_type: {marker: {"resolved": str|None, "rank": int|None,
                                    "n_tokens": int, "percentile": float|None}}}
    Percentile is (n_tokens - rank + 1) / n_tokens; 1.0 means top, 0.0 means bottom.
    """
    result: dict = {}
    for cell_type, attn in attention_by_type.items():
        if cell_type not in expected_markers:
            continue
        order = np.argsort(attn)[::-1]
        rank_of_idx = {int(idx): r + 1 for r, idx in enumerate(order)}
        n = len(token_names)
        entries: dict = {}
        for marker in expected_markers[cell_type]:
            resolved = resolve_marker_alias(marker, token_names)
            if resolved is None or resolved not in token_names:
                entries[marker] = {"resolved": None, "rank": None,
                                   "n_tokens": n, "percentile": None}
                continue
            idx = token_names.index(resolved)
            rank = rank_of_idx[idx]
            entries[marker] = {
                "resolved": resolved,
                "rank": rank,
                "n_tokens": n,
                "percentile": (n - rank + 1) / n,
            }
        result[cell_type] = entries
    return result
```

- [ ] **Step 2: Add `compute_specificity_scores` function**

Add below `compute_marker_ranks`:

```python
def compute_specificity_scores(attention_by_type: dict) -> dict:
    """
    Z-score attention per token across cell types.

    High z means the cell type attends this token unusually much vs other types —
    a better proxy for 'marker' than raw attention, which is biased toward
    globally-interesting tokens like CD45.

    Returns same shape as input: {cell_type: np.ndarray(n_tokens,)}.
    """
    cell_types = list(attention_by_type.keys())
    matrix = np.array([attention_by_type[ct] for ct in cell_types])  # (types, tokens)
    mu = matrix.mean(axis=0, keepdims=True)
    sigma = matrix.std(axis=0, keepdims=True) + 1e-8
    z = (matrix - mu) / sigma
    return {ct: z[i] for i, ct in enumerate(cell_types)}
```

- [ ] **Step 3: Wire into `main()` — emit richer validation JSON**

After the existing `validation = validate_against_markers(...)` call, before writing `marker_validation.json`, add:

```python
    ranks = compute_marker_ranks(attn_by_type_prot, _DEFAULT_MARKERS, protein_names)
    specificity = compute_specificity_scores(attn_by_type_prot)
    top_tokens_spec = get_top_tokens(specificity, protein_names, top_k=args.top_k)
    validation_spec = validate_against_markers(
        top_tokens_spec, _DEFAULT_MARKERS, token_names=protein_names
    )
    ranks_spec = compute_marker_ranks(specificity, _DEFAULT_MARKERS, protein_names)

    enriched = {
        ct: {
            "top_10_recall": validation.get(ct, {}).get("recall"),
            "top_10_recall_specificity": validation_spec.get(ct, {}).get("recall"),
            "marker_ranks_raw": ranks.get(ct, {}),
            "marker_ranks_specificity": ranks_spec.get(ct, {}),
        }
        for ct in _DEFAULT_MARKERS
    }
```

Replace the existing `json.dump(validation, ...)` with `json.dump(enriched, f, indent=2)`.

Also add a specificity-based heatmap before the existing protein heatmap calls:

```python
    plot_per_celltype_top_heatmap(
        specificity, protein_names,
        title="Protein attention SPECIFICITY (z-scored across cell types, per-row top-K)",
        save_path=str(out / "attention_heatmap_protein_specificity.png"),
        top_k_per_row=args.top_k_per_row,
    )
    print(f"Saved: {out / 'attention_heatmap_protein_specificity.png'}")
```

- [ ] **Step 4: Run against existing checkpoint**

```bash
python -m src.attention_analysis \
  --checkpoint_dir results/checkpoints/contrastive_tf_seed42_20260414_194858 \
  --data_path data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz \
  --head_reduction max
```

Expected: produces `attention_heatmap_protein_specificity.png` plus an enriched `marker_validation.json` with per-marker `rank`, `percentile`, and `resolved` fields.

- [ ] **Step 5: Eyeball the enriched JSON**

```bash
python -c "import json; d=json.load(open('results/checkpoints/contrastive_tf_seed42_20260414_194858/marker_validation.json')); print(json.dumps({k:v['marker_ranks_specificity'] for k,v in d.items()}, indent=2))"
```

Expected: ranks should be small (<= 20 out of 134) for most canonical markers under the specificity view, materially better than under raw attention.

- [ ] **Step 6: Commit**

```bash
git add src/attention_analysis.py
git commit -m "feat(attention): rank-based marker reporting + specificity z-score view"
```

---

## Task 3: Per-(layer, head) best-rank marker search

**Files:**
- Modify: `src/attention_analysis.py` (add helper + output)

**Rationale:** `reduce_per_head_attention(..., "max")` collapses the head axis *before* ranking, which can hide a marker that is top-1 in one head when another head has a higher peak on a different token. Instead, for each (cell type, marker) we want: the *best rank* achievable by *any single (layer, head)* combination.

- [ ] **Step 1: Add `best_rank_across_heads` function**

Add below `compute_specificity_scores`:

```python
def best_rank_across_heads(
    attention_per_head: np.ndarray,   # (cells, layers, heads, tokens)
    labels: np.ndarray,
    label_names: dict,
    expected_markers: dict,
    token_names: list[str],
) -> dict:
    """
    For each (cell_type, marker), find the (layer, head) whose cell-type-mean
    attention gives that marker the best (lowest) rank, and report it.

    Returns:
      {cell_type: {marker: {"resolved": str|None,
                             "best_rank": int|None,
                             "best_layer": int|None,
                             "best_head": int|None,
                             "n_tokens": int}}}
    """
    if attention_per_head.ndim != 4:
        raise ValueError(f"Expected 4-d per-head attn, got {attention_per_head.shape}")
    n_cells, n_layers, n_heads, n_tokens = attention_per_head.shape
    result: dict = {}
    for label_int, label_str in label_names.items():
        if label_str not in expected_markers:
            continue
        mask = labels == int(label_int)
        if mask.sum() == 0:
            continue
        # (layers, heads, tokens) mean attention for this cell type
        mean_attn = attention_per_head[mask].mean(axis=0)
        entries: dict = {}
        for marker in expected_markers[label_str]:
            resolved = resolve_marker_alias(marker, token_names)
            if resolved is None or resolved not in token_names:
                entries[marker] = {"resolved": None, "best_rank": None,
                                   "best_layer": None, "best_head": None,
                                   "n_tokens": n_tokens}
                continue
            tok_idx = token_names.index(resolved)
            best_rank = None
            best_lh = (None, None)
            for layer in range(n_layers):
                for head in range(n_heads):
                    row = mean_attn[layer, head]
                    # rank of tok_idx: 1 + count of tokens with strictly greater attn
                    rank = 1 + int((row > row[tok_idx]).sum())
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_lh = (layer, head)
            entries[marker] = {
                "resolved": resolved, "best_rank": best_rank,
                "best_layer": best_lh[0], "best_head": best_lh[1],
                "n_tokens": n_tokens,
            }
        result[label_str] = entries
    return result
```

- [ ] **Step 2: Call it from `main()` and save**

Inside the `if attn_by_type_prot_per_head is not None:` block, add after existing per-head validation:

```python
        best_rank = best_rank_across_heads(
            attn_protein_per_head, labels, label_names,
            _DEFAULT_MARKERS, protein_names,
        )
        best_rank_path = out / "marker_best_rank_per_head.json"
        with open(best_rank_path, "w", encoding="utf-8") as f:
            json.dump(best_rank, f, indent=2)
        print(f"Saved: {best_rank_path}")
        print("\n=== Best (layer, head) per marker ===")
        for ct, markers in best_rank.items():
            for m, info in markers.items():
                if info["best_rank"] is None:
                    continue
                print(f"  {ct} / {m} ({info['resolved']}): "
                      f"rank={info['best_rank']}/{info['n_tokens']} "
                      f"@layer{info['best_layer']} head{info['best_head']}")
```

- [ ] **Step 3: Run and inspect**

Re-run the command from Task 2 Step 4. Confirm `marker_best_rank_per_head.json` appears. Expected: NK/CD56 reaches rank <= 10 in some head even though the aggregate per-head-max hid it; CD3 appears rank <= 5 for all T subtypes in some head.

- [ ] **Step 4: Commit**

```bash
git add src/attention_analysis.py
git commit -m "feat(attention): best-rank-across-heads marker search"
```

---

## Task 4: Causal attribution via token ablation

**Files:**
- Create: `src/attribution_ablation.py`

**Rationale:** Attention is known to be an unreliable explanation (Jain & Wallace 2019). For causal importance, zero out each protein token, run the frozen encoder + classifier, record the per-cell-type logit drop. This gives a model-grounded "if this marker were missing, how much would this cell type's score fall?" score. Same format as attention outputs so downstream plots reuse.

- [ ] **Step 1: Create `src/attribution_ablation.py`**

Write file with this content:

```python
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

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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

    rna_emb = np.load(checkpoint_dir / "test_rna_embeddings.npy")
    assert rna_emb.shape[0] == test_protein.shape[0], \
        f"RNA emb N={rna_emb.shape[0]} vs protein N={test_protein.shape[0]}"

    classifier = ClassificationHead(
        input_dim=2 * args["embedding_dim"],
        n_classes=stage_b["n_classes"],
        hidden_dim=args.get("classifier_hidden", 64),
        dropout=args.get("classifier_dropout", 0.2),
    ).to(device).eval()
    classifier.load_state_dict(stage_b["classifier_state_dict"])

    N, P = test_protein.shape
    baseline_logits = np.zeros((N, stage_b["n_classes"]), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = start + batch_size
            p = torch.tensor(test_protein[start:end], dtype=torch.float32, device=device)
            r = torch.tensor(rna_emb[start:end], dtype=torch.float32, device=device)
            zp = protein_encoder(p)
            logits = classifier(torch.cat([r, zp], dim=1))
            baseline_logits[start:end] = logits.cpu().numpy()

    drops = np.zeros((N, P), dtype=np.float32)
    correct_logit_baseline = baseline_logits[np.arange(N), test_labels]

    with torch.no_grad():
        for tok in range(P):
            ablated = test_protein.copy()
            ablated[:, tok] = 0.0
            for start in range(0, N, batch_size):
                end = start + batch_size
                p = torch.tensor(ablated[start:end], dtype=torch.float32, device=device)
                r = torch.tensor(rna_emb[start:end], dtype=torch.float32, device=device)
                zp = protein_encoder(p)
                logits = classifier(torch.cat([r, zp], dim=1)).cpu().numpy()
                drops[start:end, tok] = (
                    correct_logit_baseline[start:end] - logits[np.arange(end - start), test_labels[start:end]]
                )
            if (tok + 1) % 10 == 0 or tok == P - 1:
                print(f"  ablated token {tok + 1}/{P}")

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
```

- [ ] **Step 2: Run it**

```bash
python -m src.attribution_ablation \
  --checkpoint_dir results/checkpoints/contrastive_tf_seed42_20260414_194858 \
  --data_path data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz \
  --batch_size 256
```

Expected: ~5-10 minutes on GPU, longer on CPU. Produces `ablation_logit_drop_per_cell.npy` (N_test × 134) and `ablation_logit_drop_per_type.npy` (n_types × 134). Prints top-10 markers per cell type. Canonical markers should appear prominently: T cells → CD3/CD4/CD8; NK → CD56/CD16; B → CD19/CD20/IgM; HSC → CD38; erythroid → CD71.

- [ ] **Step 3: Write a small scoring script, integrated into attention_analysis**

Modify `src/attention_analysis.py`: at the end of `main()`, add:

```python
    ablation_path = ckpt / "ablation_logit_drop_per_type.npy"
    order_path = ckpt / "ablation_per_type_order.json"
    if ablation_path.exists() and order_path.exists():
        abl = np.load(ablation_path)
        with open(order_path) as f:
            order = json.load(f)
        abl_by_type = {ct: abl[i] for i, ct in enumerate(order)}
        abl_ranks = compute_marker_ranks(abl_by_type, _DEFAULT_MARKERS, protein_names)
        abl_path = out / "marker_ranks_ablation.json"
        with open(abl_path, "w") as f:
            json.dump(abl_ranks, f, indent=2)
        print(f"Saved: {abl_path}")
        plot_per_celltype_top_heatmap(
            abl_by_type, protein_names,
            title="Per-protein logit drop by cell type (token ablation)",
            save_path=str(out / "attention_heatmap_protein_ablation.png"),
            top_k_per_row=args.top_k_per_row,
        )
        print(f"Saved: {out / 'attention_heatmap_protein_ablation.png'}")
```

- [ ] **Step 4: Rerun attention_analysis to pick up the ablation files**

```bash
python -m src.attention_analysis \
  --checkpoint_dir results/checkpoints/contrastive_tf_seed42_20260414_194858 \
  --data_path data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz \
  --head_reduction max
```

- [ ] **Step 5: Commit**

```bash
git add src/attribution_ablation.py src/attention_analysis.py
git commit -m "feat: causal marker attribution via token ablation on frozen classifier"
```

---

## Task 5: Review

**Files:**
- Create: `tasks/2026-04-15-review.md`

- [ ] **Step 1: Write review doc**

Contents: for each cell type in `_DEFAULT_MARKERS`, tabulate:
- best attention rank (raw), best rank (specificity), best rank (per-head), best rank (ablation)
- Mark which markers are now recovered that weren't before.

- [ ] **Step 2: Commit**

```bash
git add tasks/2026-04-15-review.md
git commit -m "docs: attention-analysis review on latest contrastive_tf checkpoint"
```

---
