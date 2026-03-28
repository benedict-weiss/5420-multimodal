# BIOLOGY.md — Biological Context for Implementation

## What CITE-seq Measures

CITE-seq simultaneously captures two modalities from the same cell:
- **RNA (GEX)**: mRNA transcript counts via scRNA-seq. ~20,000 possible genes, high-dimensional, sparse (many zeros due to dropout — low-abundance transcripts lost during capture). Noisy proxy for protein levels. Does not reflect post-translational modifications.
- **Surface protein (ADT)**: antibody-derived tag counts. Oligonucleotide-conjugated antibodies bind cell surface proteins, barcodes are sequenced alongside RNA. Only ~100-200 proteins measured (limited by antibody panel), but more direct measurement of functional cell state than RNA. Has its own noise: protein-specific background signal from non-specific antibody binding.

The two modalities have fundamentally different statistical properties. RNA is high-dimensional and sparse. Protein is low-dimensional but more stable. They don't always agree — mRNA abundance is a noisy predictor of protein abundance due to translational regulation, protein half-life differences, and post-translational modifications.

## Cell Types in BMMC Data

Bone marrow contains hematopoietic cells at all differentiation stages. Expected cell types in this dataset (verify exact labels after loading):

**Stem/Progenitor cells:**
- HSC (hematopoietic stem cell) — markers: CD34+, CD38-
- MPP (multipotent progenitor) — CD34+, CD38+
- CMP (common myeloid progenitor)
- GMP (granulocyte-monocyte progenitor)
- MEP (megakaryocyte-erythroid progenitor)
- CLP (common lymphoid progenitor)

**Myeloid lineage:**
- CD14+ monocytes — markers: CD14, CD64, CD11b
- CD16+ monocytes — markers: CD16, CD14low
- Dendritic cells (cDC1, cDC2, pDC)

**Lymphoid lineage:**
- Naive B cells — CD19, CD20, IgD
- Memory B cells — CD19, CD27
- Plasma cells — CD38high, CD138
- Naive CD4 T cells — CD3, CD4, CD45RA
- Memory CD4 T cells — CD3, CD4, CD45RO
- Naive CD8 T cells — CD3, CD8, CD45RA
- Effector CD8 T cells — CD3, CD8
- Regulatory T cells — CD3, CD4, CD25, FOXP3 (RNA)
- NK cells — CD56, CD16, CD3-
- MAIT cells — difficult to separate from other T cells by RNA alone

**Erythroid lineage:**
- Erythroid precursors — CD71, CD235a

## Known Difficult Classification Cases

These are cell types where RNA alone struggles and protein data should help:
- **T cell subtypes**: CD4 vs CD8, naive vs memory vs effector — very similar transcriptomes, well-separated by CD4/CD8/CD45RA/CD45RO protein markers
- **NK vs cytotoxic T cells**: overlapping gene expression, distinguishable by CD56+/CD3- (NK) vs CD56-/CD3+ (T)
- **Monocyte subtypes**: CD14+ classical vs CD16+ non-classical — protein markers are more reliable than RNA
- **MAIT cells**: rare, hard to identify transcriptomically, identifiable by specific surface markers
- **HSC vs MPP**: very similar transcriptomes, partially separable by CD38 protein levels

These are the cell types to focus on when evaluating whether multimodal > unimodal.

## Expected Attention Patterns (for Transformer Validation)

If the transformer is learning biologically meaningful features, attention should show:

**Protein encoder — expected high-attention tokens per cell type:**
| Cell Type | Expected Top Protein Tokens |
|-----------|---------------------------|
| HSC/MPP | CD34, CD38, CD90, CD117 |
| B cells | CD19, CD20, CD79a |
| Plasma cells | CD38, CD138 |
| CD4 T cells | CD3, CD4, CD45RA or CD45RO |
| CD8 T cells | CD3, CD8, CD45RA or CD45RO |
| NK cells | CD56, CD16 |
| CD14 monocytes | CD14, CD11b, CD64, HLA-DR |
| CD16 monocytes | CD16, CD14 (low attention expected) |
| Erythroid | CD71, CD235a |

**RNA encoder — expected high-attention pathway tokens:**
| Cell Type | Expected Top Pathway Tokens (KEGG) |
|-----------|-----------------------------------|
| HSC/MPP | Hematopoietic cell lineage, Signaling pathways regulating pluripotency |
| B cells | B cell receptor signaling, NF-kappa B signaling |
| T cells | T cell receptor signaling, Th1/Th2 differentiation |
| NK cells | Natural killer cell mediated cytotoxicity |
| Monocytes | Toll-like receptor signaling, TNF signaling, Phagosome |
| Erythroid | Hematopoietic cell lineage |
| Dendritic cells | Antigen processing and presentation |

If the model attends to unrelated pathways (e.g., attending to ribosome pathway equally across all cell types), that suggests it's learning technical artifacts rather than biology.

## Preprocessing Rationale

**RNA normalization choices:**
- normalize_total(1e4) + log1p: standard scRNA-seq pipeline. Accounts for differences in sequencing depth between cells.
- HVG selection (4000): removes housekeeping genes and noise genes. 4000 is generous — 2000 is also common.
- PCA(256): standard dimensionality reduction. Captures >95% of variance. All major methods (sCIN, scGLUE, competition winners) use PCA as input.
- scale(max_value=10): centers and clips extreme values. Important for stable training.

**Protein (ADT) normalization:**
- CLR (centered log-ratio): standard for compositional data. For each cell, log-transforms counts then centers by subtracting the geometric mean across all proteins in that cell. Handles the fact that different antibodies have different capture efficiencies and background levels.
- Do NOT use the same normalization as RNA — ADT counts have different distributional properties (bimodal: background + true signal).

**Why PCA for RNA but not protein:**
- RNA has 13,953 features after HVG — too many for direct input, and correlated. PCA reduces to 256 uncorrelated components.
- Protein has 134 features — already low-dimensional and each feature is independently meaningful (a specific protein). PCA would lose per-protein interpretability.

## Batch Effects in This Dataset

The dataset was intentionally designed with nested batch effects:
- 4 preparation sites, each processing samples slightly differently
- Some donors measured at multiple sites, some at only one site
- Batch effects manifest as systematic shifts in expression levels per site

The contrastive loss should help with batch effects: if the model learns to align RNA and protein from the same cell regardless of site, it implicitly learns site-invariant representations. However, monitor batch entropy in embeddings to verify.

If batch effects are strong, consider adding batch as a covariate or using batch-aware sampling during contrastive training (ensure each mini-batch contains cells from multiple sites/donors).

## Key Biological Validation Checks

1. **PHATE sanity check**: after contrastive pretraining, PHATE of joint embeddings should show cells clustering by cell type, not by batch/site/donor
2. **Marker gene/protein verification**: the top differentially expressed features per cluster should match known markers
3. **Rare cell type recovery**: check if rare populations (pDCs, MAIT cells, early progenitors) are identifiable in the multimodal embedding but not in RNA-only
4. **Cross-modality consistency**: cells that are close in RNA space should generally be close in protein space (with exceptions for post-transcriptionally regulated proteins)

## Gene Set Resources

For pathway tokenization:
- **KEGG_2021_Human**: ~330 pathways, well-curated, good coverage of immune pathways. Start here.
- **GO_Biological_Process_2021**: thousands of terms, more granular but redundant. Use if KEGG coverage is insufficient.
- **MSigDB Hallmark**: 50 broad gene sets, less granular but very clean. Good for a simpler model.
- **Reactome_2022**: alternative to KEGG, ~1600 pathways, finer-grained.

Available via `gseapy.get_library()`. Gene names must match between the gene set library and `adata.var_names` — check for naming convention mismatches (e.g., HGNC symbols vs Ensembl IDs).

## Protein Panel Details

The BioLegend TotalSeq B Universal Human Panel v1.0 contains 134 antibodies targeting surface proteins. These are pre-selected to cover major immune lineage markers. The panel includes:
- Lineage markers (CD3, CD4, CD8, CD19, CD56, CD14, etc.)
- Activation markers (CD25, CD69, HLA-DR)
- Differentiation markers (CD34, CD38, CD45RA, CD45RO)
- Functional markers (CD16, CD11b, CD11c)

Not all 134 proteins will be informative for every cell type. Many will have near-zero expression in most cells. The transformer's attention mechanism should naturally down-weight uninformative protein tokens.
