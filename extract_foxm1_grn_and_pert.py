"""
Extract quiescent→proliferative GRN matrix + FOXM1 KO perturbation CSVs.

This model was trained on ALL 13,968 cells (quiescent + intermediate + proliferative)
with 198 genes (union of quiescent & proliferative programs).
KO gene: FOXM1 — master regulator of the proliferative program.

Creates:
  grn_matrix_foxm1.npy      — 198×198 GRN interaction matrix
  grn_genes_foxm1.csv       — gene list (column: "gene")
  foxm1_ko_perturbation.csv — per-gene per-timepoint mean_wt / mean_ko / log2fc
  foxm1_ko_umap_expr.csv    — per-cell expr_real / expr_wt / expr_ko for FOXM1
"""
import numpy as np
import pandas as pd
import anndata as ad
import os, scipy.sparse as sp

DATA_DIR = "/Users/zeev/CardamomOT/my_project/Data"
CARDAMOM = "/Users/zeev/CardamomOT/my_project/cardamomOT"

# ── 1. GRN matrix ────────────────────────────────────────────────
inter = np.load(os.path.join(CARDAMOM, "foxm1_inter_stim1.0_prior1.0.npy"))
# Shape (199, 199, 1) — first row/col = stimulus (index 0 = control gene)
# CardamomOT convention: inter[1:, 1:, 0] is the N×N interaction matrix
grn_mat = inter[1:, 1:, 0]   # 198 × 198
print("GRN matrix shape:", grn_mat.shape)

adata_src = ad.read_h5ad(os.path.join(DATA_DIR, "data_full.h5ad"))
grn_genes = list(adata_src.var_names)   # 198 genes
print(f"GRN genes: {len(grn_genes)}  |  first 5: {grn_genes[:5]}")
print(f"FOXM1 in genes: {'FOXM1' in grn_genes}")

np.save(os.path.join(DATA_DIR, "grn_matrix_foxm1.npy"), grn_mat)
pd.DataFrame({"gene": grn_genes}).to_csv(
    os.path.join(DATA_DIR, "grn_genes_foxm1.csv"), index=False)
print("Saved grn_matrix_foxm1.npy and grn_genes_foxm1.csv")

# ── 2. Perturbation CSV (per-gene, per-timepoint) ────────────────
adata_wt = ad.read_h5ad(
    os.path.join(CARDAMOM, "foxm1_adata_sim_stim1.0_prior1.0.h5ad"))
adata_ko = ad.read_h5ad(
    os.path.join(CARDAMOM, "foxm1_adata_sim_KO_FOXM1_OV_none_stim1.0_prior1.0.h5ad"))

def to_dense(X):
    return X.toarray() if sp.issparse(X) else np.array(X)

X_wt = to_dense(adata_wt.X)
X_ko = to_dense(adata_ko.X)
times_wt = adata_wt.obs["time"].values.astype(float)
times_ko = adata_ko.obs["time"].values.astype(float)
genes = list(adata_wt.var_names)
print(f"\nWT sim shape: {X_wt.shape}  |  KO sim shape: {X_ko.shape}")

rows = []
for t in sorted(np.unique(times_wt)):
    mask_wt = times_wt == t
    mask_ko = times_ko == t
    wt_mean = X_wt[mask_wt].mean(axis=0)
    ko_mean = X_ko[mask_ko].mean(axis=0)
    for i, g in enumerate(genes):
        wt, ko = float(wt_mean[i]), float(ko_mean[i])
        if wt > 1e-9:
            lfc = float(np.log2((ko + 1e-9) / (wt + 1e-9)))
        else:
            lfc = 0.0
        rows.append({"gene": g, "time": t, "mean_wt": wt, "mean_ko": ko, "log2fc": lfc})

pert_df = pd.DataFrame(rows)
pert_df.to_csv(os.path.join(DATA_DIR, "foxm1_ko_perturbation.csv"), index=False)
print(f"Saved foxm1_ko_perturbation.csv — {len(pert_df)} rows")

# Top affected genes at last timepoint
last_t = sorted(np.unique(times_wt))[-1]
top = (pert_df[pert_df["time"] == last_t]
       .assign(abs_lfc=lambda d: d["log2fc"].abs())
       .nlargest(15, "abs_lfc")[["gene", "log2fc", "mean_wt", "mean_ko"]])
print(f"\nTop 15 genes at t={last_t}:")
print(top.to_string(index=False))

# ── 3. UMAP expression CSV (FOXM1 gene only, all cells) ──────────
foxm1_idx = genes.index("FOXM1")
adata_real = ad.read_h5ad(os.path.join(DATA_DIR, "data_full.h5ad"))
X_real = to_dense(adata_real.X)
n_cells = X_real.shape[0]

umap_rows = []
for cell_i in range(n_cells):
    umap_rows.append({
        "cell":      cell_i,
        "gene":      "FOXM1",
        "expr_real": float(X_real[cell_i, foxm1_idx]),
        "expr_wt":   float(X_wt[cell_i, foxm1_idx]),
        "expr_ko":   float(X_ko[cell_i, foxm1_idx]),
        "diff":      float(X_ko[cell_i, foxm1_idx] - X_wt[cell_i, foxm1_idx]),
    })

umap_df = pd.DataFrame(umap_rows)
umap_df.to_csv(os.path.join(DATA_DIR, "foxm1_ko_umap_expr.csv"), index=False)
print(f"\nSaved foxm1_ko_umap_expr.csv — {len(umap_df)} rows")

print("\nDone! Files created:")
for f in ["grn_matrix_foxm1.npy", "grn_genes_foxm1.csv",
          "foxm1_ko_perturbation.csv", "foxm1_ko_umap_expr.csv"]:
    path = os.path.join(DATA_DIR, f)
    size = os.path.getsize(path) / 1024
    print(f"  {f}: {size:.1f} KB")
