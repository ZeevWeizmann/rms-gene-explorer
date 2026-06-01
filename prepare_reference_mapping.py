"""
Generate reference PCA model + coords for patient data projection via KNN.
Run once locally, uploads to HuggingFace.
No UMAP model pickle — uses KNN in PCA space to avoid numba/Python version issues.
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import umap
from huggingface_hub import HfApi

REPO_ID = "weizmannzeev/rms-gene-programs"
HF_TOKEN = None  # set via env: export HF_TOKEN=hf_...

print("Loading reference expression matrix...")
expr = np.load("expr_matrix_f16.npy").astype(np.float32)
gene_names = pd.read_csv("gene_names.csv")["0"].tolist()
print(f"Reference: {expr.shape[0]} cells × {expr.shape[1]} genes")

# Fit PCA on reference
print("Fitting PCA (50 components)...")
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(expr)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

# Fit UMAP on reference PCA to get reference UMAP coordinates
print("Fitting UMAP (this takes a few minutes)...")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.3,
    metric="euclidean",
    random_state=42,
    verbose=True,
)
X_umap = reducer.fit_transform(X_pca)
print("UMAP fitted.")

# Save PCA model (sklearn — Python-version safe)
print("Saving files...")
with open("reference_pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)

# Save reference PCA coords and UMAP coords as numpy (no numba, version-safe)
np.save("reference_pca_coords.npy", X_pca.astype(np.float32))
np.save("reference_umap_coords.npy", X_umap.astype(np.float32))

# Save gene names
pd.DataFrame({"gene": gene_names}).to_csv("reference_gene_names.csv", index=False)

print("Uploading to HuggingFace...")
api = HfApi()
for fname in [
    "reference_pca_model.pkl",
    "reference_pca_coords.npy",
    "reference_umap_coords.npy",
    "reference_gene_names.csv",
]:
    api.upload_file(
        path_or_fileobj=fname,
        path_in_repo=fname,
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"  Uploaded {fname}")

print("Done! Reference mapping files ready.")
