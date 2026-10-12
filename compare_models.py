#ML5: Comparison of PCA VS TRUNCATED SVD

import time
import tracemalloc
from pathlib import Path
import numpy as np 
import pandas as pd
from scipy.sparse import load_npz
from sklearn.decomposition import PCA, TruncatedSVD

print("\n--- PCA vs TruncatedSVD ---")
output_dir = Path("outputs")
X = load_npz(output_dir / "customer_product_matrix.npz")
k = 50
seed = 42
print("Training matrix shape:", X.shape)
print("Number of components:", k)

# 1. TRUNCATED SVD

print("\nTraining TruncatedSVD...")
tracemalloc.start()
start = time.perf_counter()
svd = TruncatedSVD(n_components=k,random_state=seed)
svd_embeddings = svd.fit_transform(X)
svd_time = time.perf_counter() - start
_, svd_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
svd_components=svd.components_
svd_factor_mb=(svd_embeddings.nbytes+svd_components.nbytes)/(1024 ** 2)

X_norm_sq = float(X.multiply(X).sum())
svd_projection=X@svd_components.T
svd_error_sq=(X_norm_sq+ np.sum(svd_embeddings ** 2)- 2 * np.sum(svd_embeddings * svd_projection))
svd_error = np.sqrt(max(float(svd_error_sq), 0.0) / X_norm_sq)
print(f"SVD fit time: {svd_time:.4f} seconds")
print(f"SVD Python peak memory: {svd_peak / 1024**2:.2f} MB")
print(f"SVD stored factors: {svd_factor_mb:.2f} MB")
print(f"SVD reconstruction error: {svd_error:.4f}")

#PCA

print("\nTraining PCA...")

X_dense = X.toarray()

tracemalloc.start()
start = time.perf_counter()

pca = PCA(n_components=k,svd_solver="randomized",random_state=seed)
pca_embeddings = pca.fit_transform(X_dense)
pca_time = time.perf_counter() - start
_, pca_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
pca_components = pca.components_
pca_factor_mb = (pca_embeddings.nbytes+ pca_components.nbytes+ pca.mean_.nbytes)/(1024 ** 2)
pca_variance = (pca.explained_variance_ratio_.sum() * 100)
print(f"PCA fit time: {pca_time:.4f} seconds")
print(f"PCA Python peak memory: {pca_peak / 1024**2:.2f} MB")
print(f"PCA stored factors: {pca_factor_mb:.2f} MB")
print(f"PCA explained variance: {pca_variance:.2f}%")

np.save(output_dir / f"pca_embeddings_{k}.npy",pca_embeddings)
np.save(output_dir / f"pca_components_{k}.npy",pca_components)
np.save(output_dir / f"pca_mean_{k}.npy",pca.mean_)

comparison = pd.DataFrame([{"Method": "TruncatedSVD","Components": k,"Fit Time (seconds)": round(svd_time, 4),"Python Peak Memory (MB)": round(svd_peak / 1024**2, 2),"Stored Factors (MB)": round(svd_factor_mb, 2),"Explained Variance (%)": round(svd.explained_variance_ratio_.sum() * 100, 2),"Relative Reconstruction Error": round(svd_error, 4)},{"Method": "PCA","Components": k,"Fit Time (seconds)": round(pca_time, 4),"Python Peak Memory (MB)": round(pca_peak / 1024**2, 2),"Stored Factors (MB)": round(pca_factor_mb, 2),"Explained Variance (%)": round(pca_variance, 2),"Relative Reconstruction Error": np.nan}])

comparison.to_csv(output_dir / "pca_svd_comparison.csv",index=False)
print("\n--- Preliminary PCA vs SVD Comparison ---")
print(comparison.to_string(index=False))
print("\nSaved: outputs/pca_svd_comparison.csv")
print("ML5 initial comparison completed!")

#ML5
products = np.load(output_dir / "products.npy",allow_pickle=True)
for method, components in [("TruncatedSVD", svd_components),("PCA", pca_components)]:
    first_component = components[0]
    top_indices = np.argsort(np.abs(first_component))[::-1][:10]
    print(f"\n--- {method}: Component 1 ---")
    for index in top_indices:
        print(f"Product: {products[index]},"f"Loading: {first_component[index]:.4f}")