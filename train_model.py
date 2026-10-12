#ML2: Truncated SVD

import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
print("\n--- Truncated SVD ---")
output_dir = Path("outputs")
x=load_npz(output_dir/"customer_product_matrix.npz")
print("Original matrix shape:", x.shape)
print("Non-zero entries:", x.nnz)
components_counts=[10,20,50,100,200]
results=[]
for k in components_counts:
    print(f"\nTraining TruncatedSVD with k={k}")
    start_time=time.perf_counter()
    model=TruncatedSVD(n_components=k,algorithm="randomized",n_iter=5,random_state=42)
    customer_embeddings=model.fit_transform(x)

    original_norm_squared = x.multiply(x).sum()
    reconstructed_norm_squared = np.sum(customer_embeddings ** 2)
    cross_term = np.sum(customer_embeddings * (x @ model.components_.T))
    error_squared = original_norm_squared + reconstructed_norm_squared - 2 * cross_term
    relative_error = np.sqrt(max(float(error_squared), 0.0) / float(original_norm_squared))

    training_time=time.perf_counter()-start_time
    explained_variance=(model.explained_variance_ratio_.sum()*100)
    print(f"Relative reconstruction error: {relative_error:.4f}")
    print(f"Explained variance:{explained_variance:.2f}%")
    print(f"Training time:{training_time:.2f}seconds")
    print("Reduced matrix shape:",customer_embeddings.shape)
    np.save(output_dir/f"svd_embeddings_{k}.npy",customer_embeddings)
    np.save(output_dir/f"svd_components_{k}.npy",model.components_)
    results.append({"Components":k,"Explained Variance(%)":round(explained_variance,2),"Training Time(seconds)":round(training_time,2),"Relative Reconstruction Error": round(relative_error, 4),})
results_df=pd.DataFrame(results)
results_df.to_csv(output_dir/"svd_results.csv",index=False)
print("\n--- SVD Experiment Summary ---")
print(results_df.to_string(index=False))
print("\nML2 TruncatedSVD completed!")

#ML2 Plot Reconstruction

plt.figure(figsize=(8, 5))
plt.plot(results_df["Components"],results_df["Relative Reconstruction Error"],marker="o",linewidth=2)
plt.title("TruncatedSVD Reconstruction Error vs Components")
plt.xlabel("Number of Components (k)")
plt.ylabel("Relative Frobenius Reconstruction Error")
plt.xticks([10, 20, 50, 100, 200])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir/"svd_reconstruction_error.png",dpi=300)
plt.close()
print("Reconstruction error plot saved!")