#ML6 KMeans

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
output_dir=Path("outputs")
embeddings=np.load(output_dir/"svd_embeddings_50.npy")
customer_2d=embeddings[:,:2]
KMeans=KMeans(n_clusters=4,random_state=42,n_init=10)
clusters=KMeans.fit_predict(embeddings)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(customer_2d[:, 0],customer_2d[:, 1],c=clusters,cmap="tab10",s=10,alpha=0.6)

plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.title("Customer Segmentation Using TruncatedSVD")

plt.colorbar(scatter, label="Cluster")
plt.tight_layout()

plt.savefig(output_dir / "customer_clusters.png",dpi=300)

plt.show()

print("\n--- ML6: Customer Segmentation ---")
print("Total customers:", len(clusters))
print("Number of clusters:", len(np.unique(clusters)))
for cluster_id in range(4):
    count = np.sum(clusters == cluster_id)
    print(f"Cluster {cluster_id}: {count} customers")
print("\nSaved: outputs/customer_clusters.png")


#ML6: describe each group with its top products and countries

X = load_npz(output_dir / "customer_product_matrix.npz")
customers = np.load(output_dir / "customers.npy", allow_pickle=True)
products = np.load(output_dir / "products.npy", allow_pickle=True)

sheets = pd.read_excel("data/raw/online_retail_II.xlsx",sheet_name=None,usecols=["Customer ID", "Country"])
country_data = pd.concat(sheets.values(), ignore_index=True)
country_data = country_data.dropna(subset=["Customer ID", "Country"])
country_data["Customer ID"] = pd.to_numeric(country_data["Customer ID"], errors="coerce")

customer_countries = (country_data.groupby("Customer ID")["Country"].agg(lambda values: values.value_counts().index[0]))
customer_ids = pd.to_numeric(pd.Series(customers), errors="coerce")
for cluster_id in range(4):
    mask = clusters == cluster_id
    product_counts = np.asarray(X[mask].sum(axis=0)).ravel()
    top_indices = np.argsort(product_counts)[::-1][:5]

    countries = customer_countries.reindex(customer_ids[mask]).fillna("Unknown")
    top_countries = countries.value_counts().head(5)
    print(f"\n--- Cluster {cluster_id} ---")
    print("Customers:", int(mask.sum()))
    print("Top 5 products:")
    for index in top_indices:
        print(f"  {products[index]}:"f"{int(product_counts[index])} customers")
    print("Top 5 countries:")
    for country, count in top_countries.items():
        print(f"  {country}: {count} customers")

