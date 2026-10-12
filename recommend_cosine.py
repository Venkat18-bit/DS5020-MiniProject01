# ML3: COSINE SIMILARITY RECOMMENDATIONS

import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
print("\n--- Product Cosine Similarity ---")
output_dir = Path("outputs")
X = load_npz(output_dir / "customer_product_matrix.npz")
products = np.load(output_dir / "products.npy",allow_pickle=True)
k = 50
components = np.load(output_dir / f"svd_components_{k}.npy")
product_factors = components.T
print("Product factor matrix:", product_factors.shape)
start_time = time.perf_counter()
similarity = cosine_similarity(product_factors)
np.fill_diagonal(similarity, -np.inf)

top_n = 10
top_similar = np.argsort(similarity,axis=1)[:, ::-1][:, :top_n]
print("Product neighbor matrix:", top_similar.shape)
np.save(output_dir / f"product_neighbors_{k}.npy",top_similar)

# CUSTOMER RECOMMENDATIONS

def recommend_products(customer_index, top_n=10):
    purchased = X[customer_index].indices
    scores = np.zeros(X.shape[1], dtype=np.float32)
    for product_index in purchased:
        neighbors = top_similar[product_index]
        for neighbor in neighbors:
            scores[neighbor] += similarity[product_index, neighbor]
    scores[purchased] = -np.inf
    positive_indices = np.where(np.isfinite(scores) & (scores > 0))[0]
    ranked_positive = positive_indices[
        np.argsort(scores[positive_indices])[::-1]
    ]
    if len(ranked_positive) < top_n:
        popularity = np.asarray(X.sum(axis=0)).ravel()
        popularity[purchased] = -np.inf

        if len(ranked_positive) > 0:
            popularity[ranked_positive] = -np.inf

        fallback_indices = np.argsort(popularity)[::-1][
            :top_n - len(ranked_positive)
        ]
        recommended = np.concatenate([
            ranked_positive[:top_n],
            fallback_indices
        ])
    else:
        recommended = ranked_positive[:top_n]

    return recommended, scores[recommended]

for customer_index in [0, 1, 2]:
    recommended, scores = recommend_products(customer_index)
    print(f"\nExample Customer {customer_index + 1}")
    for rank, (product_index, score) in enumerate(zip(recommended, scores),start=1):
        print(rank,products[product_index],round(float(score), 4))

all_recommendations = []
for customer_index in range(X.shape[0]):
    recommended, scores = recommend_products(customer_index)
    for rank, (product_index, score) in enumerate(zip(recommended, scores),start=1):
        all_recommendations.append({"Customer Index": customer_index,"Rank": rank,"Product": products[product_index],"Cosine Score": float(score)})
recommendations_df = pd.DataFrame(all_recommendations)
recommendations_df.to_csv(output_dir / f"cosine_recommendations_{k}.csv",index=False)
elapsed = time.perf_counter() - start_time
print("\nTotal customers:", X.shape[0])
print("Total recommendations:", len(recommendations_df))
print(f"Recommendation job time: {elapsed:.2f} seconds")
print("\nSaved cosine recommendations!")
print("ML3 Cosine Similarity completed!")

print("\n--- Recommendation Validation ---")
invalid_recommendations=0
non_positive_recommendations=0
for customer_index in range(X.shape[0]):
    recommended,scores=recommend_products(customer_index)
    purchased=set(X[customer_index].indices)
    for product_index,score in zip(recommended,scores):
        if product_index in purchased:
            invalid_recommendations += 1
        if not np.isfinite(score) or score <= 0:
            non_positive_recommendations += 1
print("Previously purchased products recommended:",
      invalid_recommendations)
print("Non-positive or invalid recommendation scores:",
      non_positive_recommendations)
assert invalid_recommendations==0
print("\nML3 recommendation validation completed!")