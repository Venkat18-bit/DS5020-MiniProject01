#ML4: Hit Rate@10 Evaluation
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
print("\n--- ML4: Hit Rate@10 Evaluation ---")
output_dir=Path("outputs")
x=load_npz(output_dir/"customer_product_matrix.npz")
customers=np.load(output_dir/"customers.npy")
products=np.load(output_dir/"products.npy",allow_pickle=True)
test_df=pd.read_csv(output_dir/"eligible_test_purchases.csv",dtype={"StockCode":str})
customer_to_index={customer: i for i, customer in enumerate(customers)}
product_to_index={str(product): i for i, product in enumerate(products)}
test_products={}
for customer,group in test_df.groupby("Customer ID"):
    if customer not in customer_to_index:
        continue
    relevant={product_to_index[code] for code in group["StockCode"] if code in product_to_index}
    if relevant:
        test_products[customer_to_index[customer]]=relevant
eligible_customers=sorted(test_products.keys())
print("Eligible evaluation customers:",len(eligible_customers))

#BESTSELLER

popularity=np.array(x.sum(axis=0)).ravel()
def bestseller_recommendations(customer_index, top_n=10):
    scores=popularity.astype(float).copy()
    scores[x[customer_index].indices]=-np.inf
    return np.argsort(scores)[::-1][:top_n]

#SVD RECOMMENDATIONS

def cosine_recommendations(customer_index, similarity, top_similar, top_n=10):
    purchased = x[customer_index].indices
    scores = np.zeros(x.shape[1], dtype=np.float32)
    for product_index in purchased:
        neighbors = top_similar[product_index]
        for neighbor in neighbors:
            scores[neighbor] += similarity[product_index, neighbor]
    scores[purchased] = -np.inf
    positive_indices = np.where(np.isfinite(scores) & (scores > 0))[0]
    ranked_positive = positive_indices[np.argsort(scores[positive_indices])[::-1]]
    if len(ranked_positive) < top_n:
        fallback_scores = popularity.astype(float).copy()
        fallback_scores[purchased] = -np.inf
        fallback_scores[ranked_positive] = -np.inf
        fallback = np.argsort(fallback_scores)[::-1][:top_n - len(ranked_positive)]
        return np.concatenate([ranked_positive[:top_n],fallback])
    return ranked_positive[:top_n]

#HIT RATE@10

def calculate_hit_rate(recommend_function):
    hits=0
    for customer_index in eligible_customers:
        recommendations=recommend_function(customer_index)
        actual_products=test_products[customer_index]
        if any(product in actual_products for product in recommendations):
            hits+=1
    return hits/len(eligible_customers)*100,hits
results=[]

baseline_rate, baseline_hits=calculate_hit_rate(bestseller_recommendations)
results.append({"Model":"Bestseller Baseline","Components":0,"Hit Rate@10 (%)": round(baseline_rate,2),"Customers Hit":baseline_hits})
print(f"\nBestseller Hit Rate@10: {baseline_rate:.2f}%")
for k in [10,20,50,100,200]:
    components = np.load(output_dir / f"svd_components_{k}.npy")
    product_factors = components.T
    norms = np.linalg.norm(product_factors, axis=1, keepdims=True)
    normalized = product_factors / np.maximum(norms, 1e-12)

    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)

    top_similar = np.argsort(similarity, axis=1)[:, ::-1][:, :10]
    start = time.perf_counter()
    hit_rate, hits = calculate_hit_rate(lambda customer_index: cosine_recommendations(customer_index, similarity, top_similar))
    evaluation_time = time.perf_counter() - start
    print(f"SVD Cosine k={k} Hit Rate@10: {hit_rate:.2f}%")
    results.append({"Model": "TruncatedSVD Cosine","Components": k,"Hit Rate@10 (%)": round(hit_rate, 2),"Customers Hit": hits,"Evaluation Time (seconds)": round(evaluation_time, 2)})

# PCA Hit Rate@10

print("\n--- ML5: PCA Hit Rate@10 ---")
k = 50
pca_components = np.load(output_dir / f"pca_components_{k}.npy")
pca_product_factors = pca_components.T

pca_norms = np.linalg.norm(pca_product_factors,axis=1,keepdims=True)
pca_normalized= pca_product_factors/np.maximum(pca_norms,1e-12)

pca_similarity = pca_normalized @ pca_normalized.T
np.fill_diagonal(pca_similarity, -np.inf)

pca_top_similar = np.argsort(pca_similarity, axis=1)[:, ::-1][:, :10]

start = time.perf_counter()
pca_hit_rate, pca_hits = calculate_hit_rate(lambda customer_index: cosine_recommendations(customer_index,pca_similarity,pca_top_similar))
pca_evaluation_time = time.perf_counter() - start
print(f"PCA Cosine k={k} Hit Rate@10: {pca_hit_rate:.2f}%")
results.append({"Model": "PCA Cosine","Components": k,"Hit Rate@10 (%)": round(pca_hit_rate, 2),"Customers Hit": pca_hits,"Evaluation Time (seconds)": round(pca_evaluation_time, 2)})
results_df=pd.DataFrame(results)
results_df.to_csv(output_dir/"evaluation_results.csv",index=False)
print("\n--- Evaluation Summary ---")
print(results_df.to_string(index=False))
print("\nSaved: outputs/evaluation_results.csv")
print("\nML4 and ML5 Hit Rate@10 Evaluation Completed!")