import pandas as pd
from pathlib import Path

#DE1: Load and inspect the dataset

file_path = Path("data/raw/online_retail_II.xlsx")
print("Loading Online Retail II dataset...")
sheets  = pd.read_excel(file_path, sheet_name=None)
for sheet_name, data in sheets.items():
    print(f"\nSheet: {sheet_name}")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
df = pd.concat(sheets.values(), ignore_index=True)
print("\n--- Combined Dataset ---")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("\nColumn Names:")
print(df.columns.tolist())
print("\nFirst Five Rows:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nData Types:")
print(df.dtypes)

#DE2: Data Cleaning

print("\n--- Data Cleaning ---")
clean_df = df.copy()
cleaning_log = []
def log_removal(step,before,after):
    removed = before - after
    cleaning_log.append({"Cleaning Step": step, "Rows Removed": removed, "Rows Remaining": after})
    print(f"{step}:{removed}rows removed")
before=len(clean_df)
clean_df=clean_df[~clean_df["Invoice"].astype(str).str.upper().str.startswith("C")].copy()
log_removal("Cancelled invoices", before, len(clean_df))
before=len(clean_df)
clean_df=clean_df[clean_df["Quantity"]>0].copy()
log_removal("Non-positive quantities",before,len(clean_df))
before=len(clean_df)
clean_df=clean_df[clean_df["Price"]>0].copy()
log_removal("Non-positive prices",before,len(clean_df))
before=len(clean_df)
non_product_codes=["POST","POSTAGE", "M", "D", "BANK CHARGES","DOT","CRUK","C2","ADJUST","ADJSUT2"]
clean_df=clean_df[~clean_df["StockCode"].astype(str).str.upper().isin(non_product_codes)].copy()
log_removal("Non-product codes",before,len(clean_df))
before=len(clean_df)
clean_df=clean_df.dropna(subset=["Customer ID"]).copy()
log_removal("Missing Customer IDs",before,len(clean_df))
before=len(clean_df)
clean_df=clean_df.dropna(subset=["Description"]).copy()
log_removal("Missing Descriptions",before,len(clean_df))
clean_df["Customer ID"]=clean_df["Customer ID"].astype(int)
clean_df["StockCode"]=clean_df["StockCode"].astype(str).str.strip()
print("\n--- Cleaned Dataset ---")
print("Original rows:",len(df))
print("Cleaned rows:",len(clean_df))
print("Total rows removed:",len(df)-len(clean_df))
print("\nMissing values after cleaning:")
print(clean_df.isnull().sum())

from pathlib import Path
output_dir=Path("outputs")
output_dir.mkdir(exist_ok=True)
log_df=pd.DataFrame(cleaning_log)
log_df.to_csv(output_dir/"cleaning_log.csv",index=False)
print("\nCleaning log saved successfully!")
print("DE2 Data Cleaning completed!")

#DE3: Train/Test Split

print("\n--- Train/Test Split ---")
first_date=clean_df["InvoiceDate"].min()
last_date=clean_df["InvoiceDate"].max()
print("First transaction:",first_date)
print("Last transaction:",last_date)
cutoff_date=last_date-pd.DateOffset(months=2)
print("train/test cutoff:", cutoff_date)
train_df=clean_df[clean_df["InvoiceDate"]<cutoff_date].copy()
test_df=clean_df[clean_df["InvoiceDate"]>=cutoff_date].copy()
print("\nTraining rows:",len(train_df))
print("Testing rows:",len(test_df))
print("\nTraining date range:")
print(train_df["InvoiceDate"].min(), "to", train_df["InvoiceDate"].max())
print("\nTesting date range:")
print(test_df["InvoiceDate"].min(), "to", test_df["InvoiceDate"].max())
train_customers=set(train_df["Customer ID"])
train_products=set(train_df["StockCode"])
eligible_test =test_df[test_df["Customer ID"].isin(train_customers)].copy()
eligible_test = eligible_test[eligible_test["StockCode"].isin(train_products)].copy()
previous_purchases=set(zip(train_df["Customer ID"],train_df["StockCode"]))
eligible_test=eligible_test[~pd.Series(list(zip(eligible_test["Customer ID"],eligible_test["StockCode"])),index=eligible_test.index).isin(previous_purchases)].copy()
print("\n--- Evaluation Eligibility ---")
print("Eligible test transaction rows:", len(eligible_test))
print("Eligible test customers:",eligible_test["Customer ID"].nunique())
print("Eligible test products:",eligible_test["StockCode"].nunique())
split_summary=pd.DataFrame({"Metric":["First transaction", "Last transaction", "Train/Test cutoff", "Training rows", "Testing rows", "Eligible test rows", "Eligible test customers", "Eligible test products"], "Value": [str(first_date), str(last_date), str(cutoff_date), len(train_df), len(test_df), len(eligible_test), eligible_test["Customer ID"].nunique(), eligible_test["StockCode"].nunique()]})
split_summary.to_csv(output_dir/"split_summary.csv",index=False)
print("\nsplit summary saved!")
print("DE3 Train/Test Split completed!")

#save eligible test data for ML4

eligible_test[["Customer ID","StockCode"]].drop_duplicates().to_csv(output_dir/"eligible_test_purchases.csv",index=False)
print("Eligible test purchases saved for ML4!")

#DE4: Sparse CustomerXProduct Matrix

print("\n--- Sparse CustomerXProduct Matrix ---")
import numpy as np
from scipy.sparse import csr_matrix
matrix_df=train_df[["Customer ID", "StockCode"]].copy()
matrix_df["StockCode"]=matrix_df["StockCode"].astype(str)
matrix_df=matrix_df.drop_duplicates()
customers=np.sort(matrix_df["Customer ID"].unique())
products=np.sort(matrix_df["StockCode"].unique())
customer_to_index={customer:i for i, customer in enumerate(customers)}
product_to_index={product:i for i, product in enumerate(products)}
row_indices=matrix_df["Customer ID"].map(customer_to_index).to_numpy()
col_indices=(matrix_df["StockCode"].astype(str).map(product_to_index).to_numpy())
values=np.ones(len(matrix_df),dtype=np.float32)
customer_product_matrix=csr_matrix((values,(row_indices,col_indices)),shape=(len(customers),len(products)),dtype=np.float32)
print("Number of customers:",len(customers))
print("Number of products:",len(products))
print("Matrix shape:",customer_product_matrix.shape)
print("Non-zero entries:",customer_product_matrix.nnz)
print("\nDE4 Sparse CustomerXProduct Matrix completed!")

#DE5: Matrix Sparsity And Memory

print("\n--- Matrix Sparsity and Memory ---")
num_customers,num_products=customer_product_matrix.shape
total_entries=num_customers*num_products
non_zero_entries=customer_product_matrix.nnz
density=(non_zero_entries/total_entries)*100
sparsity=100-density
dense_memory_bytes=total_entries*np.dtype(np.float32).itemsize
sparse_memory_bytes=(customer_product_matrix.data.nbytes+customer_product_matrix.indices.nbytes+customer_product_matrix.indptr.nbytes)
dense_memory_mb=dense_memory_bytes/(1024**2)
sparse_memory_mb=sparse_memory_bytes/(1024**2)
memory_savings=((dense_memory_mb-sparse_memory_mb)/dense_memory_mb)*100
print("Matrix shape:",customer_product_matrix.shape)
print("Total possible entries:",total_entries)
print("Non-zero entries:",non_zero_entries)
print(f"Density: {density:.2f}%")
print(f"Sparsity: {sparsity:.2f}%")
print(f"Dense memory: {dense_memory_mb:.2f} MiB")
print(f"Sparse memory: {sparse_memory_mb:.2f} MiB")
print(f"Memory saved: {memory_savings:.2f}%")
matrix_stats = pd.DataFrame(
    {
        "Metric": [
            "Customers",
            "Products",
            "Total possible entries",
            "Non-zero entries",
            "Density (%)",
            "Sparsity (%)",
            "Dense memory (MiB)",
            "Sparse memory (MiB)",
            "Memory saved (%)",
        ],
        "Value": [
            num_customers,
            num_products,
            total_entries,
            non_zero_entries,
            round(density, 2),
            round(sparsity, 2),
            round(dense_memory_mb, 2),
            round(sparse_memory_mb, 2),
            round(memory_savings, 2),
        ]
    }
)
matrix_stats.to_csv(output_dir/"matrix_stats.csv",index=False)
print("\nMatrix statistics saved!")
print("DE5 Matrix Sparsity and Memory completed!")

from scipy.sparse import save_npz
save_npz(output_dir/"customer_product_matrix.npz", customer_product_matrix)
np.save(output_dir/"customers.npy", customers)
np.save(output_dir/"products.npy", products)
print("\nTraining matrix and mappings saved!")