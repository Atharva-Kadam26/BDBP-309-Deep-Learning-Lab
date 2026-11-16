import h5py

file_path = "GSE92743_Broad_Affymetrix_training_Level3_Q2NORM_n100000x12320.gctx"

with h5py.File(file_path, "r") as f:
    gene_ids = f["0"]["META"]["ROW"]["id"][:]

print("Total gene IDs:", len(gene_ids))
print("First 20 gene IDs:")

for gene_id in gene_ids[:20]:
    print(gene_id)
