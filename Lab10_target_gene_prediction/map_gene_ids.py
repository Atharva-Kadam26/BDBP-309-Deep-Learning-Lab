
import h5py
import requests
import csv

# 1. Load Entrez Gene IDs from GCTX
file_path = "GSE92743_Broad_Affymetrix_training_Level3_Q2NORM_n100000x12320.gctx"

with h5py.File(file_path, "r") as f:
    gene_ids = f["0"]["META"]["ROW"]["id"][:]

print("Total gene IDs:", len(gene_ids))

# 2. Connect to MyGene.info
url = "https://mygene.info/v3/query"

gene_mapping = {}

# 3. Process IDs in batches of 500
for i in range(0, len(gene_ids), 500):

    batch = gene_ids[i:i + 500]

    parameters = {
        "q": ",".join(str(x) for x in batch),
        "scopes": "entrezgene",
        "fields": "entrezgene,symbol",
        "species": "human"
    }

    response = requests.post(url, data=parameters, timeout=60)
    response.raise_for_status()

    results = response.json()

    for result in results:
        if "symbol" in result and "entrezgene" in result:
            gene_mapping[str(result["entrezgene"])] = result["symbol"]

    print("Processed:", min(i + 500, len(gene_ids)), "/", len(gene_ids))

# 4. Save gene annotations
with open("gene_annotations.tsv", "w", newline="") as file:

    writer = csv.writer(file, delimiter="\t")

    writer.writerow(["Entrez_ID", "Gene_Symbol"])

    for gene_id in gene_ids:
        gene_id = str(gene_id)
        writer.writerow([gene_id, gene_mapping.get(gene_id, "Not_found")])

print("\nGene mapping completed!")
print("Mapped genes:", len(gene_mapping))
print("Unmapped genes:", len(gene_ids) - len(gene_mapping))

# 5. Load D-GEX landmark genes
landmark_genes = set()

with open("D-GEX-master/map_lm.txt") as file:
    for line in file:
        landmark_genes.add(line.strip().split("\t")[0])

# 6. Load D-GEX target genes
target_genes = set()

with open("D-GEX-master/map_tg.txt") as file:
    for line in file:
        target_genes.add(line.strip().split("\t")[0])

# 7. Find overlapping genes
dataset_genes = set(gene_mapping.values())

landmark_overlap = dataset_genes.intersection(landmark_genes)
target_overlap = dataset_genes.intersection(target_genes)

print("\nMapping comparison:")
print("Landmark genes available:", len(landmark_overlap))
print("Target genes available:", len(target_overlap))

# 8. Save overlap results
with open("gene_overlap_summary.txt", "w") as file:
    file.write("Landmark genes available: " + str(len(landmark_overlap)) + "\n")
    file.write("Target genes available: " + str(len(target_overlap)) + "\n")

    file.write("\nLandmark gene symbols:\n")
    for gene in sorted(landmark_overlap):
        file.write(gene + "\n")

    file.write("\nTarget gene symbols:\n")
    for gene in sorted(target_overlap):
        file.write(gene + "\n")

print("\nResults saved successfully!")
