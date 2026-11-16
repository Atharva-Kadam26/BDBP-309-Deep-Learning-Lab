# Import required libraries
import os
import numpy as np
import matplotlib.pyplot as plt

from Bio import SeqIO
from gensim.models import Word2Vec

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


# Get the FASTA file path
base_dir = os.path.dirname(os.path.abspath(__file__))
fasta_file = os.path.join(base_dir, "proteins.fasta")


# Read protein sequences from FASTA file
protein_sequences = {}

for record in SeqIO.parse(fasta_file, "fasta"):

    sequence = str(record.seq).upper()

    # Remove invalid amino acid symbols
    sequence = "".join(
        amino_acid for amino_acid in sequence
        if amino_acid in "ACDEFGHIKLMNPQRSTVWY"
    )

    if len(sequence) >= 3:
        protein_sequences[record.id] = sequence


print("Total protein sequences:", len(protein_sequences))


# Convert protein sequences into overlapping 3-mers
protein_kmers = {}

for protein_id, sequence in protein_sequences.items():

    kmers = []

    for i in range(len(sequence) - 2):
        kmer = sequence[i:i + 3]
        kmers.append(kmer)

    protein_kmers[protein_id] = kmers


# Prepare data for Word2Vec
training_data = list(protein_kmers.values())


# Train Word2Vec model
biological_model = Word2Vec(
    sentences=training_data,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    epochs=100
)


print("\nWord2Vec model trained successfully!")


# Generate one embedding for each protein
protein_embeddings = {}

for protein_id, kmers in protein_kmers.items():

    vectors = []

    for kmer in kmers:
        vectors.append(biological_model.wv[kmer])

    # Average all k-mer embeddings
    protein_embedding = np.mean(vectors, axis=0)

    protein_embeddings[protein_id] = protein_embedding


# Convert embeddings into a matrix
protein_names = list(protein_embeddings.keys())

protein_vectors = np.array(
    list(protein_embeddings.values())
)


print("\nProtein embedding matrix shape:", protein_vectors.shape)


# Calculate cosine similarity
if len(protein_vectors) >= 2:

    similarity_matrix = cosine_similarity(protein_vectors)

    print("\nCosine Similarity Matrix:")
    print(similarity_matrix)

    print(
        "\nSimilarity between",
        protein_names[0],
        "and",
        protein_names[1],
        ":",
        similarity_matrix[0][1]
    )


# Visualize protein embeddings using PCA
if len(protein_vectors) >= 2:

    pca = PCA(n_components=2)

    reduced_embeddings = pca.fit_transform(protein_vectors)

    plt.figure(figsize=(10, 7))

    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1]
    )

    for i, name in enumerate(protein_names):
        plt.annotate(
            name,
            (reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        )

    plt.title("Protein Sequence Embeddings using Word2Vec and PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()
    plt.show()