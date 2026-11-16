# Import required libraries
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models, transforms

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Get the image folder path
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "Images")


# Load pretrained ResNet18 model
weights = models.ResNet18_Weights.DEFAULT
image_model = models.resnet18(weights=weights)

# Remove the final classification layer
image_model.fc = nn.Identity()

image_model = image_model.to(device)
image_model.eval()


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Extract embeddings from images
image_embeddings = {}
image_names = []

for filename in os.listdir(image_dir):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        image_path = os.path.join(image_dir, filename)

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Extract embedding
        with torch.no_grad():
            embedding = image_model(image_tensor)

        # Convert embedding to NumPy array
        embedding = embedding.cpu().numpy().flatten()

        image_embeddings[filename] = embedding
        image_names.append(filename)

        print(filename, "-> Embedding shape:", embedding.shape)


# Convert embeddings into a matrix
image_vectors = np.array(list(image_embeddings.values()))

print("\nEmbedding matrix shape:", image_vectors.shape)


# Calculate cosine similarity
if len(image_vectors) >= 2:

    similarity_matrix = cosine_similarity(image_vectors)

    print("\nCosine Similarity Matrix:")
    print(similarity_matrix)

    # Display similarity between the first two images
    print(
        "\nSimilarity between",
        image_names[0],
        "and",
        image_names[1],
        ":",
        similarity_matrix[0][1]
    )


# Visualize embeddings using PCA
if len(image_vectors) >= 2:

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(image_vectors)

    plt.figure(figsize=(10, 7))

    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1]
    )

    for i, name in enumerate(image_names):
        plt.annotate(
            name,
            (reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        )

    plt.title("Image Embeddings using ResNet18 and PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()
    plt.show()