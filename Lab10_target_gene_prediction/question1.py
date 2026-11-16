
"""
Lab 10 - Exercise 1

Question:
Develop a model to predict target genes from a set of landmark genes.

The model takes the expression values of landmark genes as input
and predicts the expression values of target genes.

This is a multi-output regression problem.
"""

import torch
import torch.nn as nn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(901, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 8518)
        )

    def forward(self, x):

        return self.network(x)



# Load cleaned landmark and target gene matrices

X = np.load("X_landmark.npy")
y = np.load("Y_target.npy")

print("Input shape:", X.shape)
print("Target shape:", y.shape)

print("Total samples:", X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Input shape:", X_train.shape)
print("Training Target shape:", y_train.shape)

print("Testing Input shape:", X_test.shape)
print("Testing Target shape:", y_test.shape)




scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

print("\nAfter standardization:")

print("X_train mean:", np.mean(X_train))
print("X_train standard deviation:", np.std(X_train))

print("y_train mean:", np.mean(y_train))
print("y_train standard deviation:", np.std(y_train))



X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)




train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print("\nDataLoaders created successfully!")



device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)



model = NeuralNetwork().to(device)

print("\nModel initialized successfully!")

print(model)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 20

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward propagation

        predictions = model(X_batch)

        # Calculate loss

        loss = loss_fn(predictions, y_batch)

        # Backpropagation

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Training Loss: {average_loss:.4f}"
    )


model.eval()

total_test_loss = 0

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Generate predictions

        predictions = model(X_batch)

        # Calculate testing loss

        loss = loss_fn(predictions, y_batch)

        total_test_loss += loss.item()


average_test_loss = total_test_loss / len(test_loader)

print("\nOverall Test MSE:", average_test_loss)

from sklearn.metrics import r2_score, mean_absolute_error

# Store predictions and actual values

all_predictions = []
all_targets = []

model.eval()

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)

        predictions = model(X_batch)

        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(y_batch.numpy())

# Combine all batches

all_predictions = np.concatenate(all_predictions, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# evaluation metrics

r2 = r2_score(all_targets, all_predictions)
mae = mean_absolute_error(all_targets, all_predictions)

print("\nAdditional Evaluation Metrics:")
print("R2 Score:", r2)
print("Mean Absolute Error:", mae)


# R2 score for each target gene
gene_r2_scores = r2_score(
    all_targets,
    all_predictions,
    multioutput="raw_values"
)

#  average gene-wise R2 score
average_gene_r2 = np.mean(gene_r2_scores)

#genes with positive R2 scores
positive_genes = np.sum(gene_r2_scores > 0)

# with negative R2 scores
negative_genes = np.sum(gene_r2_scores < 0)

print("\nGene-wise Evaluation:")
print("Average gene-wise R2:", average_gene_r2)
print("Genes with positive R2:", positive_genes)
print("Genes with negative R2:", negative_genes)
print("Total target genes:", len(gene_r2_scores))

print("\nGene-wise R2 Distribution:")
print("Minimum R2:", np.min(gene_r2_scores))
print("Maximum R2:", np.max(gene_r2_scores))
print("Median R2:", np.median(gene_r2_scores))
print("25th percentile:", np.percentile(gene_r2_scores, 25))
print("75th percentile:", np.percentile(gene_r2_scores, 75))

