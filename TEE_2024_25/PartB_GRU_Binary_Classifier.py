# B3_ Build an GRIJ binary classifier from the DNA sequence data.
# a. Write a python function to generate a dataset of 500 DNA sequences The six
# sequences
# have corresponding labels (y) as 1, 1, O, 1, O, 1, Repeat these four sequences 100 times to generate a sample of 600
# sequences and labels.
# b .Train a GRU classifier using PyTorch to predict labels for a given DNA sequence. Use
# the above generated DNA sequence dataset. Divide the dataset into train and 20%
# test sets. Train the model for 50 epochs. Report the training loss for every IO epochs.
# Test the trained model using the 20% data and report the accuracy.


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# Part A: Generate DNA dataset

def generate_dataset():
    sequences = [
        "CAGTAGCTAGCT",
        "CGTAGCTGAGCA",
        "CGTACGTAGCTA",
        "TGCATGCATGCA",
        "GCTACGTAGGCA",
        "GCTACGTAGCTA"
    ]

    labels = [1, 1, 0, 1, 0, 1]

    DNA = []
    Y = []

    for i in range(6):
        for j in range(100):
            DNA.append(sequences[i])
            Y.append(labels[i])

    return DNA, Y

DNA, Y = generate_dataset()

print("Total sequences:", len(DNA))
print("Total labels:", len(Y))

# Part B: One-hot encoding

mapping = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1]
}

X = []

for sequence in DNA:
    encoded_sequence = []

    for base in sequence:
        encoded_sequence.append(mapping[base])

    X.append(encoded_sequence)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

print("Input shape:", X.shape)
print("Label shape:", Y.shape)

# Part C: Split dataset into 80% training and 20% testing

dataset = TensorDataset(X, Y)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size]
)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True
)

test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False
)

# Part D: Build GRU classifier

class GRUClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=4,
            hidden_size=16,
            batch_first=True
        )

        self.fc = nn.Linear(16, 1)

    def forward(self, x):

        output, hidden = self.gru(x)

        last_hidden = hidden[-1]

        prediction = self.fc(last_hidden)

        return prediction.squeeze(1)

# Part E: Initialize model, loss and optimizer

model = GRUClassifier()

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)

# Part F: Train the model for 50 epochs

epochs = 50

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for X_batch, Y_batch in train_loader:

        prediction = model(X_batch)

        loss = loss_fn(prediction, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Training Loss: {average_loss:.4f}"
        )

# Part G: Test the model and calculate accuracy

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for X_batch, Y_batch in test_loader:

        prediction = model(X_batch)

        predicted_labels = (torch.sigmoid(prediction) >= 0.5).float()

        correct += (predicted_labels == Y_batch).sum().item()

        total += Y_batch.size(0)

accuracy = (correct / total) * 100

print("Test Accuracy:", accuracy, "%")

