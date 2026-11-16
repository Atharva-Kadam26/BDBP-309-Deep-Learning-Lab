# A1. Build an LSTM Binary Classifier from DNA Sequence Data
# (a) Write a Python function to generate a dataset of 400 DNA sequences. The four sequences:
# - "ACGTAGCTAGCT"
# - "GCTAGCTAGGCA"
# - "CGTACGTAGCTA"
# - "TGCATGCATGCA"
# have corresponding labels (y) as 0, 1, 0, 1, respectively. Repeat these four sequences 100 times to generate a sample of 400 sequences and labels. (10 Marks)
#
# (b) Use the above-generated DNA sequence dataset. Train an LSTM classifier using PyTorch to predict labels for a given DNA sequence.
# - Divide the dataset into 80% training and 20% test sets.
# - Train the model for 50 epochs.
# - Report the training loss for every 10 epochs.
# - Test the trained model using the 20% data and report the accuracy.


# Generate DNA dataset

def generate_dataset():

    sequences = [
        "ACGTAGCTAGCT",
        "GCTAGCTAGGCA",
        "CGTACGTAGCTA",
        "TGCATGCATGCA"
    ]

    labels = [0, 1, 0, 1]

    DNA = []
    Y = []

    for i in range(4):
        for j in range(100):
            DNA.append(sequences[i])
            Y.append(labels[i])

    return DNA, Y


DNA, Y = generate_dataset()

print("Total sequences:", len(DNA))
print("Total labels:", len(Y))


import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, random_split

# Generate dataset

def generate_dataset():

    sequences = [
        "ACGTAGCTAGCT",
        "GCTAGCTAGGCA",
        "CGTACGTAGCTA",
        "TGCATGCATGCA"
    ]

    labels = [0, 1, 0, 1]

    DNA = []
    Y = []

    for i in range(4):
        for j in range(100):
            DNA.append(sequences[i])
            Y.append(labels[i])

    return DNA, Y


DNA, Y = generate_dataset()

# One-hot encoding

mapping = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1]
}

X = []

for sequence in DNA:

    encoded = []

    for base in sequence:
        encoded.append(mapping[base])

    X.append(encoded)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

print("Input shape:", X.shape)
print("Label shape:", Y.shape)

# Split dataset into training and testing

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
    test_dataset, batch_size=32
)

# Build LSTM model

class LSTMClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=16,
            batch_first=True
        )

        self.fc = nn.Linear(16, 1)

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        last_hidden = hidden[-1]

        prediction = self.fc(last_hidden)

        return prediction.squeeze(1)


# Initialize model

model = LSTMClassifier()

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)

# Train model for 50 epochs

for epoch in range(50):

    model.train()

    total_loss = 0

    for X_batch, Y_batch in train_loader:

        prediction = model(X_batch)

        loss = loss_fn(prediction, Y_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch+1}/50], "
            f"Training Loss: {total_loss/len(train_loader):.4f}"
        )


# Test model

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for X_batch, Y_batch in test_loader:

        prediction = model(X_batch)

        predicted_labels = (
            torch.sigmoid(prediction) >= 0.5
        ).float()

        correct += (predicted_labels == Y_batch).sum().item()

        total += Y_batch.size(0)


accuracy = (correct / total) * 100

print("Test Accuracy:", accuracy, "%")