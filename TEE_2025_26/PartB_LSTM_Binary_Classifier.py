
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# --------------------------------
# 1. Generate DNA Sequence Dataset
# --------------------------------

def generate_dataset():

    sequences = [
        "ACGTAGCTAGCT",
        "GCTAGCTAGGCA",
        "CGTACGTAGCTA",
        "TGCATGCATGCA"
    ]

    labels = [0, 1, 0, 1]

    X = sequences * 100
    Y = labels * 100

    return X, Y


X, Y = generate_dataset()

print("Total sequences:", len(X))
print("Total labels:", len(Y))


# --------------------------------
# 2. Encode DNA Sequences
# --------------------------------

mapping = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3
}

encoded_sequences = []

for seq in X:

    encoded = []

    for nucleotide in seq:
        encoded.append(mapping[nucleotide])

    encoded_sequences.append(encoded)

X = torch.tensor(encoded_sequences, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.float32)

print("Encoded data shape:", X.shape)
print("Labels shape:", Y.shape)


# --------------------------------
# 3. Split Dataset into Train and Test
# --------------------------------

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

print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))


# --------------------------------
# 4. Define LSTM Model
# --------------------------------

class LSTMClassifier(nn.Module):

    def __init__(self):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(4, 16)

        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=32,
            batch_first=True
        )

        self.fc = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.embedding(x)

        output, (hidden, cell) = self.lstm(x)

        x = hidden[-1]

        x = self.fc(x)

        x = self.sigmoid(x)

        return x


# --------------------------------
# 5. Train LSTM Model
# --------------------------------

model = LSTMClassifier()

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)

epochs = 50

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for sequences, labels in train_loader:

        outputs = model(sequences)

        labels = labels.unsqueeze(1)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:

        print("Epoch [{}/50], Loss: {:.4f}".format(
            epoch + 1,
            running_loss / len(train_loader)
        ))


# --------------------------------
# 6. Test LSTM Model
# --------------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for sequences, labels in test_loader:

        outputs = model(sequences)

        predicted = (outputs >= 0.5).float()

        labels = labels.unsqueeze(1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print("\nTest Accuracy: {:.2f}%".format(accuracy))
