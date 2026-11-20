
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


# 1. Load the dataset

file = open("../data/q1_data/splice.data", "r")

sequences = []
labels = []

for line in file:

    parts = line.strip().split(",", 2)

    label = parts[0].strip()
    sequence = parts[2].strip().upper()

    sequences.append(sequence)
    labels.append(label)

file.close()

print("Total sequences:", len(sequences))


# 2. Preprocess the sequences

# Handle standard and ambiguous DNA bases

base_map = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],

    "D": [1/3, 0, 1/3, 1/3],
    "N": [1/4, 1/4, 1/4, 1/4],
    "S": [0, 1/2, 1/2, 0],
    "R": [1/2, 0, 1/2, 0]
}

X = []

for sequence in sequences:

    encoded_sequence = []

    for base in sequence:

        encoded_sequence.extend(base_map[base])

    X.append(encoded_sequence)

X = np.array(X, dtype=np.float32)

# Encode class labels

label_map = {
    "EI": 0,
    "IE": 1,
    "N": 2
}

y = []

for label in labels:
    y.append(label_map[label])

y = np.array(y)

print("Input shape:", X.shape)
print("Label shape:", y.shape)


# 3. Train-test split (80:20)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Convert to PyTorch tensors

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# 4. Create DataLoader

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


# 5. Build the Feed Forward Neural Network

class FFN(nn.Module):

    def __init__(self):
        super(FFN, self).__init__()

        self.fc1 = nn.Linear(240, 128)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return x


# 6. Initialize model, loss and optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FFN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Using device:", device)


# 7. Train the model

epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    total = 0

    for inputs, targets in train_loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward propagation

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        # Backpropagation

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        total += targets.size(0)

    epoch_loss = running_loss / total

    print(
        "Epoch [{}/{}], Loss: {:.4f}".format(
            epoch + 1,
            epochs,
            epoch_loss
        )
    )


# 8. Test the trained model

model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():

    X_test = X_test.to(device)

    outputs = model(X_test)

    _, predicted = torch.max(outputs, 1)

    all_predictions = predicted.cpu().numpy()
    all_targets = y_test.numpy()


# 9. Calculate Accuracy and Macro F1-score

accuracy = accuracy_score(all_targets, all_predictions)

macro_f1 = f1_score(
    all_targets,
    all_predictions,
    average="macro"
)

print("\nTest Results")

print("Test Accuracy: {:.4f}".format(accuracy))

print("Macro F1-score: {:.4f}".format(macro_f1))


# Classification report

print("\nClassification Report:")

print(
    classification_report(
        all_targets,
        all_predictions,
        labels=[0, 1, 2],
        target_names=["EI", "IE", "N"],
        zero_division=0
    )
)
