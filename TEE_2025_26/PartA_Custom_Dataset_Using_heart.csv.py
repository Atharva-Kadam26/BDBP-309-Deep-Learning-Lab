
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

# Load dataset
data = pd.read_csv("heart.csv", index_col=0)

# Separate features and target
X = data.drop("AHD", axis=1)
Y = data["AHD"]

# Convert categorical input features into numerical values
X = pd.get_dummies(X, dtype=float)

# Convert target labels into numerical values
Y = Y.map({"No": 0, "Yes": 1})

# Convert to NumPy arrays
X = X.values
Y = Y.values

# Custom Dataset class
class HeartDataset(Dataset):

    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


# Create dataset
dataset = HeartDataset(X, Y)

# Split dataset into 70% training and 30% testing
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size]
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True
)

test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False
)

# Get one batch
train_features, train_labels = next(iter(train_loader))
test_features, test_labels = next(iter(test_loader))

# Print sizes and datatypes
print("Training features size:", train_features.size())
print("Training features datatype:", train_features.dtype)

print("Training labels size:", train_labels.size())
print("Training labels datatype:", train_labels.dtype)

print("Testing features size:", test_features.size())
print("Testing features datatype:", test_features.dtype)

print("Testing labels size:", test_labels.size())
print("Testing labels datatype:", test_labels.dtype)
