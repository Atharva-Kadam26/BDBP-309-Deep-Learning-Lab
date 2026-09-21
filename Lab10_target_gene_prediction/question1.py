"""
Lab 10 - Exercise 1

Question:
Develop a model to predict target genes from a set of landmark genes.

The model takes the expression values of landmark genes as input
and predicts the expression values of target genes.

This is a multi-output regression problem.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------
# Parameters
# --------------------------------------------------

INPUT_SIZE = 943
OUTPUT_SIZE = 4760

HIDDEN_SIZE = 512
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001


# --------------------------------------------------
# Deep Learning Model
# --------------------------------------------------

class GeneExpressionModel(nn.Module):

    def __init__(self):

        super(GeneExpressionModel, self).__init__()

        self.linear1 = nn.Linear(
            INPUT_SIZE,
            HIDDEN_SIZE
        )

        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(
            HIDDEN_SIZE,
            HIDDEN_SIZE
        )

        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(
            HIDDEN_SIZE,
            OUTPUT_SIZE
        )

    def forward(self, x):

        x = self.linear1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.relu2(x)

        output = self.linear3(x)

        return output


# --------------------------------------------------
# Load Data
# --------------------------------------------------

# Replace these with your actual processed data files.

X_train = torch.load("X_train.pt")
Y_train = torch.load("Y_train.pt")

X_test = torch.load("X_test.pt")
Y_test = torch.load("Y_test.pt")


train_dataset = TensorDataset(
    X_train,
    Y_train
)

test_dataset = TensorDataset(
    X_test,
    Y_test
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else torch.device("cpu")
)

print(f"Using {device} device")


# --------------------------------------------------
# Model
# --------------------------------------------------

model = GeneExpressionModel().to(device)

print(model)


# --------------------------------------------------
# Loss and Optimizer
# --------------------------------------------------

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device)

        prediction = model(X)

        loss = loss_fn(
            prediction,
            y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss /
        len(train_loader)
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Training Loss: {average_loss:.6f}"
    )


# --------------------------------------------------
# Testing
# --------------------------------------------------

model.eval()

test_loss = 0

with torch.no_grad():

    for X, y in test_loader:

        X = X.to(device)
        y = y.to(device)

        prediction = model(X)

        loss = loss_fn(
            prediction,
            y
        )

        test_loss += loss.item()


test_loss = test_loss / len(test_loader)

print(
    f"\nTest MSE: {test_loss:.6f}"
)

print("\nEND")