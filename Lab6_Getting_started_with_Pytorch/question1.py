"""
Lab 6 - Getting started with PyTorch

Exercise 2:
Learn about Tensors, Datasets and Dataloaders, Transforms,
BuildModel, Autograd, Optimization and Save & Load model.
"""

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


# --------------------------------------------------
# 1. TENSORS
# --------------------------------------------------

X = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0]
])

y = torch.tensor([
    [3.0],
    [7.0],
    [11.0],
    [15.0]
])

print("X:")
print(X)

print("\ny:")
print(y)

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)


# --------------------------------------------------
# 2. DATASET
# --------------------------------------------------

dataset = TensorDataset(X, y)

print("\nNumber of samples:", len(dataset))

print("First sample:")
print(dataset[0])


# --------------------------------------------------
# 3. DATALOADER
# --------------------------------------------------

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

print("\nBatches:")

for batch_X, batch_y in dataloader:

    print("X batch:")
    print(batch_X)

    print("y batch:")
    print(batch_y)


# --------------------------------------------------
# 4. BUILD MODEL
# --------------------------------------------------

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.linear1 = nn.Linear(
            in_features=2,
            out_features=4
        )

        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(
            in_features=4,
            out_features=1
        )

    def forward(self, x):

        x = self.linear1(x)

        x = self.relu(x)

        output = self.linear2(x)

        return output


model = NeuralNetwork()

print("\nModel:")
print(model)


# --------------------------------------------------
# 5. LOSS FUNCTION
# --------------------------------------------------

loss_fn = nn.MSELoss()


# --------------------------------------------------
# 6. OPTIMIZATION
# --------------------------------------------------

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


# --------------------------------------------------
# 7. TRAINING + AUTOGRAD
# --------------------------------------------------

epochs = 1000

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch_X, batch_y in dataloader:

        prediction = model(batch_X)

        loss = loss_fn(
            prediction,
            batch_y
        )

        optimizer.zero_grad()

        # Autograd calculates gradients
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    if (epoch + 1) % 200 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Average Loss: {average_loss:.6f}"
        )


# --------------------------------------------------
# 8. TEST MODEL
# --------------------------------------------------

model.eval()

with torch.no_grad():

    prediction = model(X)

print("\nPredictions:")

for i in range(len(X)):

    print(
        f"Input: {X[i].numpy()} "
        f"Actual: {y[i].item():.2f} "
        f"Predicted: {prediction[i].item():.2f}"
    )


# --------------------------------------------------
# 9. SAVE MODEL
# --------------------------------------------------

torch.save(
    model.state_dict(),
    "model.pth"
)

print("\nModel saved as model.pth")


# --------------------------------------------------
# 10. LOAD MODEL
# --------------------------------------------------

loaded_model = NeuralNetwork()

loaded_model.load_state_dict(
    torch.load(
        "model.pth",
        weights_only=True
    )
)

loaded_model.eval()

with torch.no_grad():

    loaded_prediction = loaded_model(X)

print("\nPredictions from loaded model:")

print(loaded_prediction)