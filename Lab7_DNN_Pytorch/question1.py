"""
Lab 7 - Deep Network Model Development using PyTorch

Exercise 1:
Build an end-to-end working deep learning model using PyTorch library.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

def load_data():

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    training_data = datasets.MNIST(
        root="../Lab6_Getting_started_with_Pytorch/data",
        train=True,
        download=False,
        transform=transform
    )

    test_data = datasets.MNIST(
        root="../Lab6_Getting_started_with_Pytorch/data",
        train=False,
        download=False,
        transform=transform
    )

    return training_data, test_data


# --------------------------------------------------
# 2. BUILD MODEL
# --------------------------------------------------

class NeuralNetwork(nn.Module):

    def __init__(self):

        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(
            in_features=28 * 28,
            out_features=512
        )

        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(
            in_features=512,
            out_features=512
        )

        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(
            in_features=512,
            out_features=10
        )


    def forward(self, x):

        x = self.flatten(x)

        x = self.linear1(x)

        x = self.relu1(x)

        x = self.linear2(x)

        x = self.relu2(x)

        output = self.linear3(x)

        return output


# --------------------------------------------------
# 3. TRAIN MODEL
# --------------------------------------------------

def train(
    mydataloader,
    model,
    loss_fn,
    optimizer,
    device,
    epochs
):

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for X, y in mydataloader:

            X, y = X.to(device), y.to(device)

            # Prediction
            prediction = model(X)

            # Calculate loss
            loss = loss_fn(
                prediction,
                y
            )

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(mydataloader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Training Loss: {average_loss:.4f}"
        )


# --------------------------------------------------
# 4. TEST MODEL
# --------------------------------------------------

def test(
    mydataloader,
    model,
    loss_fn,
    device
):

    size = len(mydataloader.dataset)

    num_batches = len(mydataloader)

    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():

        for X, y in mydataloader:

            X, y = X.to(device), y.to(device)

            prediction = model(X)

            test_loss += loss_fn(
                prediction,
                y
            ).item()

            correct += (
                (prediction.argmax(dim=1) == y)
                .type(torch.float)
                .sum()
                .item()
            )

    test_loss = test_loss / num_batches

    accuracy = correct / size

    print("\nTest Results:")

    print(
        f"Accuracy: {100 * accuracy:.1f}%"
    )

    print(
        f"Average Loss: {test_loss:.4f}"
    )


# --------------------------------------------------
# 5. MAIN
# --------------------------------------------------

def main():

    # Load dataset
    training_data, test_data = load_data()

    # Batch size
    batch_size = 64

    # Training DataLoader
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True
    )

    # Test DataLoader
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    # Device
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else torch.device("cpu")
    )

    print(f"Using {device} device")

    # Create model
    model = NeuralNetwork().to(device)

    print("\nModel:")
    print(model)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01
    )

    # Train model
    train(
        train_dataloader,
        model,
        loss_fn,
        optimizer,
        device,
        epochs=5
    )

    # Test model
    test(
        test_dataloader,
        model,
        loss_fn,
        device
    )

    print("\nEND")


if __name__ == "__main__":
    main()
