import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


def main():

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_loader = DataLoader(
        training_data,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=64,
        shuffle=False
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 10

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            pred = model(X)

            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Training Loss: {average_loss:.3f}"
        )

    # Testing
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(device)
            y = y.to(device)

            pred = model(X)

            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

    accuracy = 100 * correct / total

    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()