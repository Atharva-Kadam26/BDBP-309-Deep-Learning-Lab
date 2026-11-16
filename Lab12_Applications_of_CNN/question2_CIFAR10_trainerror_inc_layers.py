"""Implement CNN using PyTorch for image classification using cifar10 dataset -
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html . Plot train error vs
increasing number of layers. After some point, the training error increases with the
number of layers."""

import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 


class NeuralNetwork(nn.Module):
    def __init__(self,num_layers):
        super(NeuralNetwork,self).__init__()
        self.num_layers = num_layers

        # Convolutional layers

        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, padding=1
        )
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1
        )
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1
        )
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1
        )
        self.batchnorm4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1
        )
        self.batchnorm5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1
        )
        self.batchnorm6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.maxpool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Makes the output size fixed
        self.adaptivepool = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=32,
            out_features=128
        )

        self.relu7 = nn.ReLU()

        self.fc2 = nn.Linear(
            in_features=128,
            out_features=10
        )

    def forward(self, x):

        # Layer 1

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)

        # Layer 2

        if self.num_layers >= 2:
            x = self.conv2(x)
            x = self.batchnorm2(x)
            x = self.relu2(x)

        # Layer 3

        if self.num_layers >= 3:
            x = self.conv3(x)
            x = self.batchnorm3(x)
            x = self.relu3(x)

        # Layer 4

        if self.num_layers >= 4:
            x = self.conv4(x)
            x = self.batchnorm4(x)
            x = self.relu4(x)

        # Layer 5

        if self.num_layers >= 5:
            x = self.conv5(x)
            x = self.batchnorm5(x)
            x = self.relu5(x)

        # Layer 6

        if self.num_layers >= 6:
            x = self.conv6(x)
            x = self.batchnorm6(x)
            x = self.relu6(x)

        # Pooling

        x = self.maxpool(x)

        # Adaptive pooling

        x = self.adaptivepool(x)

        # Fully connected layers

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu7(x)

        output = self.fc2(x)

        return output

def load_data():

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(
            torch.float32,
            scale=True
        ),
        v2.Normalize(
            mean=(0.5,0.5,0.5),
            std=(0.5,0.5,0.5)
        )
    ])

    training_data = datasets.CIFAR10(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    test_data = datasets.CIFAR10(
        root='./data',
        train=False,
        transform=transform,
        download=True
    )

    return training_data, test_data

def train(
        mydataloader,
        model,
        optimizer,
        device,
        epochs,
        loss_fn
):
    for epoch in range(epochs):
        model.train()

        total_loss = 0

        correct = 0
        total = 0
        for batch, (X,y) in enumerate(mydataloader):
            X, y = X.to(device), y.to(device)
            prediction = model(X)
            loss = loss_fn(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (prediction.argmax(1) == y).sum().item()
            total += y.size(0)

        average_loss = (total_loss / len(mydataloader))
        training_accuracy = (correct / total)
        training_error = (1 - training_accuracy)
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Training loss: {average_loss:.4f} "
            f"Training accuracy: "
            f"{training_accuracy * 100:.2f}%"
        )

        print(
            f"Training error: "
            f"{training_error * 100:.2f}%"
        )

        return training_error * 100

def test(
        mydataloader,
        model,
        device,
        loss_fn
):

    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for X, y in mydataloader:

            X, y = X.to(device), y.to(device)

            prediction = model(X)

            test_loss += loss_fn(
                prediction,
                y
            ).item()

            correct += (
                prediction.argmax(1) == y
            ).sum().item()

            total += y.size(0)

    test_loss = (
        test_loss / len(mydataloader)
    )

    accuracy = (
        correct / total
    )

    print(
        f"Test accuracy: "
        f"{accuracy * 100:.2f}% "
        f"Average loss: "
        f"{test_loss:.4f}"
    )


def main():

    training_data, test_data = load_data()

    batch_size = 64

    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else torch.device('cpu')
    )

    print(f"Using device: {device}")

    loss_fn = nn.CrossEntropyLoss()

    # Number of convolutional layers
    number_of_layers = [1, 2, 3, 4, 5, 6]

    training_errors = []

    epochs = 5

    for num_layers in number_of_layers:

        print("\n================================")
        print(
            f"Training CNN with "
            f"{num_layers} convolutional layers"
        )
        print("================================")

        model = NeuralNetwork(
            num_layers
        ).to(device)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9
        )

        training_error = train(
            train_dataloader,
            model,
            optimizer,
            device,
            epochs,
            loss_fn
        )

        training_errors.append(
            training_error
        )

        test(
            test_dataloader,
            model,
            device,
            loss_fn
        )

    # Print final results

    print("\nFINAL RESULTS")

    for layers, error in zip(
            number_of_layers,
            training_errors
    ):

        print(
            f"Layers: {layers} "
            f"Training error: {error:.2f}%"
        )

    # Plot training error

    plt.plot(
        number_of_layers,
        training_errors,
        marker='o'
    )

    plt.xlabel(
        "Number of convolutional layers"
    )

    plt.ylabel(
        "Training error (%)"
    )

    plt.title(
        "Training Error vs Number of CNN Layers"
    )

    plt.xticks(number_of_layers)

    plt.grid(True)

    plt.show()

    print("\nEND")


if __name__ == '__main__':
    main()






