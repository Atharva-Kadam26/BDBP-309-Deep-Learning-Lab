"""Download MNIST dataset and implement a MNIST classifier using CNN PyTorch library."""

import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch import nn
from torch.utils.data import DataLoader


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=64 * 13 * 13, out_features=128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        output = self.fc2(x)
        return output

def load_data():

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32 ,scale=True)
    ])

    training_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    return training_data, test_data

def train(
        mydataloader,
        model,
        optimizer,
        device,
        epochs,
        loss_fn,
):

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch, (X, y) in enumerate(mydataloader):
            X,y  = X.to(device), y.to(device)

            prediction = model(X)
            loss = loss_fn(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(mydataloader)
        print(
            f"Epoch [{epoch+1}/{epochs}]"
            f"Training loss: {average_loss:.4f} "
        )

def test(
        mydataloader,
        model,
        device,
        loss_fn,
):

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in mydataloader:

            X,y = X.to(device), y.to(device)
            prediction = model(X)
            test_loss += loss_fn(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float32).sum().item()

    test_loss = test_loss / len(mydataloader)
    correct = correct / len(mydataloader)
    print(
        f"Test loss: \n"
        f"Accuracy: {(correct*100)}%"
        f"Average Loss: {test_loss:.4f} "
    )

def main():
    training_data, test_data = load_data()
    batch_size = 64

    train_dataloader = DataLoader(training_data, batch_size=batch_size,shuffle=True)

    test_dataloader = DataLoader(test_data, batch_size=batch_size,shuffle=False)

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else torch.device('cpu')
    )

    print(f"Using device: {device}")
    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train(train_dataloader, model, optimizer, device,epochs=10, loss_fn=loss_fn)
    test(test_dataloader, model, device, loss_fn)

    print('\nEND')

if __name__ == '__main__':
    main()






