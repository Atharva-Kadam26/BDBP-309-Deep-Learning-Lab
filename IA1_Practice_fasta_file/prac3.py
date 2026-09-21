import torch
from torch import nn, device, optim
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import v2
from torchvision import datasets


class ConvolutedNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutedNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3
        )


        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding = 1
        )

        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2,
        )

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(
            in_features=64 * 7 * 7,
            out_features=128
        )

        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(
            in_features=128,
            out_features=10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        return x

def load_data():

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32,scale = True)
    ])

    training_data = datasets.CIFAR10(
        root = './data',
        train = True,
        download = True,
        transform = transform
    )

    test_data = datasets.CIFAR10(
        root = './data',
        train = False,
        download = True,
        transform = transform
    )

    return training_data, test_data


def training(
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
        for batch,(X,y) in enumerate(mydataloader):
            X,y = X.to(device), y.to(device)

            prediction = model(X)
            loss = loss_fn(prediction,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(mydataloader)
        print(
            f"Epochs [{epoch +1}/{epochs}]"
            f"Training Error: {average_loss:.3f}"
        )

def test(
        mydataloader,
        model,
        device,
        loss_fn
):
    num_batches = len(mydataloader)
    size = len(mydataloader.dataset)
    model.eval()

    test_loss = 0
    correct = 0
    for X,y in mydataloader:
        X, y = X.to(device), y.to(device)
        prediction = model(X)
        loss = loss_fn(prediction,y)
        test_loss += loss.item()
        correct += (prediction.argmax(dim=1)==y).type(torch.float).sum().item()
    average_loss = test_loss / num_batches
    correct = correct / size

    print(
        f"Test Error:\n"
        f"Accuracy: {(100*correct):.3f}\n"
        f"Average loss: {average_loss:.3f}"
    )

def main():
    training_data, test_data = load_data()


    loss_fn = nn.CrossEntropyLoss()
    device = ( torch.accelerator.current_accelerator().type
               if torch.accelerator.is_available()
               else torch.device('cpu')
    )
    model = ConvolutedNeuralNetwork().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    batch_size = 64
    train_loader = DataLoader(
        training_data,
        batch_size = batch_size,
        shuffle = True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size = batch_size,
        shuffle = False
    )

    training(
        train_loader,
        model,
        optimizer,
        device,
        epochs = 5,
        loss_fn= loss_fn
    )

    test(
        test_loader,
        model,
        device,
        loss_fn
    )
if __name__ == '__main__':
    main()





