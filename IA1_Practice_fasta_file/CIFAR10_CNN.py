import torch
from sympy import roots
from sympy.core.random import shuffle
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 32,
            kernel_size = 3
        )

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )


        self.relu2 = nn.ReLU()

        #maxpool
        self.maxpool = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2,
        )

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(
            in_features = 64 * 15 * 15,
            out_features = 128
        )

        self.relu3 = nn.ReLU()

        self.linear2 = nn.Linear(
           in_features = 128,
            out_features = 10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x= self.relu3(x)

        output = self.linear2(x)

        return output


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

        for batch,(X,y) in enumerate(mydataloader):

            X,y = X.to(device), y.to(device)

            #forward pass
            prediction = model(X)

            loss = loss_fn(prediction, y)

            #backpropagation
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()  #update the weights

            total_loss += loss.item()

        average_loss = total_loss / len(mydataloader)
        print(
            f"Epoch [{epoch +1}/{epoch}]"
            f"Training error: {average_loss}"
        )

def test(
        mydataloader,
        model,
        device,
        loss_fn
):
    size = len(mydataloader.dataset)
    num_batches = len(mydataloader)

    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X,y in mydataloader:
            X,y = X.to(device),y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred,y).item()

            correct += (
            (pred.argmax(dim=1) ==y).type(torch.float32).sum().item()
            )

    test_loss = test_loss / num_batches
    correct = correct/size

    print(
        f"Test error: \n"
        f"Accuracy: {(100 * correct)}%"
        f"Average Loss: {test_loss}\n"
    )

def main():
    training_data, test_data = load_data()
    batch_size = 64

    train_dataloader = DataLoader(
        training_data,
        batch_size = batch_size,
        shuffle = True,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size = batch_size,
        shuffle = False
    )

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else torch.device('cpu')
    )


    print(f"Using {device} device")

    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    train(train_dataloader,
          model,
          optimizer,
          device,
          epochs = 5,
          loss_fn = loss_fn)

    test(
        test_dataloader,
        model,
        device,
        loss_fn
    )

    print('\n END')

if __name__ == '__main__':
    main()








