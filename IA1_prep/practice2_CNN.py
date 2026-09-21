import torch
from orca.notification_messages import size
from sklearn.externals.array_api_compat.dask.array import zeros
from torch.utils.data import DataLoader,Dataset
from torch import nn, device
from torchvision import datasets
from torchvision.transforms import v2

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(
            in_features=28*28,
            out_features=512,
        )

        self.batchnorm1 = nn.BatchNorm1d(512)

        self.relu1=nn.ReLU()

        self.dropout1 = nn.Dropout(p=0.5)

        self.linear2 = nn.Linear(
            in_features=512,
            out_features=512,
        )

        self.batchnorm2 = nn.BatchNorm1d(512)

        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(p=0.5)

        self.linear3 = nn.Linear(
            in_features=512,
            out_features=10,
        )


    def forward(self,x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        return x

def load_data():
    transform = v2.compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32,scale = True)
    ])

    training_data = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    test_data = datasets.MNIST(
        root='./data',
        train=False,
        transform=transform,
        download=True
    )


    return training_data,test_data

def tain(
        mydataloader,
        model,
        optimizer,
        epochs,
        loss_fn,
):
    for epoch in range(epochs):
        model.train()

        total_loss = 0
        for batch,(X,y) in enumerate(mydataloader):
            X,y = X.to(device),y.to(device)
            prediction = model(X)
            loss = loss_fn(prediction,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(mydataloader)

        print(
            f"Epoch [{epoch+1}/{epochs}]"
            f"Training loss: {average_loss:.3f}"
        )

def test(
        mydataloader,
        model,
        device,
        loss_fn,
):
    num_batches = len(mydataloader)
    model.eval()

    test_loss , correct = 0 , 0
    with torch.no_grad():
        for X,y in mydataloader:
            X,y = X.to(device),y.to(device)
            pred = model(X)
            loss = loss_fn(pred,y)
            test_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()
    test_loss /= num_batches
    correct = correct/len(mydataloader)

    print(
        f"Test Error:\n"
        f"Accuracy: {correct*100:.2f}%"
        f"Average loss: {test_loss:.3f}\n"
    )


def probability_distribution(
        mydataloader,
        model,
        device,
):
    model.eval()

    X,y =


