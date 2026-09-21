import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import v2




class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(28*28, 512)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.linear2 = nn.Linear(512, 512)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.linear3 = nn.Linear(512, 10)


    def forward(self, x):
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

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float,scale=True),
    ])



    training_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='./data', train=False,download = True, transform=transform)

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
        for batch,(X,y) in enumerate(mydataloader):
            X, y = X.to(device), y.to(device)
            prediction = model(X)
            loss = loss_fn(prediction,y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(mydataloader)

        print(
            f"Epoch [{epoch +1}/{epochs}], Training loss: {average_loss:.4f}"
        )


def test(
        mydataloader,
        model,
        device,
        loss_fn,
):
    model.eval()
    total_loss = 0
    correct = 0
    for X,y in mydataloader:
        X, y = X.to(device), y.to(device)
        prediction = model(X)
        loss = loss_fn(prediction,y)
        total_loss += loss.item()
        correct += (prediction.argmax(1) == y).float().sum()
    average_loss = total_loss / len(mydataloader)
    correct = correct / len(mydataloader.dataset)

    print(
        f"Test Error:"
        f"Average loss: {average_loss:.4f}"
        f"Accuracy: {correct*100:.2f}%"
    )


def probability_distribution(
        mydataloader,
        model,
        device ):
    model.eval()

    X,y = next(iter(mydataloader))
    X = X.to(device)

    with torch.no_grad():
        prediction = model(X)
        probabilities = torch.softmax(prediction, dim=1)

    class_names = mydataloader.dataset.classes

    print("\nProbability distribution for first test image")
    for i in range(10):
        print(
            f"{class_names[i]}: {probabilities[0][i].item():.2f}"
        )


    predicted_class = probabilities[0].argmax().item()
    actual_class = y[0].item()

    print(
        f"Predicted class: {class_names[predicted_class]}"
    )

    print(
        f"Actual class: {class_names[actual_class]}"
    )


    print(
        f"\nSum of probabilities:",probabilities[0].sum().item()
    )

def main():

    training_data, test_data = load_data()
    device = (torch.accelerator.current_accelerator().type
              if torch.accelerator.is_available() else torch.device('cpu'))

    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    train(train_loader, model, optimizer, device,10, loss_fn=loss_fn)
    test(test_loader, model, device,loss_fn)
    probability_distribution(test_loader, model, device)

if __name__ == "__main__":
    main()