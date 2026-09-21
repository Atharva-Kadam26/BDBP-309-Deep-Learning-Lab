import torch
from progressbar.utils import deltas_to_seconds
from statsmodels.miscmodels.try_mlecov import MLEGLS
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.transforms import v2

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=28 * 28 , out_features=512)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(in_features=512, out_features=512)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(in_features=512, out_features=10)

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
        v2.ToImage(),v2.ToDtype(torch.float32,scale = True)
    ])

    training_data = datasets.FashionMNIST(root = './data', train = True, transform = transform, download = True)
    test_data = datasets.FashionMNIST(root = './data', train = False, transform = transform, download = False)

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
            loss = loss_fn(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(mydataloader)

        print(
            f"Epoch [{epoch+1}/{epochs}] Training Loss: {avg_loss:.3f}"
        )

def test(mydataloader, model, device,loss_fn):
    model.eval()

    test_loss =0
    correct = 0
    for X,y in mydataloader:
        X, y = X.to(device), y.to(device)
        prediction = model(X)
        loss = loss_fn(prediction, y)
        test_loss += loss.item()
        correct += (prediction.argmax(1)==y).type(torch.float).sum().item()

    test_loss = test_loss / len(mydataloader)
    correct = correct / len(mydataloader.dataset)

    print(
        f"Test Error:\n"
        f"Accuracy: {(100*correct):.3f}\n"
        f"Loss: {test_loss:.3f}"
    )

# def probability_distribbution(mydataloader, model, device):
#     model.eval()
#
#     X,y = next(iter(mydataloader))
#
#     X = X.to(device)
#     with torch.no_grad():
#
#         pred = model(X)
#         probabilities = torch.softmax(pred, dim = 1)
#
#     class_names = mydataloader.dataset.classes
#
#     print("\n Probability Distribution for first test image:)")
#     for i in range(10):
#         print(f"{class_names[i]}: {probabilities[0][i].item():.3f}")
#
#     predicted_class = probabilities[0].argmax().item()
#
#     actual_class = y[0].item()
#
#     print("\n Predicted class:",
#           class_names[predicted_class])
#
#     print("\n Actual class:",
#           class_names[actual_class])
#
#     print("\nSum of probabilities:",
#           probabilities[0].sum().item())


def  probability_distribution(mydataloader, model, device):
    model.eval()

    X,y = next(iter(mydataloader))
    X = X.to(device)

    with torch.no_grad():
        prediction = model(X)
        probabilities = torch.softmax(prediction, dim=1)

    class_names = mydataloader.dataset.classes

    print("\nProbability Distribution for first test image:")

    for i in range(10):
        print(
            f"{class_names[i]:.12s}"
            f"{probabilities[0][i].item():.3f}"
        )

    predicted_class = probabilities[0].argmax().item()
    actual_class = y[0].item()

    print("\nPredicted class:", class_names[predicted_class])
    print("Actual class:", class_names[actual_class])
    print("\nSum of probabilities:", probabilities[0].sum().item())





def main():
    training_data, test_data = load_data()

    batch_size = 64
    device =  torch.device(
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else torch.device('cpu')
    )


    model = NeuralNetwork().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_fn = nn.CrossEntropyLoss()

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    train(train_dataloader,
          model,
          optimizer,
          device,
          epochs=10,
          loss_fn=loss_fn
          )

    test(
        test_dataloader,
        model,
        device,
        loss_fn=loss_fn
    )


    probability_distribution(test_dataloader, model, device)


if __name__ == "__main__":
    main()






