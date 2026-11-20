import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets
from torchvision.transforms import transforms , v2
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,padding = 1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,padding = 1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(128*3*3, 256)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(256, 10)


    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x= self.maxpool(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)

        x= self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x= self.relu3(x)

        x = self.linear2(x)

        return x

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def load_data():
    transform = v2.Compose([
        transforms.ToTensor(),
        v2.ToDtype(torch.float32,scale=True)
    ])

    training_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    return training_data, test_data

def train(
        my_dataloader,
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

            for batch, (X, y) in enumerate(my_dataloader):
                X, y = X.to(device), y.to(device)

                prediction = model(X)
                loss = loss_fn(prediction, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                correct += (prediction.argmax(1) == y).sum().item()
                total += y.size(0)

            average_loss = total_loss / len(my_dataloader)
            accuracy = 100 * correct / total

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Loss: {average_loss:.4f} "
                f"Training Accuracy: {accuracy:.2f}%"
            )


def test(
        my_dataloader,
        model,
        device,
        loss_fn
):
    size = len(my_dataloader.dataset)
    num_batches = len(my_dataloader)

    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X,y in my_dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float32).sum().item()

    test_loss /= num_batches
    correct /= size

    print(
        f"Test Error: \n"
        f"Accuracy: {(100*correct)}%"
        f"Avg loss: {test_loss:.4f}"
    )

def main():
    training_data, test_data = load_data()
    batch_size = 128

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle= True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle= False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(train_dataloader,model,optimizer,device,epochs=5,loss_fn=loss_fn)
    test(test_dataloader,model,device,loss_fn)


    # ==================================
    # RESNET18 TRANSFER LEARNING
    # ==================================

    # Transformations
    resnet_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    # Load CIFAR10
    resnet_train_data = datasets.CIFAR10(
        './data', train=True, download=True,
        transform=resnet_transform
    )

    resnet_test_data = datasets.CIFAR10(
        './data', train=False, download=True,
        transform=resnet_transform
    )

    resnet_train_loader = DataLoader(
        resnet_train_data, batch_size=64, shuffle=True
    )

    resnet_test_loader = DataLoader(
        resnet_test_data, batch_size=64, shuffle=False
    )

    # Load pretrained ResNet18
    resnet = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Freeze pretrained layers
    for param in resnet.parameters():
        param.requires_grad = False

    # Replace final layer
    resnet.fc = nn.Linear(resnet.fc.in_features, 10)

    resnet = resnet.to(device)

    # Loss and optimizer
    resnet_loss = nn.CrossEntropyLoss()

    resnet_optimizer = optim.Adam(
        resnet.fc.parameters(), lr=0.001
    )

    # Train ResNet18
    resnet.eval()

    for epoch in range(5):

        total_loss = 0
        correct = 0
        total = 0

        for X, y in resnet_train_loader:

            X, y = X.to(device), y.to(device)

            prediction = resnet(X)

            loss = resnet_loss(prediction, y)

            resnet_optimizer.zero_grad()
            loss.backward()
            resnet_optimizer.step()

            total_loss += loss.item()

            correct += (prediction.argmax(1) == y).sum().item()
            total += y.size(0)

        accuracy = 100 * correct / total

        print(
            f"ResNet Epoch [{epoch+1}/5] "
            f"Loss: {total_loss/len(resnet_train_loader):.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    # Test ResNet18
    print("\nResNet18 Test Results:")

    test(
        resnet_test_loader,
        resnet,
        device,
        resnet_loss
    )

if __name__ == "__main__":
    main()


