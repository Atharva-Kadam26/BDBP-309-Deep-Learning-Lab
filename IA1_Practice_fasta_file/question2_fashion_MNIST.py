import torch
from torch import nn,device
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.transforms import v2

#Defining Neural Network

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(
            in_features = 28 * 28 ,
            out_features = 512
        )

        self.batchnorm1 = nn.BatchNorm1d(num_features = 512)

        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(
            in_features = 512,
            out_features = 512
        )

        self.batchnorm2 = nn.BatchNorm1d(num_features = 512)

        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(
            in_features = 512,
            out_features = 10
        )
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self,x):
        x = self.flatten(x)
        x = self.linear1(x)
        x= self.batchnorm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        output = self.linear3(x)

        return output


def load_data():

        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32,scale=True)
        ])

        training_data = datasets.FashionMNIST(
            root = './data',
            train = True,
            download = True,
            transform = transform
        )

        test_data = datasets.FashionMNIST(
            root = './data',
            train = False,
            download = True,
            transform = transform
        )

        return training_data, test_data

def train(mydataloader,model,loss_fn,optimizer,device,epochs):
    size = len(mydataloader.dataset)

    for epoch in range(epochs):
        model.train()

        total_loss = 0

        for batch, (X,y) in enumerate(mydataloader):
            X,y = X.to(device),y.to(device)

            prediction = model(X)

            loss = loss_fn(prediction,y)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(mydataloader)

        print(
                f'Epoch [{epoch + 1}/{epochs}]'
                f'Training Loss: {average_loss:.3f}'
            )


def test(mydataloader,model,loss_fn,device):
    size = len(mydataloader.dataset)
    num_batches = len(mydataloader)
    model.eval()

    test_loss , correct = 0, 0

    with torch.no_grad():
        for X,y in mydataloader:
            X,y = X.to(device),y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred,y).item()
            correct += (
            (pred.argmax(dim = 1) == y)
            .type(torch.float)
                .sum().item()
            )

    test_loss = test_loss / num_batches
    correct = correct / size

    print(
        f"Test Error:\n"
        f"Accuracy: {(100 * correct):> 0.1f}%,"
        f"Average loss: {test_loss:>.8f}\n"
    )


def probability_distribution(mydataloader,model,device):

    model.eval()

    #Get one batch from test data
    X,y = next(iter(mydataloader))

    X = X.to(device)

    with torch.no_grad():

        #get model output
        pred = model(X)

        #convert logits into probabilities
        probabilities = torch.softmax(pred,dim = 1)


#     #Fashion -MNIST class names
    class_names = mydataloader.dataset.classes


    print("\nProbability distribution for first test image:")

    for i in range(10):
        print(
            f"{class_names[i]:12s}:"
            f"{probabilities[0][i].item():.4f}"
        )



#     #Predicted class
    predicted_class = probabilities[0].argmax().item()

    actual_class = y[0].item()

    print(
        "\n Predicted class:",
        class_names[predicted_class]
    )


    print(
        "Actual class:",
        class_names[actual_class]
    )


    #checking probability sum
    print(
        "\nSum of probabilities:",
        probabilities[0].sum().item()
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
        else "cpu"
    )

    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)


    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = 1e-3,
    )

    train(
        train_dataloader,
        model,
        loss_fn,
        optimizer,
        device=device,
        epochs=10
    )


    test(
        test_dataloader,
        model,
        loss_fn,
        device=device
    )


    probability_distribution(test_dataloader,model,device)

    print("\n END")

if __name__ == "__main__":
    main()



