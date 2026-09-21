import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    random_state=42,
)


X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(in_features=20, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=2)

        self.relu = nn.ReLU()

    def forward(self,x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

#Training function

def training_model(optimizer_name):
    model = NeuralNetwork()
    criterion = nn.CrossEntropyLoss()

    #Selecting optimizer

    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.01)

    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    elif optimizer_name == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)

    elif optimizer_name == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=0.01)


    #training

    losses = []

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()

        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())


    model.eval()

    with torch.no_grad():
        output = model(X_test)

        predictions = torch.argmax(output, dim=1)

        accuracy = (predictions == y_test).float().mean()

    print(
        f"{optimizer_name}: "
        f"Accuracy = {accuracy.item() * 100:.2f}%"
    )

    return losses


#Compare optimizers

sgd_loss = training_model('SGD')
momentum_loss = training_model('Momentum')
adagrad_loss = training_model('AdaGrad')
adam_loss = training_model('Adam')


#Plot Loss

plt.plot(sgd_loss,label = 'SGD')
plt.plot(momentum_loss,label = 'Momentum')
plt.plot(adagrad_loss,label = 'AdaGrad')
plt.plot(adam_loss,label = 'Adam')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Optimizer Comparison")

plt.legend()
plt.show()



