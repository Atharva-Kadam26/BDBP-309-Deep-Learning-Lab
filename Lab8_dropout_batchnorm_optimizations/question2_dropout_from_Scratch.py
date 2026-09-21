import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X,y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    random_state=42,
)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Dropout from scratch

class DropoutFromScratch(nn.Module):
    def __init__(self,p =0.5):
        super().__init__()
        self.p = p

    def forward(self,x):
        if not self.training:
            return x

        #create random mask
        mask = (torch.rand_like(x) > self.p).float()

        #Apply mask
        return x * mask / (1- self.p)

#Neural Network
class DropoutNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()

        self.dropout1 = DropoutFromScratch(p=0.5)
        self.dropout2 = DropoutFromScratch(p=0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)

        x = self.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)

        return x

#Training

model = DropoutNetwork()

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    output = model(X_train).squeeze()
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}]," f"Loss: {loss.item():.4f}")


#Testing

model.eval()

with torch.no_grad():
    output = model(X_test).squeeze()

    predictions= (torch.sigmoid(output) >= 0.5).float()

    accuracy= (predictions == y_test).float().mean()

print(f"Test Accuracy: {accuracy.item()*100:.2f}%")


