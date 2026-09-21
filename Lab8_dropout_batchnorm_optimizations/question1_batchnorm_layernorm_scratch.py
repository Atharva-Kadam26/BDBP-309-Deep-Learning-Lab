import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Dataset
# --------------------------------------------------

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    random_state=42
)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --------------------------------------------------
# Batch Normalization from Scratch
# --------------------------------------------------

class BatchNormFromScratch(nn.Module):

    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        self.momentum = momentum
        self.eps = eps

        self.register_buffer(
            "running_mean",
            torch.zeros(num_features)
        )

        self.register_buffer(
            "running_var",
            torch.ones(num_features)
        )

    def forward(self, x):

        if self.training:

            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            x_norm = (
                (x - batch_mean)
                / torch.sqrt(batch_var + self.eps)
            )

            self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * batch_mean
            )

            self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * batch_var
            )

        else:

            x_norm = (
                (x - self.running_mean)
                / torch.sqrt(self.running_var + self.eps)
            )

        return self.gamma * x_norm + self.beta


# --------------------------------------------------
# Layer Normalization from Scratch
# --------------------------------------------------

class LayerNormFromScratch(nn.Module):

    def __init__(self, num_features, eps=1e-5):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):

        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)

        x_norm = (
            (x - mean)
            / torch.sqrt(var + self.eps)
        )

        return self.gamma * x_norm + self.beta


# --------------------------------------------------
# Network using BatchNorm
# --------------------------------------------------

class BatchNormNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(20, 64)
        self.bn1 = BatchNormFromScratch(64)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = BatchNormFromScratch(32)

        self.fc3 = nn.Linear(32, 2)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return x


# --------------------------------------------------
# Network using LayerNorm
# --------------------------------------------------

class LayerNormNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(20, 64)
        self.ln1 = LayerNormFromScratch(64)

        self.fc2 = nn.Linear(64, 32)
        self.ln2 = LayerNormFromScratch(32)

        self.fc3 = nn.Linear(32, 2)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.ln1(self.fc1(x)))
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)

        return x


# --------------------------------------------------
# Training function
# --------------------------------------------------

def train_model(model, epochs=50):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        output = model(X_train)

        loss = criterion(output, y_train)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], "
                f"Loss: {loss.item():.4f}"
            )

    model.eval()

    with torch.no_grad():

        output = model(X_test)

        predictions = torch.argmax(output, dim=1)

        accuracy = (
            predictions == y_test
        ).float().mean()

    print(f"Test Accuracy: {accuracy.item()*100:.2f}%")


# --------------------------------------------------
# Run BatchNorm
# --------------------------------------------------

print("----- Batch Normalization -----")

bn_model = BatchNormNetwork()
train_model(bn_model)


# --------------------------------------------------
# Run LayerNorm
# --------------------------------------------------

print("\n----- Layer Normalization -----")

ln_model = LayerNormNetwork()
train_model(ln_model)