
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader


# 1. Load the datasets

train_data = np.load("data/q2_data/train_20000.npz")
val_data = np.load("data/q2_data/val_3000.npz")
test_data = np.load("data/q2_data/test_3000.npz")

x_train = train_data["images"]
y_train = train_data["labels"]

x_val = val_data["images"]
y_val = val_data["labels"]

x_test = test_data["images"]
y_test = test_data["labels"]

print("Training data:", x_train.shape, y_train.shape)
print("Validation data:", x_val.shape, y_val.shape)
print("Testing data:", x_test.shape, y_test.shape)


# 2. Preprocess the data

class PathMNISTDataset(Dataset):

    def __init__(self, images, labels):

        self.images = torch.tensor(images, dtype=torch.float32)
        self.images = self.images / 255.0

        # Convert NHWC to NCHW
        self.images = self.images.permute(0, 3, 1, 2)

        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


train_dataset = PathMNISTDataset(x_train, y_train)
val_dataset = PathMNISTDataset(x_val, y_val)
test_dataset = PathMNISTDataset(x_test, y_test)


# 3. Create DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# 4. Build the CNN model

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 9)

    def forward(self, x):

        x = self.pool1(self.relu1(self.conv1(x)))

        x = self.pool2(self.relu2(self.conv2(x)))

        x = self.pool3(self.relu3(self.conv3(x)))

        x = self.adaptive_pool(x)

        x = self.flatten(x)

        x = self.relu4(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# 5. Initialize model, loss and optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Using device:", device)


# 6. Train the model

epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward propagation
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_accuracy = 100 * correct / total


    # Validation

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / val_total
    val_accuracy = 100 * val_correct / val_total

    print("Epoch [{}/{}]".format(epoch + 1, epochs))
    print("Training Loss: {:.4f}".format(train_loss))
    print("Training Accuracy: {:.2f}%".format(train_accuracy))
    print("Validation Loss: {:.4f}".format(val_loss))
    print("Validation Accuracy: {:.2f}%".format(val_accuracy))
    print("------------------------------------")


# 7. Evaluate on the test dataset

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        test_total += labels.size(0)

        test_correct += (predicted == labels).sum().item()


test_accuracy = 100 * test_correct / test_total

print("Final Test Accuracy: {:.2f}%".format(test_accuracy))
