


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import models

# --------------------------------
# 1. Load CIFAR10 Dataset
# --------------------------------

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Use smaller dataset for faster training
train_dataset = Subset(train_dataset, range(10000))
test_dataset = Subset(test_dataset, range(2000))

train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True
)

test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False
)

# --------------------------------
# 2. Define CNN Model
# --------------------------------

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# --------------------------------
# 3. Train CNN Model
# --------------------------------

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3

for epoch in range(epochs):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print("CNN Epoch [{}/{}], Loss: {:.4f}, Training Accuracy: {:.2f}%".format(
        epoch + 1, epochs, running_loss / len(train_loader), accuracy
    ))


# --------------------------------
# 4. Test CNN Model
# --------------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

cnn_accuracy = 100 * correct / total

print("\nCNN Test Accuracy: {:.2f}%".format(cnn_accuracy))


# --------------------------------
# 5. Load Pretrained ResNet18
# --------------------------------

resnet = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze all pretrained layers
for param in resnet.parameters():
    param.requires_grad = False

# Modify final layer for 10 CIFAR10 classes
resnet.fc = nn.Linear(resnet.fc.in_features, 10)

# Use smaller image size
resnet_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Reload datasets with ResNet transforms
resnet_train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=resnet_transform
)

resnet_test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=resnet_transform
)

# Use smaller subsets
resnet_train_dataset = Subset(resnet_train_dataset, range(10000))
resnet_test_dataset = Subset(resnet_test_dataset, range(2000))

resnet_train_loader = DataLoader(
    resnet_train_dataset, batch_size=128, shuffle=True
)

resnet_test_loader = DataLoader(
    resnet_test_dataset, batch_size=128, shuffle=False
)

# --------------------------------
# 6. Train ResNet18
# --------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)

for epoch in range(3):

    resnet.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in resnet_train_loader:

        outputs = resnet(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print("ResNet Epoch [{}/3], Loss: {:.4f}, Training Accuracy: {:.2f}%".format(
        epoch + 1, running_loss / len(resnet_train_loader), accuracy
    ))


# --------------------------------
# 7. Test ResNet18
# --------------------------------

resnet.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in resnet_test_loader:

        outputs = resnet(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

resnet_accuracy = 100 * correct / total

print("\nResNet18 Test Accuracy: {:.2f}%".format(resnet_accuracy))


# --------------------------------
# 8. Compare Both Models
# --------------------------------

print("\n----- Model Comparison -----")
print("CNN Test Accuracy: {:.2f}%".format(cnn_accuracy))
print("ResNet18 Test Accuracy: {:.2f}%".format(resnet_accuracy))

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torchvision import models
#
# # --------------------------------
# # 1. Load CIFAR10 Dataset
# # --------------------------------
#
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])
#
# train_dataset = torchvision.datasets.CIFAR10(
#     root="./data",
#     train=True,
#     download=True,
#     transform=transform
# )
#
# test_dataset = torchvision.datasets.CIFAR10(
#     root="./data",
#     train=False,
#     download=True,
#     transform=transform
# )
#
# train_loader = DataLoader(
#     train_dataset, batch_size=64, shuffle=True
# )
#
# test_loader = DataLoader(
#     test_dataset, batch_size=64, shuffle=False
# )
#
# # --------------------------------
# # 2. Define CNN Model
# # --------------------------------
#
# class CNN(nn.Module):
#
#     def __init__(self):
#         super(CNN, self).__init__()
#
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(2, 2)
#
#         self.fc1 = nn.Linear(128 * 4 * 4, 256)
#         self.fc2 = nn.Linear(256, 10)
#
#     def forward(self, x):
#
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = self.pool(self.relu(self.conv3(x)))
#
#         x = x.view(x.size(0), -1)
#
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#
#         return x
#
#
# # --------------------------------
# # 3. Train CNN Model
# # --------------------------------
#
# model = CNN()
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# epochs = 5
#
# for epoch in range(epochs):
#
#     model.train()
#
#     running_loss = 0
#     correct = 0
#     total = 0
#
#     for images, labels in train_loader:
#
#         outputs = model(images)
#
#         loss = criterion(outputs, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#         _, predicted = torch.max(outputs, 1)
#
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     accuracy = 100 * correct / total
#
#     print("CNN Epoch [{}/{}], Loss: {:.4f}, Training Accuracy: {:.2f}%".format(
#         epoch + 1, epochs, running_loss / len(train_loader), accuracy
#     ))
#
#
# # --------------------------------
# # 4. Test CNN Model
# # --------------------------------
#
# model.eval()
#
# correct = 0
# total = 0
#
# with torch.no_grad():
#
#     for images, labels in test_loader:
#
#         outputs = model(images)
#
#         _, predicted = torch.max(outputs, 1)
#
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# cnn_accuracy = 100 * correct / total
#
# print("\nCNN Test Accuracy: {:.2f}%".format(cnn_accuracy))
#
#
# # --------------------------------
# # 5. Load Pretrained ResNet18
# # --------------------------------
#
# resnet = models.resnet18(
#     weights=models.ResNet18_Weights.DEFAULT
# )
#
# # Modify final layer for 10 CIFAR10 classes
# resnet.fc = nn.Linear(resnet.fc.in_features, 10)
#
# # ResNet18 expects larger images
# resnet_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])
#
# # Reload datasets with ResNet transforms
# resnet_train_dataset = torchvision.datasets.CIFAR10(
#     root="./data",
#     train=True,
#     download=False,
#     transform=resnet_transform
# )
#
# resnet_test_dataset = torchvision.datasets.CIFAR10(
#     root="./data",
#     train=False,
#     download=False,
#     transform=resnet_transform
# )
#
# resnet_train_loader = DataLoader(
#     resnet_train_dataset, batch_size=64, shuffle=True
# )
#
# resnet_test_loader = DataLoader(
#     resnet_test_dataset, batch_size=64, shuffle=False
# )
#
# # --------------------------------
# # 6. Train ResNet18
# # --------------------------------
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(resnet.parameters(), lr=0.001)
#
# for epoch in range(5):
#
#     resnet.train()
#
#     running_loss = 0
#     correct = 0
#     total = 0
#
#     for images, labels in resnet_train_loader:
#
#         outputs = resnet(images)
#
#         loss = criterion(outputs, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#         _, predicted = torch.max(outputs, 1)
#
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     accuracy = 100 * correct / total
#
#     print("ResNet Epoch [{}/5], Loss: {:.4f}, Training Accuracy: {:.2f}%".format(
#         epoch + 1, running_loss / len(resnet_train_loader), accuracy
#     ))
#
#
# # --------------------------------
# # 7. Test ResNet18
# # --------------------------------
#
# resnet.eval()
#
# correct = 0
# total = 0
#
# with torch.no_grad():
#
#     for images, labels in resnet_test_loader:
#
#         outputs = resnet(images)
#
#         _, predicted = torch.max(outputs, 1)
#
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# resnet_accuracy = 100 * correct / total
#
# print("\nResNet18 Test Accuracy: {:.2f}%".format(resnet_accuracy))
#
#
# # --------------------------------
# # 8. Compare Both Models
# # --------------------------------
#
# print("\n----- Model Comparison -----")
# print("CNN Test Accuracy: {:.2f}%".format(cnn_accuracy))
# print("ResNet18 Test Accuracy: {:.2f}%".format(resnet_accuracy))
#
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# import torchvision.models as models
#
# from torch.utils.data import DataLoader, Subset
#
#
# # -----------------------------------------
# # 1. Device configuration
# # -----------------------------------------
#
# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )
#
# print("Device:", device)
#
#
# # -----------------------------------------
# # 2. Load CIFAR-10 dataset for CNN
# # -----------------------------------------
#
# transform_cnn = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(
#         (0.5, 0.5, 0.5),
#         (0.5, 0.5, 0.5)
#     )
# ])
#
# train_dataset = torchvision.datasets.CIFAR10(
#     root="./data",
#     train=True,
#     download=True,
#     transform=transform_cnn
# )
#
# test_dataset = torchvision.datasets.CIFAR10(
#     root="./data",
#     train=False,
#     download=True,
#     transform=transform_cnn
# )
#
#
# # Select smaller subsets
# train_dataset = Subset(train_dataset, range(10000))
# test_dataset = Subset(test_dataset, range(2000))
#
#
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=64,
#     shuffle=True
# )
#
# test_loader = DataLoader(
#     test_dataset,
#     batch_size=64,
#     shuffle=False
# )
#
#
# # -----------------------------------------
# # 3. Define CNN model
# # -----------------------------------------
#
# class CNN(nn.Module):
#
#     def __init__(self):
#         super(CNN, self).__init__()
#
#         self.conv_layers = nn.Sequential(
#
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
#
#         self.fc_layers = nn.Sequential(
#             nn.Linear(128 * 4 * 4, 256),
#             nn.ReLU(),
#             nn.Linear(256, 10)
#         )
#
#     def forward(self, x):
#
#         x = self.conv_layers(x)
#
#         x = x.view(x.size(0), -1)
#
#         x = self.fc_layers(x)
#
#         return x
#
#
# # -----------------------------------------
# # 4. Train and test CNN
# # -----------------------------------------
#
# cnn = CNN().to(device)
#
# criterion = nn.CrossEntropyLoss()
#
# optimizer = optim.Adam(
#     cnn.parameters(),
#     lr=0.001
# )
#
# epochs = 5
#
# for epoch in range(epochs):
#
#     cnn.train()
#
#     running_loss = 0
#     correct = 0
#     total = 0
#
#     for images, labels in train_loader:
#
#         images = images.to(device)
#         labels = labels.to(device)
#
#         outputs = cnn(images)
#
#         loss = criterion(outputs, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#         _, predicted = torch.max(outputs, 1)
#
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     accuracy = 100 * correct / total
#
#     print(
#         f"CNN Epoch [{epoch + 1}/{epochs}], "
#         f"Loss: {running_loss / len(train_loader):.4f}, "
#         f"Training Accuracy: {accuracy:.2f}%"
#     )
#
#
# # CNN testing
# cnn.eval()
#
# correct = 0
# total = 0
#
# with torch.no_grad():
#
#     for images, labels in test_loader:
#
#         images = images.to(device)
#         labels = labels.to(device)
#
#         outputs = cnn(images)
#
#         _, predicted = torch.max(outputs, 1)
#
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# cnn_accuracy = 100 * correct / total
#
# print("\nCNN Test Accuracy:", round(cnn_accuracy, 2), "%")
#
#
# # -----------------------------------------
# # 5. Load CIFAR-10 dataset for ResNet
# # -----------------------------------------
#
# transform_resnet = transforms.Compose([
#
#     transforms.Resize((224, 224)),
#
#     transforms.ToTensor(),
#
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])
#
# train_dataset_resnet = torchvision.datasets.CIFAR10(
#     root="./data",
#     train=True,
#     download=True,
#     transform=transform_resnet
# )
#
# test_dataset_resnet = torchvision.datasets.CIFAR10(
#     root="./data",
#     train=False,
#     download=True,
#     transform=transform_resnet
# )
#
#
# # Same smaller subsets
# train_dataset_resnet = Subset(
#     train_dataset_resnet, range(10000)
# )
#
# test_dataset_resnet = Subset(
#     test_dataset_resnet, range(2000)
# )
#
#
# train_loader_resnet = DataLoader(
#     train_dataset_resnet,
#     batch_size=64,
#     shuffle=True
# )
#
# test_loader_resnet = DataLoader(
#     test_dataset_resnet,
#     batch_size=64,
#     shuffle=False
# )
#
#
# # -----------------------------------------
# # 6. Define ResNet18 model
# # -----------------------------------------
#
# resnet = models.resnet18(
#     weights=models.ResNet18_Weights.DEFAULT
# )
#
# resnet.fc = nn.Linear(
#     resnet.fc.in_features, 10
# )
#
# resnet = resnet.to(device)
#
#
# # -----------------------------------------
# # 7. Train ResNet
# # -----------------------------------------
#
# criterion = nn.CrossEntropyLoss()
#
# optimizer = optim.Adam(
#     resnet.parameters(),
#     lr=0.001
# )
#
# for epoch in range(epochs):
#
#     resnet.train()
#
#     running_loss = 0
#     correct = 0
#     total = 0
#
#     for images, labels in train_loader_resnet:
#
#         images = images.to(device)
#         labels = labels.to(device)
#
#         outputs = resnet(images)
#
#         loss = criterion(outputs, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#         _, predicted = torch.max(outputs, 1)
#
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     accuracy = 100 * correct / total
#
#     print(
#         f"ResNet Epoch [{epoch + 1}/{epochs}], "
#         f"Loss: {running_loss / len(train_loader_resnet):.4f}, "
#         f"Training Accuracy: {accuracy:.2f}%"
#     )
#
#
# # -----------------------------------------
# # 8. Test ResNet
# # -----------------------------------------
#
# resnet.eval()
#
# correct = 0
# total = 0
#
# with torch.no_grad():
#
#     for images, labels in test_loader_resnet:
#
#         images = images.to(device)
#         labels = labels.to(device)
#
#         outputs = resnet(images)
#
#         _, predicted = torch.max(outputs, 1)
#
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# resnet_accuracy = 100 * correct / total
#
# print("\nResNet Test Accuracy:", round(resnet_accuracy, 2), "%")
#
#
# # -----------------------------------------
# # 9. Compare models
# # -----------------------------------------
#
# print("\n========== MODEL COMPARISON ==========")
#
# print("CNN Test Accuracy:", round(cnn_accuracy, 2), "%")
#
# print("ResNet Test Accuracy:", round(resnet_accuracy, 2), "%")
