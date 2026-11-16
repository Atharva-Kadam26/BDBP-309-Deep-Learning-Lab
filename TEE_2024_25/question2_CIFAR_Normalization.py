#A2. Load the CIFAR-IO dataset and normalize the pixel values to the range [O, 11. You are allowed to
# use PyTorch library functions.

import torch
import torchvision
import torchvision.transforms as transforms

# Convert images into tensors and normalize to [0,1]
transform = transforms.ToTensor()

# Load CIFAR-10 training dataset
train_data = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Load CIFAR-10 testing dataset
test_data = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Display one image
image, label = train_data[0]

print("Image shape:", image.shape)
print("Label:", label)
print("Minimum pixel value:", image.min())
print("Maximum pixel value:", image.max())
