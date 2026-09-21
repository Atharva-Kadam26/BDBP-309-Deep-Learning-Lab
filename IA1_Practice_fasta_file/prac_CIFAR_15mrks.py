# import torch
# from torchvision import datasets, transforms
#
# transform = transforms.ToTensor()
#
# train_dataset = datasets.CIFAR10(
#     root="./data", train=True, download=True, transform=transform
# )
#
# test_dataset = datasets.CIFAR10(
#     root="./data", train=False, download=True, transform=transform
# )
#
# print(f"Number of training samples:",len(train_dataset))
# print(f"Number of test samples:",len(test_dataset))
#
# image,label = train_dataset[0]
#
# print('Image shape:',image.shape)
# print('Label:',label)
# print('Minimum Pixel Value:',image.min())
# print('Maximum Pixel Value:',image.max())

import torch
from torchvision import transforms,datasets

transform = transforms.ToTensor()
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

print("Number of samples in training set:",len(train_set))
print("Number of samples in test set:",len(test_set))

image,label = train_set[0]

print("Image shape:",image.shape)
print("Label:",label)
print('Minimum Pixel Value:',image.min())
print('Maximum Pixel Value:',image.max())



