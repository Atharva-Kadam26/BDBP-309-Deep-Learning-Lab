"""
Lab 6 - Getting started with PyTorch

Exercise 3:
Loading custom dataset using PyTorch.
Create a custom Dataset class, apply transforms,
and use DataLoader to load the images in batches.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


# --------------------------------------------------
# 1. CUSTOM DATASET
# --------------------------------------------------

class CustomImageDataset(Dataset):

    def __init__(self, image_dir, transform=None):

        self.image_dir = image_dir
        self.transform = transform

        self.image_files = []
        self.labels = []

        class_names = ["cats", "dogs"]

        for label, class_name in enumerate(class_names):

            class_dir = os.path.join(
                image_dir,
                class_name
            )

            for file in os.listdir(class_dir):

                if file.endswith(".png") or file.endswith(".jpg"):

                    self.image_files.append(
                        os.path.join(class_dir, file)
                    )

                    self.labels.append(label)


    def __len__(self):

        return len(self.image_files)


    def __getitem__(self, index):

        image_path = self.image_files[index]

        label = self.labels[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return image, label


# --------------------------------------------------
# 2. TRANSFORMS
# --------------------------------------------------

transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.ToTensor()
])


# --------------------------------------------------
# 3. LOAD CUSTOM DATASET
# --------------------------------------------------

image_dir = "./custom_image_dataset"

dataset = CustomImageDataset(
    image_dir=image_dir,
    transform=transform
)


print("Number of images:", len(dataset))


# --------------------------------------------------
# 4. DISPLAY ONE SAMPLE
# --------------------------------------------------

image, label = dataset[0]

print("\nFirst image:")
print(image)

print("\nFirst image shape:")
print(image.shape)

print("\nFirst image label:")
print(label)


# --------------------------------------------------
# 5. DATALOADER
# --------------------------------------------------

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


# --------------------------------------------------
# 6. LOAD BATCH
# --------------------------------------------------

print("\nBatch information:")

for images, labels in dataloader:

    print("Images shape:")
    print(images.shape)

    print("Labels:")
    print(labels)

    break


print("\nEND")