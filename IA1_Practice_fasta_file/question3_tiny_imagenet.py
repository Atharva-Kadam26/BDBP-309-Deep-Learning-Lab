import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from PIL import Image
import os


# ---------------------------------------------------------
# CNN MODEL
# ---------------------------------------------------------

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.relu3 = nn.ReLU()

        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Reduces feature map to 1 x 1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Tiny ImageNet has 200 classes
        self.fc = nn.Linear(
            in_features=128,
            out_features=200
        )


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.adaptive_pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


# ---------------------------------------------------------
# VALIDATION DATASET
# ---------------------------------------------------------

class TinyImageNetValDataset(Dataset):

    def __init__(
            self,
            val_dir,
            class_to_idx,
            transform=None
    ):

        self.val_dir = val_dir
        self.class_to_idx = class_to_idx
        self.transform = transform

        annotation_file = os.path.join(
            val_dir,
            "val_annotations.txt"
        )

        self.samples = []

        with open(annotation_file, "r") as file:

            for line in file:

                parts = line.strip().split("\t")

                image_name = parts[0]
                class_name = parts[1]

                image_path = os.path.join(
                    val_dir,
                    "images",
                    image_name
                )

                label = class_to_idx[class_name]

                self.samples.append(
                    (image_path, label)
                )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():

    transform = v2.Compose([

        v2.ToImage(),

        v2.ToDtype(
            torch.float32,
            scale=True
        )
    ])


    dataset_path = (
        r"C:\Datasets\tiny-imagenet-200"
        r"\tiny-imagenet-200"
    )


    # Training data
    training_data = datasets.ImageFolder(

        root=os.path.join(
            dataset_path,
            "train"
        ),

        transform=transform
    )


    # Validation/Test data
    test_data = TinyImageNetValDataset(

        val_dir=os.path.join(
            dataset_path,
            "val"
        ),

        class_to_idx=training_data.class_to_idx,

        transform=transform
    )


    return training_data, test_data


# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------

def train(
        mydataloader,
        model,
        loss_fn,
        optimizer,
        device,
        epochs
):

    size = len(mydataloader.dataset)

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        correct = 0


        for X, y in mydataloader:

            X = X.to(device)
            y = y.to(device)


            # Forward pass
            pred = model(X)


            # Calculate loss
            loss = loss_fn(
                pred,
                y
            )


            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            # Statistics
            total_loss += loss.item()

            correct += (
                (pred.argmax(1) == y)
                .sum()
                .item()
            )


        average_loss = (
            total_loss /
            len(mydataloader)
        )


        accuracy = (
            100 * correct /
            size
        )


        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {average_loss:.4f} "
            f"Training Accuracy: {accuracy:.2f}%"
        )


# ---------------------------------------------------------
# TESTING
# ---------------------------------------------------------

def test(
        mydataloader,
        model,
        loss_fn,
        device
):

    size = len(mydataloader.dataset)

    num_batches = len(mydataloader)


    model.eval()

    test_loss = 0

    correct = 0


    with torch.no_grad():

        for X, y in mydataloader:

            X = X.to(device)
            y = y.to(device)


            pred = model(X)


            test_loss += (
                loss_fn(
                    pred,
                    y
                ).item()
            )


            correct += (
                (pred.argmax(1) == y)
                .sum()
                .item()
            )


    test_loss = (
        test_loss /
        num_batches
    )


    accuracy = (
        100 * correct /
        size
    )


    print(
        f"Test Accuracy: "
        f"{accuracy:.2f}%"
    )


    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    # Load dataset
    training_data, test_data = load_data()


    print(
        "Size of training data:",
        len(training_data)
    )


    print(
        "Size of test data:",
        len(test_data)
    )


    # Batch size
    batch_size = 128


    # Training DataLoader
    train_loader = DataLoader(

        training_data,

        batch_size=batch_size,

        shuffle=True
    )


    # Testing DataLoader
    test_loader = DataLoader(

        test_data,

        batch_size=batch_size,

        shuffle=False
    )


    # Device
    device = (

        torch.accelerator.current_accelerator().type

        if torch.accelerator.is_available()

        else torch.device("cpu")
    )


    print(
        f"\nUsing {device} device."
    )


    # Create CNN
    model = CNN().to(device)


    print(model)


    # Loss function
    loss_fn = nn.CrossEntropyLoss()


    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )


    # Train
    train(

        train_loader,

        model,

        loss_fn,

        optimizer,

        device,

        epochs=1
    )


    # Test
    test(

        test_loader,

        model,

        loss_fn,

        device
    )


    print("\nEND")


# ---------------------------------------------------------
# RUN PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":
    main()