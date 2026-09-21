import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from PIL import Image
import os


class TinyImageNetValDataset(Dataset):

    def __init__(self, val_dir, class_to_idx, transform=None):

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


def load_data():

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(
            torch.float32,
            scale=True
        )
    ])

    dataset_path = (
        "/home/ibab/Downloads/tiny-imagenet-200"
    )

    training_data = datasets.ImageFolder(
        root=os.path.join(
            dataset_path,
            "train"
        ),
        transform=transform
    )

    test_data = TinyImageNetValDataset(
        val_dir=os.path.join(
            dataset_path,
            "val"
        ),
        class_to_idx=training_data.class_to_idx,
        transform=transform
    )

    return training_data, test_data


class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv_layers = nn.Sequential(

            # Convolution Layer 1
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Convolution Layer 2
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Convolution Layer 3
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Linear(
                128,
                200
            )
        )

    def forward(self, x):

        x = self.conv_layers(x)

        x = self.classifier(x)

        return x


def train(
        mydataloader,
        device,
        loss_fn,
        optimizer,
        epochs,
        model
):

    size = len(mydataloader.dataset)

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        correct = 0

        for batch, (X, y) in enumerate(mydataloader):

            X = X.to(device)
            y = y.to(device)

            pred = model(X)

            loss = loss_fn(pred, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            correct += (
                (pred.argmax(1) == y)
                .sum()
                .item()
            )

        average_loss = (
            total_loss / len(mydataloader)
        )

        accuracy = (
            100 * correct / size
        )

        print(
            f"Epoch: {epoch + 1}/{epochs} "
            f"Train Loss: {average_loss:.3f} "
            f"Training Accuracy: {accuracy:.4f}%"
        )


def test(
        mydataloader,
        device,
        loss_fn,
        model
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

            test_loss += loss_fn(
                pred,
                y
            ).item()

            correct += (
                (pred.argmax(1) == y)
                .sum()
                .item()
            )

    test_loss = (
        test_loss / num_batches
    )

    accuracy = (
        100 * correct / size
    )

    print(
        f"Test Accuracy: {accuracy:.4f}%"
    )

    print(
        f"Test Loss: {test_loss:.3f}"
    )


def main():

    training_data, test_data = load_data()

    print(
        f"Size of training data: "
        f"{len(training_data)}"
    )

    print(
        f"Size of test data: "
        f"{len(test_data)}"
    )

    batch_size = 128

    train_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else torch.device("cpu")
    )

    print(
        f"\nUsing {device} device"
    )

    model = CNN()

    model = model.to(device)

    print(model)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    train(
        train_loader,
        device,
        loss_fn,
        optimizer,
        epochs=10,
        model=model
    )

    test(
        test_loader,
        device,
        loss_fn,
        model
    )

    print("\nEND")


if __name__ == "__main__":
    main()