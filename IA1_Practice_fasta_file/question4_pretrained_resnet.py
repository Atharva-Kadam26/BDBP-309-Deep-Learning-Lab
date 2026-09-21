import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights
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


def load_data(weights):

    # Get preprocessing directly from torchvision
    transform = weights.transforms()

    dataset_path = (
        r"C:\Datasets\tiny-imagenet-200"
        r"\tiny-imagenet-200"
    )

    # Training dataset
    training_data = datasets.ImageFolder(
        root=os.path.join(
            dataset_path,
            "train"
        ),
        transform=transform
    )

    # Validation/Test dataset
    test_data = TinyImageNetValDataset(
        val_dir=os.path.join(
            dataset_path,
            "val"
        ),
        class_to_idx=training_data.class_to_idx,
        transform=transform
    )

    return training_data, test_data


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

            # Forward pass
            pred = model(X)

            # Calculate loss
            loss = loss_fn(pred, y)

            # Backward pass
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # Calculate loss
            total_loss += loss.item()

            # Calculate correct predictions
            correct += (
                (pred.argmax(1) == y)
                .sum()
                .item()
            )

            # Print progress
            if batch % 50 == 0:

                print(
                    f"Batch {batch}/{len(mydataloader)}"
                )

        # Average loss for epoch
        average_loss = (
            total_loss / len(mydataloader)
        )

        # Training accuracy
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

            # Forward pass
            pred = model(X)

            # Calculate test loss
            test_loss += loss_fn(
                pred,
                y
            ).item()

            # Calculate correct predictions
            correct += (
                (pred.argmax(1) == y)
                .sum()
                .item()
            )

    # Average test loss
    test_loss = (
        test_loss / num_batches
    )

    # Test accuracy
    accuracy = (
        100 * correct / size
    )

    print(
        f"Test Accuracy: "
        f"{accuracy:.4f}%"
    )

    print(
        f"Test Loss: "
        f"{test_loss:.3f}"
    )


def main():

    # Pretrained ResNet18 weights
    weights = ResNet18_Weights.DEFAULT

    # Load datasets
    training_data, test_data = load_data(weights)

    print(
        f"Size of training data: "
        f"{len(training_data)}"
    )

    print(
        f"Size of test data: "
        f"{len(test_data)}"
    )

    # Batch size
    batch_size = 64

    # Training DataLoader
    train_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True
    )

    # Test DataLoader
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
        f"\nUsing {device} device"
    )

    # Load pretrained ResNet18
    model = resnet18(
        weights=weights
    )

    # Freeze pretrained layers
    for param in model.parameters():

        param.requires_grad = False

    # Replace final FC layer
    model.fc = nn.Linear(
        model.fc.in_features,
        200
    )

    # Move model to device
    model = model.to(device)

    print(model)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Train only the final FC layer
    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=1e-3
    )

    # Train
    train(
        train_loader,
        device,
        loss_fn,
        optimizer,
        epochs=1,
        model=model
    )

    # Test
    test(
        test_loader,
        device,
        loss_fn,
        model
    )

    print("\nEND")


if __name__ == "__main__":

    main()