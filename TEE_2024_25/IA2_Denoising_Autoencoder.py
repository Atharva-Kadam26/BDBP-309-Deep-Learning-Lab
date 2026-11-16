# A2. Build a Simple Denoising Autoencoder Network for Image Reconstruction on the MNIST Dataset
# (a) Train the denoising autoencoder and evaluate it by reconstructing some test images.
#
# (b) Experiment with a bottleneck layer of different sizes (e.g., 32, 64, and 128) and discuss how the size of the latent space impacts the reconstruction quality.


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

# 1. Load MNIST dataset

transform = transforms.ToTensor()

train_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_data = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data, batch_size=128, shuffle=True
)

test_loader = DataLoader(
    test_data, batch_size=128, shuffle=False
)


# 2. Define Autoencoder

class Autoencoder(nn.Module):

    def __init__(self, bottleneck_size):

        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),

            nn.Linear(256, bottleneck_size),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 256),
            nn.ReLU(),

            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded


# 3. Experiment with different bottleneck sizes

bottleneck_sizes = [32, 64, 128]

models = {}

for size in bottleneck_sizes:

    print("\nBottleneck size:", size)

    model = Autoencoder(size)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )

    # Training for 10 epochs

    for epoch in range(10):

        model.train()

        total_loss = 0

        for images, labels in train_loader:

            # Flatten images
            images = images.view(-1, 784)

            # Add Gaussian noise
            noisy_images = images + 0.3 * torch.randn_like(images)

            noisy_images = torch.clamp(
                noisy_images, 0, 1
            )

            # Reconstruct clean images
            output = model(noisy_images)

            loss = loss_fn(output, images)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch [{epoch+1}/10], Loss: {average_loss:.4f}"
        )

    models[size] = model


# 4. Evaluate using test images

images, labels = next(iter(test_loader))

images = images[:5]

# Add noise to test images

noisy_images = images + 0.3 * torch.randn_like(images)

noisy_images = torch.clamp(noisy_images, 0, 1)


# 5. Plot original, noisy and reconstructed images

plt.figure(figsize=(12, 10))

# Original images

for i in range(5):

    plt.subplot(5, 5, i+1)

    plt.imshow(images[i].squeeze(), cmap='gray')

    plt.title("Original")

    plt.axis('off')


# Noisy images

for i in range(5):

    plt.subplot(5, 5, i+6)

    plt.imshow(noisy_images[i].squeeze(), cmap='gray')

    plt.title("Noisy")

    plt.axis('off')


# Reconstructed images for each bottleneck size

for row, size in enumerate(bottleneck_sizes):

    model = models[size]

    model.eval()

    with torch.no_grad():

        noisy = noisy_images.view(-1, 784)

        reconstructed = model(noisy)

        reconstructed = reconstructed.view(-1, 1, 28, 28)

    for i in range(5):

        plt.subplot(5, 5, (row+2)*5 + i+1)

        plt.imshow(
            reconstructed[i].squeeze(),
            cmap='gray'
        )

        plt.title("Size " + str(size))

        plt.axis('off')


plt.tight_layout()

plt.show()

print("END")

