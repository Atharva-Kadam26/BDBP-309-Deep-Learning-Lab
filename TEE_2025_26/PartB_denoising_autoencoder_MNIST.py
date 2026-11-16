
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

# --------------------------------
# 1. Load MNIST Dataset
# --------------------------------

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True
)

test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False
)


# --------------------------------
# 2. Define Denoising Autoencoder
# --------------------------------

class Autoencoder(nn.Module):

    def __init__(self, latent_size):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, latent_size),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = x.view(x.size(0), 784)

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded


# --------------------------------
# 3. Train and Evaluate for Different
#    Latent Space Sizes
# --------------------------------

latent_sizes = [32, 64, 128]

results = {}

for latent_size in latent_sizes:

    print("\nTraining Autoencoder with latent size:", latent_size)

    model = Autoencoder(latent_size)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    # Training
    for epoch in range(epochs):

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            # Original images
            original = images.view(images.size(0), 784)

            # Add random noise
            noisy = original + 0.3 * torch.randn_like(original)

            # Keep pixel values between 0 and 1
            noisy = torch.clamp(noisy, 0, 1)

            # Reconstruct original images from noisy input
            outputs = model(noisy.view(-1, 1, 28, 28))

            loss = criterion(outputs, original)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print("Epoch [{}/{}], Loss: {:.4f}".format(
            epoch + 1, epochs, running_loss / len(train_loader)
        ))

    # --------------------------------
    # 4. Evaluate on Test Dataset
    # --------------------------------

    model.eval()

    test_loss = 0

    with torch.no_grad():

        for images, labels in test_loader:

            original = images.view(images.size(0), 784)

            noisy = original + 0.3 * torch.randn_like(original)

            noisy = torch.clamp(noisy, 0, 1)

            outputs = model(noisy.view(-1, 1, 28, 28))

            loss = criterion(outputs, original)

            test_loss += loss.item()

    test_loss = test_loss / len(test_loader)

    results[latent_size] = (model, test_loss)

    print("Latent Size:", latent_size)
    print("Test Reconstruction Loss:", round(test_loss, 4))



# --------------------------------
# 5. Display Reconstruction Results
# --------------------------------

images, labels = next(iter(test_loader))

original = images[:5].view(5, 784)

noisy = original + 0.3 * torch.randn_like(original)
noisy = torch.clamp(noisy, 0, 1)

plt.figure(figsize=(12, 10))

# Original images
for i in range(5):
    plt.subplot(5, 5, i + 1)
    plt.imshow(original[i].view(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

# Noisy images
for i in range(5):
    plt.subplot(5, 5, i + 6)
    plt.imshow(noisy[i].view(28, 28), cmap="gray")
    plt.title("Noisy")
    plt.axis("off")

# Reconstructed images
row = 2

for latent_size in latent_sizes:

    model = results[latent_size][0]
    model.eval()

    with torch.no_grad():
        reconstructed = model(noisy.view(-1, 1, 28, 28))

    for i in range(5):
        plt.subplot(5, 5, row * 5 + i + 1)
        plt.imshow(reconstructed[i].view(28, 28), cmap="gray")
        plt.title("Latent " + str(latent_size))
        plt.axis("off")

    row += 1

plt.tight_layout()
plt.show()
