# B4. Train a GAN model for a simulated dataset of random noise samples that resemble a ID Gaussian
# distribution. The generator should learn to produce data samples that resemble real samples from a
# Gaussian distribution, and the discriminator will learn to distinguish between real and fake samples,
# You can use the following assumptions and hyperparameters in your implementation.
# The generator should have Linear - ReLLl - Linear - ReLlJ - Linear layers. The discriminator should
# have Linear - LeakyRelu - Linear - LeakyRelu - Linear - Sigmoid.
# latent dim
# = 10 # Dimension of the latent vector (input to the generator)
# hidden_dim
# 128 # Hidden layer size
# output_dim = 1 # Output dimension (ID data point for simplicity)
# batch size = 64
# num_epochs 5000
# learning_rate = 0.0002
# Plot the real data and the generated data for visualization


import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Hyperparameters

latent_dim = 10
hidden_dim = 128
output_dim = 1
batch_size = 64
num_epochs = 5000
learning_rate = 0.0002


# Generate real Gaussian data

real_data = torch.randn(10000, output_dim)


# Generator

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# Discriminator

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Initialize models

generator = Generator()
discriminator = Discriminator()


# Loss function and optimizers

loss_fn = nn.BCELoss()

optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=learning_rate
)

optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=learning_rate
)


# Training the GAN

for epoch in range(num_epochs):

    # Select real samples
    indices = torch.randint(0, len(real_data), (batch_size,))
    real_samples = real_data[indices]

    # Create random noise
    noise = torch.randn(batch_size, latent_dim)

    # Generate fake samples
    fake_samples = generator(noise)

    # Labels
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    # Train Discriminator

    real_output = discriminator(real_samples)
    fake_output = discriminator(fake_samples.detach())

    loss_real = loss_fn(real_output, real_labels)
    loss_fake = loss_fn(fake_output, fake_labels)

    loss_D = loss_real + loss_fake

    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()

    # Train Generator

    noise = torch.randn(batch_size, latent_dim)
    fake_samples = generator(noise)

    output = discriminator(fake_samples)

    loss_G = loss_fn(output, real_labels)

    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    # Print loss every 500 epochs

    if (epoch + 1) % 500 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Discriminator Loss: {loss_D.item():.4f}, "
            f"Generator Loss: {loss_G.item():.4f}"
        )


# Generate samples after training

generator.eval()

with torch.no_grad():
    noise = torch.randn(10000, latent_dim)
    generated_data = generator(noise)


# Plot real and generated data

plt.figure(figsize=(8, 5))

plt.hist(
    real_data.numpy(),
    bins=50,
    density=True,
    alpha=0.5,
    label="Real Gaussian Data"
)

plt.hist(
    generated_data.numpy(),
    bins=50,
    density=True,
    alpha=0.5,
    label="Generated Data"
)

plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Real vs Generated Gaussian Distribution")
plt.legend()

plt.show()

