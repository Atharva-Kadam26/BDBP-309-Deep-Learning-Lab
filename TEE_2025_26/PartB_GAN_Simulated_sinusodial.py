
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --------------------------------
# 1. Generate Sinusoidal Dataset
# --------------------------------

x = np.linspace(0, 2 * np.pi, 1024)

y = np.sin(x)

real_data = np.column_stack((x, y))

real_data = torch.tensor(real_data, dtype=torch.float32)

print("Dataset shape:", real_data.shape)

# Plot original data
plt.figure(figsize=(8, 4))
plt.scatter(real_data[:, 0], real_data[:, 1], s=10)
plt.title("Original Sinusoidal Data")
plt.xlabel("X")
plt.ylabel("Sin(X)")
plt.show()


# --------------------------------
# 2. Define Generator
# --------------------------------

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)


# --------------------------------
# 3. Define Discriminator
# --------------------------------

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# --------------------------------
# 4. Initialize GAN
# --------------------------------

generator = Generator()
discriminator = Discriminator()

criterion = nn.BCELoss()

g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

epochs = 300
batch_size = 64

# --------------------------------
# 5. Train GAN
# --------------------------------

for epoch in range(epochs):

    g_loss_total = 0
    d_loss_total = 0

    indices = torch.randperm(len(real_data))

    for i in range(0, len(real_data), batch_size):

        batch_indices = indices[i:i + batch_size]

        real_samples = real_data[batch_indices]

        current_batch = real_samples.size(0)

        real_labels = torch.ones(current_batch, 1)
        fake_labels = torch.zeros(current_batch, 1)

        # Train Discriminator
        noise = torch.randn(current_batch, 2)

        fake_samples = generator(noise)

        real_output = discriminator(real_samples)
        fake_output = discriminator(fake_samples.detach())

        real_loss = criterion(real_output, real_labels)
        fake_loss = criterion(fake_output, fake_labels)

        d_loss = real_loss + fake_loss

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        noise = torch.randn(current_batch, 2)

        fake_samples = generator(noise)

        output = discriminator(fake_samples)

        g_loss = criterion(output, real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()

    if (epoch + 1) % 10 == 0:

        print(
            "Epoch [{}/300], D Loss: {:.4f}, G Loss: {:.4f}".format(
                epoch + 1,
                d_loss_total / len(range(0, len(real_data), batch_size)),
                g_loss_total / len(range(0, len(real_data), batch_size))
            )
        )


# --------------------------------
# 6. Generate 100 New Samples
# --------------------------------

generator.eval()

with torch.no_grad():

    noise = torch.randn(100, 2)

    generated_samples = generator(noise)

print("\nGenerated samples shape:", generated_samples.shape)


# --------------------------------
# 7. Plot Generated Samples
# --------------------------------

generated_samples = generated_samples.numpy()

plt.figure(figsize=(8, 5))

plt.scatter(
    real_data[:, 0].numpy(),
    real_data[:, 1].numpy(),
    s=10,
    label="Real Data"
)

plt.scatter(
    generated_samples[:, 0],
    generated_samples[:, 1],
    s=20,
    label="Generated Data"
)

plt.title("GAN Generated Sinusoidal Samples")
plt.xlabel("X")
plt.ylabel("Sin(X)")
plt.legend()
plt.show()
