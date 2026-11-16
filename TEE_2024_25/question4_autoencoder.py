# A4 Create an Autoencoder class with Linear-ReLU-Linear-ReLU for both encoder and decoder. Use
# this class to generate a latent representation for an input, X. X is a 734 dimensional vector. Hidden
# dimension size is 128 and the latent dimension size is 32. You just need to implement a forward pass
# trainingand no need to train the network. Generate a single input sample of size 784-dim vector at
# random and generate the latent representation for this input vector



import torch
import torch.nn as nn

# Create Autoencoder class
class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.ReLU()
        )

    def forward(self, x):

        # Get latent representation
        latent = self.encoder(x)

        # Reconstruct input
        output = self.decoder(latent)

        return latent, output


# Create model
model = Autoencoder()

# Generate random input of size 784
x = torch.rand(1, 784)

# Forward pass
latent, output = model(x)

print("Input shape:", x.shape)
print("Latent representation:", latent)
print("Latent shape:", latent.shape)
print("Output shape:", output.shape)
