# A3. Implement a forward pass of a single layer neural network in Python. Input features 3, output = 1
# (scalar value). You are allowed to use pyTorch functions.


import torch
import torch.nn as nn

# Create a single-layer neural network
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # 3 input features and 1 output
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        output = self.fc(x)
        return output

# Create model
model = NeuralNetwork()

# Give input with 3 features
x = torch.tensor([[1.0, 2.0, 3.0]])

# Perform forward pass
y = model(x)

print("Input:", x)
print("Scalar Output:", y.item())
print("Output shape:", y.shape)
