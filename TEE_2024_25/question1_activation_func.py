# A1 Implement the following functions in Python from scratch. Do not use any library functions, You are
# allowed to use numpy and matplotlib. Generate 100 equally spaced values between -10 and IO, Call
# this list as z Implement the following functions and its derivative. Use z as input and plot both the
# function outputs and its derivative outputs.
# Sigmoid
# Tanh
# ReLU (Rectified Linear Unit)


import numpy as np
import matplotlib.pyplot as plt

# Generate 100 equally spaced values
z = np.linspace(-10, 10, 100)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of sigmoid
def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Plot sigmoid
plt.figure(figsize=(6, 4))
plt.plot(z, sigmoid(z))
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.grid()
plt.show()

# Plot sigmoid derivative
plt.figure(figsize=(6, 4))
plt.plot(z, sigmoid_derivative(z))
plt.title("Sigmoid Derivative")
plt.xlabel("z")
plt.ylabel("Derivative")
plt.grid()
plt.show()


# Tanh function
def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

# Derivative of tanh
def tanh_derivative(z):
    return 1 - tanh(z)**2

# Plot tanh
plt.figure(figsize=(6, 4))
plt.plot(z, tanh(z))
plt.title("Tanh Function")
plt.xlabel("z")
plt.ylabel("Tanh(z)")
plt.grid()
plt.show()

# Plot tanh derivative
plt.figure(figsize=(6, 4))
plt.plot(z, tanh_derivative(z))
plt.title("Tanh Derivative")
plt.xlabel("z")
plt.ylabel("Derivative")
plt.grid()
plt.show()


# ReLU function
def relu(z):
    return np.maximum(0, z)

# Derivative of ReLU
def relu_derivative(z):
    return np.where(z > 0, 1, 0)

# Plot ReLU
plt.figure(figsize=(6, 4))
plt.plot(z, relu(z))
plt.title("ReLU Function")
plt.xlabel("z")
plt.ylabel("ReLU(z)")
plt.grid()
plt.show()

# Plot ReLU derivative
plt.figure(figsize=(6, 4))
plt.plot(z, relu_derivative(z))
plt.title("ReLU Derivative")
plt.xlabel("z")
plt.ylabel("Derivative")
plt.grid()
plt.show()




