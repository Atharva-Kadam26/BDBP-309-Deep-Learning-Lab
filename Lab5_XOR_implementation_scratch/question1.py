"""Implement a 2-layer (input layer, hidden layer and output layer) neural network from scratch for the XOR operation. This includes implementing forward and backward passes from scratch. The truth table for XOR is given by"""

import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z,-500,500)))

np.random.seed(42)

input_size = 2
hidden_size = 4
output_size = 1

#Input to Hidden
W1 = np.random.randn(input_size,hidden_size)
b1 = np.zeros((1, hidden_size))

#hidden to output
W2 = np.random.randn(hidden_size,output_size)
b2 = np.zeros((1, output_size))

def binary_cross_entropy(y,y_pred):
    epsilon = 1e-9

    loss = -np.mean(
        y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

    return loss

learning_rate = 1
epochs = 1000

for epoch in range(epochs):

    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)

    Z2 = A1 @ W2 + b2
    y_pred = sigmoid(Z2)

    loss = binary_cross_entropy(y,y_pred)

    #Output layer
    dZ2 = y_pred - y

    dW2 = (A1.T @ dZ2) / len(X)
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    # Hidden layer
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * A1 * (1 - A1)

    dW1 = (X.T @ dZ1) / len(X)
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    #update weights

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    #Print loss
    if epoch % 100 == 0 :
        print(f"Epoch {epoch}, Loss = {loss:.6f}")

print("\nFinal Results")

for i in range(len(X)):
    print(
        f"Input: {X[i]} "
        f"Actual: {y[i, 0]} "
        f"Predicted: {y_pred[i, 0]:.6f}"
    )


