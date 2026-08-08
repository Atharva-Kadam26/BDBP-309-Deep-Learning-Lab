# from random import randint
#
# import numpy as np
#
# X = np.random.uniform(0,4,4)
# # w = np.random.uniform(0,4,4)
# print(X)
#
# # HL = int(input("Enter the number of Hidden Layers:"))
# # for a in range(HL):
# #     num_neu_per_lay = int(input(f"Enter the number of Neurons required in Layer{a}"))
#
# num_layers = []
# HL = int(input("Enter the number of Hidden Layers:"))
# for a in range(HL):
#     layers=[]
#     num_neu_per_lay = int(input(f"Enter the number of Neurons required in Layer{a+1}:"))
#     layers.append(num_neu_per_lay)
#     num_layers.append(layers)
# print(num_layers)
#
# # for i in range(X.shape[0]):
# #     #Z = w1x1 + w2x2 + B
# #     for j in range(len(num_layers)):
# #         for k in range(j):
# #             W = 2
# #             w_x = X[i] * W
# W = [[[0.1,-0.1,0.2],[-1.1,0.4,1.1]] , [[0.2,0.1,-0.2],[0.3,-0.1,-0.1]] , [[0.3,-0.3]]]
#
# # W_2 = [[0.2,0.1,-0.2],
# #        [0.3,-0.1,-0.1]]
# #
# # W_3 = [[0.3,-0.3]]
#
#
# X = [0.3,-1.2]
#
# for i in W:
#     Z = np.dot(X,i)
#
#
#
# print(z)
#
# # for i in range(len(X)):
# #     for j in range(len(W_1[i])):



#
"""Implement forward pass for the above two networks. Print activation values for each
neuron at each layer. Print the loss value (y^)."""

import numpy as np

np.random.seed(42)

def sigmoid_func(z):
    return 1/(1+np.exp(-z))
#
# layer_structure = [4,3,2,1]

num_input = int(input("How many input features?"))
num_layers = int(input("Please enter the number of layers:"))

layer_structure = [num_input]

# Weights = []
# Bias =[]

for i in range(num_layers):
    neurons = int(input(f"Please enter the number of neurons in Layer {i+1} :"))
    layer_structure.append(neurons)

# print(layer_structure)

weights = []
biases = []

for i in range(len(layer_structure)-1):
    neurons_in = layer_structure[i]
    neurons_out = layer_structure[i+1]
    weights.append(np.random.randn(neurons_out, neurons_in)*0.5)
    biases.append(np.zeros((neurons_out,1)))


print(f"\nEnter {num_input} input values")
x_values = []
for i in range(num_input):
    value = float(input(f"x {i +1}: "))
    x_values.append(value)

X = np.array(x_values).reshape(-1,1) #turning the values into vector

a = X
for i in range(len(weights)):
    z = weights[i] @ a + biases[i]   # "@" is used for matrix multiplication
    a = sigmoid_func(z)         #activation function
    print(f"Layer {i +1} : z = {z.flatten()} , activation = {a.flatten()}")


y = a
print("\nFinal output: ",y.flatten())


