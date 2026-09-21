# import numpy as np
# import matplotlib.pyplot as plt
#
# #Generate 100 equally spaced values btwn -10 and 10
# z = np.linspace(-10,10,100)
#
# def sigmoid(z):
#     return 1/(1+np.exp(-z))
#
# def sigmoid_derivative(z):
#     s = sigmoid(z)
#     return s*(1-s)
#
# def tanh(z):
#     return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
#
# def tanh_derivative(z):
#     t = tanh(z)
#     return 1 - t**2
#
#
# def relu(z):
#     return np.maximum(0,z)
#
# def relu_derivative(z):
#     return np.where(z>0 , 1,0)
#
# sigmoid_output = sigmoid(z)
# sigmoid_derivative = sigmoid_derivative(z)
#
# tanh_output = tanh(z)
# tanh_derivative = tanh_derivative(z)
#
# relu_output = relu(z)
# relu_derivative = relu_derivative(z)
#
# plt.figure(figsize=(10,6))
#
# plt.plot(z,sigmoid_output,label="sigmoid")
# plt.plot(z,sigmoid_derivative,label="sigmoid_derv")
#
# plt.xlabel("z")
# plt.ylabel("output")
# plt.title("Sigmoid nd its derv")
# plt.legend()
# plt.grid()
# plt.show()
#
#
# plt.figure(figsize=(10,6))
# plt.plot(z,tanh_output,label="tanh")
# plt.plot(z,tanh_derivative,label="tanh_derv")
# plt.xlabel("z")
# plt.ylabel("output")
# plt.title("Tanh nd its derv")
# plt.legend()
# plt.grid()
# plt.show()
#
#
#
#
# plt.figure(figsize=(10,6))
# plt.plot(z,relu_output,label="relu")
# plt.plot(z,relu_derivative,label="relu_derv")
# plt.title("ReLU nd its derv")
# plt.legend()
# plt.grid()
# plt.show()
#
#
#
import numpy as np
import matplotlib.pyplot as plt

z =np.linspace(-10,10,100)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derv(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))

def tanh_derv(z):
    t = tanh(z)
    return 1 - t**2

def relu(z):
    return np.maximum(0, z)

def relu_derv(z):
    return np.where(z>0,1,0)


sigmoid_out = sigmoid(z)
sigmoid_derv = sigmoid_derv(z)

tanh_out = tanh(z)
tanh_derv = tanh_derv(z)

relu_out = relu(z)
relu_derv = relu_derv(z)

plt.figure(figsize=(10,6))
plt.plot(z,sigmoid_out,label="sigmoid")
plt.plot(z,sigmoid_derv,label="sigmoid_derv")
plt.title("Sigmoid and its derivative")
plt.legend()
plt.grid(True)
plt.show()