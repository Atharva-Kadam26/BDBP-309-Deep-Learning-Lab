# Implement the following functions in Python from scratch. Do not use any library
# functions. You are allowed to use numpy and matplotlib. Generate 100 equally spaced
# values between -10 and 10. Call this list as z. Implement the following functions and its
# derivative. Use class notes to find the expression for these functions. Use z as input and
# plot both the function outputs and its derivative outputs. Upload your code into Github
# and share it with me.
# a.​ Sigmoid
# b.​ Tanh
# c.​ ReLU (Rectified Linear Unit)
# d.​ Leaky ReLU
# e.​ Softmax (no need for visualization)

import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-10,10,100)

def sigmoid(x):
    sig = 1/(1 + np.exp(-x))
    return sig

def sigmoid_derivative(x):
    g_x = sigmoid(x) * (1-sigmoid(x))
    return g_x


def tanh(x):
    tan_f = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    return tan_f

def tanh_derivative(x):
    der_func = 1 - (tanh(x))**2
    return der_func

def ReLU(x):
    return np.maximum(0, x)


def ReLU_derivative(x):
    return np.where((x > 0), 1, 0)

def leaky_relu(x, alpha=0.01):
    # if x > 0:
    #     return x
    # else:
    #     return alpha * x
    return np.where(x > 0, x, alpha*x)


def leaky_relu_derivative(x, alpha=0.01):
    # if z > 0:
    #     return 1
    # else:
    #     return alpha
    return np.where(x > 0, 1, alpha)


def softmax(x):
    h = x - np.max(x)
    exp_x = np.exp(h)
    return exp_x / np.sum(exp_x)

def softmax_derivative(x):
    return np.sum(np.exp(x))

def main():
    print("Numbers from -10 to 10 are:",z)
    print("\n")
    sig = sigmoid(z)

    sig_derivative = sigmoid_derivative(z)


    plt.plot(z,sig)
    plt.title('Sigmoid Function')
    plt.xlabel('z')
    plt.ylabel('Output')
    plt.show()


    plt.plot(z,sig_derivative)
    plt.title('Sigmoid Derivative Function')
    plt.xlabel('z')
    plt.ylabel('Output(Derivative)')
    plt.show()


    relu = ReLU(z)
    relu_der = ReLU_derivative(z)
    plt.plot(z,relu)
    plt.title('ReLU Function')
    plt.xlabel('z')
    plt.ylabel('Output')
    plt.show()


    plt.plot(z,relu_der)
    plt.title('ReLU Derivative Function')
    plt.xlabel('z')
    plt.ylabel('Output(Derivative)')
    plt.show()



    #leaky ReLU
    print("Leaky ReLU Function")
    print(leaky_relu(z))
    print("Leaky ReLU Derivative Function")
    print(leaky_relu_derivative(z))
    plt.plot(z,leaky_relu(z))
    plt.title('Leaky ReLU Function')
    plt.xlabel('z')
    plt.ylabel('Output')
    plt.show()
    plt.plot(z,leaky_relu_derivative(z))
    plt.title('Leaky ReLU Derivative Function')
    plt.xlabel('z')
    plt.ylabel('Output(Derivative)')
    plt.show()

if __name__ == '__main__':
    main()