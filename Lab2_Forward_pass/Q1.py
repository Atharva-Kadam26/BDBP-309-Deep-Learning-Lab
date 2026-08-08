# Consider the following two networks. W is a matrix, x is a vector, z is a vector, and a is a
# vector. y^ is a scalar and a final prediction. Initialize x, w randomly, z is a dot product of x
# and w, a is ReLU(z). Initialize X and W randomly. Every neuron has a bias term.


import numpy as np

# x=[1,2,3,4]
w=[0.1,0.2,0.3,0.4]
B = 0.15

X = np.random.uniform(0,4,4)
w = np.random.uniform(0,4,4)

print("random weights are:",w)
print("random values of x are:",X)


def forward_pass(X,w):
    z = (np.dot(X,w)) + B
    a = 1 / (1 + np.exp(-z))
    return a,z


def main():
    a,z = forward_pass(X,w)
    print("The function's value is:",z)
    print("The ReLU's value is:",a)

if __name__ == "__main__":
    main()


    
