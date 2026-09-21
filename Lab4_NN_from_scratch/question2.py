import numpy as np
import matplotlib.pyplot as plt

#Creating a Deep Neural Network

#input
X = np.array([
    [0,0,1],
    [1,1,1],
    [1,0,1],
    [0,1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

layer_sizes = [3,6,6,6,6,1]

#weights for 4 layers
np.random.seed(42)

num_layers = len(layer_sizes)-1 #number of weight matrices needed

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_derivative(z):
    a = sigmoid(z)
    return a * (1 - a) #a = sigmoid(z)

def cross_entropy_loss(y,y_pred):
    loss = np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1- y_pred + 1e-8))
    return - loss

weights = []
biases = []

for i in range(num_layers):
    weights.append(np.random.randn(layer_sizes[i],layer_sizes[i+1]) * 0.5 )
    # so for layer_sizes it will start with (3,6) and go on and we add 0.5 to scale down the values
    biases.append(np.zeros((1,layer_sizes[i+1])) * 0.5)


def forward(X,weights,biases):
    activations = [X]  #holding the output of every layer including I/P layer
    zs = []


    for i in range(num_layers):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        a = sigmoid(z)
        zs.append(z)
        activations.append(a)

    return activations, zs

def backward(X,activations,y,weights):
    m = X.shape[0] #m = 4

    dW = [None] * num_layers
    db = [None] * num_layers

    #output layer gradient
    dz = activations[-1] - y   #dz = y^ - y

    #a5 = prediction = y^

    for i in range(num_layers -1,-1,-1): #backward goes W5 → W4 → W3 → W2 → W1

        #Gradients of weights
        dW[i] = np.dot(activations[i].T,dz)/m #dWi = A^T i-1 dzi / m
        db[i] = np.sum(dz ,axis = 0,keepdims = True) /m
        # dz = loss of all samples , axis = 0 means add these vvertically across samples

        if i > 0:
            da = np.dot(dz,weights[i].T)  #da i-1 = dzi Wi ^T
            dz = da * activations[i] * (1 - activations[i])  #chain rule dz = da x sig_derivative

    return dW,db


# activations,zs = forward(X,weights,biases)
#
# dW,db = backward(X,activations,y,weights)

#Training

learning_rate = 0.1
iterations = 1000

loss_history = []

# Store gradient magnitude for every layer at every iteration
gradient_history = [[] for _ in range(num_layers)]

for iteration in range(iterations):

    #Forward pass
    activations, zs = forward(X,weights,biases)

    #Prediction
    y_pred = activations[-1]

    #Calculate loss
    loss = cross_entropy_loss(y,y_pred)

    loss_history.append(loss)


    #Backward pass
    dW,db = backward(X,activations,y,weights)

    #store gradient magnitude
    for i in range(num_layers):
        magnitude = np.mean(np.abs(dW[i]))
        gradient_history[i].append(magnitude)


    #update weights and biases
    for i in range(num_layers):
        weights[i] = weights[i] - learning_rate * dW[i]
        biases[i] = biases[i] - learning_rate * db[i]

#plot gradient magnitude throughout training

plt.figure(figsize=(10,6))

for i in range(num_layers):
    plt.plot(range(1,iterations +1),gradient_history[i],label = "Layer " + str(i +1))

plt.xlabel("Iterations")
plt.ylabel("Mean absolute gradient")
plt.title("Gradient Magnitude during Training")

plt.yscale("log")

plt.legend()
plt.grid(True)

plt.show()




plt.figure(figsize = (8,5))

plt.plot(range(1,iterations + 1), loss_history)

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss over 1000 iterations")

plt.grid(True)

plt.show()

activations,zs = forward(X,weights,biases)
y_pred = activations[-1]

print("\nFinal Predictions:")
print(y_pred)

print("\nActual Values:")
print(y)



for i in range(num_layers):
    print("\nLayer", i +1 )
    print("Weights:")
    print(weights[i])

    print("Gradients:")
    print(dW[i])

    print("Biases:")
    print(biases[i])

    print("Bias gradients:")
    print(db[i])


#visualizing weights

fig,axes = plt.subplots(1,num_layers ,figsize=(15,4))

for i in range(num_layers):

    im = axes[i].imshow(weights[i],aspect='auto')

    axes[i].set_title("Layer" + str(i + 1) + "Weights")

    axes[i].set_xlabel("Output neurons")
    axes[i].set_ylabel("Input neurons")

    plt.colorbar(im,ax = axes[i])

plt.tight_layout()
plt.show()

#visualizing weight gradients

fig,axes = plt.subplots(1,num_layers,figsize=(15,4))

for i in range(num_layers):

    im = axes[i].imshow(dW[i],aspect='auto')

    axes[i].set_title("Layer" + str(i + 1) + "Gradients")
    axes[i].set_xlabel("Output neurons")
    axes[i].set_ylabel("Input neurons")

    plt.colorbar(im,ax = axes[i])

plt.tight_layout()
plt.show()

#calculating gradient magnitude for each layer

gradient_magnitude = []

for i in range(num_layers):

    magnitude = np.mean(np.abs(dW[i]))

    gradient_magnitude.append(magnitude)

#Plotting gradient magnitude

plt.figure(figsize =(8,5))

plt.plot(
    range(1,num_layers + 1),
    gradient_magnitude,
    marker = 'o',
)

plt.xlabel("Layer")
plt.ylabel("Mean Absolute Gradient")
plt.title("Gradient Magnitude Across Layers")

plt.xticks(range(1, num_layers + 1))

plt.grid(True)

plt.show()





