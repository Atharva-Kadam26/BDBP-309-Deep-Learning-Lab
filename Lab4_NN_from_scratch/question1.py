import numpy as np
import matplotlib.pyplot as plt

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

#initializing weights and biases
np.random.seed(42)

#z = XW + b
W = np.random.randn(3,1)  #matrix of random numbers with 3 rows and 1 column
b = 0

print("Initial weights:")
print(W)

print("Initial bias:")
print(b)

def sigmoid(z):
    return 1/(1+np.exp(-z))


#loss function
#l = -1/m ∑ [ylog(^y) + (1-y)log(1-^y)]
def cross_entropy_loss(y,y_pred):
    l = np.mean(y * np.log(y_pred + 1e-8) + (1-y) * np.log(1- y_pred + 1e-8))
    #np.mean was used as the loss is being calculated for all 4 training samples
    #we added 1e-8 to avoid log(0) so 1e-8 = 0.00000001
    return -l

learning_rate = 0.1
iterations = 1000

losses = []

#training_phase:
for i in range(iterations):

    #Forward Pass
    z = np.dot(X,W) + b
    #z = x1w1 + x2w2 + x3w3 + b

    y_pred = sigmoid(z)

    #Loss:
    loss = cross_entropy_loss(y,y_pred)
    losses.append(loss)

    #Backward Pass
    dz = y_pred - y

    dW = np.dot(X.T,dz) / len(X)   #gradients of the weights
    #dW = X^T dz/ m , m = num of training samples

    # X = 4 × 3
    # X.T = 3 × 4
    # dz = 4 × 1
    #Therefore,
    # X.T × dz
    # (3 × 4) × (4 × 1)
    # = 3 × 1

    db = np.mean(dz)    #gradients of the bias
    #db = 1/m ∑ dz

    #since the bias is a single value we get one gradient value

    #Updating the weights
    #This is gradient descent

    # W_new = W_old - alpha * dW  , where dW is the

    W = W - learning_rate * dW

    #Updating the bias
    b = b - learning_rate * db
    #b_new = b_old - alpha * db

    #loss:
    if i % 100 == 0:
        print("Iteration:", i , "Loss:",loss)


#Testing Phase:

#We perform forward pass one final time using the trained weights
z = np.dot(X,W) + b
y_pred = sigmoid(z)

predictions = (y_pred >= 0.5).astype(int)
#we convert probabilities into classes, if the prob. is > 0.5 = 1 and if it is < 0.5 =0

print("\n Final weights:")
print(W)

print("\n Final bias:")
print(b)

print("\nPredicted probabilities:")
print(y_pred)

print("\nPredicted classes:")
print(predictions)

print("\nActual classes:")
print(y)


#Plotting the loss

plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()



