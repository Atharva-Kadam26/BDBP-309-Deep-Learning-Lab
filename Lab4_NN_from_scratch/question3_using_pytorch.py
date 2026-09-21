import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



X = np.array([
[0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
],dtype=np.float32)

y = np.array([
    [0],
    [1],
    [1],
    [0]
],dtype=np.float32)

X = torch.tensor(X)
y = torch.tensor(y)

layer_size = [3,6,6,6,6,6,1]

class DeepNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(3,6),
            nn.Sigmoid(),

            nn.Linear(6, 6),
            nn.Sigmoid(),

            nn.Linear(6, 6),
            nn.Sigmoid(),

            nn.Linear(6, 6),
            nn.Sigmoid(),

            nn.Linear(6, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.network(x)

model = DeepNeuralNetwork()

learning_rate = 0.1
iterations = 1000

criterion = nn.BCELoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

#training the neural network
loss_history = []
for iteration in range(iterations):

    #forward pass
    y_pred = model(X)

    #calculate loss
    loss = criterion(y_pred, y)

    #clear previous gradients
    optimizer.zero_grad()

    loss.backward()

    #Update weights
    optimizer.step()

    #Storing loss
    loss_history.append(loss.item())


#Plotting training loss
plt.figure(figsize=(10,6))
plt.plot(
    range(1,iterations+1),
    loss_history,
)

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training loss')

plt.grid(True)
plt.show()


#Making final predictions
with torch.no_grad():
    predictions = model(X)

print("Predictions")
print(predictions)

print("\nRounded Predictions")
print(torch.round(predictions))

#checking gradients
print("\nGradients:")

gradient_values = []

for name, parameter in model.named_parameters():

    if "weight" in name and parameter.grad is not None:
        print(
            name,
            torch.mean(torch.abs(parameter.grad)).item()
        )

        gradient_values.append(
            torch.mean(torch.abs(parameter.grad)).item()
        )


plt.figure(figsize=(10,6))

plt.plot(range(1,len(gradient_values) +1 ),
     gradient_values,
         marker="o")

plt.xlabel('Layer')
plt.ylabel('Mean Absolute Gradient')
plt.title('Gradients Across Layers')

plt.yscale('log')
plt.grid(True)
plt.show()

for i, layer in enumerate(model.network): #goes thru all the layers

    if isinstance(layer, nn.Linear):  #then chooses only the layer which is linear
        weights = layer.weight.detach().numpy()  #detach removes the tensor as we only are visualizing and then we convert the tensor back to numpy
        plt.figure(figsize=(6,6))
        plt.imshow(weights,aspect="auto") #weight matrix to heatmap
        plt.colorbar()

        plt.title("Weights of layer " + str(i// 2+1))

        plt.xlabel("output neurons")
        plt.ylabel("input neurons")

        plt.show()


print("\nFinal Results")

with torch.no_grad(): #only training no gradients needed

    predictions = model(X) #run the 4 XOR inputs through the trained network

for i in range(len(X)):

    print(
        "Input:", X[i].numpy(),
        "Actual:", y[i].item(),
        "Predicted:", predictions[i].item()
    )



