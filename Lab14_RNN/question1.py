# # 1.​ Implement vanilla RNN from scratch -
# # https://medium.com/@thisislong/building-a-recurrent-neural-network-from-scratch-ba9b2
# # 7a42856
#



# 1.​ Implement vanilla RNN from scratch -
# https://medium.com/@thisislong/building-a-recurrent-neural-network-from-scratch-ba9b2
# 7a42856

import numpy as np

#Creating Dataset
np.random.seed(42)

X = np.random.randn(2000,3,1) #generates 2000 samples, each containing 3 time steps and 1 feature per time step
y = np.sum(X,axis=1)  #sums the three values in every sample to create the target output.


#Defining RNN Dimensions
input_size = 1
hidden_size = 8
output_size = 1

#initialising weights and biases
Wxh = np.random.randn(hidden_size, input_size) * 0.1
Whh = np.random.randn(hidden_size, hidden_size) * 0.1
Why = np.random.randn(output_size, hidden_size) * 0.1

#biases are initialised to zeroes
bh = np.zeros((hidden_size,1))
by = np.zeros((output_size,1))

#Setting the learning rate
learning_rate = 0.01
epochs = 100

#Training the RNN
for epoch in range(epochs):
    total_loss = 0

    for i in range(len(X)):

        #forward pass
        h = np.zeros((hidden_size,1))
        hidden_states = [h]

        for t in range(3):
            x_t = X[i,t].reshape(input_size,1)
            h = np.tanh(Wxh @ x_t + Whh @ h + bh)
            # h_t=tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
            hidden_states.append(h)

        predictions = Why @ h + by
        target = y[i].reshape(output_size,1)

        #calculate loss
        #L = 1/2 ( y^ - y)^2
        error = predictions - target
        loss = 0.5 * np.sum(error ** 2)
        total_loss += loss

        # Backpropagation through time
        # These variables store the gradients of the loss with respect to each parameter.

        dWxh = np.zeros_like(Wxh)
        dWhh = np.zeros_like(Whh)
        dWhy = np.zeros_like(Why)
        dbh = np.zeros_like(bh)
        dby = np.zeros_like(by)

        dWhy += error @ h.T   #dL/dy^ = y^ - y
        dby += error

        dh = Why.T @ error

        for t in reversed(range(1, 4)):
            h_t = hidden_states[t]
            h_prev = hidden_states[t - 1]
            x_t = X[i, t - 1].reshape(input_size, 1)

            dh_raw = dh * (1 - h_t ** 2)

            dWxh += dh_raw @ x_t.T # it accumulates the input to hidden weight gradient
            dWhh += dh_raw @ h_prev.T  #accumulates the recurrent weight gradient
            dbh += dh_raw #accumulates the hidden bias gradient

            dh = Whh.T @ dh_raw #passes the error backward to the preceding hidden state
         # d/dz tanh(z) = 1 - tanh^2(z)

        # Gradient clipping
        for grad in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(grad, -1, 1, out=grad)

        # Update weights and biases
        Wxh -= learning_rate * dWxh
        Whh -= learning_rate * dWhh
        Why -= learning_rate * dWhy
        bh -= learning_rate * dbh
        by -= learning_rate * dby

    if (epoch + 1) % 10 == 0:
        average_loss = total_loss / len(X)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.10f}")


# Check predictions on training samples
print("\nTraining predictions:")

for i in range(5):
    h = np.zeros((hidden_size, 1))

    for t in range(3):
        x_t = X[i, t].reshape(input_size, 1)
        h = np.tanh(Wxh @ x_t + Whh @ h + bh)

    prediction = Why @ h + by

    print(
        f"Actual: {y[i].item():.4f}, "
        f"Predicted: {prediction.item():.4f}"
    )


# Test the trained RNN
test_sequence = np.array([[0.2], [0.4], [0.3]])
h = np.zeros((hidden_size, 1))

for t in range(3):
        x_t = test_sequence[t].reshape(input_size, 1)
        h = np.tanh(Wxh @ x_t + Whh @ h + bh)

prediction = Why @ h + by

print("\nTest sequence:", test_sequence.flatten())
print("Predicted sum:", prediction.item())
print("Actual sum:", np.sum(test_sequence))


# import numpy as np
#
# #Creating Dataset
# np.random.seed(42)
#
# X = np.random.randn(2000,3,1) #generates 200 samples, each containing 3 time steps and 1 feature per time step
# y = np.sum(X,axis=1)  #sums the three values in every sample to create the target output.
#
#
# #Defining RNN Dimensions
# input_size = 1
# hidden_size = 8
# output_size = 1
#
# #initialising weights and biases
# Wxh = np.random.randn(hidden_size, input_size) * 0.1
# Whh = np.random.randn(hidden_size, hidden_size) * 0.1
# Why = np.random.randn(output_size, hidden_size) * 0.1
#
# #biases are initialised to zeroes
# bh = np.zeros((hidden_size,1))
# by = np.zeros((output_size,1))
#
# #Setting the learning rate
# learning_rate = 0.01
# epochs = 100
#
# #Training the RNN
# for epoch in range(epochs):
#     total_loss = 0
#
#     for i in range(len(X)):
#
#         #forward pass
#         h = np.zeros((hidden_size,1))
#         hidden_states = [h]
#
#         for t in range(3):
#             x_t = X[i,t].reshape(input_size,1)
#             h = np.tanh(Wxh @ x_t + Whh @ h + bh)
#             # h_t=tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
#             hidden_states.append(h)
#
#         predictions = Why @ h + by
#         target = y[1].reshape(output_size,1)
#
#         #calculate loss
#         #L = 1/2 ( y^ - y)^2
#         error = predictions - target
#         loss = 0.5 * np.sum(error ** 2)
#         total_loss += loss
#
#         # Backpropagation through time
#         # These variables store the gradients of the loss with respect to each parameter.
#
#         dWxh = np.zeros_like(Wxh)
#         dWhh = np.zeros_like(Whh)
#         dWhy = np.zeros_like(Why)
#         dbh = np.zeros_like(bh)
#         dby = np.zeros_like(by)
#
#         dWhy += error @ h.T   #dL/dy^ = y^ - y
#         dby += error
#
#
#         dh = Why.T @ error
#
#         for t in reversed(range(1, 4)):
#             h_t = hidden_states[t]
#             h_prev = hidden_states[t - 1]
#             x_t = X[i, t - 1].reshape(input_size, 1)
#
#             dh_raw = dh * (1 - h_t ** 2)
#
#             dWxh += dh_raw @ x_t.T # it accumulates the input to hidden weight gradient
#             dWhh += dh_raw @ h_prev.T  #accumulates the recurrent weight gradient
#             dbh += dh_raw #accumulates the hidden bias gradient
#
#             dh = Whh.T @ dh_raw #passes the error backward to the preceding hidden state
#          # d/dz tanh(z) = 1 - tanh^2(z)
#
#         # Update weights and biases
#         Wxh -= learning_rate * dWxh
#         Whh -= learning_rate * dWhh
#         Why -= learning_rate * dWhy
#         bh -= learning_rate * dbh
#         by -= learning_rate * dby
#
#     if (epoch + 1) % 10 == 0:
#         average_loss = total_loss / len(X)
#         print(f"Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}")
#
#
# # Test the trained RNN
# test_sequence = np.array([[0.2], [0.4], [0.3]])
# h = np.zeros((hidden_size, 1))
#
# for t in range(3):
#         x_t = test_sequence[t].reshape(input_size, 1)
#         h = np.tanh(Wxh @ x_t + Whh @ h + bh)
#
# prediction = Why @ h + by
#
# print("\nTest sequence:", test_sequence.flatten())
# print("Predicted sum:", prediction.item())
# print("Actual sum:", np.sum(test_sequence))
#
#
#
#
#
#
