
import numpy as np

# Inverted Dropout function
def inverted_dropout(X, dropout_rate=0.5, training=True):

    # During testing, return original input
    if training == False:
        return X

    # Calculate keep probability
    keep_prob = 1 - dropout_rate

    # Generate random dropout mask
    mask = np.random.rand(*X.shape) < keep_prob

    # Apply mask and scale remaining neurons
    output = X * mask / keep_prob

    return output


# Input data
X = np.array([
    [0.8, 0.6, 0.9, 0.4],
    [0.5, 0.7, 0.3, 0.9],
    [0.2, 0.8, 0.6, 0.5]
])

print("Original Input:")
print(X)

# Apply inverted dropout during training
output = inverted_dropout(X, dropout_rate=0.5, training=True)

print("\nOutput after Dropout:")
print(output)

# Apply during testing
test_output = inverted_dropout(X, dropout_rate=0.5, training=False)

print("\nOutput during Testing:")
print(test_output)
