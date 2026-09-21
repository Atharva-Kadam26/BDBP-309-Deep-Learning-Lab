import numpy as np
import matplotlib.pyplot as plt


def simulate_gradient(depth, weight, activation_derivative):

    gradient = 1.0
    gradients = []

    for layer in range(depth):

        gradient = (
            gradient
            * weight
            * activation_derivative
        )

        gradients.append(abs(gradient))

    return gradients


# Number of layers
depth = 50


# --------------------------------------------------
# Vanishing Gradient
# --------------------------------------------------

vanishing = simulate_gradient(
    depth=depth,
    weight=0.5,
    activation_derivative=0.25
)


# --------------------------------------------------
# Exploding Gradient
# --------------------------------------------------

exploding = simulate_gradient(
    depth=depth,
    weight=10,
    activation_derivative=0.25
)


# --------------------------------------------------
# Print Gradient Values
# --------------------------------------------------

print("VANISHING GRADIENT VALUES")

for layer, gradient in enumerate(vanishing, start=1):

    print(
        f"Layer {layer}: "
        f"{gradient:.10f}"
    )


print("\nEXPLODING GRADIENT VALUES")

for layer, gradient in enumerate(exploding, start=1):

    print(
        f"Layer {layer}: "
        f"{gradient:.10f}"
    )


# --------------------------------------------------
# Plot
# --------------------------------------------------

layers = range(1, depth + 1)

plt.figure(figsize=(10, 6))

plt.plot(
    layers,
    vanishing,
    label="Vanishing Gradient"
)

plt.plot(
    layers,
    exploding,
    label="Exploding Gradient"
)

plt.xlabel("Number of Layers")
plt.ylabel("Gradient Magnitude")

plt.title(
    "Vanishing and Exploding Gradients"
)

# Log scale makes both effects visible
plt.yscale("log")

plt.legend()
plt.grid(True)

plt.show()