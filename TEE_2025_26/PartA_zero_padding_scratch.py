
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def cnn_padding(image_path, padding=2, displayimage=True):

    # Read image in grayscale
    image = Image.open(image_path).convert("L")

    # Convert image into NumPy array
    image = np.array(image)

    # Print original image size
    print("Original image size:", image.shape)

    # Get dimensions
    rows, cols = image.shape

    # Create zero padded image
    padded_image = np.zeros(
        (rows + 2 * padding, cols + 2 * padding),
        dtype=image.dtype
    )

    # Place original image in the center
    padded_image[padding:padding+rows,
                 padding:padding+cols] = image

    # Print padded image size
    print("Padded image size:", padded_image.shape)

    # Display images
    if displayimage:
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(padded_image, cmap="gray")
        plt.title("Padded Image")
        plt.axis("off")

        plt.show()

    return padded_image


# Function call
padded = cnn_padding("image.png")
