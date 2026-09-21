import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def cnn_padding(image_path, padding=2, displayimage=True):

    image = np.array(Image.open(image_path))

    padded_image = np.pad(
        image,
        ((padding, padding), (padding, padding), (0, 0)),
        mode="constant",
        constant_values=0
    )

    print("Original image size:", image.shape)
    print("Padded image size:", padded_image.shape)

    if displayimage:
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(padded_image)
        plt.title("Padded Image")
        plt.axis("off")

        plt.show()

    return padded_image


def main():
    cnn_padding("image.jpg", padding=2, displayimage=True)


if __name__ == "__main__":
    main()