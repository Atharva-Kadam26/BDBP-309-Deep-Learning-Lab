import numpy as np
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision.transforms import v2


def convolution(image, kernel):

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # O/p = (i/p + 2P - f) / S + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):  # for moving the kernel across the image
        for j in range(output_width):

            image_region = image[
                i:i + kernel_height,
                j:j + kernel_width
            ]

            output[i, j] = np.sum(image_region * kernel)  # element wise multiplication

    return output


def maxpool(image):

    # Pool size = 2 × 2
    pool_size = 2

    # Stride = 2
    stride = 2

    image_height, image_width = image.shape

    output_height = (image_height - pool_size) // stride + 1
    output_width = (image_width - pool_size) // stride + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):  # rows
        for j in range(output_width):  # columns

            image_region = image[
                i * stride:i * stride + pool_size,
                j * stride:j * stride + pool_size
            ]

            output[i, j] = np.max(image_region)  # max pooling

    return output


def load_data():

    transform = v2.Compose([
        v2.ToImage(),
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    return dataset


def main():

    # Loading real image dataset
    dataset = load_data()

    # Taking one real image
    image, label = dataset[0]

    # Converting image to NumPy array
    image = np.array(image)

    # Remove channel dimension
    image = image.squeeze()

    print("Actual label:", label)
    print("Input image shape:", image.shape)

    # Define convolution kernel
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    # Convolution
    convolution_output = convolution(image, kernel)

    print("After convolution:", convolution_output.shape)

    # Max pooling
    maxpool_output = maxpool(convolution_output)

    print("After maxpool:", maxpool_output.shape)

    # Print convolution values
    print("Convolution minimum value:", convolution_output.min())
    print("Convolution maximum value:", convolution_output.max())

    # Display Images
    plt.figure(figsize=(12, 4))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Convolution output
    plt.subplot(1, 3, 2)

    limit = max(
        abs(convolution_output.min()),
        abs(convolution_output.max())
    )

    plt.imshow(
        convolution_output,
        cmap='gray',
        vmin=-limit,
        vmax=limit
    )

    plt.title("After Convolution")
    plt.axis('off')

    # Max pooling output
    plt.subplot(1, 3, 3)
    plt.imshow(maxpool_output, cmap='gray')
    plt.title("After Max Pooling")
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    main()