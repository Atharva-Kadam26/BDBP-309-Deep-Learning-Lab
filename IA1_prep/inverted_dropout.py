import numpy as np


def inverted_dropout(x, p=0.5, training=True):

    if not training:
        return x

    mask = np.random.binomial(1, 1 - p, size=x.shape)

    output = x * mask / (1 - p)

    return output


def main():

    x = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ], dtype=float)

    print("Original input:")
    print(x)

    output = inverted_dropout(x, p=0.5, training=True)

    print("\nAfter Inverted Dropout:")
    print(output)


if __name__ == "__main__":
    main()