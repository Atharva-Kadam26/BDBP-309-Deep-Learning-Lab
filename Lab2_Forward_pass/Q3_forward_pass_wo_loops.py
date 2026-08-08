"""Implement the forward pass using vectorized operations, i.e. W should be a matrix, x, z
and a are vectors. The implementation should not contain any loops."""

import numpy as np

np.random.seed(42)


class forward_pass:
    def __init__(self, X):
        self.X = X
        self.config = self.build_config()
        self.layers = self.config["Layers"]

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def build_config(self):
        num_layers = int(input("Enter the number of layers: "))
        config = {}
        config['Layers'] = num_layers
        for i in range(num_layers):
            neurons = int(input(f"Enter the number of neurons in Layer {i + 1}: "))
            if i == 0:
                config[f"Weight {i + 1}"] = np.random.random((neurons, len(self.X)))
                config[f"Bias {i + 1}"] = np.random.random(neurons)
            else:
                config[f"Weight {i + 1}"] = np.random.random((neurons, len(config[f"Weight {i}"])))
                config[f"Bias {i + 1}"] = np.random.random(neurons)
        return config

    def forward_pass(self, X=None, current_layer=1, config=None):
        if config is None:
            config = self.config
        if X is None:
            X = self.X

        wei = config[f"Weight {current_layer}"]
        bia = config[f"Bias {current_layer}"]
        a = self.sigmoid((wei @ X) + bia)

        print(f"Layer {current_layer}: a = {a}")

        if config["Layers"] == current_layer:
            return a

        return self.forward_pass(a, current_layer=current_layer + 1, config=config)

    def validate_simple_example(self):
        W1 = np.array([[1, 0], [0, 1]])
        b1 = np.array([0, 0])
        W2 = np.array([[1, 1]])
        b2 = np.array([0])

        # keys now match exactly what forward_pass looks for
        config = {
            "Layers": 2,
            "Weight 1": W1, "Bias 1": b1,
            "Weight 2": W2, "Bias 2": b2,
        }
        ans = self.forward_pass(config=config)
        return ans


def main():
    input_arr = np.array([0.3, -1.2])
    cls = forward_pass(input_arr)
    answer = cls.validate_simple_example()
    print("Final output:", answer)


if __name__ == "__main__":
    main()
