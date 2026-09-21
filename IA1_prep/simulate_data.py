import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd


class HeartDataset(Dataset):
    def __init__(self, filename):

        data = pd.read_csv(filename)

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        X = pd.get_dummies(X, dtype=float)
        y = pd.factorize(y)[0]

        self.X = torch.tensor(
            X.values,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

def main():

    dataset = HeartDataset("heart.csv")

    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    print("Dataset size:", len(dataset))
    print("Training dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    X, y = next(iter(train_loader))

    print("\nTraining DataLoader size:", len(train_loader))
    print("Test DataLoader size:", len(test_loader))

    print("\nData:")
    print(X)

    print("\nLabels:")
    print(y)

    print("\nData datatype:", X.dtype)
    print("Label datatype:", y.dtype)


if __name__ == "__main__":
    main()