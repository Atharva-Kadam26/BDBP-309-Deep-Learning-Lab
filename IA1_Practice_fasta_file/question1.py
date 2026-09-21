import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torch.nn.utils.rnn import pad_sequence


class FASTADataset(Dataset):

    def __init__(self, fasta_file):

        self.sequences = []
        self.names = []

        with open(fasta_file, "r") as file:

            current_name = None
            current_sequence = ""

            # Read file line by line
            for line in file:

                line = line.strip()

                # FASTA header starts with '>'
                if line.startswith(">"):

                    # Save previous sequence
                    if current_name is not None:
                        self.names.append(current_name)
                        self.sequences.append(current_sequence)

                    # Store header without '>'
                    current_name = line[1:]

                    # Start new sequence
                    current_sequence = ""

                else:

                    # Add sequence line
                    current_sequence = current_sequence + line

            # Save the last sequence
            if current_name is not None:
                self.names.append(current_name)
                self.sequences.append(current_sequence)

        # DNA encoding
        self.encoding = {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3,
            "N": 4
        }

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, index):

        sequence = self.sequences[index]

        # Convert DNA bases into numerical values
        encoded_sequence = []

        for base in sequence:

            en = self.encoding.get(
                base.upper(),
                4
            )

            encoded_sequence.append(en)

        # Convert to tensor
        sequence_tensor = torch.tensor(
            encoded_sequence,
            dtype=torch.long
        )

        return sequence_tensor, sequence


def collate_fn(batch):

    sequence_tensors = []
    original_sequences = []

    # Separate tensors and original sequences
    for tensor, sequence in batch:

        sequence_tensors.append(tensor)
        original_sequences.append(sequence)

    # Pad sequences to equal length
    padded_sequences = pad_sequence(
        sequence_tensors,
        batch_first=True,
        padding_value=4
    )

    return padded_sequences, original_sequences


def main():

    # Load FASTA dataset
    dataset = FASTADataset("sample.fasta")

    print(
        "Total dataset size:",
        len(dataset)
    )

    # Split dataset
    train_size = int(
        0.70 * len(dataset)
    )

    test_size = (
        len(dataset) - train_size
    )

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create train and test DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(
        "\nTraining Dataset size:",
        len(train_dataset)
    )

    print(
        "Test Dataset size:",
        len(test_dataset)
    )

    print(
        "\nTraining DataLoader size:",
        len(train_loader)
    )

    print(
        "Test DataLoader size:",
        len(test_loader)
    )

    # Get first batch
    sequence_tensor, sequences = next(
        iter(train_loader)
    )

    # Print first batch
    print(
        "\nFirst Batch Sequence Tensor:"
    )

    print(sequence_tensor)

    print(
        "\nOriginal sequences in first batch:"
    )

    for sequence in sequences:

        print(sequence)

    print(
        "\nSequence Tensor shape:"
    )

    print(sequence_tensor.shape)


if __name__ == "__main__":

    main()