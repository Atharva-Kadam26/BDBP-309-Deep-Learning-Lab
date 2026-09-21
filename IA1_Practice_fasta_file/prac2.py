import torch
from torch.utils.data import DataLoader,Dataset, random_split
from torch.nn.utils.rnn import pad_sequence

class FASTADataset(Dataset):
    def __init__(self,filename):

        self.sequences = []
        self.names = []

        with open(filename, 'r') as f:
            current_sequence = ""
            current_name = None

            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_name is not None:
                        self.sequences.append(current_sequence)
                        self.names.append(current_name)

                    current_sequence = ""
                    current_name = line[1:]

                else:
                    current_sequence += line

            if current_name is not None:
                self.sequences.append(current_sequence)
                self.names.append(current_name)

        self.encoding = {
            'A':0,
            'C':1,
            'G':2,
            'T':3,
            'N':4,
        }


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequences = self.sequences[index]

        encoded_sequence = []
        for base in sequences:
            encoding = self.encoding.get(base.upper(),4)
            encoded_sequence.append(encoding)

        sequence_tensor = torch.tensor(encoded_sequence, dtype=torch.long)

        return sequence_tensor,sequences


def collate_fn(batch):

    sequence_tensors = []
    original_sequence = []

    for tensor,sequence in batch:
        sequence_tensors.append(tensor)
        original_sequence.append(sequence)

    padded_sequence = pad_sequence(sequence_tensors, batch_first=True, padding_value=4)

    return padded_sequence, original_sequence

def main():

    dataset = FASTADataset("sample.fasta")
    print("\nSize of original dataset:", len(dataset))

    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size

    training_data,test_data = random_split(dataset, [train_size, test_size])

    print("\nSize of training dataset:", len(training_data))
    print("\nSize of test dataset:", len(test_data))


    train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=2, shuffle=False, collate_fn=collate_fn)

    print("Size of training dataloader:", len(train_dataloader))
    print("Size of test dataloader:", len(test_dataloader))

    print("\nSequence tensor of the first batch:")
    tensor,sequence = next(iter(train_dataloader))
    print(tensor)

    print("\nSequence of the first batch:")
    for seq in sequence:
        print(seq)

    print("\nShape of the tensor of  first batch:")
    print(tensor.size())

if __name__ == "__main__":
    main()







