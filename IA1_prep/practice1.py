import torch
from torch.utils.data import Dataset, DataLoader,random_split
from torch.nn.utils.rnn import pad_sequence

class FastaDataset(Dataset):
    def __init__(self, fasta_file):

        self.sequences =[]
        self.names = []

        with open(fasta_file,'r') as f:
            current_sequence = ''
            current_name = None
            for line in f:

                if line.startswith('>'):
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

        self.encoding ={
            'A':0,
            'C':1,
            'G':2,
            'T':3,
            'N':4,
        }

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        encoded_sequence =[]
        for base in sequence:
            encoding = self.encoding.get(base.upper(),4)
            encoded_sequence.append(encoding)

        sequence_tensor = torch.tensor(encoded_sequence,dtype = torch.long)
        return sequence_tensor, sequence

def collate_fn(batch):
    sequence_tensors = []
    original_sequence = []

    for tensor,sequence in batch:
        sequence_tensors.append(tensor)
        original_sequence.append(sequence)

    padded_sequence = pad_sequence(sequence_tensors,batch_first = True,padding_value = 4)
    return padded_sequence, original_sequence

def main():

    dataset = FastaDataset('sample.fasta')
    training_size = int(len(dataset)*0.7)
    test_size = len(dataset) - training_size
    training_data,test_data = random_split(dataset,[training_size,test_size],generator=torch.Generator().manual_seed(42))

    print("\nSize of original dataset",len(dataset))
    print("\nSize of training dataset",len(training_data))
    print("\nSize of test dataset",len(test_data))

    train_dataloader = DataLoader(dataset=training_data,batch_size=2,shuffle=True,collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=test_data,batch_size=2,shuffle=True,collate_fn=collate_fn)

    print("\n Size of training dataloader",len(train_dataloader))
    print("\n Size of test dataloader",len(test_dataloader))

    tensor,sequence = next(iter(train_dataloader))
    print("Tensor of first batch")
    print(tensor)

    print("\n Original sequence of first batch")
    for i in sequence:
        print(i[:20])  #Note the sequences have been restricted to printing only the first 20 bases

    print("\nSize of the tensor of first batch",tensor.shape)


if __name__ == "__main__":
    main()





