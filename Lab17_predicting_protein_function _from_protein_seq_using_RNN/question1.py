import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

base_dir = os.path.dirname(os.path.abspath(__file__))

fasta_file = os.path.join(base_dir, "train_sequences.fasta")
terms_file = os.path.join(base_dir, "train_terms.tsv")


#loading the GO annotations
terms_df = pd.read_csv(terms_file, sep="\t")

print(terms_df.head())
print(terms_df.shape)

def read_fasta(file_path):

    sequences = {}

    with open(file_path, "r") as file:

        protein_id = None
        sequence = ""

        for line in file:

            line = line.strip()

            if line.startswith(">"):

                if protein_id is not None:
                    sequences[protein_id] = sequence

                header = line[1:]

                if "|" in header:
                    protein_id = header.split("|")[1]
                else:
                    protein_id = header.split()[0]

                sequence = ""

            else:
                sequence += line

        if protein_id is not None:
            sequences[protein_id] = sequence

    return sequences


sequences = read_fasta(fasta_file)
print("Total protein sequences:", len(sequences))

for protein_id, sequence in list(sequences.items())[:5]:
    print(protein_id, sequence[:50])

#processing the protein sequences:
amino_acids = "ACDEFGHIKLMNPQRSTVWY"

vocab = {
    amino_acid: index + 1
    for index, amino_acid in enumerate(amino_acids)
}

vocab["<PAD>"] = 0
vocab["<UNK>"] = 21

vocab_size = len(vocab)
print("Vocabulary size:", vocab_size)


def encode_sequence(sequence):

    encoded = []
    for amino_acid in sequence:
        if amino_acid in vocab:
            encoded.append(vocab[amino_acid])

        else:
            encoded.append(vocab["<UNK>"])

    return encoded


#preparing protein function labels
#note that this is just an experimental subset
top_terms = terms_df["term"].value_counts().head(30).index.tolist()

terms_df = terms_df[
    terms_df["term"].isin(top_terms)
]
print("Number of selected GO terms:", len(top_terms))

top_terms = terms_df["term"].value_counts().head(30).index.tolist()

terms_df = terms_df[
    terms_df["term"].isin(top_terms)
]

print("Number of selected GO terms:", len(top_terms))

protein_labels = {}

for _, row in terms_df.iterrows():

    protein_id = row["EntryID"]
    go_term = row["term"]

    if protein_id not in protein_labels:
        protein_labels[protein_id] = []

    protein_labels[protein_id].append(go_term)

#creating mapping for GO Terms
term_to_index = {
    term: index
    for index, term in enumerate(top_terms)
}

num_classes = len(term_to_index)

print("Number of functional classes:", num_classes)

#[1, 0, 1, 0, 1, 0, ...]
#Here the first, third and fifth GO terms are associated with that protein.

#combining sequences and labels
max_samples = 10000
max_length = 1000

protein_ids = []

for protein_id in protein_labels:

    if protein_id in sequences:
        protein_ids.append(protein_id)

protein_ids = protein_ids[:max_samples]

print("Proteins available for training:", len(protein_ids))

X = []
y = []
for protein_id in protein_ids:
    sequence = sequences[protein_id]
    encoded = encode_sequence(sequence)
    encoded = encoded[:max_length]
    X.append(encoded)
    label_vector = np.zeros(num_classes)

    for term in protein_labels[protein_id]:
        if term in term_to_index:
            label_vector[term_to_index[term]] = 1
    y.append(label_vector)

print("Total sequences:", len(X))
print("Total labels:", len(y))


#defining the dataset
class ProteinDataset(Dataset):

    def __init__(self, sequences, labels):

        self.sequences = sequences
        self.labels = labels

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, index):

        sequence = self.sequences[index]

        label = self.labels[index]

        return (
            torch.tensor(sequence, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )


def collate_fn(batch):

    sequences, labels = zip(*batch)

    lengths = torch.tensor(
        [len(sequence) for sequence in sequences]
    )

    max_len = max(lengths)

    padded_sequences = torch.zeros(
        len(sequences),
        max_len,
        dtype=torch.long
    )

    for i, sequence in enumerate(sequences):

        padded_sequences[i, :len(sequence)] = sequence

    labels = torch.stack(labels)

    return padded_sequences, lengths, labels


#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


train_dataset = ProteinDataset(X_train, y_train)
test_dataset = ProteinDataset(X_test, y_test)

batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)


#bidirectional LSTM :
class ProteinLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):

        super(ProteinLSTM, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            hidden_dim * 2,
            num_classes
        )

    def forward(self, x, lengths):

        embedded = self.embedding(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        output, (hidden, cell) = self.lstm(packed)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1
        )

        output = self.fc(combined)

        return output


embedding_dim = 64
hidden_dim = 128

model = ProteinLSTM(
    vocab_size,
    embedding_dim,
    hidden_dim,
    num_classes
).to(device)

print(model)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


#training the model
epochs = 10

train_losses = []

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for sequences_batch, lengths, labels in train_loader:

        sequences_batch = sequences_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(sequences_batch, lengths)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    epoch_loss = total_loss / len(train_loader)

    train_losses.append(epoch_loss)

    print(
        f"Epoch [{epoch+1}/{epochs}], "
        f"Loss: {epoch_loss:.4f}"
    )


plt.figure(figsize=(8, 5))

plt.plot(train_losses)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Protein Function Prediction - Training Loss")

plt.show()


#evaluating the model
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for sequences_batch, lengths, labels in test_loader:

        sequences_batch = sequences_batch.to(device)

        outputs = model(sequences_batch, lengths)

        probabilities = torch.sigmoid(outputs)

        predictions = (probabilities >= 0.5).int()

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

precision = precision_score(
    all_labels,
    all_predictions,
    average="micro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="micro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="micro",
    zero_division=0
)

print("Precision:", precision)

print("Recall:", recall)

print("F1 Score:", f1)

#predict protein function for a new sequence
def predict_function(sequence):

    model.eval()

    encoded = encode_sequence(sequence)

    encoded = encoded[:max_length]

    input_sequence = torch.tensor(
        [encoded],
        dtype=torch.long
    ).to(device)

    lengths = torch.tensor([len(encoded)])

    with torch.no_grad():

        output = model(input_sequence, lengths)

        probabilities = torch.sigmoid(output)

    probabilities = probabilities.cpu().numpy()[0]

    predicted_indices = np.where(probabilities >= 0.5)[0]

    for index in predicted_indices:

        print(
            top_terms[index],
            "Probability:",
            round(probabilities[index], 4)
        )


