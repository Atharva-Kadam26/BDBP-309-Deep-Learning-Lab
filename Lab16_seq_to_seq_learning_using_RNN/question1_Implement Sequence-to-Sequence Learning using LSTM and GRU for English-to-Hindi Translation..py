
import pandas as pd
import numpy as np
import re
import random

import torch
import torch.nn as nn

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device:", device)

batch_size = 64
embed_size = 128
hidden_size = 256
epochs = 5
learning_rate = 0.001

max_vocab_size = 20000
max_length = 20

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


#loading dataset
df = pd.read_csv("Hindi_English_Truncated_Corpus.csv")

print("dataset shape:", df.shape)

print("\nfirst five rows:")
print(df.head())

print("\ncolumn names:")
print(df.columns)


#data cleaning

print("\nmissing values:")
print(df.isnull().sum())

# removing rows with missing English or Hindi sentences
df = df.dropna(
    subset=['english_sentence', 'hindi_sentence']
)

# removing duplicate English-Hindi sentence pairs
df = df.drop_duplicates(
    subset=['english_sentence', 'hindi_sentence']
)

# resetting dataframe index

df = df.reset_index(drop=True)
print("\ncleaned dataset shape:", df.shape)


#text preprocessing

def preprocess_english(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s']",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def preprocess_hindi(text):

    text = re.sub(
        r"[^\u0900-\u097F\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# applying preprocessing to English and Hindi sentences

df['english_sentence'] = df['english_sentence'].apply(
    preprocess_english
)
df['hindi_sentence'] = df['hindi_sentence'].apply(
    preprocess_hindi
)

# removing empty sentences after preprocessing
df = df[
    (df['english_sentence'] != "") &
    (df['hindi_sentence'] != "")
]

df = df.reset_index(drop=True)
print("\npreprocessed sentences:")
print(df[['english_sentence', 'hindi_sentence']].head())

print("\nfinal dataset shape:", df.shape)


#tokenization and splitting data

# special tokens

pad = "<PAD>"
sos = "<SOS>"
eos = "<EOS>"
unk = "<UNK>"


# splitting sentences into individual words

def tokenize(text):
    return text.split()


# splitting dataset into training and testing sets

train_sentences, test_sentences = train_test_split(
    df[['english_sentence', 'hindi_sentence']].values.tolist(),
    test_size=0.2,
    random_state=42
)

print("\ntraining samples:", len(train_sentences))
print("testing samples:", len(test_sentences))


#vocab. creation

def build_vocab(sentences, max_vocab_size):

    word_counts = Counter()
    # counting word frequencies
    for sentence in sentences:
        word_counts.update(
            tokenize(sentence)
        )

    # initializing vocabulary with special tokens

    vocab = {
        pad: 0,
        sos: 1,
        eos: 2,
        unk: 3
    }

    # selecting most frequent words

    for word, count in word_counts.most_common(
        max_vocab_size - 4
    ):

        vocab[word] = len(vocab)

    return vocab


# building English vocabulary using training sentences

english_vocab = build_vocab(
    [pair[0] for pair in train_sentences],
    max_vocab_size
)


# building Hindi vocabulary using training sentences

hindi_vocab = build_vocab(
    [pair[1] for pair in train_sentences],
    max_vocab_size
)


print("\nenglish vocabulary size:", len(english_vocab))
print("hindi vocabulary size:", len(hindi_vocab))


# converting sentences into numerical sequences

def sentence_to_sequence(sentence, vocab):

    tokens = tokenize(sentence)
    # adding start-of-sentence token

    sequence = [vocab[sos]]

    # converting words into numerical indices
    for word in tokens:
        sequence.append(
            vocab.get(word, vocab[unk])
        )

    # adding end-of-sentence token
    sequence.append(vocab[eos])
    return sequence



#generating sequences and filtering data

train_data = []
test_data = []


# converting training sentences

for eng, hin in train_sentences:

    eng_sequence = sentence_to_sequence(
        eng,
        english_vocab
    )

    hin_sequence = sentence_to_sequence(
        hin,
        hindi_vocab
    )

    # retaining sentence pairs within maximum length

    if len(eng_sequence) <= max_length and len(hin_sequence) <= max_length:

        train_data.append(
            (eng_sequence, hin_sequence)
        )


# converting testing sentences

for eng, hin in test_sentences:
    eng_sequence = sentence_to_sequence(
        eng,
        english_vocab
    )

    hin_sequence = sentence_to_sequence(
        hin,
        hindi_vocab
    )

    # retaining sentence pairs within maximum length
    if len(eng_sequence) <= max_length and len(hin_sequence) <= max_length:

        test_data.append(
            (eng_sequence, hin_sequence)
        )


# displaying results

print("\nusable training pairs:", len(train_data))
print("usable testing pairs:", len(test_data))

print("\nexample English sequence:")
print(train_data[0][0])

print("\nexample Hindi sequence:")
print(train_data[0][1])

#creating custom dataset :

class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data  #storing the sentence pairs

    def __len__(self):
        return len(self.data) #total num of sentence pairs

    def __getitem__(self, index): #retrieving an english-hindi sentence pair using its index
        english_sequence, hindi_sequence = self.data[index]

        return (
            torch.tensor(english_sequence, dtype=torch.long),
            torch.tensor(hindi_sequence, dtype=torch.long)
        )   #numerical sequences to tensors


# creating training and testing dataset objects

train_dataset = TranslationDataset(train_data)
test_dataset = TranslationDataset(test_data)

print("training dataset size:", len(train_dataset))
print("testing dataset size:", len(test_dataset))


#padding and collate function

def collate_fn(batch):

    english_batch = [item[0] for item in batch]
    hindi_batch = [item[1] for item in batch]

    # storing original English sentence lengths
    english_lengths = torch.tensor(
        [len(sentence) for sentence in english_batch]
    )

    # padding English sentences

    english_batch = pad_sequence(
        english_batch,
        batch_first=True,
        padding_value=0
    )

    # padding Hindi sentences
    hindi_batch = pad_sequence(
        hindi_batch,
        batch_first=True,
        padding_value=0
    )

    return english_batch, hindi_batch, english_lengths

#creating dataloaders

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


print("training batches:", len(train_loader))
print("testing batches:", len(test_loader))



#LSTM Seq to seq

#lstm encoder
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(
            input_size,
            embed_size,
            padding_idx=0
        )

        self.lstm = nn.LSTM(   #I/p dim = 128, O/p dim = 256
            embed_size,
            hidden_size,
            batch_first=True
        )

    def forward(self, source, lengths):
        embedded = self.embedding(source)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        output, (hidden, cell) = self.lstm(packed)

        return hidden, cell

#lstm decoder
# ==========================================
# SECTION 14: LSTM DECODER
# ==========================================

class LSTMDecoder(nn.Module):

    def __init__(self, output_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(
            output_size,
            embed_size,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, input_token, hidden, cell):
        input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)

        output, (hidden, cell) = self.lstm(
            embedded,
            (hidden, cell)
        )

        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell

#lstm model
class LSTMSeq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, lengths):

        batch_size = source.shape[0]
        target_length = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(
            batch_size,
            target_length - 1,
            target_vocab_size
        ).to(device)

        hidden, cell = self.encoder(source, lengths)

        input_token = target[:, 0]

        for t in range(1, target_length):
            prediction, hidden, cell = self.decoder(
                input_token,
                hidden,
                cell
            )

            outputs[:, t - 1, :] = prediction

            # teacher forcing

            input_token = target[:, t]

        return outputs



#initialising lstm .....
lstm_encoder = LSTMEncoder(
    len(english_vocab),
    embed_size,
    hidden_size
)

lstm_decoder = LSTMDecoder(
    len(hindi_vocab),
    embed_size,
    hidden_size
)

lstm_model = LSTMSeq2Seq(
    lstm_encoder,
    lstm_decoder
).to(device)

print(lstm_model)


criterion = nn.CrossEntropyLoss(
    ignore_index=0
)

optimizer = torch.optim.Adam(
    lstm_model.parameters(),
    lr=learning_rate
)

#training
def train_model(model, train_loader, optimizer, criterion):

    model.train()

    total_loss = 0

    for english_batch, hindi_batch, english_lengths in train_loader:

        english_batch = english_batch.to(device)
        hindi_batch = hindi_batch.to(device)
        english_lengths = english_lengths.to(device)

        optimizer.zero_grad()

        outputs = model(
            english_batch,
            hindi_batch,
            english_lengths
        )

        outputs = outputs.reshape(
            -1,
            outputs.shape[-1]
        )

        targets = hindi_batch[:, 1:].reshape(-1)

        loss = criterion(outputs, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


#evaluation

def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():

        for english_batch, hindi_batch, english_lengths in test_loader:
            english_batch = english_batch.to(device)
            hindi_batch = hindi_batch.to(device)
            english_lengths = english_lengths.to(device)

            outputs = model(
                english_batch,
                hindi_batch,
                english_lengths
            )

            outputs = outputs.reshape(
                -1,
                outputs.shape[-1]
            )

            targets = hindi_batch[:, 1:].reshape(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(test_loader)



lstm_train_losses = []
lstm_test_losses = []

for epoch in range(epochs):

    train_loss = train_model(
        lstm_model,
        train_loader,
        optimizer,
        criterion
    )

    test_loss = evaluate_model(
        lstm_model,
        test_loader,
        criterion
    )

    lstm_train_losses.append(train_loss)
    lstm_test_losses.append(test_loss)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Training Loss: {train_loss:.4f} "
        f"Testing Loss: {test_loss:.4f}"
    )



#GRU encoder

class GRUEncoder(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(
            input_size,
            embed_size,
            padding_idx=0
        )

        self.gru = nn.GRU(
            embed_size,
            hidden_size,
            batch_first=True
        )

    def forward(self, source, lengths):
        embedded = self.embedding(source)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        output, hidden = self.gru(packed)

        return hidden


#GRU decoder
class GRUDecoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(
            output_size,
            embed_size,
            padding_idx=0
        )

        self.gru = nn.GRU(
            embed_size,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, input_token, hidden):

        input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)
        output, hidden = self.gru(
            embedded,
            hidden
        )
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden

#GRU seq to seq model
class GRUSeq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, lengths):

        batch_size = source.shape[0]
        target_length = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(
            batch_size,
            target_length - 1,
            target_vocab_size
        ).to(device)

        hidden = self.encoder(source, lengths)

        input_token = target[:, 0]

        for t in range(1, target_length):

            prediction, hidden = self.decoder(
                input_token,
                hidden
            )

            outputs[:, t - 1, :] = prediction

            # teacher forcing

            input_token = target[:, t]
        return outputs


#initializing GRU
gru_encoder = GRUEncoder(
    len(english_vocab),
    embed_size,
    hidden_size
)

gru_decoder = GRUDecoder(
    len(hindi_vocab),
    embed_size,
    hidden_size
)

gru_model = GRUSeq2Seq(
    gru_encoder,
    gru_decoder
).to(device)

print(gru_model)


gru_criterion = nn.CrossEntropyLoss(
    ignore_index=0
)

gru_optimizer = torch.optim.Adam(
    gru_model.parameters(),
    lr=learning_rate
)

#training GRU
gru_train_losses = []
gru_test_losses = []

for epoch in range(epochs):

    train_loss = train_model(
        gru_model,
        train_loader,
        gru_optimizer,
        gru_criterion
    )

    test_loss = evaluate_model(
        gru_model,
        test_loader,
        gru_criterion
    )

    gru_train_losses.append(train_loss)
    gru_test_losses.append(test_loss)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Training Loss: {train_loss:.4f} "
        f"Testing Loss: {test_loss:.4f}"
    )


#COmparing LSTM vs GRU

# gru_train_losses = []
# gru_test_losses = []
#
# for epoch in range(epochs):
#
#     train_loss = train_model(
#         gru_model,
#         train_loader,
#         gru_optimizer,
#         gru_criterion
#     )
#
#     test_loss = evaluate_model(
#         gru_model,
#         test_loader,
#         gru_criterion
#     )
#
#     gru_train_losses.append(train_loss)
#     gru_test_losses.append(test_loss)
#
#     print(
#         f"Epoch [{epoch + 1}/{epochs}] "
#         f"Training Loss: {train_loss:.4f} "
#         f"Testing Loss: {test_loss:.4f}"
#     )


#translation :
idx_to_hindi = {
    index: word
    for word, index in hindi_vocab.items()
}


def translate_sentence(sentence, model):

    model.eval()

    sentence = preprocess_english(sentence)

    sequence = sentence_to_sequence(
        sentence,
        english_vocab
    )

    source = torch.tensor(
        sequence,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    lengths = torch.tensor([len(sequence)])

    with torch.no_grad():

        # encoding English sentence

        if isinstance(model, LSTMSeq2Seq):

            hidden, cell = model.encoder(
                source,
                lengths
            )

        else:

            hidden = model.encoder(
                source,
                lengths
            )

        # starting with SOS token

        input_token = torch.tensor(
            [hindi_vocab[sos]],
            dtype=torch.long
        ).to(device)

        translated_words = []

        # generating Hindi tokens

        for _ in range(max_length):

            if isinstance(model, LSTMSeq2Seq):

                prediction, hidden, cell = model.decoder(
                    input_token,
                    hidden,
                    cell
                )

            else:

                prediction, hidden = model.decoder(
                    input_token,
                    hidden
                )

            predicted_token = prediction.argmax(1).item()

            # stopping when EOS is generated

            if predicted_token == hindi_vocab[eos]:
                break

            # ignoring special tokens

            if predicted_token not in [
                hindi_vocab[pad],
                hindi_vocab[sos]
            ]:

                translated_words.append(
                    idx_to_hindi[predicted_token]
                )

            # feeding predicted token into next step

            input_token = torch.tensor(
                [predicted_token],
                dtype=torch.long
            ).to(device)

    return " ".join(translated_words)

#testing translation
english_sentence = "how are you"

lstm_translation = translate_sentence(
    english_sentence,
    lstm_model
)

gru_translation = translate_sentence(
    english_sentence,
    gru_model
)

print("English sentence:", english_sentence)

print("\nLSTM translation:")
print(lstm_translation)

print("\nGRU translation:")
print(gru_translation)

# LSTM vs GRU PERFORMANCE COMPARISON

import pandas as pd

comparison = pd.DataFrame({
    "Model": ["LSTM", "GRU"],
    "Final Training Loss": [
        lstm_train_losses[-1],
        gru_train_losses[-1]
    ],
    "Final Testing Loss": [
        lstm_test_losses[-1],
        gru_test_losses[-1]
    ],
    "Minimum Testing Loss": [
        min(lstm_test_losses),
        min(gru_test_losses)
    ]
})

print("\nLSTM vs GRU Performance Comparison:")
print(comparison)




# Exercise 3: VANISHING GRADIENT DEMONSTRATION
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining parameters
input_size = 32
hidden_size = 64
sequence_length = 50
batch_size = 1

print("Device:", device)

# defining LSTM model
lstm = nn.LSTM(
    input_size,
    hidden_size,
    batch_first=True
).to(device)


# defining GRU model
gru = nn.GRU(
    input_size,
    hidden_size,
    batch_first=True
).to(device)

# generating random input sequence

input_sequence = torch.randn(
    batch_size,
    sequence_length,
    input_size
).to(device)

print("Input shape:", input_sequence.shape)
#the O/P shape will be (1,50,32), this represents one sequence containing 50 time steps, with 32 features at each time step.


# function to calculate gradient magnitudes

def calculate_gradients(model, model_type):

    x = input_sequence.clone().detach().requires_grad_(True)

    # passing input through the model
    if model_type == "LSTM":
        output, (hidden, cell) = model(x)
    else:
        output, hidden = model(x)

    # calculating loss using final time step
    loss = output[:, -1, :].pow(2).mean()

    # performing backpropagation
    loss.backward()

    # calculating gradient magnitude at each input time step
    gradient_norms = torch.norm(
        x.grad,
        dim=2
    ).squeeze(0)

    return gradient_norms.detach().cpu().numpy()

# calculating gradients for LSTM
lstm_gradients = calculate_gradients(
    lstm,
    "LSTM"
)


# calculating gradients for GRU
gru_gradients = calculate_gradients(
    gru,
    "GRU"
)


# plotting gradient magnitudes
plt.figure(figsize=(10, 6))
plt.plot(
    lstm_gradients,
    label="LSTM"
)

plt.plot(
    gru_gradients,
    label="GRU"
)

plt.yscale("log")

plt.xlabel("Time Steps")
plt.ylabel("Gradient Magnitude (Log Scale)")

plt.title("Vanishing Gradient Comparison: LSTM vs GRU")

plt.legend()
plt.grid()

plt.show()

