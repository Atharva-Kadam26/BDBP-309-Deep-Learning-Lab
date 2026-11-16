
"""Implement vanilla RNN using PyTorch for image captioning. You can use the PyTorch
library"""

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import pandas as pd
import os
import re
from collections import Counter

#Image transformation
transform = transforms.Compose([
    transforms.Resize((224,224)),   #resizes every image into 224 pixels in Height and width
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], #mean for red, green , blue
                         std=[0.229, 0.224, 0.225])
])

class Vocabulary:
    def __init__(self):
        # we convert every word into a numerical representation
        self.word_to_index ={
            "<PAD>":0,   #Padding token
            "<START>":1, #Beginning of sentence
            "<END>":2,   #Ending of sentence
            "<UNK>":3    #Unknown word
        }

        #We then generate a reverse dictionary, this is necessary because after generating numerical predictions, we need to convert them back into English words.
        self.index_to_word ={
            0: "<PAD>",
            1: "<START>",
            2: "<END>",
            3: "<UNK>"
        }

    def build_vocabulary(self,captions):
        counter = Counter()  #to store the frequency of each word
        for caption in captions:
            words = re.findall(r"\w+", caption.lower())   #extracting individual words
            counter.update(words)
        # A Dog is Playing! becomes ['a','dog','is','playing']
        for word in counter:    #iterating thru unique words
            if word not in self.word_to_index:
                index = len(self.word_to_index)
                self.word_to_index[word] = index
                self.index_to_word[index] = word

    def numericalize(self,caption):
        words = re.findall(r"\w+", caption.lower())
        tokens =[self.word_to_index["<START>"]]

        for word in words:
            tokens.append(self.word_to_index.get(word,3))
        tokens.append(self.word_to_index["<END>"])
        return tokens

    def __len__(self):
        return len(self.word_to_index)

# loading the dataset

base_dir = os.path.dirname(os.path.abspath(__file__))

image_dir = os.path.join(base_dir, "Images")

caption_file = os.path.join(base_dir, "captions.txt")


# reading caption records

data = pd.read_csv(caption_file)

print(data.head())

print("Total captions:", len(data))


# removing missing records

data = data.dropna(subset=["image", "caption"])

data["caption"] = data["caption"].astype(str)


# extracting captions

all_captions = data["caption"].tolist()


# building vocabulary

vocabulary = Vocabulary()

vocabulary.build_vocabulary(all_captions)


print("Vocabulary size:", len(vocabulary))


class ImageCaptionDataset(Dataset):
    def __init__(self, dataframe, image_dir, vocabulary, transform):
        self.data = dataframe
        self.image_dir = image_dir
        self.vocabulary = vocabulary
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image_name = self.data.iloc[index]["image"]
        caption = self.data.iloc[index]["caption"]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        caption = self.vocabulary.numericalize(caption)
        caption = torch.tensor(caption, dtype=torch.long)

        return image, caption

from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):

    images = []
    captions = []
    for image, caption in batch:
        images.append(image)
        captions.append(caption)


    images = torch.stack(images)


    captions = pad_sequence(
        captions,
        batch_first=True,
        padding_value=0
    )

    return images, captions


# creating dataset
dataset = ImageCaptionDataset(
    data,
    image_dir,
    vocabulary,
    transform
)

# creating dataloader
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)


print("Total batches:", len(loader))

class RNNImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        # CNN encoder
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.encoder = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        # freezing CNN parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # initializing hidden state

        self.init_hidden = nn.Linear(512, hidden_dim)


        # word embedding

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )


        # vanilla RNN layer

        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )


        # output layer

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions):
        with torch.no_grad():
            features = self.encoder(images)

        features = features.reshape(features.size(0), -1)

        hidden = torch.tanh(self.init_hidden(features))
        hidden = hidden.unsqueeze(0)

        captions = captions[:, :-1]
        embeddings = self.embedding(captions)

        output, hidden = self.rnn(
            embeddings,
            hidden
        )

        output = self.fc(output)

        return output

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


model = RNNImageCaptioningModel(
    vocab_size=len(vocabulary),
    embedding_dim=256,
    hidden_dim=256
).to(device)


print(model)

criterion = nn.CrossEntropyLoss(ignore_index=0)


optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

import time

num_epochs = 2

for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    model.encoder.eval()
    total_loss = 0

    for batch_idx, (images, captions) in enumerate(loader):
        images = images.to(device)
        captions = captions.to(device)

        # forward propagation
        outputs = model(images, captions)

        # target sequence
        targets = captions[:, 1:]

        # calculating loss
        loss = criterion(
            outputs.reshape(-1, outputs.size(2)),
            targets.reshape(-1)
        )


        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    end_time = time.time()
    epoch_time = (end_time - start_time) / 60


    print(
        f"Epoch: {epoch + 1}, "
        f"Loss: {total_loss / len(loader):.4f}, "
        f"Time: {epoch_time:.2f} minutes"
    )


def generate_caption(model, image_path, vocabulary):

    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encoder(image)
        features = features.reshape(features.size(0), -1)
        hidden = torch.tanh(model.init_hidden(features))
        hidden = hidden.unsqueeze(0)


        word = torch.tensor(
            [[vocabulary.word_to_index["<START>"]]]
        ).to(device)

        caption = []

        for i in range(30):
            embedding = model.embedding(word)

            output, hidden = model.rnn(
                embedding,
                hidden
            )

            output = model.fc(output.squeeze(1))
            predicted_word = output.argmax(1).item()

            if predicted_word == vocabulary.word_to_index["<END>"]:
                break
            caption.append(
                vocabulary.index_to_word[predicted_word]
            )
            word = torch.tensor(
                [[predicted_word]]
            ).to(device)

    return " ".join(caption)


# selecting a test image
test_image = os.path.join(
    image_dir,
    data.iloc[0]["image"]
)

# generating caption

generated_caption = generate_caption(
    model,
    test_image,
    vocabulary
)

print("Generated Caption:", generated_caption)

# saving trained model

torch.save(
    model.state_dict(),
    "rnn_captioning.pth"
)
