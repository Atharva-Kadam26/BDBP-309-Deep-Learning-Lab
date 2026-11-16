# Implement GAN using PyTorch for image captioning.


import os
import re
import random
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.nn.utils.rnn import pad_sequence


# 1. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# 2. Vocabulary
captions_df = pd.read_csv("captions.csv")

word_count = Counter()

for caption in captions_df["caption"]:
    words = re.findall(r"\w+", caption.lower())
    word_count.update(words)

vocab = {
    "<PAD>": 0,
    "<START>": 1,
    "<END>": 2,
    "<UNK>": 3
}

for word, count in word_count.items():
    if count >= 2:
        vocab[word] = len(vocab)

vocab_size = len(vocab)

print("Vocabulary size:", vocab_size)


# 3. Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 4. Dataset
class FlickrDataset(Dataset):

    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = os.path.join("images", row["image"])

        image = Image.open(image_path).convert("RGB")
        image = transform(image)

        words = re.findall(r"\w+", row["caption"].lower())

        caption = [vocab["<START>"]]

        for word in words:
            caption.append(vocab.get(word, vocab["<UNK>"]))

        caption.append(vocab["<END>"])

        return image, torch.tensor(caption)


# 5. Collate function
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
        padding_value=vocab["<PAD>"]
    )

    return images, captions


# Use a subset to keep training manageable
unique_images = captions_df["image"].unique()

random.seed(42)
random.shuffle(unique_images)

selected_images = unique_images[:1000]

train_df = captions_df[
    captions_df["image"].isin(selected_images)
].reset_index(drop=True)

dataset = FlickrDataset(train_df)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)


# 6. CNN Encoder
class CNNEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)

        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        for parameter in self.cnn.parameters():
            parameter.requires_grad = False

    def forward(self, images):

        with torch.no_grad():
            features = self.cnn(images)

        features = features.view(features.size(0), -1)

        return features


# 7. LSTM Generator
class CaptionGenerator(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, 256, padding_idx=vocab["<PAD>"]
        )

        self.image_to_hidden = nn.Linear(512, 256)
        self.image_to_cell = nn.Linear(512, 256)

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            batch_first=True
        )

        self.fc = nn.Linear(256, vocab_size)

    def forward(self, features, captions):

        embeddings = self.embedding(captions[:, :-1])

        hidden = torch.tanh(self.image_to_hidden(features))
        cell = torch.tanh(self.image_to_cell(features))

        output, _ = self.lstm(
            embeddings,
            (hidden.unsqueeze(0), cell.unsqueeze(0))
        )

        output = self.fc(output)

        return output


# 8. Discriminator
class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, 256, padding_idx=vocab["<PAD>"]
        )

        self.fc = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, features, captions):

        word_embeddings = self.embedding(captions)

        mask = (captions != vocab["<PAD>"]).float()

        sentence_embedding = (
            (word_embeddings * mask.unsqueeze(2)).sum(dim=1)
            / mask.sum(dim=1, keepdim=True).clamp(min=1)
        )

        combined = torch.cat(
            (features, sentence_embedding), dim=1
        )

        return self.fc(combined)


# 9. Initialize models
encoder = CNNEncoder().to(device)

generator = CaptionGenerator(vocab_size).to(device)

discriminator = Discriminator().to(device)


# 10. Loss and optimizers
criterion = nn.CrossEntropyLoss(
    ignore_index=vocab["<PAD>"]
)

adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(
    generator.parameters(), lr=0.0002
)

optimizer_D = optim.Adam(
    discriminator.parameters(), lr=0.0002
)


# 11. Training
epochs = 5

for epoch in range(epochs):

    generator.train()
    discriminator.train()

    total_g_loss = 0
    total_d_loss = 0

    for images, captions in dataloader:

        images = images.to(device)
        captions = captions.to(device)

        features = encoder(images)

        # Generator output
        generated = generator(features, captions)

        # Real caption representation
        real_score = discriminator(features, captions)

        # Predicted tokens for adversarial training
        predicted_tokens = generated.detach().argmax(dim=2)

        # Add start token
        start_tokens = torch.full(
            (predicted_tokens.size(0), 1),
            vocab["<START>"],
            device=device
        )

        fake_captions = torch.cat(
            (start_tokens, predicted_tokens), dim=1
        )

        fake_score = discriminator(features, fake_captions)

        real_labels = torch.ones_like(real_score)
        fake_labels = torch.zeros_like(fake_score)

        # Train discriminator
        loss_real = adversarial_loss(real_score, real_labels)
        loss_fake = adversarial_loss(fake_score, fake_labels)

        loss_D = (loss_real + loss_fake) / 2

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Train generator using caption loss
        generated = generator(features, captions)

        target = captions[:, 1:]

        loss_caption = criterion(
            generated.reshape(-1, vocab_size),
            target.reshape(-1)
        )

        # Generator tries to fool discriminator
        predicted_tokens = generated.argmax(dim=2)

        fake_captions = torch.cat(
            (start_tokens, predicted_tokens), dim=1
        )

        fake_score = discriminator(features, fake_captions)

        loss_G_adv = adversarial_loss(
            fake_score,
            torch.ones_like(fake_score)
        )

        loss_G = loss_caption + 0.1 * loss_G_adv

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        total_g_loss += loss_G.item()
        total_d_loss += loss_D.item()

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Generator Loss: {total_g_loss / len(dataloader):.4f} "
        f"Discriminator Loss: {total_d_loss / len(dataloader):.4f}"
    )


# 12. Generate caption for an image
def generate_caption(image_path):

    encoder.eval()
    generator.eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        features = encoder(image)

        hidden = torch.tanh(
            generator.image_to_hidden(features)
        ).unsqueeze(0)

        cell = torch.tanh(
            generator.image_to_cell(features)
        ).unsqueeze(0)

        input_word = torch.tensor(
            [[vocab["<START>"]]], device=device
        )

        result = []

        for _ in range(25):

            embedding = generator.embedding(input_word)

            output, (hidden, cell) = generator.lstm(
                embedding, (hidden, cell)
            )

            prediction = generator.fc(output[:, -1, :])

            predicted_word = prediction.argmax(dim=1).item()

            if predicted_word == vocab["<END>"]:
                break

            if predicted_word == vocab["<PAD>"]:
                break

            reverse_vocab = {v: k for k, v in vocab.items()}

            word = reverse_vocab.get(predicted_word, "<UNK>")

            result.append(word)

            input_word = torch.tensor(
                [[predicted_word]], device=device
            )

    return " ".join(result)


# 13. Test the model
test_image = os.path.join(
    "images", captions_df.iloc[0]["image"]
)

caption = generate_caption(test_image)

print("\nGenerated Caption:")
print(caption)


# 14. Save models
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")

print("\nModels saved successfully!")
