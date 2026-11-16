import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import pandas as pd
import os
import re
from collections import Counter

transform = transforms.Compose([
    transforms.Resize((224,224)),  #resizes all images to 224 x 224 pixels
    transforms.ToTensor(),  #converts images to PyTorch Tensors and scales pixel values to 0-1
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class Vocabulary:
    def __init__(self):

        self.word_to_index = {
            "<PAD>": 0,
            "<START>": 1,
            "<END>": 2,
            "<UNK>": 3
        }

        self.index_to_word = {
            0: "<PAD>",
            1: "<START>",
            2: "<END>",
            3: "<UNK>"
        }

    def build_vocabulary(self, captions):

        counter = Counter()

        for caption in captions:
            words = re.findall(r"\w+", caption.lower())

            counter.update(words)

        for word in counter:

            if word not in self.word_to_index:
                index = len(self.word_to_index)

                self.word_to_index[word] = index

                self.index_to_word[index] = word

    def numericalize(self, caption):

        words = re.findall(r"\w+", caption.lower())

        tokens = [self.word_to_index["<START>"]]

        for word in words:
            tokens.append(self.word_to_index.get(word, 3))

        tokens.append(self.word_to_index["<END>"])

        return tokens

    def __len__(self):

        return len(self.word_to_index)


#Creating the custom dataset
class ImageCaptionDataset(Dataset):

    def __init__(self, image_dir, caption_file, vocabulary, transform):

        self.image_dir = image_dir

        self.data = pd.read_csv(caption_file)

        self.vocabulary = vocabulary

        self.transform = transform  #stores the image preprocessing pipeline

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):
        image_name = self.data.iloc[index, 0]  #retrieves the image filename from the first column
        caption = self.data.iloc[index, 1]     #retrives its caption from the second column
        image_path = os.path.join(self.image_dir, image_name)  #creates the complete image path
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)   #applies image preprocessing
        caption = self.vocabulary.numericalize(caption) #converts the caption into numerical IDs
        caption = torch.tensor(caption)   #converts the numerical sequence into a tensor

        return image, caption
# This class allows PyTorch to load images and their corresponding captions as individual training samples.


#Padding and Dataloader
def collate_fn(batch):  #to combine multiple samples into one batch

    images = []   #to store images
    captions = []  #to store captions

    for image, caption in batch:  #iterating thru every sample in the batch
        images.append(image)
        captions.append(caption)

    images = torch.stack(images)  #combines image tensors into a single batch tensor

    captions = nn.utils.rnn.pad_sequence(    #padding shorter captions to match the longest caption in the batch
        captions,
        batch_first=True, #makes the first dimension represent the batch size
        padding_value=0  #using the pad token to fill the shorter sequences
    )

    return images, captions

#before padding
# [1, 4, 5, 2]
# [1, 4, 6, 7, 8, 2]


# after padding
# [1, 4, 5, 2, 0, 0]
# [1, 4, 6, 7, 8, 2]


#dataset loading
image_dir = "Images"

caption_file = "captions.txt"

data = pd.read_csv(caption_file)   #reading all caption records

vocabulary = Vocabulary()    #creating vocab object

vocabulary.build_vocabulary(data["caption"].tolist())   #creating vocab using all available captions

dataset = ImageCaptionDataset(   #creates the custom dataset
    image_dir,
    caption_file,
    vocabulary,
    transform
)

loader = DataLoader(     #creates a dataloader for training
    dataset,
    batch_size=16,     #16 image caption records in each batch
    shuffle=False,
    collate_fn=collate_fn
)

print("Vocabulary size:", len(vocabulary))
print("Total caption records:", len(dataset))


#LSTM for Image captioning
class LSTMImageCaptioningModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):  #vocab_size = total no. of unique tokens in vocab ; embedding_dim:dim of each word embedding;hidden_dim:no. of features in the LSTM hidden state and cell state
        super().__init__()

        # CNN Encoder
        resnet = models.resnet18(  #ResNet18 is a Convolutional Neural Network (CNN) that extracts visual features from images.
            weights=models.ResNet18_Weights.DEFAULT
        )


        #removing the classification layer
        self.encoder = nn.Sequential(
            *list(resnet.children())[:-1]
        )
        #resnet.children() retrieves the individual ResNet18 layers
        #[:-1]: removing the final classification layer
        #nn.Sequential(): combining the remaining layers
        #We remove the classification layer because we do not want ImageNet class predictions. We want image features.

        for param in self.encoder.parameters():
            param.requires_grad = False   #Iterates through all CNN parameters and disables gradient updates
            #This means the pretrained ResNet18 weights remain unchanged during our training.

        # Initialize LSTM hidden state
        self.init_hidden = nn.Linear(512, hidden_dim)  #Creates a fully connected layer that converts the 512-dimensional image features into the initial LSTM hidden state.
        #ho = tanh(Whf + bh)

        # Initialize LSTM cell state
        self.init_cell = nn.Linear(512, hidden_dim)

        # Word embedding: we convert the numerical word IDs into dense vectors
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,  #I/P dim of each word vector
            hidden_dim,     #size of hidden state nd cell state
            batch_first=True  #I/P format has batch size as the first dimension
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)  #It converts the LSTM hidden representation into scores for every word in the vocabulary.

    def forward(self, features, captions):

        hidden = torch.tanh(self.init_hidden(features))

        cell = torch.tanh(self.init_cell(features))
        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)
        #For a batch of 16 images and hidden dimension 256, both states change from: (16,256) -> (1,16,256)

        captions = captions[:, :-1] #removing the final token from each caption so that the LSTM can receive previous words and learn to predict the next word
        embeddings = self.embedding(captions)

        output, (hidden, cell) = self.lstm(   #we pass word embeddings, initial hidden state and initial cell state
            embeddings,
            (hidden, cell)
        )

        output = self.fc(output)  #generating word prediction score
        return output


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = LSTMImageCaptioningModel(
    vocab_size=len(vocabulary),
    embedding_dim=256,
    hidden_dim=256
).to(device)

print(model)

# ####### Added bcoz of some error
# checkpoint = torch.load(
#     "lstm_captioning_checkpoint.pth",
#     map_location=device
# )
# ########

# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()

criterion = nn.CrossEntropyLoss(ignore_index=0)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)


#extracting image features using the pretrained ResNet18
features_list = []
captions_list = []

model.encoder.eval()

with torch.no_grad():

    for images, captions in loader:

        images = images.to(device)

        features = model.encoder(images)

        features = features.reshape(features.size(0), -1)

        for i in range(features.size(0)):
            features_list.append(features[i].cpu())
            captions_list.append(captions[i].cpu())


#Padding and Dataloader for extracted features
def collate_feature_fn(batch):

    features = []
    captions = []

    for feature, caption in batch:
        features.append(feature)
        captions.append(caption)

    features = torch.stack(features)

    captions = pad_sequence(
        captions,
        batch_first=True,
        padding_value=0
    )

    return features, captions


feature_dataset = list(zip(features_list, captions_list))

feature_loader = DataLoader(
    feature_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_feature_fn
)


#training the LSTM model
num_epochs = 10

for epoch in range(num_epochs):

    model.train()
    total_loss = 0

    for features, captions in feature_loader:

        features = features.to(device)
        captions = captions.to(device)

        outputs = model(features, captions)
        targets = captions[:, 1:]

        loss = criterion(
            outputs.reshape(-1, outputs.size(2)),
            targets.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_loss = total_loss / len(feature_loader)

    print(
        "Epoch:", epoch + 1,
        "Loss:", epoch_loss
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss
        },
        "lstm_captioning_checkpoint.pth"
    )


#now since we have trained the LSTM model ,
#evaluation -
#generate captions for new images:
def generate_caption(model, image_path, vocabulary):

    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encoder(image) #extract image features using ResNet18
        features = features.reshape(features.size(0), -1)  #CNN o/p -> 2D feature matrix

        hidden = torch.tanh(model.init_hidden(features))  #initialising the LSTM hidden state using image features
        cell = torch.tanh(model.init_cell(features))  #same as above but initialising LSTM cell state

        hidden = hidden.unsqueeze(0)

        cell = cell.unsqueeze(0)

        word = torch.tensor(
            [[vocabulary.word_to_index["<START>"]]]
        ).to(device)

        caption = []
        for i in range(30):   #allowing a max. of 30 generation steps

            embedding = model.embedding(word)

            output, (hidden, cell) = model.lstm(  #processing the current word using the prev hidden and cell states
                embedding,
                (hidden, cell)
            )

            output = model.fc(output.squeeze(1))  #converting LSTM o/p into vocab prediction scores
            predicted_word = output.argmax(1).item()  #selects the word with the highest prediction score

            if predicted_word == vocabulary.word_to_index["<END>"]:
                break

            caption.append(vocabulary.index_to_word[predicted_word])  #converting the predicted ID into an English word and stores it

            word = torch.tensor([[predicted_word]]).to(device)  #feeding the predicted word back into the LSTM for the next prediction

    return " ".join(caption)  #combining the generated words into a readable sentence


#testing and saving the model
generated_caption = generate_caption(
    model,
    "Images/103106960_e8a41d64f8.jpg",
    vocabulary
)

print("Generated Caption:", generated_caption)

torch.save(
    model.state_dict(),
    "lstm_captioning.pth"
)













# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# import torchvision.models as models
#
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data import TensorDataset
# from torch.nn.utils.rnn import pad_sequence
# from PIL import Image
# import pandas as pd
# import os
# import re
# from collections import Counter
#
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
#
#
# class Vocabulary:
#     def __init__(self):
#
#         self.word_to_index = {
#             "<PAD>": 0,
#             "<START>": 1,
#             "<END>": 2,
#             "<UNK>": 3
#         }
#
#         self.index_to_word = {
#             0: "<PAD>",
#             1: "<START>",
#             2: "<END>",
#             3: "<UNK>"
#         }
#
#     def build_vocabulary(self, captions):
#
#         counter = Counter()
#
#         for caption in captions:
#             words = re.findall(r"\w+", caption.lower())
#
#             counter.update(words)
#
#         for word in counter:
#
#             if word not in self.word_to_index:
#                 index = len(self.word_to_index)
#
#                 self.word_to_index[word] = index
#
#                 self.index_to_word[index] = word
#
#     def numericalize(self, caption):
#
#         words = re.findall(r"\w+", caption.lower())
#
#         tokens = [self.word_to_index["<START>"]]
#
#         for word in words:
#             tokens.append(self.word_to_index.get(word, 3))
#
#         tokens.append(self.word_to_index["<END>"])
#
#         return tokens
#
#     def __len__(self):
#
#         return len(self.word_to_index)
#
#
# class ImageCaptionDataset(Dataset):
#
#     def __init__(self, image_dir, caption_file, vocabulary, transform):
#
#         self.image_dir = image_dir
#
#         self.data = pd.read_csv(caption_file)
#
#         self.vocabulary = vocabulary
#
#         self.transform = transform
#
#     def __len__(self):
#
#         return len(self.data)
#
#     def __getitem__(self, index):
#         image_name = self.data.iloc[index, 0]
#         caption = self.data.iloc[index, 1]
#         image_path = os.path.join(self.image_dir, image_name)
#         image = Image.open(image_path).convert("RGB")
#         image = self.transform(image)
#         caption = self.vocabulary.numericalize(caption)
#         caption = torch.tensor(caption)
#
#         return image, caption
#
#
# def collate_fn(batch):
#
#     images = []
#     captions = []
#
#     for image, caption in batch:
#         images.append(image)
#         captions.append(caption)
#
#     images = torch.stack(images)
#
#     captions = nn.utils.rnn.pad_sequence(
#         captions,
#         batch_first=True,
#         padding_value=0
#     )
#
#     return images, captions
#
#
# image_dir = "Images"
#
# caption_file = "captions.txt"
#
# data = pd.read_csv(caption_file)
#
# vocabulary = Vocabulary()
#
# vocabulary.build_vocabulary(data["caption"].tolist())
#
# dataset = ImageCaptionDataset(
#     image_dir,
#     caption_file,
#     vocabulary,
#     transform
# )
#
# loader = DataLoader(
#     dataset,
#     batch_size=16,
#     shuffle=False,
#     collate_fn=collate_fn
# )
#
# print("Vocabulary size:", len(vocabulary))
# print("Total caption records:", len(dataset))
#
#
# class LSTMImageCaptioningModel(nn.Module):
#
#     def __init__(self, vocab_size, embedding_dim, hidden_dim):
#         super().__init__()
#
#         resnet = models.resnet18(
#             weights=models.ResNet18_Weights.DEFAULT
#         )
#
#         self.encoder = nn.Sequential(
#             *list(resnet.children())[:-1]
#         )
#
#         for param in self.encoder.parameters():
#             param.requires_grad = False
#
#         self.init_hidden = nn.Linear(512, hidden_dim)
#
#         self.init_cell = nn.Linear(512, hidden_dim)
#
#         self.embedding = nn.Embedding(
#             vocab_size,
#             embedding_dim,
#             padding_idx=0
#         )
#
#         self.lstm = nn.LSTM(
#             embedding_dim,
#             hidden_dim,
#             batch_first=True
#         )
#
#         self.fc = nn.Linear(hidden_dim, vocab_size)
#
#     def forward(self, features, captions):
#
#         hidden = torch.tanh(self.init_hidden(features))
#
#         cell = torch.tanh(self.init_cell(features))
#         hidden = hidden.unsqueeze(0)
#         cell = cell.unsqueeze(0)
#
#         captions = captions[:, :-1]
#         embeddings = self.embedding(captions)
#
#         output, (hidden, cell) = self.lstm(
#             embeddings,
#             (hidden, cell)
#         )
#
#         output = self.fc(output)
#         return output
#
#
# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )
#
# model = LSTMImageCaptioningModel(
#     vocab_size=len(vocabulary),
#     embedding_dim=256,
#     hidden_dim=256
# ).to(device)
#
# print(model)
#
# ####### Added bcoz of some error
# checkpoint = torch.load(
#     "lstm_captioning_checkpoint.pth",
#     map_location=device
# )
# ########
#
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()
#
# criterion = nn.CrossEntropyLoss(ignore_index=0)
#
# optimizer = optim.Adam(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=0.001
# )
#
#
# features_list = []
# captions_list = []
#
# model.encoder.eval()
#
# with torch.no_grad():
#
#     for images, captions in loader:
#
#         images = images.to(device)
#
#         features = model.encoder(images)
#
#         features = features.reshape(features.size(0), -1)
#
#         for i in range(features.size(0)):
#             features_list.append(features[i].cpu())
#             captions_list.append(captions[i].cpu())
#
#
# def collate_feature_fn(batch):
#
#     features = []
#     captions = []
#
#     for feature, caption in batch:
#         features.append(feature)
#         captions.append(caption)
#
#     features = torch.stack(features)
#
#     captions = pad_sequence(
#         captions,
#         batch_first=True,
#         padding_value=0
#     )
#
#     return features, captions
#
#
# feature_dataset = list(zip(features_list, captions_list))
#
# feature_loader = DataLoader(
#     feature_dataset,
#     batch_size=16,
#     shuffle=True,
#     collate_fn=collate_feature_fn
# )
#
#
# num_epochs = 10
#
# for epoch in range(num_epochs):
#
#     model.train()
#     total_loss = 0
#
#     for features, captions in feature_loader:
#
#         features = features.to(device)
#         captions = captions.to(device)
#
#         outputs = model(features, captions)
#         targets = captions[:, 1:]
#
#         loss = criterion(
#             outputs.reshape(-1, outputs.size(2)),
#             targets.reshape(-1)
#         )
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#     epoch_loss = total_loss / len(feature_loader)
#
#     print(
#         "Epoch:", epoch + 1,
#         "Loss:", epoch_loss
#     )
#
#     torch.save(
#         {
#             "epoch": epoch + 1,
#             "model_state_dict": model.state_dict(),
#             "optimizer_state_dict": optimizer.state_dict(),
#             "loss": epoch_loss
#         },
#         "lstm_captioning_checkpoint.pth"
#     )
#
#
# def generate_caption(model, image_path, vocabulary):
#
#     model.eval()
#     image = Image.open(image_path).convert("RGB")
#     image = transform(image).unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         features = model.encoder(image)
#         features = features.reshape(features.size(0), -1)
#
#         hidden = torch.tanh(model.init_hidden(features))
#         cell = torch.tanh(model.init_cell(features))
#
#         hidden = hidden.unsqueeze(0)
#
#         cell = cell.unsqueeze(0)
#
#         word = torch.tensor(
#             [[vocabulary.word_to_index["<START>"]]]
#         ).to(device)
#
#         caption = []
#         for i in range(30):
#
#             embedding = model.embedding(word)
#
#             output, (hidden, cell) = model.lstm(
#                 embedding,
#                 (hidden, cell)
#             )
#
#             output = model.fc(output.squeeze(1))
#             predicted_word = output.argmax(1).item()
#
#             if predicted_word == vocabulary.word_to_index["<END>"]:
#                 break
#
#             caption.append(vocabulary.index_to_word[predicted_word])
#
#             word = torch.tensor([[predicted_word]]).to(device)
#
#     return " ".join(caption)
#
#
# generated_caption = generate_caption(
#     model,
#     "Images/103106960_e8a41d64f8.jpg",
#     vocabulary
# )
#
# print("Generated Caption:", generated_caption)
#
# torch.save(
#     model.state_dict(),
#     "lstm_captioning.pth"
# )











# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# import torchvision.models as models
#
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import pandas as pd
# import os
# import re
# from collections import Counter
#
# transform = transforms.Compose([
#     transforms.Resize((224,224)),  #resizes all images to 224 x 224 pixels
#     transforms.ToTensor(),  #converts images to PyTorch Tensors and scales pixel values to 0-1
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
#
#
# class Vocabulary:
#     def __init__(self):
#
#         self.word_to_index = {
#             "<PAD>": 0,
#             "<START>": 1,
#             "<END>": 2,
#             "<UNK>": 3
#         }
#
#         self.index_to_word = {
#             0: "<PAD>",
#             1: "<START>",
#             2: "<END>",
#             3: "<UNK>"
#         }
#
#     def build_vocabulary(self, captions):
#
#         counter = Counter()
#
#         for caption in captions:
#             words = re.findall(r"\w+", caption.lower())
#
#             counter.update(words)
#
#         for word in counter:
#
#             if word not in self.word_to_index:
#                 index = len(self.word_to_index)
#
#                 self.word_to_index[word] = index
#
#                 self.index_to_word[index] = word
#
#     def numericalize(self, caption):
#
#         words = re.findall(r"\w+", caption.lower())
#
#         tokens = [self.word_to_index["<START>"]]
#
#         for word in words:
#             tokens.append(self.word_to_index.get(word, 3))
#
#         tokens.append(self.word_to_index["<END>"])
#
#         return tokens
#
#     def __len__(self):
#
#         return len(self.word_to_index)
#
# #Creating the custom dataset
# class ImageCaptionDataset(Dataset):
#
#     def __init__(self, image_dir, caption_file, vocabulary, transform):
#
#         self.image_dir = image_dir
#
#         self.data = pd.read_csv(caption_file)
#
#         self.vocabulary = vocabulary
#
#         self.transform = transform  #stores the image preprocessing pipeline
#
#     def __len__(self):
#
#         return len(self.data)
#
#     def __getitem__(self, index):
#         image_name = self.data.iloc[index, 0]  #retrieves the image filename from the first column
#         caption = self.data.iloc[index, 1]     #retrives its caption from the second column
#         image_path = os.path.join(self.image_dir, image_name)  #creates the complete image path
#         image = Image.open(image_path).convert("RGB")
#         image = self.transform(image)   #applies image preprocessing
#         caption = self.vocabulary.numericalize(caption) #converts the caption into numerical IDs
#         caption = torch.tensor(caption)   #converts the numerical sequence into a tensor
#
#         return image, caption
# # This class allows PyTorch to load images and their corresponding captions as individual training samples.
#
# #Padding and Dataloader
# def collate_fn(batch):  #to combine multiple samples into one batch
#
#     images = []   #to store images
#     captions = []  #to store captions
#
#     for image, caption in batch:  #iterating thru every sample in the batch
#         images.append(image)
#         captions.append(caption)
#     images = torch.stack(images)  #combines image tensors into a single batch tensor
#
#     captions = nn.utils.rnn.pad_sequence(    #padding shorter captions to match the longest caption in the batch
#         captions,
#         batch_first=True, #makes the first dimension represent the batch size
#         padding_value=0  #using the pad token to fill the shorter sequences
#     )
#
#     return images, captions
#
# #before padding
# # [1, 4, 5, 2]
# # [1, 4, 6, 7, 8, 2]
#
#
# # after padding
# # [1, 4, 5, 2, 0, 0]
# # [1, 4, 6, 7, 8, 2]
#
#
# #dataset loading
# image_dir = "Images"
#
# caption_file = "captions.txt"
#
# data = pd.read_csv(caption_file)   #reading all caption records
#
# vocabulary = Vocabulary()    #creating vocab object
#
# vocabulary.build_vocabulary(data["caption"].tolist())   #creating vocab using all available captions
#
# dataset = ImageCaptionDataset(   #creates the custom dataset
#     image_dir,
#     caption_file,
#     vocabulary,
#     transform
# )
#
# loader = DataLoader(     #creates a dataloader for training
#     dataset,
#     batch_size=32,     #32 image caption records in each batch
#     shuffle=True,
#     collate_fn=collate_fn
# )
#
# print("Vocabulary size:", len(vocabulary))
# print("Total caption records:", len(dataset))
#
#
# #LSTM for Image captioning
# class LSTMImageCaptioningModel(nn.Module):
#
#     def __init__(self, vocab_size, embedding_dim, hidden_dim):  #vocab_size = total no. of unique tokens in vocab ; embedding_dim:dim of each word embedding;hidden_dim:no. of features in the LSTM hidden state and cell state
#         super().__init__()
#
#         # CNN Encoder
#         resnet = models.resnet18(  #ResNet18 is a Convolutional Neural Network (CNN) that extracts visual features from images.
#             weights=models.ResNet18_Weights.DEFAULT
#         )
#
#
#         #removing the classification layer
#         self.encoder = nn.Sequential(
#             *list(resnet.children())[:-1]
#         )
#         #resnet.children() retrieves the individual ResNet18 layers
#         #[:-1]: removing the final classification layer
#         #nn.Sequential(): combining the remaining layers
#         #We remove the classification layer because we do not want ImageNet class predictions. We want image features.
#
#         for param in self.encoder.parameters():
#             param.requires_grad = False   #Iterates through all CNN parameters and disables gradient updates
#             #This means the pretrained ResNet18 weights remain unchanged during our training.
#
#         # Initialize LSTM hidden state
#         self.init_hidden = nn.Linear(512, hidden_dim)  #Creates a fully connected layer that converts the 512-dimensional image features into the initial LSTM hidden state.
#         #ho = tanh(Whf + bh)
#
#         # Initialize LSTM cell state
#         self.init_cell = nn.Linear(512, hidden_dim)
#
#         # Word embedding: we convert the numerical word IDs into dense vectors
#         self.embedding = nn.Embedding(
#             vocab_size,
#             embedding_dim,
#             padding_idx=0
#         )
#
#         # LSTM layer
#         self.lstm = nn.LSTM(
#             embedding_dim,  #I/P dim of each word vector
#             hidden_dim,     #size of hidden state nd cell state
#             batch_first=True  #I/P format has batch size as the first dimension
#         )
#
#         # Output layer
#         self.fc = nn.Linear(hidden_dim, vocab_size)  #It converts the LSTM hidden representation into scores for every word in the vocabulary.
#
#     def forward(self, images, captions):
#
#         with torch.no_grad():  #diabling gradient as CNN is frozen
#             features = self.encoder(images)
#         features = features.reshape(features.size(0), -1)   #flatten the CNN O/P into 2D tensor
#
#         hidden = torch.tanh(self.init_hidden(features))
#
#         cell = torch.tanh(self.init_cell(features))
#         hidden = hidden.unsqueeze(0)
#         cell = cell.unsqueeze(0)
#         #For a batch of 32 images and hidden dimension 256, both states change from: (32,256) -> (1,32,256)
#
#         captions = captions[:, :-1] #removing the final token from each caption so that the lSTM can receive previous words nd learn to predict the next word
#         embeddings = self.embedding(captions)
#
#         output, (hidden, cell) = self.lstm(   #we pass word embeddings, initial hidden state and initial cell state
#             embeddings,
#             (hidden, cell)
#         )
#
#         output = self.fc(output)  #generating word prediction score
#         return output
#
#
# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )
#
# model = LSTMImageCaptioningModel(
#     vocab_size=len(vocabulary),
#     embedding_dim=256,
#     hidden_dim=256
# ).to(device)
#
# print(model)
#
# criterion = nn.CrossEntropyLoss(ignore_index=0)
#
# optimizer = optim.Adam(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=0.001
# )
#
# #training the LSTM model
# num_epochs = 10
# for epoch in range(num_epochs):
#
#     model.train()
#     total_loss = 0
#
#     for images, captions in loader:  #loading one batch of image-caption pairs
#
#         images = images.to(device)
#         captions = captions.to(device)
#
#         outputs = model(images, captions)
#         targets = captions[:, 1:]
#         loss = criterion(
#             outputs.reshape(-1, outputs.size(2)),
#             targets.reshape(-1)
#         )
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#
#     print(
#         "Epoch:", epoch + 1,
#         "Loss:", total_loss / len(loader)
#     )
#
#
#
# #now since we have trained the LSTM model ,
# #evaluation -
# #generate captions for new images:
# def generate_caption(model, image_path, vocabulary):
#
#     model.eval()
#     image = Image.open(image_path).convert("RGB")
#     image = transform(image).unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         features = model.encoder(image) #extract image features using ResNet18
#         features = features.reshape(features.size(0), -1)  #Cnn o/p -> 2D feature matrix
#
#         hidden = torch.tanh(model.init_hidden(features))  #initialising the lstm hidden state using image features
#         cell = torch.tanh(model.init_cell(features))  #same as above but initialising lstm cell state
#
#         hidden = hidden.unsqueeze(0)
#
#         cell = cell.unsqueeze(0)
#
#         word = torch.tensor(
#             [[vocabulary.word_to_index["<START>"]]]
#         ).to(device)
#
#         caption = []
#         for i in range(30):   #allowing a max. of 30 generation steps
#
#             embedding = model.embedding(word)
#
#             output, (hidden, cell) = model.lstm(  #processing the current word using the prev hidden and cell states
#                 embedding,
#                 (hidden, cell)
#             )
#
#             output = model.fc(output.squeeze(1))  #converting LSTM o/p into vocab prediction scores
#             predicted_word = output.argmax(1).item()  #selects the word with the highest prediction score
#
#             if predicted_word == vocabulary.word_to_index["<END>"]:
#                 break
#
#             caption.append(vocabulary.index_to_word[predicted_word])  #converting the predicted ID into an english word and stores it
#
#             word = torch.tensor([[predicted_word]]).to(device)  #feeding the predicted word back into the LSTM for the next prediction
#
#     return " ".join(caption)  #combining the generated words into a readable sentence
#
#
#
# #testing and saving the model
# generated_caption = generate_caption(
#     model,
#     "Images/image1.jpg",
#     vocabulary
# )
#
# print("Generated Caption:", generated_caption)
#
# torch.save(
#     model.state_dict(),
#     "lstm_captioning.pth"
# )