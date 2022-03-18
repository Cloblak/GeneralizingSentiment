from os.path import join
from google.colab import drive

ROOT = "/content/drive"
drive.mount(ROOT, force_remount=False)

!ls


%cd /content/drive/"My Drive"/AIPI540_NLP/data


import os
import numpy as np
import pandas as pd
import string
import time
import copy
from sklearn.linear_model import LogisticRegression
import urllib.request
import zipfile

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch import nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm



import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = pd.read_parquet("full_raw_data.parquet.gzip")
print(data.count())

data.head()


data['sentiment'] = data['sentiment'].map({"neutral":1,"positive":2,"negative":3})

# Divide training data by class
df_class3 = data[data['sentiment'] == 3]
df_class1 = data[data['sentiment'] == 1]
df_class2 = data[data['sentiment'] == 2]


# Count the number in each class
count_class0 = (data['sentiment'] ==0 ).sum()
count_class1 = (data['sentiment'] ==1 ).sum()
count_class2 = (data['sentiment'] ==2 ).sum()


# Resample class 0 so count is equal to class 1
df_class2_under = df_class2.sample(n=count_class0)
df_class1_under = df_class1.sample(n=count_class0)
# Add the undersampled class 0 together with class 1
df_undersampled = pd.concat([df_class2_under,df_class1_under,df_class0])


df_undersampled.sentiment.value_counts()
data = df_undersampled


# split dataset
train_df ,test_df = train_test_split(data,test_size=0.2)



# Put data in iterator form needed to create PyTorch Datasets from data
train_iter = [(label,text) for label,text in zip(train_df['sentiment'].to_list(),train_df['text'].to_list())]
test_iter = [(label,text) for label,text in zip(test_df['sentiment'].to_list(),test_df['text'].to_list())]

# Create PyTorch Datasets from iterators
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)

# Split training data to get a validation set
num_train = int(len(train_dataset) * 0.95)
split_train_dataset, split_valid_dataset = random_split(train_dataset, [num_train, len(train_dataset) - num_train])


# Function to tokenize the text
def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)


# Build vocabulary from tokens of training set
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


# Define collate_batch function to get single collated tensor for batch in form needed by nn.EmbeddingBag
def collate_batch(batch, tokenizer, vocab):
    # Pipelines for processing text and labels
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1

    label_list, text_list, offsets = [], [], [0]
    # Iterate through batch, processing text and adding text, labels and offsets to lists
    for (label, text) in batch:
        label_list.append(label_pipeline(label))
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)



batch_size = 64
# Create training, validation and test set DataLoaders using custom collate_batch function
train_dataloader = DataLoader(split_train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn = lambda x: collate_batch(x,tokenizer,vocab))
val_dataloader = DataLoader(split_valid_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn = lambda x: collate_batch(x,tokenizer,vocab))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=True, collate_fn = lambda x: collate_batch(x,tokenizer,vocab))

# Set up dict for dataloaders to use in training
train_dataloaders = {'train':train_dataloader,'val':val_dataloader}

# Store size of training and validation sets
dataset_sizes = {'train':len(split_train_dataset),'val':len(split_valid_dataset)}


# Define the model
class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        # Embedding layer
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean",sparse=True)
        # Fully connected final layer to convert embeddings to output predictions
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)



def train_model(model, criterion, optimizer, dataloaders, scheduler, device, num_epochs=5):
    model = model.to(device) # Send model to GPU if available
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Get the input images and labels, and send to GPU if available
            for (labels, text, offsets) in dataloaders[phase]:
                text = text.to(device)
                labels = labels.to(device)
                offsets = offsets.to(device)

                # Zero the weight gradients
                optimizer.zero_grad()

                # Forward pass to get outputs and calculate loss
                # Track gradient only for training data
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(text,offsets)
                    loss = criterion(outputs, labels)

                    # Backpropagation to get the gradients with respect to each weight
                    # Only if in train
                    if phase == 'train':
                        loss.backward()
                        # Update the weights
                        optimizer.step()

                # Convert loss into a scalar and add it to running_loss
                running_loss += loss.item() * labels.size(0)
                # Track number of correct predictions
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            # Step along learning rate scheduler when in train
            if phase == 'train':
                scheduler.step()

            # Calculate and display average loss and accuracy for the epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} loss: {:.4f} accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # If model performs better on val set, save weights as the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation set accuracy: {:3f}'.format(best_acc))

    # Load the weights from best model
    model.load_state_dict(best_model_wts)

    return model

# Instantiate the model
num_classes = len(set([label for (label, _) in train_iter]))
vocab_size = len(vocab)
embed_dim = 64 # Set desired document embedding size
nn_model = TextClassificationModel(vocab_size, embed_dim, num_classes)

# Set hyperparameters
epochs = 10 # epoch
learning_rate = 1.  # learning rate

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(nn_model.parameters(), lr=learning_rate)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1 , gamma=0.8)

# Train the model
nn_model = train_model(nn_model, criterion, optimizer, train_dataloaders, lr_scheduler, device, num_epochs=10)


def evaluate(dataloader, model):
    # Generate predictions and calculate accuracy
    nn_model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model.forward(text, offsets)
            #loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


# Evaluate performance on the test dataset
accu_test = evaluate(test_dataloader, nn_model)
print('test set accuracy {:8.3f}'.format(accu_test))


# save model
filename = 'fullmodel1.pt'

torch.save(nn_model, filename)






































