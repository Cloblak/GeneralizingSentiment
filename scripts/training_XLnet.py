# mount to drive
from os.path import join
from google.colab import drive

ROOT = "/content/drive"
drive.mount(ROOT, force_remount=False)


!ls

# change directory
%cd /content/drive/"My Drive"/AIPI540_NLP/data

# import package
import math
import os
import numpy as np
import pandas as pd
import string
import time
import copy
from sklearn.linear_model import LogisticRegression
import urllib.request
import zipfile
from torch.utils.tensorboard import SummaryWriter
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch import nn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import torch.nn.functional as F
from tqdm import tqdm,trange
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer
import sentencepiece
import tensorflow as tf

from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences


import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
% matplotlib inline


# set and check device
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

# ceck data, label and balance
data = pd.read_parquet("full_raw_data.parquet.gzip")
print(data.count())


data['sentiment'] = data['sentiment'].map({"neutral":1,"positive":2,"negative":0})

# Divide training data by class
df_class0 = data[data['sentiment'] == 0]
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


df = data.sample(frac = 0.2)

len(df)

data = df

# Get sentence data
sentences = data.text.values
sentences = [sentence + " [SEP] [CLS]" for sentence in sentences]


# Get tag labels data
labels = data.sentiment.values
print(labels[0])


sentences[0]



# set tokeniziation
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

tokenized_texts = [tokenizer.tokenize(sent) for sent in tqdm(sentences)]
print ("Tokenize the first sentence:")
print (tokenized_texts[0])

# set max lenth
MAX_LEN = 128

# tokenize
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tqdm(tokenized_texts)]


input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in tqdm(input_ids):
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)

#Use train_test_split to split our data into train and validation sets for training

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                            random_state=56, test_size=0.2)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=56, test_size=0.2)

# Convert all of our data into torch tensors, the required datatype for our model

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)


# Select a batch size for training. For fine-tuning with XLNet, the authors recommend a batch size of 32, 48, or 128. We will use 32 here to avoid memory issues.
batch_size = 32

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


# Load XLNEtForSequenceClassification, the pretrained XLNet model with a single linear classification layer on top.

model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=3)
model.cuda()

# set optimizer parameters
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]


# This variable contains all of the hyperparemeter information our training loop needs
optimizer = AdamW(optimizer_grouped_parameters,
                     lr=2e-5)

# Store our loss and accuracy for plotting
train_loss_set = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 2

# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc="Epoch"):

    # Training

    # Set our model to training mode (as opposed to evaluation mode)
    model.train()

    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    # Train the data for one epoch
    for step, batch in tqdm(enumerate(train_dataloader)):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        logits = outputs[1]
        train_loss_set.append(loss.item())
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss / nb_tr_steps))




# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for i, batch in enumerate(validation_dataloader):
      batch = tuple(t.to(device) for t in batch)
      # Unpack the inputs from our dataloader
      b_input_ids, b_input_mask, b_labels = batch
      # Forward pass
      outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
      # print (outputs)
      prediction = torch.argmax(outputs[0],dim=1)
      total += b_labels.size(0)
      correct+=(prediction==b_labels).sum().item()

print('Test Accuracy of the model on vla data is: {} %'.format(100 * correct / total))

# save model
%cd /content/drive/"My Drive"/AIPI540_NLP


torch.save(model.state_dict(),'/model_1percent.pt')


filename = 'pretrained2.pt'

torch.save(model, filename)


