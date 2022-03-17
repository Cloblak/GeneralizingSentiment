import os
import numpy as np
import pandas as pd
import string
import time
import copy
from sklearn.linear_model import LogisticRegression
import urllib.request
import zipfile
from model import TextClassificationModel
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch import nn
from sklearn.model_selection import train_test_split
import warnings
from transformers import XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer
from keras.preprocessing.sequence import pad_sequences
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        # Embedding layer
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean", sparse=True)
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


def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)


def read_vocab(path):
    vocab = dict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            index, token = line.split('\t')
            vocab[token] = int(index)
    return vocab


def predict(text):
    #     labeled_df = pd.read_parquet("full_raw_data.parquet.gzip")
    #     labeled_df['sentiment'] = labeled_df['sentiment'].map({"neutral":1,"positive":2,"negative":0})
    #     train_df ,test_df = train_test_split(labeled_df,test_size=0.2)
    #     train_iter = [(label,text) for label,text in zip(train_df['sentiment'].to_list(),train_df['text'].to_list())]

    #     # Build vocabulary from tokens of training set
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
    tokens = tokenizer.tokenize(text)
    input_id = [tokenizer.convert_tokens_to_ids(tokens)]
    MAX_LEN = 128
    input_id = pad_sequences(input_id, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    input_id_t = torch.tensor(input_id).to(device)
    # Model class must be defined somewhere

    model = "../models/pretrained1.pt"
    print("Loading:", model)

    net = torch.load(model)

    net = net.to(device)
    # https://pytorch.org/docs/stable/torchvision/models.html
    attention_masks = []
    for seq in input_id:
      seq_mask = [float(i > 0) for i in seq]
      attention_masks.append(seq_mask)
    mask = torch.tensor(attention_masks).to(device)




    out = net(input_id_t, token_type_ids=None, attention_mask=mask)

    classes = ["negative", "neutral", "positive"]

    prob = torch.nn.functional.softmax(out[0], dim=1)[0] * 100
    print(prob)
    _, indices = torch.sort(out[0], descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:3]]
