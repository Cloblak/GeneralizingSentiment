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
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch import nn
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def yield_tokens(data_iter,tokenizer):
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
    vocab = vocab.load("vocab_obj.pth")
    vocab.set_default_index(vocab["<unk>"])
    vocab_size = len(vocab)
    print(vocab_size)
    embed_dim = 64
    num_classes = 3
    # Model class must be defined somewhere
    #net = TextClassificationModel(vocab_size, embed_dim, num_classes)

    model = "fullmodel1.pt"
    print("Loading:", model)
    
    net = torch.load(model)
    
    net = net.to(device)
    # https://pytorch.org/docs/stable/torchvision/models.html


    text_pipeline = lambda x: vocab(tokenizer(x))
    processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64).to(device)
    
    
    
    offset = torch.tensor([0]).to(device)
    
    net.eval()
    
    out = net.forward(processed_text,offset)
        
    
    classes = ["neutral", "postive", "negative"]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    print(prob)
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:3]]
