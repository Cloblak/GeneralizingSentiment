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

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def yield_tokens(data_iter,tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)

labeled_df = pd.read_csv('clean_text.csv')
data_iter = [(label,text) for label,text in zip(labeled_df['sentiment_id'].to_list(),labeled_df['clean_text'].to_list())]
# Build vocabulary from tokens of training set
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(yield_tokens(data_iter,tokenizer), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


def predict(text):
    # Model class must be defined somewhere
    net = TextClassificationModel()

    model = "model.pt"
    print("Loading:", model)
    net.load_state_dict(
        torch.load("models/" + model, map_location=torch.device("cpu"))
    )

    # https://pytorch.org/docs/stable/torchvision/models.html


    text_pipeline = lambda x: vocab(tokenizer(x))
    processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
    net.eval()
    out = net(processed_text)

    classes = ["0", "1", "2"]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    print(prob)
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:3]]