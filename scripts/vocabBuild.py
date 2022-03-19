import torch
import os
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
from sklearn.model_selection import train_test_split

def save_vocab(vocab, path):
    with open(path, 'w+', encoding='utf-8') as f:     
        for token, index in vocab.stoi.items():
            f.write(f'{index}\t{token}\n')

def yield_tokens(data_iter,tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)

if __name__ == "__main__":
    print(os.getcwd())
    labeled_df = pd.read_parquet("../data/data_prepipeline/full_raw_data.parquet.gzip")
    labeled_df['sentiment'] = labeled_df['sentiment'].map({"neutral":1,"positive":2,"negative":0})
    train_df ,test_df = train_test_split(labeled_df,test_size=0.2)
    train_iter = [(label,text) for label,text in zip(train_df['sentiment'].to_list(),train_df['text'].to_list())]
  
  
    # Build vocabulary from tokens of training set
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter,tokenizer), specials=["<unk>"])

    torch.save(vocab, "vocab_obj.pth")