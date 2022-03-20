#This notebook was used to investiagte and build the cleaned and prepped data for NLP operations.

#The goal is to take in three different datasets and standarize the sentiment from


import pandas as pd
import pyarrow as pa
import os

# os.chdir("../data/raw_unformated/")
os.getcwd()

# Twitter Data Prep

twitter_columns = ['text', 'airline_sentiment']

twitter_df = pd.read_csv('../data/raw_unformated/Tweets.csv', header=0, usecols=twitter_columns)
twitter_df.rename(columns={"airline_sentiment": "sentiment", "input": "text"}, inplace=True)


twitter_df

# Rotten Tomato Data Prep

"""
Citation:
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts
Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)

General Information About the Data:
sentiment_labels.txt contains all phrase ids and the corresponding sentiment labels, separated by a vertical line.
Note that you can recover the 5 classes by mapping the positivity probability using the following cut-offs:
[0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
for very negative, negative, neutral, positive, very positive, respectively.
Please note that phrase ids and sentence ids are not the same.
"""

rot_df = pd.read_json('../data/raw_unformated/reviews_Movies_and_TV_5.json.gz', lines=True, compression='gzip')


# print df
rot_df

# count value of the feature
rot_df['overall'].value_counts()


rot_df_final = pd.DataFrame()
rot_df_final["sentiment"] = rot_df["overall"]
rot_df_final["text"] = rot_df["reviewText"]
rot_df_final["sentiment"] = rot_df_final["sentiment"].replace(1, "negative")
rot_df_final["sentiment"] = rot_df_final["sentiment"].replace(3, "neutral")
rot_df_final["sentiment"] = rot_df_final["sentiment"].replace(5, "positive")

label_list = ["negative", "neutral", "positive"]

# keep only 1, 3, and 5 ratings
rot_df_final = rot_df_final[rot_df_final["sentiment"].isin(label_list)]

rot_df_final


rot_df_final[rot_df_final["sentiment"] == "neutral"]


# Amazon Review Prep

# HuggyFace News Sentiment

import json

"""
Citation:
@InProceedings{Hamborg2021b,
  author    = {Hamborg, Felix and Donnay, Karsten},
  title     = {NewsMTSC: (Multi-)Target-dependent Sentiment Classification in News Articles},
  booktitle = {Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2021)},
  year      = {2021},
  month     = {Apr.},
  location  = {Virtual Event},
}
"""
hf_1 = pd.read_json('../data/raw_unformated/news_sentiment_newsmtsc/dev.jsonl', lines=True)
hf_2 = pd.read_json('../data/raw_unformated/news_sentiment_newsmtsc/test.jsonl', lines=True)
hf_3 = pd.read_json('../data/raw_unformated/news_sentiment_newsmtsc/train.jsonl', lines=True)

# concate
huggyface_df = pd.concat([hf_1, hf_2, hf_3])

# drop duplicate
huggyface_df = huggyface_df.drop_duplicates()

huggyface_df


huggyface_df_final = pd.DataFrame()
huggyface_df_final["sentiment"] = huggyface_df['polarity']
huggyface_df_final["text"] = huggyface_df['sentence']
huggyface_df_final = huggyface_df_final.replace(-1, "negative")
huggyface_df_final = huggyface_df_final.replace(0, "neutral")
huggyface_df_final = huggyface_df_final.replace(1, "positive")
huggyface_df_final


# Combine all datasets

final_df = pd.concat([twitter_df, rot_df_final, huggyface_df_final])

# see the df
final_df

# save the data
final_df.to_parquet("../data/data_prepipeline/full_raw_data.parquet.gzip", compression='gzip')





