import pandas as pd
import os



if __name__ == "__main__":
    
    print("Building Labeled Data....")
    
    # Adding Twitter Data Prep
    
    os.chdir("..")

    twitter_columns = ['text', 'airline_sentiment']

    twitter_df = pd.read_csv('data/raw_unformated/Tweets.csv', header=0, usecols=twitter_columns)
    twitter_df.rename(columns={"airline_sentiment": "sentiment", "input": "text"}, inplace=True)

    # Adding Amazon Movie and TV Data Prep

    """
    Citation:
    Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering
    R. He, J. McAuley
    WWW, 2016
    pdf

    Image-based recommendations on styles and substitutes
    J. McAuley, C. Targett, J. Shi, A. van den Hengel
    SIGIR, 2015
    pdf
    """

    amz_df = pd.read_json('data/raw_unformated/reviews_Movies_and_TV_5.json.gz', lines=True, compression='gzip')

    amz_df_final = pd.DataFrame()
    amz_df_final["sentiment"] = amz_df["overall"]
    amz_df_final["text"] = amz_df["reviewText"]
    amz_df_final["sentiment"] = amz_df_final["sentiment"].replace(1, "negative")
    amz_df_final["sentiment"] = amz_df_final["sentiment"].replace(3, "neutral")
    amz_df_final["sentiment"] = amz_df_final["sentiment"].replace(5, "positive")

    label_list = ["negative", "neutral", "positive"]

    # keep only 1, 3, and 5 ratings
    amz_df_final =amz_df_final[amz_df_final["sentiment"].isin(label_list)]

    # Adding HuggyFace

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
    hf_1 = pd.read_json('data/raw_unformated/news_sentiment_newsmtsc/dev.jsonl', lines=True)
    hf_2 = pd.read_json('data/raw_unformated/news_sentiment_newsmtsc/test.jsonl', lines=True)
    hf_3 = pd.read_json('data/raw_unformated/news_sentiment_newsmtsc/train.jsonl', lines=True)

    huggyface_df = pd.concat([hf_1, hf_2, hf_3])

    huggyface_df = huggyface_df.drop_duplicates()

    huggyface_df_final = pd.DataFrame()
    huggyface_df_final["sentiment"] = huggyface_df['polarity']
    huggyface_df_final["text"] = huggyface_df['sentence']
    huggyface_df_final = huggyface_df_final.replace(-1, "negative")
    huggyface_df_final = huggyface_df_final.replace(0, "neutral")
    huggyface_df_final = huggyface_df_final.replace(1, "positive")

    # Combine all datasets

    final_df = pd.concat([twitter_df, amz_df_final, huggyface_df_final])

    final_df.to_parquet("data/data_prepipeline/full_raw_data.parquet.gzip", compression='gzip')
    
    print("Updated Labeled Data Now Saved in data/data_prepipeline")