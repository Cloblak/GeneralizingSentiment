# import packages
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings('ignore')

# change the dirctory to import local pacakges
os.chdir("scripts")
import data_preprocessing
os.chdir("..")


if __name__ == "__main__":
    
    print("Modeling Data With None DL Approach: Logistic Regression...")
    
    # import data
    labeled_df = pd.read_parquet("data/data_prepipeline/full_raw_data.parquet.gzip")
    
    labeled_df = labeled_df.sample(75000)
    
    cleaned_df = data_preprocessing.proprocessing(labeled_df.text)
    labeled_df["text_cleaned"] = cleaned_df.str[0]
    
    labeled_df.dropna(inplace=True)
    
    # Class count
    labeled_df['sentiment_id'] = labeled_df['sentiment']
    labeled_df['sentiment_id'] = labeled_df['sentiment_id'].replace("negative", 0)
    labeled_df['sentiment_id'] = labeled_df['sentiment_id'].replace("neutral", 1)
    labeled_df['sentiment_id'] = labeled_df['sentiment_id'].replace("positive", 2)
    
    count_class_2, count_class_1, count_class_0 = labeled_df.sentiment_id.value_counts()

    print(f'Postive Class Count {count_class_2}')
    print(f'Neutral Class Count {count_class_1}')
    print(f'Negative Class Count {count_class_0}')
    
    df_class_0 = labeled_df[labeled_df['sentiment_id'] == 0]
    df_class_1 = labeled_df[labeled_df['sentiment_id'] == 1]
    df_class_2 = labeled_df[labeled_df['sentiment_id'] == 2]

    df_class_1_under = df_class_1.sample(count_class_0)
    df_class_2_under = df_class_2.sample(count_class_0)
    undersample_df_cleaned = pd.concat([df_class_0, df_class_1_under, df_class_2_under], axis=0)

    print('Random under-sampling:')
    print(undersample_df_cleaned.sentiment.value_counts())

    undersample_df_cleaned.sentiment.value_counts().plot(kind='bar', title='Count (target)')

    X = undersample_df_cleaned['text_cleaned'] # Collection of documents
    y = undersample_df_cleaned['sentiment_id'] # Target or the labels we want to predict (i.e., the 13 different complaints of products)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.25,
                                                        random_state = 0)
    
    X = undersample_df_cleaned['text_cleaned'] # Collection of documents
    y = undersample_df_cleaned['sentiment_id'] # Target or the labels we want to predict (i.e., the 13 different complaints of products)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.25,
                                                        random_state = 0)
    
    # Load pre-trained model
    senttrans_model = SentenceTransformer('all-MiniLM-L6-v2',device=device)

    # Create embeddings for training set text
    X_train = X_train.values.tolist()
    X_train = [senttrans_model.encode(doc) for doc in X_train]

    # Create embeddings for test set text
    X_test = X_test.values.tolist()
    X_test = [senttrans_model.encode(doc) for doc in X_test]


    logreg_model = LogisticRegression(solver='saga')
    logreg_model.fit(X_train, y_train)
    preds = logreg_model.predict(X_train)
    acc = sum(preds==y_train)/len(y_train)
    print('Accuracy on the training set is {:.3f}'.format(acc))
    