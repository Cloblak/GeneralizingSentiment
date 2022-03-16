
# mount to my google drive
from os.path import join
from google.colab import drive

ROOT = "/content/drive"
drive.mount(ROOT, force_remount=False)

# change the directory
!ls

%cd /content/drive/"My Drive"/AIPI540_NLP/data

# import required packages
import numpy as np
import pandas as pd
import pyarrow as pa
import string
import time
import urllib.request
import zipfile
import torch

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier



import warnings
warnings.filterwarnings('ignore')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_parquet("full_raw_data.parquet.gzip")
print(data.count())


data.head()

# summary of our image classes
target_count = data.sentiment.value_counts()
print('Class positive:', target_count["positive"])
print('Class neutral:', target_count["neutral"])
print('Class negative:', target_count["negative"])


target_count.plot(kind='bar', title='Count by Class');


X_train,X_test,y_train,y_test = train_test_split(data['text'].values.tolist(), data['sentiment'], random_state=0,test_size=0.2)


# Load pre-trained model
senttrans_model = SentenceTransformer('paraphrase-MiniLM-L6-v2',device=device)

# Create embeddings for training set text
#X_train = train_df['full_text'].values.tolist()
X_train = [senttrans_model.encode(doc) for doc in tqdm(X_train)]

# Create embeddings for test set text
#X_test = test_df['full_text'].values.tolist()
X_test = [senttrans_model.encode(doc) for doc in tqdm(X_test)]


# Train a classification model using logistic regression classifier
logreg_model = LogisticRegression(solver='saga')
logreg_model.fit(X_train,y_train)
preds = logreg_model.predict(X_train)
acc = sum(preds==y_train)/len(y_train)
print('Accuracy on the training set is {:.3f}'.format(acc))


# mean and std of the training score
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


# show learning curve
plt.subplots(1, figsize=(10,10))
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()