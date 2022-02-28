import string
string.punctuation


# defining the function to remove punctuation

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
# storing the puntuation free text

data['clean_punc']= data['text'].apply(lambda x:remove_punctuation(x))

data['lower']= data['clean_punc'].apply(lambda x: x.lower())

# defining function for tokenization

import re


def tokenization(text):
    tokens = re.split('W+',text)
    return tokens

#applying function to the column

data['token']= data['lower'].apply(lambda x: tokenization(x))

#importing nlp library
import nltk
#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

#applying the function
data['stopwords_free']= data['token'].apply(lambda x:remove_stopwords(x))

from nltk.stem import WordNetLemmatizer
#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

#defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
data['lemmatized']=data['stopwords_free'].apply(lambda x:lemmatizer(x))