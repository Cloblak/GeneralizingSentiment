import string
import re
import nltk
from nltk.stem import WordNetLemmatizer


# defining the function to remove punctuation

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree


# defining function for tokenization



def tokenization(text):
    tokens = re.split(' ',text)
    return tokens


nltk.download('omw-1.4')

#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

#defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

# data should be the text column from the dataset
def preprocessing(data):
    data= data.apply(lambda x:remove_punctuation(x))
    data= data.apply(lambda x: x.lower())
    data= data.apply(lambda x: tokenization(x))
    data= data.apply(lambda x:remove_stopwords(x))
    data= data.apply(lambda x:lemmatizer(x))
    
    return data
