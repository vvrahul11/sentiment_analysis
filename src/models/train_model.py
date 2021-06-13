import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

#nltk.download('stopwords')

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import spacy

import re
from scipy.sparse.construct import random

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torchtext

from src.features.build_features import bag_of_words
from src.data.preprocessing import preprocessing, remove_whitespaces
from src.visualization.visualize import plot_confusion_matrix
from src.data.preprocessing import datasplit
from src.models.ml_models import gaussiannb

def analyze_restaurant_reviews(path):
    # Read data
    dataset = pd.read_csv(path, delimiter = '\t', quoting = 3)
    # prepare data
    corpus = preprocessing(dataset, 'Review')
    # bag of words operation
    X, y = bag_of_words(dataset, corpus)
    # split data
    X_train, X_test, y_train, y_test = datasplit(X, y)
    # get prediction
    y_pred = gaussiannb(X_train, y_train, X_test)
    # compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))

    plot_confusion_matrix(cm, classes=['Good','Bad'],normalize=False,
                      title='Confusion matrix with all features')

def tweet_clean(text):
    """ clean tweet """
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = re.sub(r'https?:/\/\S+', ' ', text)
    return text.strip()

def tokenizer(s):
    nlp = spacy.load('en_core_web_sm', disable=['parser',
                                    'tagger',
                                    'ner'])
    return [w.text.lower() for w in nlp(tweet_clean(s))]

def analyze_tweets(path):
    import seaborn as sns
    # Read data
    tweets = pd.read_csv(path, error_bad_lines = False)
    tweets = tweets.drop(columns = ['ItemID', 'SentimentSource'], axis = 1)

    fig = plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=tweets.Sentiment.unique(),
                    y=tweets.Sentiment.value_counts())
    ax.set(xlabel='Labels')

    train, test = train_test_split(tweets, test_size=0.2, random_state=42)
    train.reset_index(drop=True)
    test.reset_index(drop=True)

    train.to_csv('data/external/tweets/train.csv', index=False)
    test.to_csv('data/external/tweets/test.csv' , index=False)

    TEXT = torchtext.data.Field(tokenize = tokenizer)
    LABEL = torchtext.data.LabelField(dtype=torch.float)

    datafields = [('Sentiment', LABEL), ('Sentiment', TEXT)]

    trn, tst = torchtext.data.TabularDataset.split(path='data/external/tweets/',
                                                train='train_tweets.csv',
                                                test = 'test_tweets.csv',
                                                format='csv',
                                                skip_header=True,
                                                fields=datafields)
    print(f"Number of training examples: {len(trn)}")
    print(f"Number of testing examples: {len(tst)}")

    TEXT.build_vocab(trn, max_size= 25000,
                        vectors="glove.68.100d",
                        unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(trn)
    print(TEXT.vocab.freqs.most_common(50))


if __name__ == '__main__':
    #analyze_restaurant_reviews('data/raw/Restaurant_Reviews.tsv')
    analyze_tweets('data/external/tweets/tweets.csv')
    #Sentiment Analysis of IMDB reviews