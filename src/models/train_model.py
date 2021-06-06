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

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torchtext

from src.features.build_features import bag_of_words
from src.data.preprocessing import preprocessing, remove_whitespaces



def datasplit(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)
    return X_train, X_test, y_train, y_test


def gaussiannb(X_train, y_train, X_test):
     classifier = GaussianNB()
     classifier.fit(X_train, y_train)
     y_pred = classifier.predict(X_test)

     return y_pred

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Ref:http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.figure(figsize=(8,6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=15)
    plt.xlabel('Predicted label',fontsize=15)
    plt.show()

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
    analyze_restaurant_reviews('data/raw/Restaurant_Reviews.tsv')
    #analyze_tweets('data/external/tweets/tweets.csv')
    #Sentiment Analysis of IMDB reviews