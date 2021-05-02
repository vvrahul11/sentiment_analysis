import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

#nltk.download('stopwords')

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import re

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from src.features.build_features import preprocessing, bag_of_words, remove_whitespaces


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

if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    # Read data
    dataset = pd.read_csv('C:/RahulGit/sentiment_analysis/data/raw/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
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