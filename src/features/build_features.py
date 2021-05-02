import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

#nltk.download('stopwords')

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import re

import spacy
from spacy.symbols import  ORTH

nlp = spacy.load("en_core_web_sm")

def preprocessing(dataset, column):    
    
    corpus = []
    for i in range(0, 1000):
        review = re.sub('[^a-zA-Z]', ' ', dataset[column][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    return corpus


def bag_of_words(dataset, corpus):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values
    return X, y
        
def label_encode(data):
    """ 
    data = string: 'dog bites man. man bites dog. dog eats meat. man eats food'
    """
    #Label Encoding
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print("Label Encoded:",integer_encoded)

    return integer_encoded
    
def onehot_encode(data):
    """ 
    data = string: 'dog bites man. man bites dog. dog eats meat. man eats food'
    """
    values = sentence_segmenter(data)
     #One-Hot Encoding
    onehot_encoder = OneHotEncoder()
    onehot_encoded = onehot_encoder.fit_transform(data).toarray()
    print("Onehot Encoded Matrix:\n",onehot_encoded)

    return onehot_encoded


def remove_whitespaces(data_string, column = None):
    """ 
    Remove white space from a string or a dataframe 
        
    # string
    original_string = "Lorem    ipsum        ... no, really, it kept going...          malesuada enim feugiat.         Integer imperdiet    erat."
    # dataframe
    df = pd.DataFrame({'message': [original_string[0:50], original_string[0:50]]})
    # short dataframe
    #df = original_string[0:50]

    # remove whitespaces
    data_strings = remove_whitespaces(df, 'message')
    
    """ 

    if type(data_string) is str:
        data_string = ' '.join(data_string.split())

    elif type(data_string) is pd.DataFrame:
        try:
            data_string[column] = (data_string[column].str.split()).str.join(' ')
        except:
            print("Column name is not provided")
        
    return data_string

def count_words(filepath, words_list):
    """

    Parameters
    ----------
    filepath : str
        Path to text file
        
    words_list : list of str
        Count the total number of appearance of these words
        

    Returns
    -------
    n : int
        Total number of times the words appears
        
    Usage: 
    count_words('../alice.txt', ['cat', 'dog'])

    """
    # Open the text file
    with open(filepath) as file:
        text = file.read()

    n = 0
    for word in text.split():
        # Count the number of times the words in the list appear
        if word.lower() in words_list:
            n += 1

    print('Lewis Carroll uses the word "cat" {} times'.format(n))

    return n

def remove_digits(corpus):
    """ 
    Remove digits (ex: 4, 12) from a sentence
    Arguments:
        corpus: string
    """
    corpus = re.sub(r'\d+', '', corpus)

    return corpus

def remove_punctuations(corpus):
    #removing punctuations
    import string
    corpus = corpus.translate(str.maketrans('', '', string.punctuation))
    return corpus

def tokenize(corpus, nltk = True, spacy = False):
    from pprint import pprint
    
    if nltk == True:
        ##NLTK
        import nltk
        from nltk.corpus import stopwords
        nltk.download('stopwords')
        nltk.download('punkt')
        from nltk.tokenize import word_tokenize
        stop_words_nltk = set(stopwords.words('english'))

        tokenized_corpus_nltk = word_tokenize(corpus)
        print("\nNLTK\nTokenized corpus:",tokenized_corpus_nltk)
        tokenized_corpus_without_stopwords = [i for i in tokenized_corpus_nltk if not i in stop_words_nltk]
        print("Tokenized corpus without stopwords:",tokenized_corpus_without_stopwords)

        return tokenized_corpus_without_stopwords

    if spacy == True:
        ##SPACY 
        from spacy.lang.en.stop_words import STOP_WORDS
        import spacy
        spacy_model = spacy.load('en_core_web_sm')

        stopwords_spacy = spacy_model.Defaults.stop_words
        print("\nSpacy:")
        tokenized_corpus_spacy = word_tokenize(corpus)
        print("Tokenized Corpus:",tokenized_corpus_spacy)
        tokens_without_sw= [word for word in tokenized_corpus_spacy if not word in stopwords_spacy]

        print("Tokenized corpus without stopwords",tokens_without_sw)

        return tokens_without_sw

def pos_tagging(corpus_original, nltk = True, spacy = False):
    if nltk == True:
        #POS tagging using spacy
        doc = spacy_model(corpus_original) 
        
        print("POS Tagging using spacy:")  
        # Token and Tag 
        for token in doc: 
            print(token,":", token.pos_)

    if spacy == True:
        #pos tagging using nltk
        nltk.download('averaged_perceptron_tagger')
        print("POS Tagging using NLTK:")
        pprint(nltk.pos_tag(word_tokenize(corpus_original)))


def stem_corpus(doc):
    """ 
    Stemming is just a simpler version of lemmatization where we are interested
    in stripping the suffix at the end of the word. When stemming we are interesting
    in reducing the inflected or derived word to it's base form. 
    Arguments:
        doc: string : 'I prefer not to argue'
    """
    from nltk.stem.snowball import SnowballStemmer

    stemmer = SnowballStemmer(language='english')
    for token in doc.split(" "):
        print(token, '=>' , stemmer.stem(token))


def lemmatise_corpus(corpus):
    """ 
    Lemmatization is the process where we take individual tokens
    from a sentence and we try to reduce them to their base form.
    The process that makes this possible is having a vocabulary and 
    performing morphological analysis to remove inflectional endings. 
    The output of the lemmatization process (as shown in the figure above) 
    is the lemma or the base form of the word. For instance, a lemmatization 
    process reduces the inflections, "am", "are", and "is", to the base form, "be".
    """
    from spacy.lemmatizer import Lemmatizer
    from spacy.lookups import Lookups

    ## lemmatization
    doc = nlp(corpus)
    for word in doc:
        print(word.text, "=>", word.lemma_)

def sentence_segmenter(corpus):
    """ 
    corpus: "I love coding and programming. I also love sleeping!"
    """
    words = []
    ## load the language model
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(corpus)
    for sent in doc.sents:
        words.append(sent.text.split())
        
    print(words)

def add_to_tokenizer(string, word, word_split):
    """
    custom way to add words to tokenizer
    Args:
        string ([str]): [word to be learned]
        words_list ([list]): [list of words]

    Example :
        string = "gimme that"
        word = "gimme"
        word_split = ["gim", "me"]
    """
    
    doc = nlp(string) 
    # phrase to tokenize
    print([w.text for w in doc]) #['gimme', 'that]

    # Add special case rule
    special_case = [{ORTH:word_split[0], ORTH:word_split[1]}]
    nlp.tokenizer.add_special_case(word, special_case)

    # check new tokenization
    print([w.text for w in nlp(string)])


