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

def bag_of_words(dataset, corpus):
    """[Perofrm bag of words modeling]

    Args:
        dataset ([pd.DataFrame]): [Pandas dataframe]
        corpus : [corpus]

    Returns:
        [X, y]: [Returns features and target]
    """
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

def word_sentence_tokenization(text):
    """
    Get a text with many sentences, apply nltk sentence followed by word_tokenize and lower case
    """
    sentences = [
        [word.lower() for word in nltk.word_tokenize(sentence)]
        for sentence in nltk.sent_tokenize(text)
    ]
    return sentences

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


def create_cooccurrence_matrix(sentences, window_size=2):
    """Create co occurrence matrix from given list of sentences.
    Reference: https://stackoverflow.com/questions/20574257/constructing-a-co-occurrence-matrix-in-python-pandas
    
    Returns:
    - vocabs: dictionary of word counts
    - co_occ_matrix_sparse: sparse co occurrence matrix

    Example:
    ===========
    sentences = ['I love nlp',    'I love to learn',
                 'nlp is future', 'nlp is cool']

    vocabs,co_occ = create_cooccurrence_matrix(sentences)

    df_co_occ  = pd.DataFrame(co_occ.todense(),
                              index=vocabs.keys(),
                              columns = vocabs.keys())

    df_co_occ = df_co_occ.sort_index()[sorted(vocabs.keys())]

    df_co_occ.style.applymap(lambda x: 'color: red' if x>0 else '')   

    """
    import scipy
    import nltk

    vocabulary = {}
    data = []
    row = []
    col = []

    tokenizer = nltk.tokenize.word_tokenize

    for sentence in sentences:
        sentence = sentence.strip()
        tokens = [token for token in tokenizer(sentence) if token != u""]
        for pos, token in enumerate(tokens):
            i = vocabulary.setdefault(token, len(vocabulary))
            start = max(0, pos-window_size)
            end = min(len(tokens), pos+window_size+1)
            for pos2 in range(start, end):
                if pos2 == pos:
                    continue
                j = vocabulary.setdefault(tokens[pos2], len(vocabulary))
                data.append(1.)
                row.append(i)
                col.append(j)

    cooccurrence_matrix_sparse = scipy.sparse.coo_matrix((data, (row, col)))
    return vocabulary, cooccurrence_matrix_sparse

def stem_words(words):
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]