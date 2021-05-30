import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import pandas as pd
import emoji
import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocessing(dataset, column):    
    """[This function preprocess a document. It makes the sentence
    in lowercase, split a sentence, perform stemming, join them back together]

    Args:
        dataset ([pd.DataFrame]): [pandas dataframe]
        column ([string]): [Name of the column to be processed]

    Returns:
        [list]: [A corpus]
    """
    
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


def remove_emojis(text):
    
    emoji_pattern = re.compile("[" "\U0001F1E0-\U0001F6FF" "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r"", text)
    text = "".join([x for x in text if x not in emoji.UNICODE_EMOJI])
    return text

def remove_mentions_hashtags(text):
    text = re.sub(r"@(\w+)", " ", text)
    text = re.sub(r"#(\w+)", " ", text)
    return text

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

def remove_punctuations_numbers(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    words = (''.join(nopunct)).split()

def remove_stopwords(words):
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    words = [w for w in words if len(w) > 2]  # remove a,an,of etc.

def preprocess_text(text, remove_stop = True, stem_words = False, remove_mentions_hashtags = True):
    """
    eg:
    input: preprocess_text("@water #dream hi hello where are you going be there tomorrow happening happen happens",  
    stem_words = True) 
    output: ['tomorrow', 'happen', 'go', 'hello']
    """
    import re
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from nltk.stem.porter import PorterStemmer
    import emoji
    import string    
  
    # Remove emojis
    emoji_pattern = re.compile("[" "\U0001F1E0-\U0001F6FF" "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r"", text)
    text = "".join([x for x in text if x not in emoji.UNICODE_EMOJI])

    if remove_mentions_hashtags:
        text = re.sub(r"@(\w+)", " ", text)
        text = re.sub(r"#(\w+)", " ", text)

    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    words = (''.join(nopunct)).split()

    if(remove_stop):
        words = [w for w in words if w not in ENGLISH_STOP_WORDS]
        words = [w for w in words if len(w) > 2]  # remove a,an,of etc.

    if(stem_words):
        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words]

    return list(words) 

    

if __name__ == '__main__':
    corpus = 'Today is a good day, tomorrow is going to be a good day'
    vocab = preprocess_text(corpus, remove_stop=True)
    print(vocab)