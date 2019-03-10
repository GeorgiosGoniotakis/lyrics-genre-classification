import re

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


def to_lower(s):
    """Converts a given string to lower case."""
    return s.lower()


def rem_punct(s):
    """Removes the punctuation from a given string."""
    return re.sub(r'[^\w\s]', '', s)


def rem_digits(s):
    """Removes digits from a given string"""
    return s.replace('\d+', '')


def rem_swords(s):
    """Removes the stop words from a given string."""
    stop_words = stopwords.words('english')
    return " ".join([item for item in s.split() if item not in stop_words])

def rem_other(s):
    """Replaces a word with a space."""
    rem_words = ["\n"]
    return " ".join([item for item in s.split() if item not in rem_words])


def stem_words(s):
    """Performs stemming on a given string."""
    stemmer = SnowballStemmer('english')
    return " ".join([stemmer.stem(w) for w in s.split()])


def standard_preprocessing(s):
    """Invokes all the necessary methods to perform data preprocessing."""
    return stem_words(rem_swords(rem_digits(rem_punct(rem_other(to_lower(s))))))


def preprocess_data(data, col=None):
    """Preprocesses a given DataFrame or string entry.
    It applies conversion to lowercase, removes punctuation, removes digits,
    removes stop words and stems the words.
    Args:
        data: A pandas DataFrame or a single string.
        col: The name of the column that needs NLP preprocessing.
    Returns:
        The resulting data set.
    """
    if col:
        data[col] = data[col].apply(standard_preprocessing)
    else:
        data = standard_preprocessing(data)

    return data