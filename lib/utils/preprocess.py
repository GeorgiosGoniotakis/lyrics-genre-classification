import re

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import spelling_corr
import contractions
import unidecode


def to_lower(s):
    """Converts a given string to lower case."""
    return s.lower()


def rem_punct(s):
    """Removes the punctuation from a given string."""
    return re.sub(r'[^\w\s]', '', s)


def rem_digits(s):
    """Removes digits from a given string"""
    return re.sub('\d+', '', s)


def rem_swords(s):
    """Removes the stop words from a given string."""
    stop_words = stopwords.words('english')
    return " ".join([item for item in s.split() if item not in stop_words])


def rem_other(s):
    """Replaces a word with a space."""
    rem_words = ["\n"]
    return " ".join([item for item in s.split() if item not in rem_words])


def rem_accents(s):
    """Removes accents such as é and ë"""
    return unidecode.unidecode(s)


def stem_words(s):
    """Performs stemming on a given string."""
    stemmer = SnowballStemmer('english')
    return " ".join([stemmer.stem(w) for w in s.split()])


def fix_typos(s):
    """Fixes common spelling mistakes"""
    final = ''
    words = s.split()
    for w in words:
        final += spelling_corr.correction(w) + ' '
    return final


def fix_contractions(s):
    """Replaces contractions with full phrase e.g. 'won't' becomes 'will not'"""
    return contractions.fix(s)

def remove_symbols(s):
    """Removes remaining symbols such as $,% etc."""
    return


def standard_preprocessing(s):
    """Invokes all the necessary methods to perform data preprocessing."""
    return stem_words(rem_swords(rem_digits(rem_punct(rem_other(to_lower(s))))))


def custom_preprocessing(s, filters):
    """Invokes the functions specified on the given string
    Filters that can be applied:
        'lower': converts string to lower case
        'punct': removes punctuation
        'digits': removes digits
        'stop': removes stop words
        'stem': applies stemming
        'custom': removes symbols as defined in rem_other
    Args:
        s: String to which the preprocessing should be applied
        filters: A list of strings of preprocessing functions to be applied to the string
    Returns:
        The preprocessed string
    """
    functions_dict = {'lower': to_lower, 'punct': rem_punct, 'digits': rem_digits,
                      'stop': rem_swords, 'custom': rem_other, 'stem': stem_words,
                      'typos': fix_typos, 'contractions': fix_contractions,
                      'accents': rem_accents}
    # Iterate through filters list and apply each function to string
    for f in filters:
        # Check that the filter is valid, otherwise print error message
        if f in functions_dict:
            fun = functions_dict[f]
            s = fun(s)
        else:
            print(f + ' : Function not recognised')
    return s


def preprocess_data(data, filters=None, col=None):
    """Preprocesses a given DataFrame or string entry.
    If a list of functions is not specified,
    it applies conversion to lowercase, removes punctuation, removes digits,
    removes stop words and stems the words.

    Functions that can be specified are the above
    Args:
        data: A pandas DataFrame or a single string.
        filters: A list of pre-processing functions to be applied to the data
        col: The name of the column that needs NLP preprocessing.
    Returns:
        The resulting data set.
    """

    # Apply pre-processing to data series
    if col:
        if filters:
            # Apply custom pre-processing as defined by user in filters parameter
            data[col] = data[col].apply(custom_preprocessing, args=(filters,))
        else:
            # Apply standard pre-processing
            data[col] = data[col].apply(standard_preprocessing)
    # Apply pre-processing to a string
    else:
        if filters:
            # Apply custom pre-processing as defined by user in filters parameter
            data = custom_preprocessing(data, filters)
        else:
            # Apply standard pre-processing
            data = standard_preprocessing(data)
    return data

