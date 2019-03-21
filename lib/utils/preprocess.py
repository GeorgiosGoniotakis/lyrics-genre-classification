import re

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from autocorrect import spell
from lib.utils.contractions import *
import unidecode
import spacy
from spacy_langdetect import LanguageDetector


def to_lower(s, out=False):
    """Converts a given string to lower case."""
    return s.lower()


def remove_punct(s, out=False):
    """Removes the punctuation from a given string."""
    if out and len(re.findall(r'[^\w\s]', s)) > 0:
        print('Number of punctuation removed: ' + str(len(re.findall(r'[^\w\s]', s))))
    return re.sub(r'[^\w\s]', '', s)


def remove_digits(s, out=False):
    """Removes digits from a given string"""
    if out and len(re.findall('\d+', s)) > 0:
        print('Number of digits removed: ' + str(len(re.findall('\d+', s))))
    return re.sub('\d+', '', s)


def remove_swords(s, out=False):
    """Removes the stop words from a given string."""
    stop_words = stopwords.words('english')
    return " ".join([item for item in s.split() if item not in stop_words])


def remove_other(s, out=False):
    """Replaces a new line with a space."""
    rem_words = ["\n"]
    return " ".join([item for item in s.split() if item not in rem_words])


def remove_accents(s, out=False):
    """Removes accents such as é and ë"""
    if out:
        if unidecode.unidecode(s) != s:
            print(s, unidecode.unidecode(s))
    return unidecode.unidecode(s)


def stem_words(s, out=False):
    """Performs stemming on a given string."""
    stemmer = SnowballStemmer('english')
    return " ".join([stemmer.stem(w) for w in s.split()])


def fix_typos(s, out=False):
    """Fixes common spelling mistakes"""
    final = ''
    for w in s.split():
        final += spell(w) + ' '
        if spell(w) != w and out:
            print('{} was corrected to: {}'.format(w, spell(w)))
    return final


def fix_contractions(s, out=False):
    """Replaces contractions with full phrase e.g. 'won't' becomes 'will not'"""
    return fix(s)


def remove_symbols(s, out=False):
    """Removes remaining symbols such as $,% etc."""
    if out and len(re.findall('(?=[^\s])\W', s)) > 0:
        print('Number of symbols removed: ' + str(len(re.findall('(?=[^\s])\W', s))))
    return re.sub('(?=[^\s])\W', '', s)


def remove_parts(s, out=False):
    """Removes text in square brackets e.g. [Verse 1]"""
    if out and len(re.findall('\[(.*?)\]', s)) > 0:
        print('Number of parts removed: ' + str(len(re.findall('\[(.*?)\]', s))))
    return re.sub('\[(.*?)\]', '', s)


def filter_langs(s, out=False):
    """Removes sentences that are not recognised as english"""
    nlp = spacy.load('en')
    nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)
    new_s = ''
    for x in s.split('\n'):
        doc = nlp(x)
        if doc._.language['language'] == 'en':
            new_s = new_s + x + ' '
    return new_s


def standard_preprocessing(s):
    """Invokes all the necessary methods to perform data preprocessing."""
    return stem_words(remove_swords(remove_digits(remove_punct(remove_other(to_lower(s))))))


def custom_preprocessing(s, filters, out=False):
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
    functions_dict = {'lower': to_lower, 'punct': remove_punct, 'digits': remove_digits,
                      'stop': remove_swords, 'custom': remove_other, 'stem': stem_words,
                      'typos': fix_typos, 'contractions': fix_contractions, 'lang': filter_langs,
                      'accents': remove_accents, 'symbols': remove_symbols, 'parts': remove_parts}
    # Iterate through filters list and apply each function to string
    for f in filters:
        # Check that the filter is valid, otherwise print error message
        if f in functions_dict:
            fun = functions_dict[f]
            if out:
                s = fun(s, out=True)
            else:
                s = fun(s)
        else:
            print(f + ' : Function not recognised')
    return s


def preprocess_data(data, filters=None, col=None, out=False):
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
            data[col] = data[col].apply(custom_preprocessing, args=(filters,), out=True)
        else:
            # Apply standard pre-processing
            data[col] = data[col].apply(standard_preprocessing)
    # Apply pre-processing to a string
    else:
        if filters:
            # Apply custom pre-processing as defined by user in filters parameter
            data = custom_preprocessing(data, filters, out=True)
        else:
            # Apply standard pre-processing
            data = standard_preprocessing(data)
    return data


def create_dict(data):
    """Returns a dictionary where key: unique artist and value: list of songs by that artist"""
    d = dict()
    for index, row in data.iterrows():
        current = row['artist']
        if current not in d.keys():
            d[current] = [row['song']]
        else:
            d[current].append(row['song'])
    return d


def find_duplicates(data):
    """Prints duplicate songs"""
    dic = create_dict(data)
    for key, value in dic.items():
        s = set([x for x in value if value.count(x) > 1])
        if s:
            print(key, s)


def remove_duplicates(data, out=False):
    """Remove duplicates"""
    duplicates = list()
    for dup in data.duplicated(subset=['song', 'artist']).iteritems():
        if dup[1]:
            duplicates.append(dup[0])
    if out:
        print(len(duplicates) + ' duplicates removed')
    data.drop(axis=1, index=duplicates, inplace=True)
    return data

