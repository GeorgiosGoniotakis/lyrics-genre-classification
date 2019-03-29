"""Run >> python -m spacy download en << to obtain the English collection of spacy."""

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from autocorrect import spell
from lib.utils.contractions import *
from lib.utils.timer import Timer
import unidecode
import spacy
from spacy_langdetect import LanguageDetector

nlp = spacy.load('en')
nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)


def to_lower(s):
    """Converts a given string to lower case."""
    return s.lower()


def remove_punct(s):
    """Removes the punctuation from a given string."""
    hits = len(re.findall(r'[.,!;:\-_]', s))
    sanitized = re.sub(r'[.,!;:\-_]', '', s)
    return sanitized, hits


def remove_digits(s):
    """Removes digits from a given string"""
    hits = len(re.findall(r'\d+', s))
    sanitized = re.sub(r'\d+', '', s)
    return sanitized, hits


def remove_swords(s):
    """Removes the stop words from a given string."""
    res = list()
    removed_cnt = 0
    stop_words = stopwords.words('english')
    for item in s.split():
        if item not in stop_words:
            res.append(item)
        else:
            removed_cnt += 1
    return " ".join(res), removed_cnt


def remove_other(s):
    """Replaces a new line with a space."""
    space_count = len(re.findall(r"\n|\r", s))
    return re.sub(r"\n|\r", " ", s), space_count


def remove_accents(s):
    """Removes accents such as é and ë"""
    accent_cnt = 0
    sanitized = unidecode.unidecode(s)
    if sanitized != s:
        accent_cnt += 1
    return sanitized, accent_cnt


def stem_words(s):
    """Performs stemming on a given string."""
    stemmer = SnowballStemmer('english')
    return " ".join([stemmer.stem(w) for w in s.split()])


def fix_typos(s):
    """Fixes common spelling mistakes"""
    typos_cnt = 0
    final = ''
    for w in s.split():
        spell_cor = spell(w)
        final += spell_cor + ' '
        if spell_cor != w:
            typos_cnt += 1
    return final, typos_cnt


def fix_contractions(s):
    """Replaces contractions with full phrase e.g. 'won't' becomes 'will not'"""
    return fix(s)


def remove_symbols(s):
    """Removes remaining symbols such as $,% etc."""
    return re.sub('(?=[^\s])\W', '', s), len(re.findall('(?=[^\s])\W', s))


def remove_parts(s):
    """Removes text in square brackets e.g. [Verse 1]"""
    return re.sub('\[(.*?)\]', '', s), len(re.findall('\[(.*?)\]', s))


def filter_langs(s):
    """Removes sentences that are not recognised as english"""
    return s, nlp(s)._.language["language"]


def remove_duplicates(data, cols):
    """Remove duplicates"""
    duplicates = list()
    for dup in data.duplicated(subset=cols).iteritems():
        if dup[1]:
            duplicates.append(dup[0])
    data.drop(axis=1, index=duplicates, inplace=True)
    return data, len(duplicates)


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

    Example:
        preprocess_data(dataset, filters=['parts', 'contractions', 'punct', 'symbols', 'digits', 'accents', 'custom',
        'typos', 'lang', 'stop', 'stem'], col='lyrics')
    """
    functions_dict = {'lower': to_lower, 'punct': remove_punct, 'digits': remove_digits,
                      'stop': remove_swords, 'custom': remove_other, 'stem': stem_words,
                      'typos': fix_typos, 'contractions': fix_contractions, 'lang': filter_langs,
                      'accents': remove_accents, 'symbols': remove_symbols, 'parts': remove_parts}

    # Remove NaN values
    data.dropna(inplace=True)
    num = data.shape[0]

    # Remove duplicates
    data, n_dup = remove_duplicates(data, ['title', 'artist'])
    print("Number of duplicate records removed: {}".format(n_dup))

    input("Press Enter to begin preprocessing...")

    # Apply pre-processing to data series
    if col:
        if filters:
            for f in filters:
                cnt = 0
                fun = functions_dict[f]
                for key, value in data.iterrows():
                    cnt += 1
                    print("Filter: {}, Processing record: {}/{}".format(f, cnt, num))
                    res = fun(value[col])
                    if len(res) == 2 and isinstance(res, tuple):
                        if isinstance(res[1], tuple):
                            data.at[key, col] = res[0]
                            data.at[key, "contr"] = res[1][0]
                            data.at[key, "slang"] = res[1][1]
                        else:
                            data.at[key, col], data.at[key, f] = res
                    else:
                        data.at[key, col] = res

    return data


# Running sample
# import pandas as pd
#
# data = pd.read_csv("../../data/200000.csv", index_col=0)
# timer = Timer()
# data = preprocess_data(data, filters=['parts', 'contractions', 'punct', 'symbols', 'digits', 'accents', 'custom',
#                                       'lang', 'stop', 'stem'], col='lyrics')
# print("Preprocessing finished in: {} mins".format(str(timer.get_time()/60)))
# data.to_csv("../../data/200000_clean.csv", index=False)
