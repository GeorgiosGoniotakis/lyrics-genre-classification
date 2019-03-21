from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

TEST_SET_SIZE = 0.2
MAX_FEATURES = 50000
RANDOM_STATE = 0

TEXT_COL = "lyrics"
TARGET_COL = "genre"


def prepare_data(data, method):
    # Unpack data
    X = data.drop(TARGET_COL, axis=1)
    y = data[TARGET_COL]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Reassemble datasets
    df_train = reassemble(X_train, y_train, "genre")
    df_test = reassemble(X_test, y_test, "genre")

    if method == "tfidf":
        tfvect = TfIdf(df_train, df_test)
        X_train = tfvect[0]
        X_test = tfvect[1]
        vectorizer = tfvect[2]  # Use to store it for later use
    elif method == "bow":
        raise NotImplementedError

    return X_train, X_test, y_train, y_test


def reassemble(X, y, target_col):
    df = X.copy()
    df[target_col] = y
    return df


def split_data(X, y):
    """
    """
    # Split the data into training and test sets
    return train_test_split(X, y,
                            test_size=TEST_SET_SIZE,
                            random_state=RANDOM_STATE)


def TfIdf(dt_train, dt_test):
    """Builds the TF-IDF matrix."""
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 4), max_df=0.90, min_df=5, max_features=MAX_FEATURES)
    X_train = tfidf_vectorizer.fit_transform(dt_train[TEXT_COL])
    X_test = tfidf_vectorizer.transform(dt_test[TEXT_COL])
    y = dt_train[TARGET_COL]
    return [X_train, X_test, tfidf_vectorizer]


def BoW(data):
    """

    Args:
        col: The column which contains the textual data
    """
    return CountVectorizer().fit_transform(data)
