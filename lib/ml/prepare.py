from lib.ml.reduction import reduce_features

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


TEST_SET_SIZE = 0.2
MAX_FEATURES = None
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
        cnt_vect = CountVectorizer()
        X_train = cnt_vect.fit_transform(X_train[TEXT_COL])
        X_test = cnt_vect.transform(X_test[TEXT_COL])
    return X_train, X_test, y_train, y_test


def prepare_data2(data, method):
    # Unpack data
    X = data.drop(TARGET_COL, axis=1)
    y = data[TARGET_COL]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Reassemble datasets
    df_train = reassemble(X_train, y_train, TARGET_COL)
    df_test = reassemble(X_test, y_test, TARGET_COL)

    if method == "tfidf":
        tfvect = TfIdf(df_train, df_test)

        X_train = tfvect[0]
        X_test = tfvect[1]
        print("Number of features before PCA. Training: {}, Test: {}".format(X_train.shape[0],
                                                                             X_test.shape[0]))

        # PCA Test
        X_train, X_test = reduce_features(X_train, X_test)
        print("Reduced # of features for training set. Training: {}, Test: {}".format(X_train.shape[0],
                                                                                      X_test.shape[0]))
        input("Waiting...")

        vectorizer = tfvect[2]  # Use to store it for later use
    elif method == "bow":
        raise NotImplementedError

    return X_train, X_test, y_train, y_test


def reassemble(X, y, target_col):
    df = X.copy()
    df[target_col] = y
    return df


def split_data(X, y):
    # Split the data into training and test sets
    return train_test_split(X, y,
                            test_size=TEST_SET_SIZE,
                            random_state=RANDOM_STATE)


def TfIdf(dt_train, dt_test):
    """Builds the TF-IDF matrix."""
    tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train = tfidf_vectorizer.fit_transform(dt_train[TEXT_COL])
    X_test = tfidf_vectorizer.transform(dt_test[TEXT_COL])
    y = dt_train[TARGET_COL]
    return [X_train, X_test, tfidf_vectorizer]


def BoW(data):
    return CountVectorizer().fit_transform(data)
