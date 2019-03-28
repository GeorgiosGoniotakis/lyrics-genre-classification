from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer


def reduce_features(X_train, X_test):
    lsa = TruncatedSVD(2, algorithm='arpack')
    norm = Normalizer(copy=False)

    X_train = lsa.fit_transform(X_train)
    X_test = lsa.transform(X_test)

    return X_train, X_test
