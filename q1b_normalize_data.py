import numpy as np


def normalize_data(X_train, X_test):
    X_train = X_train / np.max(np.abs(X_train))
    X_test = X_test / np.max(np.abs(X_test))
    return X_train, X_test
