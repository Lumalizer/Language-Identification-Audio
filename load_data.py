import numpy as np
from options import Options
from q1b_normalize_data import normalize_data


def load_data(options: Options) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sampling_rate = 8_000
    languages = ["de", "en", "es", "fr", "nl", "pt"]
    language_dict = {languages[i]: i for i in range(len(languages))}

    X_train, y_train = np.load("dataset/inputs_train_fp16.npy"), np.load(
        "dataset/targets_train_int8.npy")
    X_test, y_test = np.load("dataset/inputs_test_fp16.npy"), np.load(
        "dataset/targets_test_int8.npy")

    if options.normalize:
        X_train, X_test = normalize_data(X_train, X_test)

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

    return X_train, y_train, X_test, y_test
