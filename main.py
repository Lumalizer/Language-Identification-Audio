import numpy as np
import matplotlib.pyplot as plt
from random import choices
# random samples
# plot them


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sampling_rate = 8_000
    languages = ["de", "en", "es", "fr", "nl", "pt"]
    language_dict = {languages[i]: i for i in range(len(languages))}

    X_train, y_train = np.load("dataset/inputs_train_fp16.npy"), np.load(
        "dataset/targets_train_int8.npy")
    X_test, y_test = np.load("dataset/inputs_test_fp16.npy"), np.load(
        "dataset/targets_test_int8.npy")

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = load_data()

print(X_train.shape, y_train.shape)

random_choices = choices(range(len(X_train)), k=5)

fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')

axs[0].plot(X_train[random_choices[0]])
axs[0].set(xlabel='time (ms)', ylabel='Amplitude')


plt.subplots_adjust(hspace=.4)
plt.show()
