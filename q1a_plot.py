import matplotlib.pyplot as plt


from random import sample


def q1a_plot(X_train):
    random_choices = sample(range(len(X_train)), k=5)

    fig, axs = plt.subplots(5, figsize=(15, 10))
    fig.suptitle('Audio Samples')

    for i, rand in enumerate(random_choices):
        axs[i].plot(X_train[rand])
        axs[i].set(xlabel='time (ms)', ylabel='Amplitude')

    plt.subplots_adjust(hspace=.4)
    plt.show()
