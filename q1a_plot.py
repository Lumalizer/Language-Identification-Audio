import matplotlib.pyplot as plt
from random import sample
from options import Options


def q1a_plot(options: Options, X_train):
    random_choices = sample(range(len(X_train)), k=5)

    fig, axs = plt.subplots(5, figsize=(15, 10))
    fig.suptitle('Audio Samples')
    limits = (-.6, .6) if options.normalize else (-1, 1)

    for i, rand in enumerate(random_choices):
        axs[i].plot(X_train[rand])
        axs[i].set(xlabel='time (ms)', ylabel='Amplitude', ylim=limits)

    plt.subplots_adjust(hspace=.4)
    plt.show()
