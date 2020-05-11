import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def mean_variance_normalization(X):
    """
    Function normalizes data by subtrating the mean(mu)
    and deviding by variance(sigma)

    ARGS:
    X - numpy array
    Returns:
    X - numpy array
    """
    mean = np.mean(X)
    variance = np.std(X)

    return (X - mean) / variance


def load_data(path, output = True, normalize = True):
    """
    loads data and applies normalizaton if specified

    Note:
    I hardcoded where are labels and features, if you decide to load
    your data, don't forget to change it below

    ARGS:
    path - location of the dataset(string)
    output - bool, weather to display the data(only if 2D)
    normalize - bool, if true applies mean and variance normalization

    Returns:
    X, Y - numpy arrays
    """

    data = pd.read_csv(os.path.join(
        os.getcwd(), "data", "2d_dataset_{}.csv".format(path)))

    labels = data["y"]
    features = data[["alpha", "beta"]]

    X_train = features.to_numpy()
    y_train = labels.to_numpy()

    if normalize:
        X_train = mean_variance_normalization(X_train)

    if output:
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                    s=30, edgecolor='k')
        plt.show()

    return X_train, y_train


