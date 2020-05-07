import os
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np

for i in range(1, 9):

    data = pd.read_csv(os.path.join(
        os.getcwd(), "data", "2d_dataset_{}.csv".format(i)))

    labels = data["y"]
    features = data[["alpha", "beta"]]

    # plt.scatter(features["alpha"], features["beta"])
    # plt.show()

    X_train = features.to_numpy()
    y_train = labels.to_numpy()

    # # Plotting decision regions
    # x_min, x_max = 1, 4
    # y_min, y_max = 1, 4
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
    #                      np.arange(y_min, y_max, 0.1))
    # X = np.concatenate(
    #     (np.ones((xx.shape[0] * xx.shape[1], 1)),  np.c_[xx.ravel(), yy.ravel()]), axis=1)
    # h = np.random.rand(xx.shape[0], xx.shape[1])
    # h = h > 0.5
    # h = h.reshape(xx.shape)
    # plt.contourf(xx, yy, h)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                s=30, edgecolor='k')
    plt.xlabel("Marks obtained in 1st Exam")
    plt.ylabel("Marks obtained in 2nd Exam")
    plt.show()
