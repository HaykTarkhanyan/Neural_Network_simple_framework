import numpy as np
import pandas as pd
import os
from initializers import init_params

i = 1
data = pd.read_csv(os.path.join(
    os.getcwd(), "data", "2d_dataset_{}.csv".format(i)))

labels = data["y"].head(100)
features = data[["alpha", "beta"]].head(100)

layers = [features.shape[1], 4, 5]
in_met = 'zeros'

print(layers)


def linear_forward(W, X, b):
    return np.dot(X, W) + b.T


par = init_params(layers, in_met)


a = [0]
for i in range(len(layers)):
    a
    print(linear_forward(par["W" + str(i + 1)],
                         features, par["b" + str(i + 1)]).shape)
