import numpy as np
import pandas as pd
import os
from data_loader import load_data
from initializers import init_params
from activations import tanh_forward, sigmoid_forward


X, Y = load_data(4, output = False)

layers = [X.shape[1], 3, 2, 1]
in_met = 'he'

print(layers)


def linear_forward(X, W, b):
    """
    Applies linear forward pass -  X dot W + b

    Args:
    X - (n_l, n_l_1)
    """
    return np.dot(X, W) + b


par = init_params(layers, in_met)
a = [X, 0,0,0]

# print (par)
def forward_prop(X, params):
    caches = {}
    for i in range(len(layers)-1):
        z = linear_forward(params['W' + str(i+1)], a[i], params["b" + str(i+1)])
        a[i+1] = sigmoid_forward(z)

        caches["Z" + str(i+1)] = z
    return caches

print (forward_prop(X, par))
