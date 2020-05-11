import numpy as np
import pandas as pd
import scipy
import os
from data_loader import load_data
from initializers import init_params
from activations import tanh_forward, sigmoid_forward
from activations import activation_forward


X, Y = load_data(4, output = False, normalize = True)

X = np.random.rand(1,1) * 1000
in_met = 'he'

layers = [X.shape[1], 1, 1, 1]
activations = ['tanh', 'relu', 'relu']

def linear_forward(W, X, b):
    """
    Applies linear forward pass -  X dot W + b

    Args:
    X - numpy array, input
    W - numpy array, weight matrix
    b - numpy array, bias
    """
    return np.dot(X, W) + b


par = init_params(layers, in_met)
a = [X, 0,0,0]

# print (par)
def forward_prop(X, activations, params):
    """
    banan
    """
    caches = {}
    for i in range(len(layers)-1):
        z = linear_forward(params['W' + str(i+1)], a[i], params["b" + str(i+1)])
        a[i+1] = activation_forward(z, activations[i])

        caches["A" + str(i+1)] = a[i+1]
        caches["Z" + str(i+1)] = z
    return caches

print (forward_prop(X, activations, par))

cc = forward_prop(X, activations,par)
