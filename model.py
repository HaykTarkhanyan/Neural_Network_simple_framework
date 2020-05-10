import numpy as np
from initializers import init_params


LAYERS = [3, 5, 1]
params = {}


print(params)


class Classifier:
    def __init__(self, layers, activations, init_method):
        self.layers = layers
        self.activations = activations
        self.init_method = init_method

    def __str__():  # or repr
        pass  # add in future

    def init_parameters(self):
        self.params = init_params(self.layers, self.init_method)


clf = Classifier(LAYERS, 12, 'xavier')

clf.init_parameters()
