import numpy as np

# in brief activations are used for introducing non-linearity, so that model
# will be able to fit to more complex data
# also for bringing inputs to desired range like from 0-1 for classification

# https://www.coursera.org/learn/neural-networks-deep-learning/
# lecture/4dDC1/activation-functions
# https://www.coursera.org/learn/neural-networks-deep-learning/lecture/OASKH/
# why-do-you-need-non-linear-activation-functions

def activation_forward(z, activation_type):
    """
    Function performs activation step in forward prop, given input to
    activation function and type of activation

    Options for activ_type: "sigmoid", "tanh", "relu", "leaky_relu", "softmax"

    Args:
    z - numpy array or scalar
    method - string("sigmoid", "tanh", "relu", "leaky_relu", "softmax")

    Return:
    returns numpy array or scalar, just as input
    """

    if activation_type.lower().startswith("sigm"):
        a = sigmoid_forward(z)
    elif activation_type.lower().startswith("tan"):
        a = tanh_forward(z)
    elif activation_type.lower().startswith("soft"):
        a = softmax_forward(z)
    elif activation_type.lower().startswith("re"):
        a = relu_forward(z)
    elif activation_type.lower().startswith("l"):
        a = leaky_relu_forward(z)
    else:
        print ("Wrong input, perhaps issue is with the activation func name")

    return a


def sigmoid_forward(z):
    """
    1 / (1 + e^(-z)) - Sigmoid(Logistic) function

    Sigmoid is centered in 0.5, and it brings input to range of 0-1
    which is useful for computing probobility,

    Best use case is as a last layer in a binary classification task
    May bring a problem of vanishing gradients which will slow down learning


    ARGS:
    z - scalar or numpy array, linear part (W * X + b)
    """
    return 1 / (1 + np.exp(-z))


def tanh_forward(z):
    """
    (e^x - e^(-x)) / (e^x + e^(-x)) - Tangent Hyperbolic

    brings input to range (-1 -> 1), and is 0 centered, which gives it advanted
    over sigmoid, thats why is prefierable in hidden layers

    its actually rescaled version of the sigmoid - 2 * sigmoid(2*x) - 1

    Never use in last layer

    ARGS:
    z - scalar or numpy array, linear part (W * X + b)
    """
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def relu_forward(z):
    """
    max(0, z) - Rectified Linear Unit

    Returns maximum value of 0 and Z, is used mostly in ConvNet s,
    Has an advande of not slowing down learning.

    ARGS:
    z - scalar or numpy array, linear part (W * X + b)
    """
    return np.maximum(0, z)


def leaky_relu_forward(z):
    """
    max(0.01 * z, z) - Leaky Rectified Linear Unit

    Similar to Relu but in case of negative inputs derivative
    is 0.01 not just 0 which may fasten learning by a little bit

    Used mainly in ConvNets

    Note:
    0.01 is simply hardcoded, because it doesn't play big role

    ARGS:
    z - scalar or numpy array, linear part (W * X + b)
    """
    return np.maximum(0.01 * z, z)


def softmax_forward(z):
    """
    exp(z) / sum(exp(z)) - Softmax function

    Used in a multi-class classification problems, plays a role similar to
    sigmoid but generalizes for multiple classes

    Used in a last layer

    ARGS:
    z - scalar or numpy array, linear part (W * X + b)

    """
    exps = np.exp(z)
    sum_of_exps = np.sum(exps)

    return exps / sum_of_exps
