import numpy as np

# NOTE:
# initializing biases to 0 would work fine as well.

# https://towardsdatascience.com/
# weight-initialization-techniques-in-neural-networks-26c649eb3b78
# https://www.coursera.org/lecture/deep-neural-network/
# weight-initialization-for-deep-networks-RwqYe


def init_params(layers, method):
    """
    Function takes as input list containg numbers of hidden unit in each layer
    and initializes model occurdingly, given initilizaton method

    Options: "zeros", "uniform", "normal", "he", "xavier"

    Args:
    layers - list, tuple or set - contains numbers of hidden unit in each layer
    method - string(either "zeros", "uniform", "normal", "he", "xavier")

    Return:
    returns dictionary with keywords W1, b1, W2, b2 etc
    """
    params = {}

    if method.lower().startswith("zero"):
        for i in range(len(layers) - 1):
            params["W" + str(i + 1)], params["b" + str(i + 1)]  \
                = zeros_initializer(layers[i], layers[i + 1])

    elif method.lower().startswith("unif"):
        for i in range(len(layers) - 1):
            params["W" + str(i + 1)], params["b" + str(i + 1)]  \
                = uniform_initializer(layers[i], layers[i + 1])

    elif method.lower().startswith("norm"):
        for i in range(len(layers) - 1):
            params["W" + str(i + 1)], params["b" + str(i + 1)]  \
                = normal_initializer(layers[i], layers[i + 1])

    elif method.lower().startswith("he"):
        for i in range(len(layers) - 1):
            params["W" + str(i + 1)], params["b" + str(i + 1)]  \
                = he_initializer(layers[i], layers[i + 1], )

    elif method.lower().startswith("xav"):
        for i in range(len(layers) - 1):
            params["W" + str(i + 1)], params["b" + str(i + 1)]  \
                = xavier_initializer(layers[i], layers[i + 1])

    else:
        print ("Wrong input, perhaps issue is with the name of initializer")

    return params


def zeros_initializer(n_x, n_y):
    """
    Initilizes both weights and biases to matrices of 0 s

    Note:
    This will make NN symetric, which will cause to terrible fail

    Args:
    n_x - number of rows of the matrix (int)
    n_y - number of columns the matrix (int)

    Returns:
    Weights matrix - numpy array with shape (n_x, n_y)
    Bias vector - numpy array with shape (n_y, 1)
    """
    w = np.zeros((n_x, n_y))
    b = np.zeros((n_y, 1))

    return w, b


def uniform_initializer(n_x, n_y):
    """
    Initilizes randomly with values from uniform distribution

    Args:
    n_x - number of rows of the matrix (int)
    n_y - number of columns the matrix (int)

    Returns:
    Weights matrix - numpy array with shape (n_x, n_y)
    Bias vector - numpy array with shape (n_y, 1)
    """
    w = np.random.random((n_x, n_y))
    b = np.random.random((n_y, 1))

    return w, b


def normal_initializer(n_x, n_y):
    """
    Initilizes randomly with values from Gaussian distribution

    Args:
    n_x - number of rows of the matrix (int)
    n_y - number of columns the matrix (int)

    Returns:
    Weights matrix - numpy array with shape (n_x, n_y)
    Bias vector - numpy array with shape (n_y, 1)
    """
    w = np.random.randn(n_x, n_y)
    b = np.random.randn(n_y, 1)

    return w, b


def he_initializer(n_x, n_y, size_prev_layer):
    """
    just as other initilizers but scaled by a factor of -
     sqrt(2 / #num of neurons in previous layer)

    Note:
    sometimes number of neurons of current layer is also taken into account
    but not in my implementation

    Args:
    n_x - number of rows of the matrix (int)
    n_y - number of columns the matrix (int)

    Returns:
    Weights matrix - numpy array with shape (n_x, n_y)
    Bias vector - numpy array with shape (n_y, 1)
    """

    # num of units in previous layer is same as n_x
    w = np.random.randn(n_x, n_y) * np.sqrt(2 / n_x)
    b = np.random.randn(n_y, 1) * np.sqrt(2 / n_x)

    return w, b


def xavier_initializer(n_x, n_y):
    """
    just as "He" but with 1 instead of 2, best suited with tanh activation
     sqrt(1 / #num of neurons in previous layer)

    Note:
    sometimes number of neurons of current layer is also taken into account
    but not in my implementation

    Args:
    n_x - number of rows of the matrix (int)
    n_y - number of columns the matrix (int)

    Returns:
    Weights matrix - numpy array with shape (n_x, n_y)
    Bias vector - numpy array with shape (n_y, 1)
    """

    # num of units in previous layer is same as n_x
    w = np.random.randn(n_x, n_y) * np.sqrt(1 / n_x)
    b = np.random.randn(n_y, 1) * np.sqrt(1 / n_x)

    return w, b
