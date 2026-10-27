import numpy as np

def sigmoid(x):
    """
    Takes every real number and turns it into a value between 0 and 1.
    It represent an activated/deactivated neuron.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Measures the change rate of the curve in a specific point.
    Required during backpropagation to know how to adjust weights and inputs.
    """
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """
    0 if x is negative, x otherwise.
    Standard function for DNN. Since it is linear for positive values, it prevents computational problems.
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    0 for negative values, 1 for positive ones.
    Lets backpropagation flow to go through the active neurons without any loosing.
    """
    return np.where(x > 0, 1, 0)