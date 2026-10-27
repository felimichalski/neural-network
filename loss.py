import numpy as np

def mean_squared_error(predicted, actual):
    """
    It calculates the Mean Squared Error (MSE).
    It measures the average magnitude of the errors between predictions and actual values (used during training).
    It squares the errors to penalize larger errors more heavily.
    """
    return np.mean((predicted - actual) ** 2)

def mean_squared_error_derivative(predicted, actual):
    """
    Calculates the gradient of the loss relative to the network's output.
    Fundamental for backpropagation, it indicates direction and intensity of the error made by the neuron in the last layer.
    """
    return 2 * (predicted - actual) / predicted.shape[0]