import numpy as np

class Layer:
    def __init__(self, n_input, n_neurons, activation_fn, activation_derivative):
        """
        Initialize weights and bias considering the number of input and the number of neurons per layer.
        """
        
        # Weights matrix (initialized with values CLOSE TO zero, in order to prevent every neuron to generate the same output)
        # rows = input, columns = neurons
        self.weights = np.random.randn(n_input, n_neurons) * 0.01
        
        # Bias: a zero-vector which size is equals to the amount of neurons
        self.bias = np.zeros((1, n_neurons))
        
        self.activation_fn = activation_fn
        self.activation_derivative = activation_derivative
        
        # State
        self.input = None
        self.linear_combination = None
        self.output = None

    def forward(self, input):
        """
        Executes the neuron flow.
        1. Linear combination.
        2. Activation function.
        """
        self.input = input
        
        # Summatory of all the values with weights and bias
        self.linear_combination = np.dot(input, self.weights) + self.bias
        
        # Application of the activation function over the linear combination
        self.output = self.activation_fn(self.linear_combination)
        
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """
        Calculates the gradients of weights, bias and input and updates the parameters.
        The learning rate control how heavy is the modification over weights and bias.
        The output gradient is the error gradient of the next layer (or the loss functions if it is the last layer).
        """
        # Chain rule: combines the external error with the derivative of the activation in the linear combination.
        activation_gradient = output_gradient * self.activation_derivative(self.linear_combination)
        
        # Gradients for weights and bias
        weights_gradient = np.dot(self.input.T, activation_gradient)
        bias_gradient = np.sum(activation_gradient, axis=0, keepdims=True)
        
        # Gradient to forward to the previous layer
        input_gradient = np.dot(activation_gradient, self.weights.T)
        
        # Parameters update
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        
        return input_gradient