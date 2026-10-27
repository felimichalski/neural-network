import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons, activation_fn, activation_derivative):
        """
        Initialize weights and bias considering the number of inputs and the number of neurons per layer.
        """
        
        # Weights matrix (initialized with values CLOSE TO zero, in order to prevent every neuron to generate the same output)
        # rows = inputs, columns = neurons
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        
        # Bias: a zero-vector which size is equals to the amount of neurons
        self.bias = np.zeros((1, n_neurons))
        
        self.activation_fn = activation_fn
        self.activation_derivative = activation_derivative
        
        # State
        self.inputs = None
        self.linear_combination = None
        self.output = None

    def forward(self, inputs):
        """
        Executes the neuron flow.
        1. Linear combination.
        2. Activation function.
        """
        self.inputs = inputs
        
        # Summatory of all the values with weights and bias
        self.linear_combination = np.dot(inputs, self.weights) + self.bias
        
        # Application of the activation function over the linear combination
        self.output = self.activation_fn(self.linear_combination)
        
        return self.output