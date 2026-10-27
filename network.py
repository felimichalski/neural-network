import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        """Adds a layer to the network"""
        self.layers.append(layer)

    def forward(self, input):
        """Propagates the input data through all the layer in order"""
        current = input
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def backward(self, loss_fn_gradient, learning_rate):
        """Propagates the error backwards (backpropagation)."""
        current_gradient = loss_fn_gradient
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient, learning_rate)

    def train(self, input, labels, cycles, learning_rate, loss_fn, loss_derivative_fn):
        """Executes the training flow of the model a given number of times"""
        for i in range(cycles):
            # Obtain predictions
            predictions = self.forward(input)
            
            # Calculate the error between the obtained result and the actual values (labels)
            loss = loss_fn(predictions, labels)
            
            # Gradient of the loss
            loss_derivative = loss_derivative_fn(predictions, labels)
            
            # Execute backpropagation
            self.backward(loss_derivative, learning_rate)
            
            # Show progress (informative purpose)
            if (i + 1) % max(1, cycles // 10) == 0:
                print(f"Cycle {i + 1}/{cycles} - Loss: {loss:.6f}")