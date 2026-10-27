import numpy as np
from layer import Layer
from network import NeuralNetwork
from activation import relu, relu_derivative, sigmoid, sigmoid_derivative
from loss import mean_squared_error, mean_squared_error_derivative

# I want my network to predict the behaviour of the XOR function

# Test input
test_input = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Labels
labels = np.array([
    [0],
    [1],
    [1],
    [0]
])

def main():
    attempts = int(input("Attempts: "))

    manual_flag = input("Do you want to configure each attempt manually (A) or use a global configuration (B)? ")
    if manual_flag.lower() == "a":
        manual_flag = True
    elif manual_flag.lower() == "b":
        manual_flag = False
    else:
        print("Invalid value")
        return
    
    cycles = 0
    learning_rate = 0
    
    if not manual_flag:
        cycles = int(input("Cycles of each attempt: "))
        learning_rate = float(input("Learning rate of each attempt: "))

    for _ in range(attempts):
        net = NeuralNetwork()

        # Input layer
        # The usage of ReLu lets the network to learn non-linear relations
        net.add(Layer(n_input=2, n_neurons=4, activation_fn=relu, activation_derivative=relu_derivative))

        # Output layer (n_input depends on the output of the previous layer)
        # The activation function in the last layer depends on the problem to be solved.
        # In this case, since the output of XOR is wether 0 or 1, we use sigmoid. 
        net.add(Layer(n_input=4, n_neurons=1, activation_fn=sigmoid, activation_derivative=sigmoid_derivative))

        # Training
        print("==========================================")
        print("Starting training...")

        if manual_flag:
                cycles = int(input("Cycles of this attempt: "))
                learning_rate = float(input("Learning rate of this attempt: "))
                
        net.train(
            input=test_input,
            labels=labels,
            cycles=cycles, 
            learning_rate=learning_rate, 
            loss_fn=mean_squared_error, 
            loss_derivative_fn=mean_squared_error_derivative
        )

        # Check the behaviour of the network
        predictions = net.forward(test_input)
        print(f"\nCycles: {cycles}")
        print(f"Learning rate: {learning_rate}")
        print("Final results:")
        for i in range(len(test_input)):
            print(f"Input: {test_input[i]} -> Prediction: {predictions[i][0]:.4f} (Actual: {labels[i][0]})")
        
        
if __name__ == "__main__":
    main()