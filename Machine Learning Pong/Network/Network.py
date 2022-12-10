import numpy as np


def ReLU(x):
    return [xi if xi > 0 else 0 for xi in x]


def Sigmoid(x):
    return [0 if xi < 0 else 1 for xi in x]


class Network:
    # Input_size is the size of the initial input data
    # Shape is an array of integers with each one containing the size of a layer for the network
    # kwargs will contain the bias and weights if you want to pre-initialize the network from a previous run
    def __init__(self, input_size, shape, **kwargs):

        bias = kwargs.get("bias", None)
        weights = kwargs.get("weights", None)
        self.layers = []
        previous_layer = input_size

        # Run through each layer that needs to be created
        for i in range(len(shape)):
            layer = shape[i]

            # Check if there are preset bias and weights, otherwise randomize them as shown in class
            if bias is not None and weights is not None:
                b = bias[i]
                w = weights[i]
            else:
                b = np.zeros(layer)
                w = np.random.default_rng().normal(loc=0, scale= 2 / (layer + previous_layer), size=(layer, previous_layer))

            # If it is the last layer use "Sigmoid", otherwise use the default of ReLU
            if i == len(shape) - 1:
                self.layers.append(Layer(bias=b, weights=w, function="Sigmoid"))
            else:
                self.layers.append(Layer(bias=b, weights=w))

            previous_layer = layer

    # Given the input, run through the network to get the output
    def calc(self, inputs):
        for layer in self.layers:
            inputs = layer.activate(inputs)
        return inputs

    # Update the weights and bias in the network by some random amount
    # Can accept a mutation_rate value which is the rate of change and a change_value which is the max amount of change
    def update(self, mutation_rate, change_value):
        for layer in self.layers:
            layer.update(mutation_rate, change_value)

    def getWeights(self):
        weights = []
        biases = []
        for layer in self.layers:
            weights.append(layer.weights)
            biases.append(layer.bias)
        return weights, biases


class Layer:
    def __init__(self, bias, weights, function="ReLU"):
        self.function = function
        self.bias = bias
        self.weights = weights

    def activate(self, inputs):
        if self.function == "ReLU":
            return ReLU(np.dot(self.weights, inputs) + self.bias)
        if self.function == "Sigmoid":
            return Sigmoid(np.dot(self.weights, inputs) + self.bias)

    def update(self, mutation_rate, change_value):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                if np.random.random() < mutation_rate:
                    self.weights[i][j] += np.random.uniform(-change_value, change_value)
        for i in range(len(self.bias)):
            if np.random.random() < mutation_rate:
                self.bias[i] += np.random.uniform(-change_value, change_value)
