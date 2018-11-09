"""
A NeuralNetwork is just a collection of layers.
"""

import numpy as np


def softmax(x):
    """Compute the softmax of vector x."""
    z = x - np.max(x, axis=1).reshape(-1, 1)
    return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)


class NeuralNetwork:
    """
    A basic NeuralNetwork class
    """
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        """
        perform forward pass
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        """
        perform backward pass
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict_proba(self, inputs):
        value = softmax(self.forward(inputs))
        return value

    def predict(self, inputs):
        """
        select class with max value
        """
        value = self.predict_proba(inputs)
        predicted = np.argmax(value, axis=1)
        return predicted

    def params_and_grads(self):
        """
        iterate over layers and parameters
        """
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad
