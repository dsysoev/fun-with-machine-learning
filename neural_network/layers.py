"""
Implementation of core Layer class
and Linear, Tanh and Sigmoid activation functions
"""

import numpy as np


class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, inputs):
        """
        Produce the outputs corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, grad):
        """
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError


class Linear(Layer):
    """
    computes output = inputs @ w + b
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs):
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad):
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T


class Activation(Layer):
    """
    An activation layer just applies a function
    elementwise to its inputs
    """
    def __init__(self, f, f_prime):
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs):
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad):
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    y = tanh(x)
    return 1 - y ** 2


class Tanh(Activation):
    """
    Tanh function
    """
    def __init__(self):
        super().__init__(tanh, tanh_prime)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    y = sigmoid(x)
    return y * (1 - y)


class Sigmoid(Activation):
    """
    sigmoid function
    """
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)

def relu(x):
    return x * (x > 0)


def relu_prime(x):
    return x > 0


class Relu(Activation):
    """
    Relu function
    """
    def __init__(self):
        super().__init__(relu, relu_prime)

def leakerelu(x, alpha):
    is_positive = x > 0
    return is_positive * x + (1 - is_positive) * alpha * x


def leakerelu_prime(x, alpha):
    is_positive = x > 0
    return is_positive + (1 - is_positive) * alpha


class LeakyRelu(Activation):
    """
    Leaky Relu function
    """
    def __init__(self, alpha):
        self.alpha = alpha
        super().__init__(leakerelu, leakerelu_prime)

    def forward(self, inputs):
        self.inputs = inputs
        return self.f(inputs, self.alpha)

    def backward(self, grad):
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs, self.alpha) * grad
