"""
Implementation of Stochastic gradient descent algorithm
https://en.wikipedia.org/wiki/Stochastic_gradient_descent
"""


class Optimizer:
    def step(self, network):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, network):
        for level, (param, grad) in enumerate(network.params_and_grads()):
            # apply grads to params with learning rate
            param -= self.learning_rate * grad
