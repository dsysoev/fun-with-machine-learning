
import numpy as np


class Loss:
    def loss(self, predicted, actual):
        raise NotImplementedError

    def grad(self, predicted, actual):
        raise NotImplementedError


class MSE(Loss):
    """
    Mean squared error loss
    """
    def loss(self, predicted, actual):
        # error over all matrix
        # actial class must be as one hot vector
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted, actual):
        # calculate gradients
        return 2 * (predicted - actual)
