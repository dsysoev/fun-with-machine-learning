
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
        # actual class must be as one hot vector
        value = np.sum((predicted - actual) ** 2)
        return value

    def grad(self, predicted, actual):
        # calculate gradients
        value = 2 * (predicted - actual)
        return value


class BinaryCrossEntropy(Loss):
    """
    Binary Cross Entropy loss
    """
    def loss(self, predicted, actual):
        epsilon = 1e-5
        loss = -np.sum(actual * np.log(predicted + epsilon) + (1 - actual) * np.log(1 - predicted + epsilon))
        return loss

    def grad(self, predicted, actual):
        epsilon = 1e-3
        value = (predicted - actual) / (predicted * (1 - predicted) + epsilon)
        return value
