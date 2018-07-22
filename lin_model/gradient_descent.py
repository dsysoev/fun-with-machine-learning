
"""
implementation of gradient descent
n the solution of the linear equation
https://en.wikipedia.org/wiki/Gradient_descent#Solution_of_a_linear_system
"""

import numpy as np
from tqdm import trange


def cost_function(y_true, y_pred):
    """Computes the cost of linear regression
    """
    n, _ = y_true.shape
    cost = 1 / (2 * n) * np.sum((y_true - y_pred) ** 2)
    return cost

def normalize(X):
    """Returns a normalized version of X where
    the mean value of each feature is 0
    and the standard deviation is 1
    """
    # calculate mean value along zero axis
    X_mean = np.mean(X, axis=0)
    # calculate standard deviation along zero axis
    X_std = np.std(X, axis=0)
    # Standard scaler
    # z = (x - mu) / sigma
    X_norm = (X - X_mean) / X_std
    return X_norm, X_std

def gradient_descent(X, y, theta=None, alpha=0.05, num_epoch=10, verbose=True):
    """Performs gradient descent to learn theta
    by taking n steps
    """
    n, m = X.shape
    if theta is None:
        # set theta as random weights
        theta = np.random.rand(m + 1, 1)
    # add intercept value to X
    X0 = np.concatenate([np.ones((n, 1)), X], axis=1)
    # set dict with costs values
    cost = {}
    t = trange(num_epoch)
    for epoch in t:
        # set new theta value
        # theta = theta - alpha * 1 / m * X' * (X * theta - y)
        theta = theta - alpha * 1 / n * X0.T @ (X0 @ theta - y)
        # calculate predicted value with new theta
        y_pred = X0 @ theta
        # calculate cost value
        cost[epoch] = cost_function(y, y_pred)
        if verbose:
            t.set_description(
                "epoch: {:4d} cost: {:.4f}".format(epoch, cost[epoch]))
    return theta

def inverse_transform(theta_normalize, X, X_std):
    """transform theta normilize to original feature space
    """
    n = X_std.shape[0]
    # rescale theta
    theta_rescaled = (theta_normalize.flatten()[1:] / X_std).reshape((n, 1))
    # get predictions
    y_pred = X @ theta_rescaled
    # calculate mean bias values
    theta_0 = (y - y_pred).mean()
    # concatenate
    theta = np.concatenate([[[theta_0]], theta_rescaled], axis=0)
    return theta

if __name__ in '__main__':
    # set X vector
    X = np.array([
        [-1, 0],
        [1, 100],
        [2, 200],
        [3, 300],
        [4, 400],
        [5, 500],
    ])
    # set y based on equation y = 5 + X0 * 20 + X1 * 1
    theta_true = np.array([
        [5.], [20.], [1.]
        ])
    y = X @ theta_true[1:, :] + theta_true[0, :]
    print("X:\n{}".format(X))
    print("y:\n{}".format(y))
    # transfom all components in X to normalize view
    # for proper working of the gradient descent
    X_normalize, X_std = normalize(X)
    # obtaine theta (normalize) vector using gradient descent algorithm
    theta_normilize = gradient_descent(
        X_normalize, y, alpha=0.1, num_epoch=10000, verbose=True)
    # convert theta normilize to original feature space
    theta = inverse_transform(theta_normilize, X, X_std)
    print("\n\n")
    print("theta:\n{}".format(theta))
    print("theta_true:\n{}".format(theta_true))
