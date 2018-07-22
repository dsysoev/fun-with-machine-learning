
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class NeuralNetwork(object):
    """Neural network"""

    def __init__(self, layers, alpha=0.1, num_epoch=10, verbose=True):
        """Neural network

        layers: list of layers [Layer(), ...]
        alpha: learning rate [0, 1)
        num_epoch: number of epoch for gradient descent
        verbose: show loss value each epoch

        """
        self.layers = layers
        self.alpha = alpha
        self.num_epoch = num_epoch
        self.verbose = verbose

    def _loss(self, y_true, y_pred):
        """return MSE loss"""
        return 1 / 2 * np.sum((y_true - y_pred) ** 2)

    def _grad(self, y_true, y_pred):
        """return gradient errors"""
        return 2 * (y_pred - y_true)

    def _forward(self, X):
        """forward pass"""
        inputs = X
        for layer in self.layers:
            # calculate prediction
            inputs = layer.forward(inputs)
        return inputs

    def _backward(self, grad):
        """backward pass"""
        for layer in reversed(self.layers):
            # calculate gradient
            grad = layer.backward(grad)
        return grad

    def _step(self):
        """one gradient descent step"""
        for layer in self.layers:
            # for each layer
            for name, param in layer.params.items():
                # subtract the product of alpha by the gradient
                # from each layer parameter
                layer.params[name] -= self.alpha * layer.grads[name]

    def fit(self, X, y):
        """Fit model based on given data"""
        for epoch in range(self.num_epoch):
            # for each epoch
            # get prediction using forward pass
            y_pred = self._forward(X)
            # calculate MSE loss
            loss = self._loss(y, y_pred)
            # get gradient
            grad = self._grad(y, y_pred)
            # perform backward pass
            self._backward(grad)
            # using gradient descent
            # perform one step
            self._step()
            if self.verbose:
                print("epoch: {:4d} loss: {:.4f}".format(epoch, loss))
        return self

    def predict(self, X):
        return np.where(self._forward(X) >= 0.5, 1, 0)


class Layer(object):

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
        Backpropagate the gradient through the layer
        """
        raise NotImplementedError


class Linear(Layer):
    """Fully connected linear layer"""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad):
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T

class Tanh(Layer):
    """Tanh activation layer"""

    def forward(self, inputs):
        self.inputs = inputs
        return np.tanh(inputs)

    def backward(self, grad):
        tanh_grad = np.tanh(grad)
        return (1 - tanh_grad ** 2) * grad

class Sigmoid(Layer):
    """Sigmoid activation layer"""

    def forward(self, inputs):
        self.inputs = inputs
        return 1. / (1. + np.exp(-inputs))

    def backward(self, grad):
        sigmoid_grad = 1. / (1. + np.exp(-grad))
        return sigmoid_grad * (1 - sigmoid_grad) * grad

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
    return X_norm, X_mean, X_std


if __name__ in '__main__':
    # set seed
    np.random.seed(37)
    # load data from iris dataset
    datax, datay = load_iris(return_X_y=True)
    # solve binary classification problem
    # using neural network
    # get first and second class only
    n = datay[datay < 2].shape[0]
    # split data for train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        datax[:n], datay[:n].reshape((n, 1)), test_size=0.3, shuffle=True)
    # for proper working of the gradient descent
    # transfom X to normalize view
    X_normalize, X_mean, X_std = normalize(X_train)
    X_test_norm = (X_test - X_mean) / X_std
    # set neural network
    # with 1 hidden layer
    # and sigmoid activation function
    nn = NeuralNetwork(
        layers=[
            Linear(input_size=X_normalize.shape[1], output_size=1),
            Sigmoid(),
        ],
        num_epoch=10,
        alpha=0.1,
        verbose=True
        )
    # fit network
    nn.fit(X_normalize, y_train)
    # predict on test set
    y_pred = nn.predict(X_test_norm)
    # calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print("accuracy: {:.2f}".format(accuracy))
