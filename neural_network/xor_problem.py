"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
https://en.wikipedia.org/wiki/Exclusive_or
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data import BatchIterator, encode_labels
from layers import Linear, Sigmoid
from loss import MSE
from network import NeuralNetwork
from optimizer import SGD
from train import train


def get_xor_dataset(random_state=43, size=1000):
    """
    create dataset for XOR problem with shape (size, 2)
    """
    X = np.random.RandomState(random_state).rand(size, 2)
    X = pd.DataFrame(X, columns=['a', 'b'])
    X = X - 0.5
    y = ((X['a'] > 0) ^ (X['b'] > 0)).astype(int)

    y_enc = encode_labels(y, 2)
    return X.values, y_enc


# set random seed
np.random.seed(37)
X, y = get_xor_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=17, stratify=y)

# create a network architecture
network = NeuralNetwork([
    Linear(input_size=2, output_size=4),
    Sigmoid(),
    Linear(input_size=4, output_size=2),
])

# train our network
train(network,
      X_train,
      y_train,
      num_epochs=300,
      iterator=BatchIterator(),
      loss=MSE(),
      optimizer=SGD(learning_rate=0.05),
      verbose=True)

# get prediction
train_accuracy = network.score(X_train, y_train)
test_accuracy = network.score(X_test, y_test)
print('Final score')
print(f'Train accuracy: {train_accuracy:.4f}')
print(f'Test  accuracy: {test_accuracy:.4f}')
