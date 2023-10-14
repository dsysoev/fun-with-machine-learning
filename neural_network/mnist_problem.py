"""
Solve a classic problem (MNIST dataset) with neural networks.
https://en.wikipedia.org/wiki/MNIST_database
"""

import numpy as np

from train import train
from network import NeuralNetwork
from layers import Linear, Sigmoid, Relu, Tanh, LeakyRelu, Dropout
from optimizer import SGD
from data import BatchIterator, load_mnist, encode_labels
from loss import BinaryCrossEntropy, MSE


# set random seed
np.random.seed(137)
# load mnist data
train_inputs, train_targets = load_mnist('data/', kind='train')
test_inputs, test_targets = load_mnist('data/', kind='t10k')
# normalize input values using min max scaler style
train_inputs_normalized = train_inputs / 255.
test_inputs_normalzed = test_inputs / 255.
print('train shape {}, test shape {}'.format(train_inputs.shape, test_inputs.shape))
# one hot encoding labels
train_targets_enc = encode_labels(train_targets, 10)
test_targets_enc = encode_labels(test_targets, 10)

# create a network architecture
# simple and fast
network = NeuralNetwork([
    Dropout(prob=0.2),
    Linear(input_size=784, output_size=32, batch_norm=False),
    Relu(),
    Linear(input_size=32, output_size=10, batch_norm=False),
], training=True)

# train our network
train(
    network,
    train_inputs_normalized,
    train_targets_enc,
    num_epochs=100,
    loss=MSE(),
    iterator=BatchIterator(batch_size=32),
    optimizer=SGD(learning_rate=0.001),
    verbose=True
    )


# get final score
train_accuracy = network.score(train_inputs_normalized, train_targets_enc)
test_accuracy = network.score(test_inputs_normalzed, test_targets_enc)
print('Final score')
print(f'Train accuracy: {train_accuracy:.4f}')
print(f'Test  accuracy: {test_accuracy:.4f}')
